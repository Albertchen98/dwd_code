# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from hydra.core.config_store import ConfigStore
import os
import torch
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTModel, DINOv3ViTImageProcessorFast
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import auto_docstring
from transformers.utils.generic import check_model_inputs
from typing import Callable, Optional, List
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from cosmos_transfer2._src.predict2.utils.context_parallel import broadcast
from einops import rearrange
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.imaginaire.utils import log
import numpy as np
from torchvision.models.vision_transformer import VisionTransformer
import random
from megatron.core import parallel_state
# DINOV3-L16
INTERMEDIATE_LAYER_INDEX_DINOV3L16 = [4, 11, 17, 23]  # Use the last layer by default
HEIGHT=704
WIDTH=1280
V2_HEIGHT=704 // 16 * 14
V2_WIDTH=1280 // 16 * 14
ALTERNATIVE_CHANNEL=[3, 8, 12, 16, 20, 24, 28, 32]

def match_distribution(h, h_vit, eps=1e-6):
    """
    Match h_vit distribution to h distribution.

    Args:
        h: [B, D1, N]   (DINO features)
        h_vit: [B, D2, N] (ViT features)
    """
    # Compute global mean and std for DINO features
    mean_h = h.mean(dim=(0, 2), keepdim=True)
    std_h = h.std(dim=(0, 2), keepdim=True)

    mean_h_scalar = mean_h.mean().detach()
    std_h_scalar = std_h.mean().detach()

    # Compute mean and std for ViT features
    mean_vit = h_vit.mean(dim=(0, 2), keepdim=True)
    std_vit = h_vit.std(dim=(0, 2), keepdim=True)

    mean_vit_scalar = mean_vit.mean().detach()
    std_vit_scalar = std_vit.mean().detach()

    # Normalize and re-scale
    h_vit_normed = (h_vit - mean_vit_scalar) / (std_vit_scalar + eps)
    h_vit_aligned = h_vit_normed * std_h_scalar + mean_h_scalar

    return h_vit_aligned

class DINOv3ViTModelRI(DINOv3ViTModel):
    """RI stands for return intermediate layers."""
    def __init__(self, config):
        super().__init__(config)
        # Keep the pretrained LayerNorm, including its learned weight and bias.

    # @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[List[int]] = -1,
        use_l2_norm: bool = False,
    ) -> List[torch.Tensor]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """

        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)
        hidden_states_return = []
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )
            if i in layer_idx:
                
                hidden_states_return.append(self.norm(hidden_states)[:, 5:])
        
        if use_l2_norm:
            hidden_states_return = [F.normalize(h, dim=-1) for h in hidden_states_return]
            # hidden_states_return.append(F.normalize(hidden_states[:, 5:]))
            
        return hidden_states_return


# 默认 HF 本地权重（DINOv2-L/14 with registers）；可通过 Hydra / DINOV3Encoder(checkpoint_dir=...) 覆盖
DINOV2_WITH_REGISTERS_LARGE_DEFAULT = (
    "/horizon-bucket/saturn_v_4dlabel/008_Simulation/001_users/conglang.zhang/"
    "huggingface/dinov2-with-registers-large"
)


class DINOv2WithRegistersModelRI(nn.Module):
    """HF DINOv2-with-registers，接口对齐 DINOv3ViTModelRI（多中间层 + 去 CLS/register 前缀 patch）。"""

    def __init__(
        self,
        checkpoint_dir: str,
        dtype=torch.bfloat16,
        attn_implementation: Optional[str] = None,
    ) -> None:
        super().__init__()
        _attn = attn_implementation or os.environ.get("DINOV2_ATTN_IMPLEMENTATION", "sdpa")
        self.backbone = AutoModel.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype,
            attn_implementation=_attn,
            trust_remote_code=True,
        )
        cfg = self.backbone.config
        n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.num_prefix_tokens = 1 + n_reg
        self.layernorm = self.backbone.layernorm

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[List[int]] = None,
        use_l2_norm: bool = False,
    ) -> List[torch.Tensor]:
        del bool_masked_pos, head_mask
        L = int(self.backbone.config.num_hidden_layers)
        if layer_idx is None:
            layer_idx = [L - 1]
        # -1 表示最后一层（与 HF hidden_states 下标一致）
        layer_idx = [(L - 1 if i == -1 else i) for i in layer_idx]
        pe = self.backbone.embeddings.patch_embeddings
        patch_w = pe.projection.weight if hasattr(pe, "projection") else pe.weight
        pixel_values = pixel_values.to(patch_w.dtype)
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hs = out.hidden_states
        assert hs is not None
        n = len(hs)
        if n == L + 1:
            take = lambda i: hs[i + 1]
        elif n == L:
            take = lambda i: hs[i]
        else:
            raise RuntimeError(
                f"DINOv2 hidden_states len={n} vs num_hidden_layers={L}; "
                "请检查 transformers / 模型版本。"
            )
        hidden_states_return = []
        for i in layer_idx:
            h = self.layernorm(take(i))
            hidden_states_return.append(h[:, self.num_prefix_tokens :, :])
        if use_l2_norm:
            hidden_states_return = [F.normalize(h, dim=-1) for h in hidden_states_return]
        return hidden_states_return


class DINOV3Encoder(nn.Module):
    def __init__(
        self,
        checkpoint_dir: str,
        offload_model_to_cpu: bool=False,
        device="cuda" if torch.cuda.is_available() else "cpu",  # noqa: B008
        out_layers=[23], 
        use_l2_norm=False,
        dtype=torch.bfloat16,
        use_anyup=False,
        anyup_scale=None, #"Anyup upsampling factor(w.r.t original latent resolution 44x80) must be set."
        use_dino_pca=False, 
        pca_mean_path=None,
        pca_comp_path=None,
        use_processor=False,
        use_random_channel=False,
        max_channel=None,
        use_instance_mean=False,
        height=HEIGHT,
        width=WIDTH,
        dino_version: str = "v3",
        patch_size: Optional[int] = None,
        attn_implementation: Optional[str] = None,
        # 0：不分块；>0：pixel batch 维 (B*T) 按该大小切块跑 backbone，降低 eager attention 峰值显存。
        # 也可在训练前 export DINOV2_FORWARD_CHUNK_SIZE=4（会覆盖此处 0）。
        forward_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.offload_model = offload_model_to_cpu
        self.device = device
        self.dtype = dtype
        self.forward_chunk_size = int(forward_chunk_size)
        self.dino_version = (dino_version or "v3").strip().lower()
        if self.dino_version == "v2":
            self.patch_size = 14 if patch_size is None else int(patch_size)
            _attn_v2 = attn_implementation or os.environ.get("DINOV2_ATTN_IMPLEMENTATION", "sdpa")
            self.model = DINOv2WithRegistersModelRI(
                checkpoint_dir, dtype=self.dtype, attn_implementation=_attn_v2
            )
            self.processor = AutoImageProcessor.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
                size={"height": height, "width": width},
                do_resize=True,
                do_rescale=True,
                do_normalize=True,
                do_center_crop=False,
            )
        elif self.dino_version == "v3":
            self.patch_size = 16 if patch_size is None else int(patch_size)
            _attn_v3 = attn_implementation or os.environ.get(
                "DINOV3_ATTN_IMPLEMENTATION", "flash_attention_2"
            )
            self.model = DINOv3ViTModelRI.from_pretrained(
                checkpoint_dir, attn_implementation=_attn_v3, dtype=self.dtype
            )
            self.processor = DINOv3ViTImageProcessorFast(
                resample=PILImageResampling.BILINEAR,
                image_mean=IMAGENET_DEFAULT_MEAN,
                image_std=IMAGENET_DEFAULT_STD,
                do_resize=True,
                do_rescale=True,
                do_normalize=True,
                size={"height": height, "width": width},
            )
        else:
            raise ValueError(f"dino_version must be 'v3' or 'v2', got {dino_version!r}")
        self.height = height
        self.width = width
        # self.model.to(self.device, dtype=self.dtype).eval()
        self.intermediate_layer_idx = out_layers
        self.use_l2_norm=use_l2_norm
        self.use_processor = use_processor
        self.use_random_channel = use_random_channel
        self.max_channel = max_channel
        self.use_instance_mean = use_instance_mean # <--- [修改2] 保存参数
        self.model.eval().requires_grad_(False)
        if not self.offload_model:
            self.model = self.model.to(self.device)
            
        self.use_dino_pca = use_dino_pca
        if self.use_dino_pca:
            # register as buffers so they move with .to() and are replicated across DataParallel devices
            pca_mean = torch.from_numpy(np.load(pca_mean_path)).to(device=self.device, dtype=self.dtype)
            pca_comp = torch.from_numpy(np.load(pca_comp_path)).to(device=self.device, dtype=self.dtype)
            self.register_buffer('pca_mean', pca_mean)
            self.register_buffer('pca_comp', pca_comp)
            
        self.use_anyup = use_anyup
        self.anyup_scale = anyup_scale
        if self.use_anyup:
            assert self.use_dino_pca, "AnyUp requires DINO PCA features."
            assert self.anyup_scale, "Anyup upsampling factor(w.r.t original latent resolution 44x80) must be set."
            from anyup.anyup.model import AnyUp
            from anyup.anyup.layers import setup_cross_attention_block
            
            self.anyup = AnyUp().to(device=device, dtype=self.dtype)
            model_dict = torch.load("./model_hubs/anyup/anyup_multi_backbone.pth", map_location="cpu")
            self.anyup.load_state_dict(model_dict)
            # FIXME 这里会导致出错
            # self.anyup.cross_decode = setup_cross_attention_block(
            #     use_natten=True,
            #     qk_dim=self.anyup.cross_decode.cross_attn.attention.embed_dim,
            #     num_heads=4,
            #     window_ratio=self.anyup.cross_decode.window_ratio,
            #     use_params_from=self.anyup.cross_decode,
            # ).to(device=device, dtype=self.dtype)
            
            

    @torch.inference_mode()
    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        """Encode an image into a feature vector.
        input_img = rearrange(input_img, "b c t h w -> b t h w c")
        """
        
        if self.offload_model:
            self.model.to(self.device)
        b, c, t, h, w = input_img.shape
        # log.info(f"input_img.shape is {input_img.shape}")
        # log.info(f"input_img has the range from {input_img.min()} to {input_img.max()}")
        # input_img = rearrange(input_img, "b c t h w -> (b t) h w c")
        input_img = rearrange(input_img, "b c t h w -> (b t) c h w")
        with torch.no_grad():
            # inputs = self.processor(images=input_img, return_tensors="pt").to(self.device, dtype=self.dtype)
            # image_features = self.model(**inputs).last_hidden_state[:,5:]
            # image_features = self.model(**inputs, layer_idx=self.intermediate_layer_idx, use_l2_norm=self.use_l2_norm)
            if self.use_processor:
                # HF image preprocess uses numpy; bfloat16/float16 tensors cannot be converted to numpy.
                _proc_in = input_img.detach().float()
                inputs = self.processor(images=_proc_in, return_tensors="pt").to(self.device, dtype=self.dtype)
                image_features = self.model(**inputs, layer_idx=self.intermediate_layer_idx, use_l2_norm=self.use_l2_norm)
            else:
                # 插值到指定大小
                input_img = F.interpolate(
                    input_img,
                    size=(self.height, self.width),
                    mode='bilinear',
                    align_corners=False
                )

                chunk_sz = self.forward_chunk_size
                if chunk_sz <= 0:
                    _ev = os.environ.get("DINOV2_FORWARD_CHUNK_SIZE", "").strip()
                    chunk_sz = int(_ev) if _ev else 0
                bt = int(input_img.shape[0])
                if chunk_sz > 0 and bt > chunk_sz:
                    parts = []
                    for s in range(0, bt, chunk_sz):
                        sub = input_img[s : s + chunk_sz]
                        parts.append(
                            self.model(
                                pixel_values=sub,
                                layer_idx=self.intermediate_layer_idx,
                                use_l2_norm=self.use_l2_norm,
                            )
                        )
                    n_layers = len(parts[0])
                    image_features = [
                        torch.cat([parts[k][li] for k in range(len(parts))], dim=0) for li in range(n_layers)
                    ]
                else:
                    image_features = self.model(
                        pixel_values=input_img,
                        layer_idx=self.intermediate_layer_idx,
                        use_l2_norm=self.use_l2_norm,
                    )
            assert isinstance(image_features, list)
            image_features = torch.cat(image_features, dim=-1)
            image_features = image_features.reshape(
                image_features.shape[0],
                self.height // self.patch_size,
                self.width // self.patch_size,
                -1,
            )
            image_features = rearrange(image_features, "(b t) h w c -> b c t h w", b=b, t=t)
            # image_features = [feat.detach() for feat in image_features]
        if self.offload_model:
            self.model.to("cpu")
        # log.info(f"returned dinov3 feature maps with length: {len(image_features)}")
        if self.use_anyup:
            anyup_h = self.anyup_scale * (self.height // self.patch_size)
            anyup_w = self.anyup_scale * (self.width // self.patch_size)
            input_img = F.interpolate(
                        input_img, 
                        size=(anyup_h, anyup_w), 
                        mode='bilinear',     # 常用双线性插值，也可以用 'nearest' (更简单快点) 或 'bicubic'
                        align_corners=False  # 默认设置为 False 即可
                    )

            image_features = rearrange(image_features, "b c t h w -> (b t) c h w ", b=b, t=t)
            image_features = self.anyup(input_img, image_features, q_chunk_size=1024)
            image_features = rearrange(image_features, "(b t) c h w -> b c t h w", b=b, t=t)

        if self.use_dino_pca:
            image_features = rearrange(image_features, "b c t h w -> b t h w c")
            if self.use_instance_mean:
                # 使用当前 Batch 数据的均值 (B, T, H, W 维度求均值，保留 C)
                image_features = image_features - image_features.mean(dim=(0, 1, 2, 3), keepdim=True)
            else:
                # 使用预存的 PCA 均值
                image_features = image_features - self.pca_mean[None,None,None,None]
            # image_features = image_features - self.pca_mean[None,None,None,None]
            image_features = image_features @ self.pca_comp.T
            image_features = rearrange(image_features, "b t h w c -> b c t h w")
            if self.use_random_channel and self.training:
                random_channel = random.choice(ALTERNATIVE_CHANNEL)
                # broadcast random_channel
                if parallel_state.is_initialized():
                    cp_group = parallel_state.get_context_parallel_group()
                    random_channel = broadcast(random_channel, cp_group)
                # print(f"random_channel:{random_channel}, cp_group:{cp_group}")
                image_features[:, random_channel:] = 0
        if self.max_channel is not None:
            image_features[:, self.max_channel:] = 0 
        return image_features

DinoV3Config: LazyDict = L(DINOV3Encoder)(checkpoint_dir="./model_hubs/dinov3-vitl16", use_dino_pca=False, 
                                          pca_mean_path=None,
                                          pca_comp_path=None,)

# DINOv2-L/14 with registers（本地 HF 目录）；训练时可在 YAML 覆盖 checkpoint_dir / height / width / out_layers
# 默认用 V2_HEIGHT×V2_WIDTH（616×1120）：与 720p9:16 下 DiT latent 空间 44×80 对齐（1120/14=80）。
# 若误用 704×1280，则 W 方向 token 为 1280//14=91，会在 control 分支与主分支相加时报错。
DinoV2WithRegistersConfig: LazyDict = L(DINOV3Encoder)(
    dino_version="v2",
    checkpoint_dir=DINOV2_WITH_REGISTERS_LARGE_DEFAULT,
    patch_size=14,
    height=V2_HEIGHT,
    width=V2_WIDTH,
    use_dino_pca=False,
    pca_mean_path=None,
    pca_comp_path=None,
)

def register_dinov3_encoder():
    cs = ConfigStore.instance()
    cs.store(
        group="dinov3_encoder",
        package="model.config.dinov3_encoder",
        name="dinov3_vitl16",
        node=DinoV3Config,
    )
    cs.store(
        group="dinov3_encoder",
        package="model.config.dinov3_encoder",
        name="dinov2_with_registers_large",
        node=DinoV2WithRegistersConfig,
    )
   
@dataclass  
class VITConfig:
    """
    Configuration class for Vision Transformer.
    字段与 VisionTransformer.__init__ 参数一一对应。
    """
    # 必填参数 (无默认值)
    image_size: int
    patch_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    output_dim: int
    

class ExtraVIT(VisionTransformer):
    def __init__(self,
                 checkpoint_dir,
                 image_size, 
                 patch_size,
                 num_layers,
                 num_heads,
                 hidden_dim,
                 mlp_dim,
                 output_dim):
        super().__init__(image_size, 
                 patch_size,
                 num_layers,
                 num_heads,
                 hidden_dim,
                 mlp_dim)
        
        self.heads = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.init_from_ckpt(checkpoint_dir)
        
    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(sd, strict=False)
    
    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[1]

        # Add class token
        batch_size = x.shape[0]
        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Pass through Transformer encoder
        x = self.encoder(x)

        # Remove class token, keep patch tokens only
        x = x[:, 1:, :]  # shape: (B, 256, 384)

        # Apply head projection
        x = self.heads(x)  # shape: (B, 256, 8)

        # Transpose to (B, 8, 256)
        return x.transpose(1, 2)
    
class SVGEncoder(nn.Module):
    def __init__(
        self,
        dino_checkpoint_dir: str,
        vit_checkpoint_dir: str,
        offload_model_to_cpu: bool=False,
        device="cuda" if torch.cuda.is_available() else "cpu",  # noqa: B008
        dtype=torch.bfloat16,
        mask_ratio=0,
        use_outnorm=True,
        vit_config: VITConfig = None,
    ) -> None:
        super().__init__()
        # self.dino_checkpoint_dir = dino_checkpoint_dir
        self.offload_model = offload_model_to_cpu
        self.device = device
        self.dtype = dtype
        self.mask_ratio = mask_ratio
        self.use_outnorm = use_outnorm
        self.dinov3_model = DINOv3ViTModel.from_pretrained(dino_checkpoint_dir, attn_implementation="flash_attention_2", dtype=self.dtype)
        self.processor = DINOv3ViTImageProcessorFast(
            resample=PILImageResampling.BILINEAR,
            image_mean=IMAGENET_DEFAULT_MEAN,
            image_std=IMAGENET_DEFAULT_STD,
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            size={"height":HEIGHT, "width":WIDTH}
        )
        
        vit_config_dict = asdict(vit_config)
        self.vit = ExtraVIT(vit_checkpoint_dir, **vit_config_dict)
            
    @torch.inference_mode()
    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        """Encode an image into a feature vector.
        input_img = rearrange(input_img, "b c t h w -> b t h w c")
        """
        
        # if self.offload_model:
        #     self.model.to(self.device)
        b, c, t, h, w = input_img.shape
        # log.info(f"input_img.shape is {input_img.shape}")
        # log.info(f"input_img has the range from {input_img.min()} to {input_img.max()}")
        input_img = rearrange(input_img, "b c t h w -> (b t) h w c")
        with torch.no_grad():
            _proc_in = input_img.detach().float()
            inputs = self.processor(images=_proc_in, return_tensors="pt").to(self.device, dtype=self.dtype)
            # image_features = self.model(**inputs).last_hidden_state[:,5:]
            h = self.model(**inputs)
            h_vit = self.vit(inputs["pixel_values"])
            
            if self.training and self.mask_ratio > 0:
                B, D, N = h_vit.shape
                mask_flags = (torch.rand(B, device=self.device) < self.mask_ratio).float().view(B, 1, 1)
                mask_token_exp = self.mask_token.expand(B, D, N)
                h_vit = h_vit * (1 - mask_flags) + mask_token_exp * mask_flags

            if self.use_outnorm:
                h_vit = match_distribution(h, h_vit)
            
            h = torch.cat([h, h_vit], dim=1)
            
        h = h.view(h.shape[0], -1, int(h // 16), int(w // 16)).contiguous()
        return h
            # image_features = [feat.detach() for feat in image_features]
        # if self.offload_model:
        #     self.model.to("cpu")
        # log.info(f"returned dinov3 feature maps with length: {len(image_features)}")
        # if self.use_dino_pca:
        #     image_features = rearrange(image_features, "b c t h w -> b t h w c")
        #     image_features = image_features - self.pca_mean[None,None,None,None]
        #     image_features = image_features @ self.pca_comp.T
        #     image_features = rearrange(image_features, "b t h w c -> b c t h w")
        # return image_features





if __name__ == "__main__":
    from scripts.extract_dino_features import main
    main()
