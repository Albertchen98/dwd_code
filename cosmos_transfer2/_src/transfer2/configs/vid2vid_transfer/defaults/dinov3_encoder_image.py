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

from hydra.core.config_store import ConfigStore
import os
import torch
from transformers import DINOv3ViTModel, DINOv3ViTImageProcessorFast
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import auto_docstring
from transformers.utils.generic import check_model_inputs
from typing import Callable, Optional, List
from torch import nn
import transformer_engine as te
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
from PIL import Image
# DINOV3-L16
INTERMEDIATE_LAYER_INDEX_DINOV3L16 = [4, 11, 17, 23]  # Use the last layer by default
HEIGHT=704
WIDTH=1280
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
        # Retain the pretrained LayerNorm and its learned affine parameters.

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
    ) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.offload_model = offload_model_to_cpu
        self.device = device
        self.dtype = dtype
        self.model = DINOv3ViTModelRI.from_pretrained(checkpoint_dir, attn_implementation="flash_attention_2", dtype=self.dtype)
        self.processor = DINOv3ViTImageProcessorFast(
            resample=PILImageResampling.BILINEAR,
            image_mean=IMAGENET_DEFAULT_MEAN,
            image_std=IMAGENET_DEFAULT_STD,
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            size={"height":height, "width":width}
        )
        self.height = height
        self.width = width
        # self.model.to(self.device, dtype=self.dtype).eval()
        self.intermediate_layer_idx = out_layers
        self.use_l2_norm=use_l2_norm
        self.use_processor = use_processor
        self.use_random_channel = use_random_channel
        self.max_channel = max_channel
        self.use_instance_mean = use_instance_mean # <--- [修改2] 保存参数
        self.model.eval()
        if not self.offload_model:
            self.model = self.model.to("cuda")
            
        self.use_dino_pca = use_dino_pca
        if self.use_dino_pca:
            # register as buffers so they move with .to() and are replicated across DataParallel devices
            pca_mean = torch.from_numpy(np.load(pca_mean_path)).to(dtype=self.dtype)
            pca_comp = torch.from_numpy(np.load(pca_comp_path)).to(dtype=self.dtype)
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
            self.model.to("cuda")
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
                inputs = self.processor(images=input_img, return_tensors="pt").to(self.device, dtype=self.dtype)
                image_features = self.model(**inputs, layer_idx=self.intermediate_layer_idx, use_l2_norm=self.use_l2_norm)
            else:
                # 插值到指定大小
                input_img = F.interpolate(
                    input_img,
                    size=(self.height, self.width),
                    mode='bilinear',
                    align_corners=False
                )
                
                image_features = self.model(pixel_values=input_img, layer_idx=self.intermediate_layer_idx, use_l2_norm=self.use_l2_norm)
            assert isinstance(image_features, list)
            image_features = torch.cat(image_features, dim=-1)
            image_features = image_features.reshape(image_features.shape[0], self.height//16, self.width//16, -1)
            image_features = rearrange(image_features, "(b t) h w c -> b c t h w", b=b, t=t)
            # image_features = [feat.detach() for feat in image_features]
        if self.offload_model:
            self.model.to("cpu")
        # log.info(f"returned dinov3 feature maps with length: {len(image_features)}")
        if self.use_anyup:
            anyup_h, anyup_w = self.anyup_scale * HEIGHT//16, self.anyup_scale * WIDTH//16
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
            if self.use_random_channel:
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

def register_dinov3_encoder():
    cs = ConfigStore.instance()
    cs.store(
        group="dinov3_encoder",
        package="model.config.dinov3_encoder",
        name="dinov3_vitl16",
        node=DinoV3Config,
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
        #     self.model.to("cuda")
        b, c, t, h, w = input_img.shape
        # log.info(f"input_img.shape is {input_img.shape}")
        # log.info(f"input_img has the range from {input_img.min()} to {input_img.max()}")
        input_img = rearrange(input_img, "b c t h w -> (b t) h w c")
        with torch.no_grad():
            inputs = self.processor(images=input_img, return_tensors="pt").to(self.device, dtype=self.dtype)
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

import os
import torch
import ray
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2

# 请确保这些引入在你的环境中是可用的，或者替换为你实际的项目路径
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess

# 假设的常量，请根据实际情况修改
@ray.remote(num_gpus=1)
class ImageEncoderActor:
    def __init__(self, encoder_config):
        self.device = torch.device("cuda")
        # 假设 DINOV3Encoder 在上下文中可用
        self.encoder = DINOV3Encoder(**encoder_config).to(self.device)
        self.encoder.eval()
        
        self.height = encoder_config["height"]
        self.width = encoder_config["width"]

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True), 
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ResizePreprocess(tuple([self.height, self.width])) 
        ])

    def process_image_list(self, image_files, input_root, vis_root=None, batch_size=8):
        """
        返回: List[torch.Tensor]
        每个 Tensor 的 shape 为 (C, H, W)，位于 CPU 上。
        """
        results_list = [] # 用于在内存中收集特征
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i : i + batch_size]
            batch_tensors = []
            current_batch_names = [] # 用于可视化命名

            # 1. 加载并预处理
            for img_name in batch_files:
                img_path = os.path.join(input_root, img_name)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        tensor = self.transform(img) # (C, H, W)
                        batch_tensors.append(tensor)
                        current_batch_names.append(img_name)
                except Exception as e:
                    print(f"Error loading {img_name}: {e}")

            if not batch_tensors:
                continue

            # 2. 构造 Batch & 推理
            input_batch = torch.stack(batch_tensors).to(self.device) # (B, C, H, W)
            input_batch = input_batch.unsqueeze(2) # (B, C, 1, H, W) 适配模型

            try:
                with torch.no_grad():
                    # Output: (B, C_out, T, H_out, W_out)
                    # 这里的 T 通常是 1
                    features = self.encoder(input_batch)
                    features = features.squeeze(2) # (B, C_out, H_out, W_out)
                    
                    # 关键步骤：移动到 CPU，否则显存会爆，且 Ray 传输需要 CPU 对象
                    features = features.cpu()

                # 3. 收集结果 & 可选可视化
                for idx, feat in enumerate(features):
                    # A. 存入列表 (用于返回)
                    results_list.append(feat) # feat shape: (C, H, W)

                    # B. 可视化 (可选，生成 png)
                    if vis_root:
                        base_name = os.path.splitext(current_batch_names[idx])[0]
                        vis_save_path = os.path.join(vis_root, base_name + '.png')
                        
                        # 取前3个通道做 PCA 可视化
                        pca_3 = feat[:3, :, :].float()
                        v_min, v_max = pca_3.min(), pca_3.max()
                        if v_max - v_min > 1e-6:
                            pca_3 = (pca_3 - v_min) / (v_max - v_min)
                        else:
                            pca_3.fill_(0)
                        
                        pca_img = (pca_3.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(pca_img).save(vis_save_path)

            except Exception as e:
                print(f"Error during inference batch: {e}")
                
        # 返回收集到的所有特征张量列表
        return results_list

def main():
    ray.init()
    
    # --- 配置 ---
    input_folder = "my_experiments/exp5_upscale_gaussian/1120_DV_x4_gaussian_k17_x4_gaussian_k17"
    
    # 输出的单个 PT 文件路径
    output_pt_path = input_folder + "/all_features_BCTHW.pt"
    
    vis_folder = input_folder + "/vis" 
    
    use_l2_norm = False
    highres_scale = 1
    gpu_batch_size = 8
    
    if vis_folder:
        os.makedirs(vis_folder, exist_ok=True)
    
    # 收集图片
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    image_files.sort() # 确保时间顺序正确！
    image_files = image_files[:10] # 限制处理数量
    
    total_files = len(image_files)
    print(f"Found {total_files} images. Processing to create one BCTHW tensor.")

    # 模型配置
    global HEIGHT, WIDTH 
    HEIGHT = 704 
    WIDTH = 1280
    
    encoder_config = {
        "checkpoint_dir": "./model_hubs/dinov3-vitl16",
        "use_dino_pca": True,
        "pca_mean_path": "patch_pca_mean_32.npy",
        "pca_comp_path": "patch_pca_components_32.npy",
        "use_anyup": False,
        "anyup_scale": 8,
        "offload_model_to_cpu": False,
        "out_layers": [23],
        "dtype": torch.bfloat16,
        "use_l2_norm": use_l2_norm,
        "height": HEIGHT * highres_scale,
        "width": WIDTH * highres_scale,
    }

    # 创建 Actors
    num_gpus = torch.cuda.device_count()
    actors = [ImageEncoderActor.remote(encoder_config) for _ in range(num_gpus)]
    
    # 分发任务
    chunk_size = math.ceil(total_files / num_gpus)
    futures = []
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        if start_idx >= total_files:
            break
            
        actor_files = image_files[start_idx:end_idx]
        actor = actors[i]
        
        # 注意：不再传入 output_root，只传入 vis_root
        futures.append(
            actor.process_image_list.remote(
                actor_files, 
                input_folder, 
                vis_root=vis_folder, 
                batch_size=gpu_batch_size
            )
        )

    print("Processing started on GPUs...")
    
    # 1. 获取结果
    # nested_results 结构: [[tensor1, tensor2...], [tensorN, tensorN+1...]]
    # Ray 保证返回列表的顺序与 futures 列表顺序一致，所以时间顺序是安全的
    nested_results = ray.get(futures)
    
    print("Inference done. Aggregating tensors...")

    # 2. 展平列表
    all_features = [feat for chunk in nested_results for feat in chunk]
    
    if not all_features:
        print("No features extracted.")
        ray.shutdown()
        return

    # 3. 堆叠 (Stack) -> (T, C, H, W)
    # T = 图片数量
    try:
        tensor_TCHW = torch.stack(all_features)
        print(f"Stacked shape (TCHW): {tensor_TCHW.shape}")

        # 4. 维度变换 -> (B, C, T, H, W)
        # 假设只有一个 batch (即这一组图片是一个视频)
        # permute(1, 0, 2, 3): (T, C, H, W) -> (C, T, H, W)
        tensor_CTHW = tensor_TCHW.permute(1, 0, 2, 3)
        
        # unsqueeze(0): 增加 Batch 维 -> (1, C, T, H, W)
        final_tensor = tensor_CTHW.unsqueeze(0)
        
        print(f"Final shape (BCTHW): {final_tensor.shape}")
        
        # 5. 保存
        torch.save(final_tensor, output_pt_path)
        print(f"Saved successfully to: {output_pt_path}")

    except RuntimeError as e:
        print(f"Error during stacking/saving (possible OOM if dataset is huge): {e}")

    ray.shutdown()

if __name__ == "__main__":
    main()