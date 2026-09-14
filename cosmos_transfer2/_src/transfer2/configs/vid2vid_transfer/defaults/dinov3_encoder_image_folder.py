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
ALTERNATIVE_CHANNEL=[4, 8, 12, 16, 20, 24, 28, 32]

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
from tqdm import tqdm
from decord import VideoReader, cpu
from torchvision.transforms import v2
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess
@ray.remote(num_gpus=1)
class VideoEncoderActor:
    def __init__(self, encoder_config):
        # 在 Actor 初始化时加载模型到对应的 GPU
        self.device = torch.device("cuda")
        self.encoder = DINOV3Encoder(**encoder_config).to(self.device)
        self.encoder.eval()
        
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True), 
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ResizePreprocess(tuple([HEIGHT, WIDTH]))
        ])

    def process_video(self, video_path, feature_path, batch_size=4, max_frames=None):
        if os.path.exists(feature_path):
            print(f"Skipped {video_path}")
            return
        print(f">>> Start processing: {video_path}")
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
            n_frames = len(vr)
            
            # 限制处理帧数
            if max_frames is not None:
                n_frames = min(n_frames, max_frames)
                print(f"Processing only first {n_frames} frames")
            
            all_features = []
            
            # 按 batch_size 串行处理视频帧
            for i in tqdm(range(0, n_frames, batch_size)):
                end_idx = min(i + batch_size, n_frames)
                frame_ids = list(range(i, end_idx))
                
                # 加载并预处理
                frames = vr.get_batch(frame_ids).asnumpy()
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (B, C, H, W)
                
                # 应用 Transform (注意：transform 内部需处理好维度)
                input_tensor = self.transform(frames).to(self.device) # (B, C, H, W)
                
                # 调整为模型需要的维度 (1, C, B, H, W) -> 模拟 T 维度
                input_tensor = input_tensor.permute(1, 0, 2, 3).unsqueeze(0) 

                with torch.no_grad():
                    # 推理
                    feat = self.encoder(input_tensor) # (1, C_out, B_out, H_out, W_out)
                    all_features.append(feat.cpu())

            # 沿时间维度(dim=2)拼接
            output = torch.cat(all_features, dim=2)
            print(f"save feature to {feature_path}")
            torch.save(output, feature_path)
            
            del all_features, output
            torch.cuda.empty_cache()
            print(f"Processed {video_path}")
        
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")


def main():
    # 1. 初始化 Ray - 明确指定 GPU 数量
    import subprocess
    
    # 获取 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible:
        num_gpus = len(cuda_visible.split(','))
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible}, using {num_gpus} GPUs")
    else:
        num_gpus = torch.cuda.device_count()
        print(f"Using all {num_gpus} available GPUs")
    
    # 初始化 Ray 时指定 GPU 数量
    ray.init(num_gpus=num_gpus)
    
    use_l2_norm = False
    video_folder = "test_videos/1120_DV"
    feature_folder = "test_videos/1120_DV"
    highres_scale = 4
    max_frames = 93  # 只处理前93帧
    os.makedirs(feature_folder, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos")
    # 2. 准备模型配置
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

    # 3. 创建 Actor 池 - 使用检测到的 GPU 数量
    num_gpus = torch.cuda.device_count()
    actors = [VideoEncoderActor.remote(encoder_config) for _ in range(num_gpus)]
    
    # 4. 任务分发
    results = []
    for i, video_name in enumerate(video_files):
        video_path = os.path.join(video_folder, video_name)
        feature_path = os.path.join(feature_folder, video_name.replace('.mp4', '.pt'))
        
        # 轮询分配给 Actor
        actor = actors[i % num_gpus]
        results.append(actor.process_video.remote(video_path, feature_path, batch_size=4, max_frames=max_frames))

    # 5. 等待所有任务完成并显示进度条
    for _ in tqdm(range(len(results)), desc="Processing Videos"):
        done, results = ray.wait(results)
        # 可以 print(ray.get(done[0])) 来查看具体成功与否

    ray.shutdown()

@ray.remote(num_gpus=1)
class FrameEncoderActor:
    """单帧处理 Actor，支持分割图引导滤波"""
    def __init__(self, encoder_config, guided_filter_radius=2, guided_filter_eps=1e-5):
        import kornia
        self.device = torch.device("cuda")
        self.encoder = DINOV3Encoder(**encoder_config).to(self.device).eval()
        self.guided_filter_radius = guided_filter_radius
        self.guided_filter_eps = guided_filter_eps
        
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        
    def process_single_frame(self, frame_path, seg_path, highres_scale):
        """处理单帧图像"""
        import kornia
        from PIL import Image
        
        # 1. 读取原图
        frame_pil = Image.open(frame_path).convert("RGB")
        frame_tensor = self.transform(frame_pil).unsqueeze(0).unsqueeze(2).to(self.device)
        
        # Resize
        frame_tensor = F.interpolate(
            frame_tensor.squeeze(2),
            size=(HEIGHT * highres_scale, WIDTH * highres_scale),
            mode='bilinear',
            align_corners=False
        ).unsqueeze(2)
        
        # 2. 提取特征
        with torch.no_grad():
            feat_raw = self.encoder(frame_tensor)
        
        # 3. 应用导向滤波（如果有分割图）
        if seg_path and os.path.exists(seg_path):
            feat_h, feat_w = feat_raw.shape[3], feat_raw.shape[4]
            
            # 读取分割图
            seg_pil = Image.open(seg_path).convert("RGB")
            seg_np = np.array(seg_pil)
            seg_tensor = torch.from_numpy(seg_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            
            # Resize 到特征图尺寸
            seg_resized_4d = F.interpolate(
                seg_tensor,
                size=(feat_h, feat_w),
                mode='bilinear',
                align_corners=False
            )
            seg_guide_gray = kornia.color.rgb_to_grayscale(seg_resized_4d)
            
            # 提取特征并滤波
            feat_4d = feat_raw[0, :, 0, :, :].unsqueeze(0).float()  # [1, 32, H, W]
            
            kernel_size = self.guided_filter_radius * 2 + 1
            filtered_channels = []
            for i in range(feat_4d.shape[1]):
                src_c = feat_4d[:, i:i+1, :, :]
                filtered_c = kornia.filters.guided_blur(
                    guidance=seg_guide_gray,
                    input=src_c,
                    kernel_size=(kernel_size, kernel_size),
                    eps=self.guided_filter_eps
                )
                filtered_channels.append(filtered_c)
            
            feat_filtered = torch.cat(filtered_channels, dim=1).unsqueeze(2)  # [1, 32, 1, H, W]
            return feat_filtered.cpu()
        else:
            return feat_raw.cpu()

def process_frames_with_seg_guided_filter():
    """
    处理整个文件夹的图像,使用分割图引导滤波后提取 DINOv3 特征。
    输出: [1, 32, T, 176, 320] 其中 T 为图片数量
    """
    import glob
    from pathlib import Path
    from tqdm import tqdm
    
    # ============ 配置区 ============
    frames_dir = "/home/xuyang/chuanheng/cosmos-transfer2.5-20260104/test_videos/1120_DV/frames_png"
    seg_dir = "/home/xuyang/chuanheng/cosmos-transfer2.5-20260104/test_videos/1120_DV/inst_seg"
    output_path = "/home/xuyang/chuanheng/cosmos-transfer2.5-20260104/test_videos/1120_DV/dino_features_seg_guided.pt"
    
    # 模型配置
    highres_scale = 4
    pca_dim = 32
    guided_filter_radius = 2
    guided_filter_eps = 1e-5
    max_frames = 93  # 只处理前93帧
    
    # ============ 初始化 Ray ============
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible:
        num_gpus = len(cuda_visible.split(','))
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible}, using {num_gpus} GPUs")
    else:
        num_gpus = torch.cuda.device_count()
        print(f"Using all {num_gpus} available GPUs")
    
    ray.init(num_gpus=num_gpus)
    
    # ============ 准备数据 ============
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))[:max_frames]
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No PNG files found in {frames_dir}")
    
    print(f"Processing {len(frame_files)} frames (limited to {max_frames})")
    
    # ============ 创建 Actor 池 ============
    encoder_config = {
        "checkpoint_dir": "./model_hubs/dinov3-vitl16",
        "use_dino_pca": True,
        "pca_mean_path": "patch_pca_mean_32.npy",
        "pca_comp_path": "patch_pca_components_32.npy",
        "use_anyup": False,
        "offload_model_to_cpu": False,
        "out_layers": [23],
        "use_processor": False,
        "dtype": torch.bfloat16,
        "height": HEIGHT * highres_scale,
        "width": WIDTH * highres_scale,
    }
    
    actors = [
        FrameEncoderActor.remote(encoder_config, guided_filter_radius, guided_filter_eps) 
        for _ in range(num_gpus)
    ]
    
    # ============ 分配任务（按顺序提交并记录） ============
    task_refs = []  # 按顺序存储 ray.ObjectRef
    
    for i, frame_path in enumerate(frame_files):
        frame_name = Path(frame_path).stem
        frame_num = frame_name.split('_')[-1]
        seg_name = f"{frame_num.zfill(8)}_inst_seg_vis.png"
        seg_path = os.path.join(seg_dir, seg_name)
        
        actor = actors[i % num_gpus]
        task_ref = actor.process_single_frame.remote(frame_path, seg_path, highres_scale)
        task_refs.append(task_ref)
    
    # ============ 按提交顺序收集结果 ============
    print("Collecting results in submission order...")
    all_features = []
    for task_ref in tqdm(task_refs, desc="Processing frames"):
        feat = ray.get(task_ref)  # 阻塞等待当前任务完成
        all_features.append(feat)
    
    ray.shutdown()
    
    # ============ 拼接并保存 ============
    output_tensor = torch.cat(all_features, dim=2)
    
    print(f"Final feature shape: {output_tensor.shape}")
    print(f"Saving to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_tensor, output_path)
    
    print("Done!")

if __name__ == "__main__":
    process_frames_with_seg_guided_filter()