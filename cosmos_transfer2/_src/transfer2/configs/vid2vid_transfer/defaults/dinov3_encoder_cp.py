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
# DINOV3-L16
INTERMEDIATE_LAYER_INDEX_DINOV3L16 = [4, 11, 17, 23]  # Use the last layer by default
HEIGHT=704*4
WIDTH=1280*4
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
                image_features = self.model(pixel_values=input_img, layer_idx=self.intermediate_layer_idx, use_l2_norm=self.use_l2_norm)
            assert isinstance(image_features, list)
            image_features = torch.cat(image_features, dim=-1)
            image_features = image_features.reshape(image_features.shape[0], HEIGHT//16, WIDTH//16, -1)
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
# --- 辅助类：用于处理视频切片 ---
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess

@ray.remote(num_gpus=1)
class VideoChunkEncoderActor:
    def __init__(self, encoder_config):
        self.device = torch.device("cuda")
        # 初始化模型
        self.encoder = DINOV3Encoder(**encoder_config).to(self.device)
        self.encoder.eval()
        
        # 预处理流程
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True), 
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ResizePreprocess(tuple([encoder_config["height"], encoder_config["width"]]))
        ])

    def process_chunk(self, video_path, start_idx, end_idx, batch_size=4):
        """
        处理视频的一个片段 [start_idx, end_idx)
        """
        try:
            # 使用 decord 读取视频
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
            
            chunk_features = []
            
            # 在分配的片段内进行 batch 处理
            # 注意：这里的 range 是针对当前分片的局部循环
            for i in range(start_idx, end_idx, batch_size):
                # 计算当前 batch 的结束位置，不能超过分片的 end_idx
                current_batch_end = min(i + batch_size, end_idx)
                frame_ids = list(range(i, current_batch_end))
                
                if not frame_ids:
                    break
                
                # 1. 加载数据
                frames = vr.get_batch(frame_ids).asnumpy()
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (B, C, H, W)
                
                # 2. 预处理
                input_tensor = self.transform(frames).to(self.device) # (B, C, H, W)
                
                # 3. 调整维度 (B, C, H, W) -> (1, C, B, H, W) 
                # DINOV3Encoder 期望输入为 (Batch, Channel, Time, Height, Width)
                # 这里我们将 batch 视为 Time 维度
                input_tensor = input_tensor.permute(1, 0, 2, 3).unsqueeze(0) 

                # 4. 推理
                with torch.no_grad():
                    # feat shape: (1, C_out, B_time, H_out, W_out)
                    feat = self.encoder(input_tensor) 
                    # 立即转回 CPU 避免显存堆积
                    chunk_features.append(feat.cpu())
            
            if not chunk_features:
                return None

            # 5. 拼接当前分片内的结果 (在 Time 维度拼接, dim=2)
            # feat shape: (1, C, T, H, W)
            chunk_output = torch.cat(chunk_features, dim=2)
            return chunk_output

        except Exception as e:
            print(f"Error processing chunk {start_idx}-{end_idx} for {video_path}: {str(e)}")
            return None

def main():
    # 1. 初始化 Ray
    # 确保 ray 已经启动，且有足够的 GPU 资源
    if not ray.is_initialized():
        ray.init()
    
    # 配置参数
    use_l2_norm = False
    video_folder = "test_videos/1120_DV"
    feature_folder = "test_videos/1120_DV_features" # 建议输出到单独文件夹
    highres_scale = 1
    
    # 确保输出目录存在
    os.makedirs(feature_folder, exist_ok=True)
    
    # 扫描视频
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

    # 3. 创建 Actor 池 (8张卡)
    num_gpus = torch.cuda.device_count()
    print(f"Initializing {num_gpus} actors...")
    actors = [VideoChunkEncoderActor.remote(encoder_config) for _ in range(num_gpus)]
    
    # 4. 逐个视频处理（因为每个视频占用所有GPU）
    for video_name in tqdm(video_files, desc="Total Videos"):
        video_path = os.path.join(video_folder, video_name)
        feature_path = os.path.join(feature_folder, video_name.replace('.mp4', '.pt'))
        
        # 跳过已存在的
        if os.path.exists(feature_path):
            print(f"Skipping {video_name}, already exists.")
            continue

        try:
            # --- 步骤 A: 获取视频总帧数 ---
            # 这一步很快，就在主进程做
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            del vr # 释放资源
            
            # --- 步骤 B: 计算每个 GPU 的分片 ---
            frames_per_gpu = math.ceil(total_frames / num_gpus)
            futures = []
            
            # --- 步骤 C: 分发任务 ---
            for i in range(num_gpus):
                start_idx = i * frames_per_gpu
                end_idx = min((i + 1) * frames_per_gpu, total_frames)
                
                # 如果 start 已经超过总帧数（针对极短视频），则不再分发
                if start_idx >= total_frames:
                    continue
                
                # 将切片任务分配给第 i 个 Actor
                futures.append(actors[i].process_chunk.remote(
                    video_path, start_idx, end_idx, batch_size=4
                ))
            
            # --- 步骤 D: 等待所有分片完成 ---
            # ray.get 会阻塞直到所有结果返回
            results = ray.get(futures)
            
            # 检查是否有 None (错误情况)
            if any(r is None for r in results):
                print(f"Failed to process {video_name}: some chunks returned None.")
                continue

            # --- 步骤 E: 拼接结果并保存 ---
            # results 是一个 list of Tensors，每个 Tensor 形状是 (1, C, T_chunk, H, W)
            # 我们需要在 T 维度 (dim=2) 进行拼接
            full_feature = torch.cat(results, dim=2)
            
            print(f"Saving features for {video_name} with shape {full_feature.shape} to {feature_path}")
            torch.save(full_feature, feature_path)
            
        except Exception as e:
            print(f"Critical error processing video {video_name}: {e}")

    # 清理
    ray.shutdown()

if __name__ == "__main__":
    main()