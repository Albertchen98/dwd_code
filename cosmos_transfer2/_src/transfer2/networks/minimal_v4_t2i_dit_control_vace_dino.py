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

import math
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import transformer_engine as te
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributed import ProcessGroup, get_process_group_ranks
from torchvision import transforms
import torch.nn.functional as F

from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import (
    Attention,
)
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import Block as BaseBlock
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import (
    FinalLayer,
)
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT as BaseMiniTrainDIT
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import (
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
    replace_selfattn_op_with_sparse_attn_op,
)
from cosmos_transfer2._src.imaginaire.utils import log
import numpy as np


class DinoTokMergehead(nn.Module):
    def __init__(
        self,
        in_features: int=1024,
        compr_shallow: bool = True,
        compr_deep: bool = False,
    ) -> None:
        super().__init__()
        if compr_shallow:
            self.proj_shallow = nn.Conv3d(in_features, 64, kernel_size=1)
        else:
            self.proj_shallow = nn.Identity()
        
        if compr_deep:
            self.proj_deep = nn.Conv3d(in_features, 64, kernel_size=1)
        else:
            self.proj_deep = nn.Identity()
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Merge multiple feature maps into one.
        features: List of feature maps from different layers. Each feature map is of shape [B, C, T, H, W]
        """
        shallow_feat = features[0].contiguous()
        deep_feat = features[1].contiguous()
        shallow_feat = self.proj_shallow(shallow_feat)
        deep_feat = self.proj_deep(deep_feat)
        return torch.cat([shallow_feat, deep_feat], dim=1) # out channel 1088 
    
    def init_weights(self):
        # --- 针对 self.proj_shallow 的初始化 ---
        # 只有当模块是 nn.Conv3d 时才进行初始化
        if isinstance(self.proj_shallow, nn.Conv3d):
            # Xavier Uniform 初始化
            nn.init.xavier_uniform_(self.proj_shallow.weight)
            if self.proj_shallow.bias is not None:
                nn.init.zeros_(self.proj_shallow.bias)
        
        # --- 针对 self.proj_deep 的初始化 ---
        # 只有当模块是 nn.Conv3d 时才进行初始化
        if isinstance(self.proj_deep, nn.Conv3d):
            # Xavier Uniform 初始化
            nn.init.xavier_uniform_(self.proj_deep.weight)
            if self.proj_deep.bias is not None:
                nn.init.zeros_(self.proj_deep.bias)

       
class DinoTokMergeheadv2(nn.Module):
    def __init__(
        self,
        in_features: int=1024,
        mid_chans: int=64,
        out_chans: int=128,
        compr_shallow: bool = True,
        compr_deep: bool = False,
    ) -> None:
        super().__init__()
        fusion_dim = 0
        if compr_shallow:
            self.proj_shallow = nn.Sequential(
                # 关键：先做 InstanceNorm，把 Sim/Real 的统计差异抹平
                nn.InstanceNorm2d(in_features, affine=True), 
                # 1x1 卷积降维并选取特征，不改变空间分辨率
                nn.Conv2d(in_features, mid_chans, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            )
            fusion_dim += mid_chans
        else:
            self.proj_shallow = nn.Identity()
            fusion_dim += in_features
        
        if compr_deep:
            self.proj_deep = nn.Sequential(
                # 关键：先做 InstanceNorm，把 Sim/Real 的统计差异抹平
                nn.InstanceNorm2d(in_features, affine=True), 
                # 1x1 卷积降维并选取特征，不改变空间分辨率
                nn.Conv2d(in_features, mid_chans, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            )
            fusion_dim += mid_chans
        else:
            self.proj_deep = nn.Identity()
            fusion_dim += in_features
        # 2. 融合主干 (Fusion Body)
        # 输入通道是 mid_chans * 层数
        # concat_chans = mid_chans * 2
        
        self.fusion = nn.Sequential(
            # 3x3 卷积融合不同层的特征，感知局部几何
            nn.Conv2d(fusion_dim, out_chans, kernel_size=3, padding=1, bias=False),
            # 再次 IN，确保输出纯净
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.ReLU(inplace=True),
            # 可选：最后再加一层 1x1 调整到精确的 ControlNet 维度
            nn.Conv2d(out_chans, out_chans, kernel_size=1)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Merge multiple feature maps into one.
        features: List of feature maps from different layers. Each feature map is of shape [B, C, T, H, W]
        """
        shallow_feat = features[0] # [B, C, T, H, W]
        deep_feat = features[1]    # [B, C, T, H, W]
        
        # 1. 维度折叠: [B, C, T, H, W] -> [B*T, C, H, W]
        # 使用 rearrange 或者 view 均可，这里用 view 保持纯 PyTorch
        B, C, T, H, W = shallow_feat.shape
        
        # 必须确保内存连续才能 view
        if not shallow_feat.is_contiguous():
            shallow_feat = shallow_feat.contiguous()
        if not deep_feat.is_contiguous():
            deep_feat = deep_feat.contiguous()
            
        # 将 T 维度合并到 Batch 维度
        # permute first: (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        shallow_feat = shallow_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        deep_feat = deep_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # 2. 独立投影 (Alignment & Filter)
        shallow_feat = self.proj_shallow(shallow_feat)
        deep_feat = self.proj_deep(deep_feat)
        
        # 3. 特征拼接 (Concatenate)
        # [B*T, mid_chans, H, W] + [B*T, mid_chans, H, W]
        feat = torch.cat([shallow_feat, deep_feat], dim=1)
        
        # 4. 融合 (Fusion)
        feat = self.fusion(feat) # Output: [B*T, out_chans, H, W]
        
        # 注意：ControlNet (Stable Diffusion) 通常接受 2D Batch 输入。
        # 如果你的后续网络需要分开 B 和 T，可以在这里 reshape 回去：
        feat = feat.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        
        return feat
    
    def init_weights(self):
        """
        递归初始化所有层。
        """
        # self.apply 会自动遍历 self.proj_shallow, self.proj_deep, self.fusion
        # 以及它们内部的所有子层，并将 _init_layer 函数应用到每一个上面。
        self.apply(self._init_layer)

    def _init_layer(self, m):
        """
        针对单个层的具体初始化逻辑
        """
        # 1. 初始化卷积层 (Conv2d, Conv3d)
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            # Kaiming Normal (He Initialization)
            # 适用于 ReLU 激活函数，mode='fan_out' 保持前向传播方差一致
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            # 如果有偏置 bias，初始化为 0
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # 2. 初始化线性层 (Linear) - 以防万一你有 Linear 层
        elif isinstance(m, nn.Linear):
            # 这里的 nonlinearity 通常也设为 relu 或者 linear
            nn.init.trunc_normal_(m.weight, std=0.02) 
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 3. 初始化归一化层 (BatchNorm, InstanceNorm, GroupNorm, LayerNorm)
        # 注意：InstanceNorm2d 只有在 affine=True 时才有 weight 和 bias
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)  # 缩放系数初始化为 1 (不做改变)
            if m.bias is not None:
                nn.init.zeros_(m.bias)   # 偏移系数初始化为 0

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)          

class DinoTokMergeheadv3(nn.Module):
    def __init__(
        self,
        in_features: int=1024,
        mid_chans: int=256,
        out_chans: int=128,
        compr_shallow: bool = True,
        compr_deep: bool = False,
    ) -> None:
        super().__init__()
        fusion_dim = 0
        if compr_shallow:
            self.proj_shallow = nn.Sequential(
                # 改用 GroupNorm (num_groups=32)，比 InstanceNorm 更能保留空间结构和强度信息
                nn.Conv2d(in_features, mid_chans, kernel_size=1, bias=False),
                nn.SiLU(), # 改用 SiLU，DiT/SD 标配
            )
            fusion_dim += mid_chans
        else:
            self.proj_shallow = nn.Identity()
            fusion_dim += in_features
        
        # 2. 深层特征 (Layer 23) 处理：主要负责语义对齐
        if compr_deep:
            self.proj_deep = nn.Sequential(
                nn.Conv2d(in_features, mid_chans, kernel_size=1, bias=False),
                nn.SiLU(),
            )
            fusion_dim += mid_chans
        else:
            self.proj_deep = nn.Identity()
            fusion_dim += in_features
            
        # 融合后的通道数
        # concat_chans = mid_chans * 2
        
        # 3. 融合主干 (Fusion Body)
        # 使用 ResBlock 代替简单的 Conv，增加非线性拟合能力同时保留梯度
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, out_chans, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_chans, eps=1e-6),
            nn.SiLU()
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Merge multiple feature maps into one.
        features: List of feature maps from different layers. Each feature map is of shape [B, C, T, H, W]
        """
        shallow_feat = features[0] # [B, C, T, H, W]
        deep_feat = features[1]    # [B, C, T, H, W]
        
        # 1. 维度折叠: [B, C, T, H, W] -> [B*T, C, H, W]
        # 使用 rearrange 或者 view 均可，这里用 view 保持纯 PyTorch
        B, C, T, H, W = shallow_feat.shape
        
        # 必须确保内存连续才能 view
        if not shallow_feat.is_contiguous():
            shallow_feat = shallow_feat.contiguous()
        if not deep_feat.is_contiguous():
            deep_feat = deep_feat.contiguous()

        # 将 T 维度合并到 Batch 维度
        # permute first: (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        shallow_feat = shallow_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        deep_feat = deep_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # 2. 独立投影 (Alignment & Filter)
        shallow_feat = self.proj_shallow(shallow_feat)
        deep_feat = self.proj_deep(deep_feat)
        
        # 3. 特征拼接 (Concatenate)
        # [B*T, mid_chans, H, W] + [B*T, mid_chans, H, W]
        feat = torch.cat([shallow_feat, deep_feat], dim=1)
        
        # 4. 融合 (Fusion)
        feat = self.fusion(feat) # Output: [B*T, out_chans, H, W]
        
        # 注意：ControlNet (Stable Diffusion) 通常接受 2D Batch 输入。
        # 如果你的后续网络需要分开 B 和 T，可以在这里 reshape 回去：
        feat = feat.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        
        return feat
    
    def init_weights(self):
        """
        递归初始化所有层。
        """
        # self.apply 会自动遍历 self.proj_shallow, self.proj_deep, self.fusion
        # 以及它们内部的所有子层，并将 _init_layer 函数应用到每一个上面。
        self.apply(self._init_layer)

    def _init_layer(self, m):
        if isinstance(m, nn.Conv2d):
            # 没有前面的 Norm，Kaiming Init 更加重要
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)


class DinoTokMergeheadv4(nn.Module):
    def __init__(
        self,
        in_features: int=1024,
        mid_chans: int=64,
        out_chans: int=128,
        compr_shallow: bool = True,
        compr_deep: bool = False,
        channel_drop: bool = True,
    ) -> None:
        super().__init__()
        fusion_dim = 0
        if compr_shallow:
            self.proj_shallow = nn.Sequential(
                # 改用 GroupNorm (num_groups=32)，比 InstanceNorm 更能保留空间结构和强度信息
                nn.Conv2d(in_features, mid_chans, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, mid_chans), mid_chans),
                nn.SiLU(), # 改用 SiLU，DiT/SD 标配
            )
            fusion_dim += mid_chans
        else:
            self.proj_shallow = nn.Identity()
            fusion_dim += in_features
        
        # 2. 深层特征 (Layer 23) 处理：主要负责语义对齐
        if compr_deep:
            self.proj_deep = nn.Sequential(
                nn.Conv2d(in_features, mid_chans, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, mid_chans), mid_chans),
                nn.SiLU(),
            )
            fusion_dim += mid_chans
        else:
            self.proj_deep = nn.Identity()
            fusion_dim += in_features
            
        # 融合后的通道数
        # concat_chans = mid_chans * 2
        
        # 3. 融合主干 (Fusion Body)
        # 使用 ResBlock 代替简单的 Conv，增加非线性拟合能力同时保留梯度
        
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, out_chans, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, out_chans), out_chans, eps=1e-6),
            nn.SiLU()
        )
        if channel_drop:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = nn.Dropout(0)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Merge multiple feature maps into one.
        features: List of feature maps from different layers. Each feature map is of shape [B, C, T, H, W]
        """
        shallow_feat = features[0] # [B, C, T, H, W]
        deep_feat = features[1]    # [B, C, T, H, W]
        
        # 1. 维度折叠: [B, C, T, H, W] -> [B*T, C, H, W]
        # 使用 rearrange 或者 view 均可，这里用 view 保持纯 PyTorch
        B, C, T, H, W = shallow_feat.shape
        
        # 必须确保内存连续才能 view
        if not shallow_feat.is_contiguous():
            shallow_feat = shallow_feat.contiguous()
        if not deep_feat.is_contiguous():
            deep_feat = deep_feat.contiguous()
        if self.training:
            deep_feat = self.dropout(deep_feat)
        # 将 T 维度合并到 Batch 维度
        # permute first: (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        shallow_feat = shallow_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        deep_feat = deep_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # 2. 独立投影 (Alignment & Filter)
        shallow_feat = self.proj_shallow(shallow_feat)
        deep_feat = self.proj_deep(deep_feat)
        
        # 3. 特征拼接 (Concatenate)
        # [B*T, mid_chans, H, W] + [B*T, mid_chans, H, W]
        feat = torch.cat([shallow_feat, deep_feat], dim=1)
        
        # 4. 融合 (Fusion)
        feat = self.fusion(feat) # Output: [B*T, out_chans, H, W]
        
        # 注意：ControlNet (Stable Diffusion) 通常接受 2D Batch 输入。
        # 如果你的后续网络需要分开 B 和 T，可以在这里 reshape 回去：
        feat = feat.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        
        return feat
    
    def init_weights(self):
        """
        递归初始化所有层。
        """
        # self.apply 会自动遍历 self.proj_shallow, self.proj_deep, self.fusion
        # 以及它们内部的所有子层，并将 _init_layer 函数应用到每一个上面。
        self.apply(self._init_layer)

    def _init_layer(self, m):
        if isinstance(m, nn.Conv2d):
            # 没有前面的 Norm，Kaiming Init 更加重要
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None: nn.init.ones_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

class BasicResBlock(nn.Module):
    """
    标准的 ResNet BasicBlock 适配版: 
    Structure: Input -> Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> Add(Input) -> SiLU
    """
    def __init__(self, channels):
        super().__init__()
        # 使用 GroupNorm 而不是 BatchNorm，因为 Batch Size 可能较小
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        identity = x
        
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        
        out = out + identity  # 残差连接
        return out

class DINOPatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels: int=1024,
        out_channels: int=4096,
    ):
        """Transform the DINO feature to be aligned with video input tensors (B, T, H, W, D)
            DINO feature is originally generated with the shape as (B, Td, Hd, Wd, Dd): 
            Td is basically the number of video frames
            Hd and Wd is 1/14 of original video resolution
            Dd is 1024 for DINO-L, 768 for DINO-B and 1536 for DINO-G
        Args:
            conditioning_embedding_channels (int): the target channel size, aligned with video input tensor
            conditioning_channels (int, optional): DINO feature channel size. Defaults to 1024.
            block_out_channels (list, optional): the channel size of different conv layers. Defaults to [512, 128, 256, 256].
            t_downsampling (str, optional): the method to downsample the DINO feature along temporal dimmension, 
            either "conv" or "keyframe" or "interp". Defaults to "conv".
        """
        super().__init__()
        self.spatial_patch_size=spatial_patch_size
        if spatial_patch_size == 1:
            self.proj = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(in_channels, out_channels, bias=False),
                )
            
        elif spatial_patch_size == 2:
            self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
            
        elif spatial_patch_size == 4:
            self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b (c r m n) t h w",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Conv3d(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                bias=False,
            ),
            nn.SiLU(inplace=True),
            Rearrange("b c t h w -> b t h w c"),
        )
        elif spatial_patch_size == 4.1:
            # === 处理 dino upfactor=4的情况 ===
            self.proj = nn.Sequential(
                Rearrange("b c t h w -> (b t) c h w"), 
                
                # --- Stage 1: 129 -> 64 ---
                nn.Conv2d(in_channels, 16*4, kernel_size=3, padding=1), 
                BasicResBlock(16*4),
                
                # --- Stage 2: 64 -> 128 ---
                nn.Conv2d(16*4, 32*4, kernel_size=3, stride=2, padding=1),
                nn.SiLU(inplace=True),
                BasicResBlock(32*4),
                
                # --- Stage 3: 128 -> 384 这一步不做下采样了 ---
                nn.Conv2d(32*4, 96*4, kernel_size=3, stride=1, padding=1),
                nn.SiLU(inplace=True),
                BasicResBlock(96*4),
                
                # --- Stage 4: 384 -> 2048 ---
                # 最后一层直接下采样输出
                nn.Conv2d(96*4, out_channels, kernel_size=3, stride=2, padding=1),
                nn.SiLU(inplace=True),                
                Rearrange("(b t) c h w -> b t h w c", t=1), 
            )
        
        elif spatial_patch_size == 8:
            # 这里的实现照搬的ControlNet的源代码：目前正好是downsample 8倍
            # FIXME: 输入是B C T H W，视频应该改成3D conv
            
            self.proj = nn.Sequential(
                # 将输入 [B, C, T, H, W] 重排为 [(B*T), H, W, C]
                # 这样可以将时间维度折叠进 batch 维度，便于使用 2D 卷积处理每一帧
                Rearrange("b c t h w -> (b t) c h w"), 
                # FIXME 16 * 4 32 * 4
                # 第一组 Conv + SiLU: 保持通道数，处理特征
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), # 使用传入的 in_channels # FIXME: 使用8+1=9的in_channels，取消padding
                nn.SiLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.SiLU(),
                
                # 第二组 Conv + SiLU: 下采样 (stride=2) + 增加通道数
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # H,W -> H/2, W/2
                nn.SiLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.SiLU(),
                
                # 第三组 Conv + SiLU: 下采样 (stride=2) + 增加通道数
                nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1), # H/2,W/2 -> H/4, W/4
                nn.SiLU(),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.SiLU(),
                
                # 最后一组 Conv + SiLU: 下采样 (stride=2) + 输出到目标通道数
                # 总共下采样 2*2*2 = 8 倍，符合 spatial_patch_size=8
                nn.Conv2d(96, out_channels, kernel_size=3, stride=2, padding=1), # H/4,W/4 -> H/8, W/8
                nn.SiLU(),
                
                # 将处理后的结果从 [(B*T), Cout, H_out, W_out] 重排回 [B, T, H_out, W_out, Cout]
                # 注意：这里使用 -1 来自动推断 T 维度，使其能处理不同的 T 值
                Rearrange("(b t) c h w -> b t h w c", t=1),  # 这里只处理t=1的情况
            )    
        elif spatial_patch_size == 8.2:
             self.proj = nn.Sequential(
                Rearrange("b c t h w -> (b t) c h w"), 
                nn.Conv2d(in_channels, 16*2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16*2, 16*2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16*2, 32*2, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32*2, 32*2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32*2, 96*2, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(96*2, 96*2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(96*2, out_channels, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                Rearrange("(b t) c h w -> b t h w c", t=1), 
            )

        elif spatial_patch_size == 8.3:
            # === [Update] 使用 ResBlock 的优化版本 ===
            self.proj = nn.Sequential(
                Rearrange("b c t h w -> (b t) c h w"), 
                
                # --- Stage 1: 129 -> 64 ---
                nn.Conv2d(in_channels, 16*4, kernel_size=3, padding=1), 
                BasicResBlock(16*4),
                
                # --- Stage 2: 64 -> 128 ---
                nn.Conv2d(16*4, 32*4, kernel_size=3, stride=2, padding=1),
                nn.SiLU(inplace=True),
                BasicResBlock(32*4),
                
                # --- Stage 3: 128 -> 384 ---
                nn.Conv2d(32*4, 96*4, kernel_size=3, stride=2, padding=1),
                nn.SiLU(inplace=True),
                BasicResBlock(96*4),
                
                # --- Stage 4: 384 -> 2048 ---
                # 最后一层直接下采样输出
                nn.Conv2d(96*4, out_channels, kernel_size=3, stride=2, padding=1),
                nn.SiLU(inplace=True),                
                Rearrange("(b t) c h w -> b t h w c", t=1), 
            )

        elif spatial_patch_size == 8.1:
            self.proj = nn.Sequential(
                            Rearrange('b c t (h p1) (w p2) -> b t h w (c p1 p2)', p1=8, p2=8),
                            nn.Linear(576, 16, bias=False), # 576 -> 16
                            nn.LayerNorm(16),
                            nn.SiLU(inplace=True),                      
                            nn.Linear(16, out_channels, bias=False), 
                        )
        elif spatial_patch_size == 8.11:
            self.proj = nn.Sequential(
                            Rearrange('b c t (h p1) (w p2) -> b t h w (c p1 p2)', p1=8, p2=8),
                            nn.Linear(576, 32, bias=False), # 576 -> 16
                            nn.LayerNorm(16),
                            nn.SiLU(inplace=True),                      
                            nn.Linear(32, out_channels, bias=False), 
                        )
        elif spatial_patch_size == 8.12:
            self.proj = nn.Sequential(
                            Rearrange('b c t (h p1) (w p2) -> b t h w (c p1 p2)', p1=8, p2=8),
                            nn.Linear(576, 64, bias=False), # 576 -> 16
                            nn.LayerNorm(16),
                            nn.SiLU(inplace=True),                      
                            nn.Linear(64, out_channels, bias=False), 
                        )

    def init_weights(self) -> None:
        """
        手动初始化所有权重，确保没有遗漏。
        包含了对 ResBlock 中 GroupNorm 的处理以及 Zero Init 技巧。
        """
        def _init_weights(module):
            # 1. Linear 层初始化 (Truncated Normal)
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
            # 2. Conv 层初始化 (Kaiming Normal)
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  

            # 3. Norm 层初始化 (GroupNorm/BatchNorm)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        # A. 应用基础初始化到所有子模块
        if hasattr(self, 'proj') and self.proj is not None:
            self.proj.apply(_init_weights)

        # B. 对 BasicResBlock 应用零初始化 (Zero Init)
        # 作用：让残差块初始输出为 0，网络在训练初期等效于只有直连路径
        for m in self.modules():
            if isinstance(m, BasicResBlock):
                nn.init.constant_(m.conv2.weight, 0)
                if m.conv2.bias is not None:
                    nn.init.constant_(m.conv2.bias, 0)
    def forward(self, x):

        """
         Forward pass of the DINOPatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        # if self.use_dino_pca:
        #     x = rearrange(x, "b c t h w -> b t h w c")
        #     x = x - self.pca_mean[None,None,None,None]
        #     x = x @ self.pca_comp.T
        # log.info(f"DINOpatchembed Weights requires_grad: {self.proj[1].weight.requires_grad}")
        # breakpoint()
        x = self.proj(x)
        return x
    

class I2VCrossAttentionFull(Attention):
    """
    A modified Attention class that adds separate query, key, value projections for reference image attention.
    """

    def __init__(self, *args, img_latent_dim: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        inner_dim = self.head_dim * self.n_heads
        self.k_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.q_img = nn.Linear(self._query_dim, inner_dim, bias=False)  # NEW: separate query for image attention
        self.q_img_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)  # NEW: dedicated normalization for q_img
        self.k_img_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

    def init_weights(self) -> None:
        super().init_weights()
        torch.nn.init.trunc_normal_(self.k_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        torch.nn.init.trunc_normal_(self.v_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        torch.nn.init.trunc_normal_(
            self.q_img.weight, std=1.0 / math.sqrt(self._query_dim)
        )  # NEW: initialize q_img with correct dim
        self.q_img_norm.reset_parameters()  # NEW: initialize q_img_norm
        self.k_img_norm.reset_parameters()

    def compute_qkv(
        self, x, context, rope_emb=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_context, img_context = context
        q, k, v = super().compute_qkv(x, text_context, rope_emb)

        # Compute image-specific query, key, value - following base Attention class pattern
        q_img = self.q_img(x)  # NEW: separate query projection for image attention (same input x as regular q)
        k_img = self.k_img(img_context)
        v_img = self.v_img(img_context)

        # Rearrange q_img, k_img, v_img - same as base Attention class
        q_img, k_img, v_img = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q_img, k_img, v_img),
        )

        # Apply normalization - following base Attention class pattern
        q_img = self.q_img_norm(q_img)  # Use dedicated q_img_norm for image queries
        k_img = self.k_img_norm(k_img)  # Use dedicated k_img_norm for image keys

        # Apply rotary embeddings if needed (only for self-attention)
        # Note: q_img and k_img don't need rotary embeddings since they're for cross-attention

        return q, k, v, q_img, k_img, v_img

    def compute_attention(self, q, k, v, q_img, k_img, v_img):
        result = self.attn_op(q, k, v)  # [B, S, H, D] - text attention using shared q
        result_img = self.attn_op(q_img, k_img, v_img)  # [B, S, H, D] - image attention using separate q_img
        return self.output_dropout(self.output_proj(result + result_img))

    def forward(
        self,
        x,
        context=None,
        rope_emb=None,
    ):
        q, k, v, q_img, k_img, v_img = self.compute_qkv(x, context, rope_emb)
        return self.compute_attention(q, k, v, q_img, k_img, v_img)


# Modified BaseBlock class with share_q_in_i2v_cross_attn parameter
class Block(BaseBlock):
    """
    This is a modified version of the BaseBlock class from minimal_v4_dit.py.
    It adds a share_q_in_i2v_cross_attn parameter to the block constructor.
    If share_q_in_i2v_cross_attn is False, it uses the I2VCrossAttentionFull class instead of the I2VCrossAttention class.
    This is used to separate the query projection for image attention.

    From original BaseBlock class:
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256
        [NEW] share_q_in_i2v_cross_attn (bool): Whether to share q in i2v cross-attention. Default: True

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        share_q_in_i2v_cross_attn: bool = False,
        use_wan_fp32_strategy: bool = False,
    ):
        # Call parent constructor first
        super().__init__(
            x_dim=x_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            backend=backend,
            image_context_dim=image_context_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        # Override cross attention if we need separate q for image attention
        if image_context_dim is not None and not share_q_in_i2v_cross_attn:
            self.cross_attn = I2VCrossAttentionFull(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                img_latent_dim=image_context_dim,
                qkv_format="bshd",
            )


class ControlAwareDiTBlock(Block):
    """A modified DiTBlock for the base model that accepts control branch modulations ("hints") if available.

    Args:
        has_image_input: Whether the block accepts image input
        dim: Dimension of the input features
        num_heads: Number of attention heads
        ffn_dim: Dimension of the feed-forward network
        eps: Epsilon value for normalization
        block_id: ID of the block, used to index into the hints tensor
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        block_id: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            backend,
            image_context_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.block_id = block_id

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        hints: Optional[torch.Tensor] = None,
        control_context_scale: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass of the block.

        Args:
            x: Input tensor
            hints: Optional control signals from the control branch
            control_context_scale: Optional scaling factor for the hints
            **kwargs: Additional arguments passed to the base DiTBlock

        Returns:
            Processed tensor with optional hints added
        """
        x_B_T_H_W_D = super().forward(x_B_T_H_W_D, **kwargs)
        if self.block_id is not None and hints is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + hints[self.block_id] * control_context_scale
        return x_B_T_H_W_D


class ControlEncoderDiTBlock(Block):
    """A modified DiTBlock for the control branch that *generates* per-block control modulation signals.
      compared to the base DiTBlock, it adds skip connections and zero convolutions.

    Args:
        has_image_input: Whether the block accepts image input
        dim: Dimension of the input features
        num_heads: Number of attention heads
        ffn_dim: Dimension of the feed-forward network
        eps: Epsilon value for normalization
        block_id: ID of the block, used for skip connections
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        block_id: int = 0,
        hint_dim: Optional[int] = None,
        use_after_proj: bool = True,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            backend,
            image_context_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.block_id = block_id
        self.use_after_proj = use_after_proj

        # Zero convolution as in ControlNet
        if block_id == 0:
            self.before_proj = nn.Linear(hint_dim if hint_dim else self.x_dim, self.x_dim)
        if use_after_proj:
            self.after_proj = nn.Linear(self.x_dim, self.x_dim)

    def init_weights(self):
        super().init_weights()
        if self.use_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)
        if self.block_id == 0:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

    def forward(self, c: torch.Tensor, x_B_T_H_W_D: torch.Tensor, **kwargs):
        """
        stacks the previous block's output with the current block's output,
        so that the final output from the control branch is a stack of all the control block outputs,
        and easy to apply block-wise to base model.

        """
        if self.block_id == 0:
            c = self.before_proj(c) + x_B_T_H_W_D
            all_c = []
        elif self.use_after_proj:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        if self.use_after_proj:
            c_skip = self.after_proj(c)
            all_c += [c_skip, c]
            c = torch.stack(all_c)
        return c


# Modified BaseMiniTrainDIT class by adding reference image parameter
class MiniTrainDITImageContext(BaseMiniTrainDIT):
    """
    This is a modified version of the MiniTrainDIT class from minimal_v4_dit.py.
    It adds img_context_deep_proj and share_q_in_i2v_cross_attn functionality to the base MiniTrainDIT.

    From original MiniTrainDIT class:
    Extended MiniTrainDIT class that adds img_context_deep_proj and share_q_in_i2v_cross_attn functionality
    to the base MiniTrainDIT from minimal_v4_dit.py.

    New parameters:
        img_context_deep_proj (bool): Whether to use deep MLP projection for image context.
            - False (default): Simple projection (Linear + GELU) for backward compatibility
            - True: Deep MLP projection (Linear -> GELU -> Linear -> LayerNorm) following IP-Adapter Full style
        share_q_in_i2v_cross_attn (bool): Whether to share q in i2v cross-attention. Default: True
            - True (default): Use I2VCrossAttention (shared query between text and image attention)
            - False: Use I2VCrossAttentionFull (separate query projections for text and image attention)
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "transformer_engine",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: Optional[int] = None,  # Main flag of whether user reference image
        img_context_deep_proj: bool = False,  # work when extra_image_context_dim is not None
        share_q_in_i2v_cross_attn: bool = False,  # work when extra_image_context_dim is not None
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        sac_config: SACConfig = SACConfig(),
        n_dense_blocks: int = -1,
        gna_parameters=None,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        # Initialize the grandparent class (whatever the parent inherits from)
        super(BaseMiniTrainDIT, self).__init__()

        # Store parameters in the same order as parent class
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim
        # NEW: Our additional parameters
        self.img_context_deep_proj = img_context_deep_proj
        self.share_q_in_i2v_cross_attn = share_q_in_i2v_cross_attn
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        # Component building (same order as parent)
        self.build_patch_embed()
        # breakpoint()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels

        # Create blocks with our modified Block class
        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                    image_context_dim=None if extra_image_context_dim is None else model_channels,
                    share_q_in_i2v_cross_attn=share_q_in_i2v_cross_attn,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)

        # Create image context projection with deep support
        if extra_image_context_dim is not None:
            if img_context_deep_proj:
                # Deep MLP projection
                self.img_context_proj = nn.Sequential(
                    nn.Linear(extra_image_context_dim, extra_image_context_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(extra_image_context_dim, model_channels, bias=False),
                    nn.LayerNorm(model_channels),
                )
            else:
                # Simple projection
                self.img_context_proj = nn.Sequential(
                    nn.Linear(
                        extra_image_context_dim, model_channels, bias=True
                    ),  # help distinguish between image and video context
                    nn.GELU(),
                )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

        # Replace self-attention with sparse attention if specified
        if n_dense_blocks != -1:
            self = replace_selfattn_op_with_sparse_attn_op(self, n_dense_blocks, gna_parameters=gna_parameters)

        self._is_context_parallel_enabled = False

    def init_weights(self):
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()

        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        # Handle image context projection initialization
        if self.extra_image_context_dim is not None:
            if self.img_context_deep_proj:
                # Initialize deep projection with proper scaling
                for layer in self.img_context_proj:
                    if isinstance(layer, nn.Linear):
                        std = 1.0 / math.sqrt(layer.in_features)
                        torch.nn.init.trunc_normal_(layer.weight, std=std, a=-3 * std, b=3 * std)
                    elif isinstance(layer, nn.LayerNorm):
                        layer.reset_parameters()
            else:
                # Simple projection initialization (same as parent)
                self.img_context_proj[0].reset_parameters()


class MinimalV4ControlVaceDiT(MiniTrainDITImageContext):
    """
    Adding control branch to the base model.
    """

    def __init__(
        self,
        *args,
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        vace_has_mask: bool = False,
        vace_block_every_n: int = 2,
        condition_strategy: Literal["spaced", "first_n"] = "spaced",
        num_max_modalities: int = 8,
        use_input_hint_block: bool = False,
        spatial_compression_factor: int = 8,
        num_control_branches: int = 1,
        separate_embedders: bool = False,
        use_after_proj_for_multi_branch: bool = True,
        timestep_scale: float = 1.0,  # Add timestep scaling for rectified flow
        **kwargs,
    ):
        """
        vace_block_every_n: create one control block every n base model blocks
        vace_has_mask: if true, control branch latent is [inactive, reactive, mask] as in VACE paper. Otherwise, just the latent of the control input
        condition_strategy: How the control blocks correspond to the base model blocks. "first_n" conditions first n base model blocks.
            "spaced" conditions every vace_block_every_n base model block. E.g. vace_block_every_n=2, condition_strategy="spaced" means control block 0
            controls base block 0 and 2, control block 1 controls base block 2, etc.
        """

        assert "in_channels" in kwargs, "in_channels must be provided"

        # kwargs["in_channels"] += 1  # Add 1 for the condition mask
        nf = kwargs["model_channels"]
        hint_nf = kwargs.pop("hint_nf", [nf, nf, nf, nf, nf, nf, nf, nf])
        self.dino_ctrl_channels = kwargs.pop("dino_ctrl_channels", None)
        self.dino_merge_type = kwargs.pop("dino_merge_type", "v1")
        self.dino_upfactor = kwargs.pop("dino_upfactor", None)
        self.use_dino_merge = kwargs.pop("use_dino_merge", None)
        self.dino_merge_config = kwargs.pop("dino_merge_config", None)
        self.use_dino_pca = kwargs.pop("use_dino_pca", None)
        
        self.use_anyup_hint_block = kwargs.pop("use_anyup_hint_block", False)
        self.num_control_branches = num_control_branches
        self.use_after_proj_for_multi_branch = use_after_proj_for_multi_branch
        self.timestep_scale = timestep_scale  # Store timestep scale for rectified flow
        super().__init__(
            *args,
            crossattn_emb_channels=crossattn_emb_channels,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )

        self.crossattn_emb_channels = crossattn_emb_channels
        self.mlp_ratio = mlp_ratio

        # if vace_has_mask, the control latent is 16 + 64 (for mask)
        self.vace_has_mask = vace_has_mask
        self.num_max_modalities = num_max_modalities
        # in_channels = self.in_channels - 1  # subtract the condition mask
        in_channels = self.in_channels
        self.vace_in_channels = (in_channels + spatial_compression_factor**2) if vace_has_mask else in_channels
        self.vace_in_channels *= num_max_modalities
        # self.vace_in_channels += 1  # adding the condition mask back

        # for finding corresponding control block with base model block.
        self.condition_strategy = condition_strategy
        if self.condition_strategy == "spaced":
            # base block k uses the 2k'th element in the hint list, {0:0, 2:1, 4:2, ...}, as in VACE paper
            self.control_layers = [i for i in range(0, self.num_blocks, vace_block_every_n)]
            self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}
        elif self.condition_strategy == "first_n":
            # condition first n base model blocks, where n is number of control blocks
            self.control_layers = list(range(0, self.num_blocks // vace_block_every_n))
            self.control_layers_mapping = {i: i for i in range(len(self.control_layers))}
        else:
            raise ValueError(f"Invalid condition strategy: {self.condition_strategy}")
        assert 0 in self.control_layers

        # Input hint block
        self.use_input_hint_block = use_input_hint_block
        if use_input_hint_block:
            assert self.num_control_branches == 1, "input hint block is not supported for multi-branch"
            input_hint_block = []
            nonlinearity = nn.SiLU()
            for i in range(len(hint_nf) - 1):
                input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
            self.input_hint_block = nn.Sequential(*input_hint_block)

        # -------- Base model --------

        # Base model blocks. Overwrite them to enable accepting the control branch modulations ("hints").
        # Shape remains the same as the base model so we can load pretrained weights.
        self.blocks = nn.ModuleList(
            [
                ControlAwareDiTBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers else None,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                )
                for i in range(self.num_blocks)
            ]
        )

        # -------- Control branch --------
        self.separate_embedders = separate_embedders
        if separate_embedders:
            self.t_embedder_for_control_branch = nn.Sequential(
                Timesteps(self.model_channels),
                TimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora),
            )
            self.t_embedding_norm_for_control_branch = te.pytorch.RMSNorm(self.model_channels, eps=1e-6)

        self.build_patch_embed_dino_vace()

        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                setattr(
                    self,
                    f"control_blocks_{nc}",
                    nn.ModuleList(
                        [
                            ControlEncoderDiTBlock(
                                x_dim=self.model_channels,
                                context_dim=self.crossattn_emb_channels,
                                num_heads=self.num_heads,
                                mlp_ratio=self.mlp_ratio,
                                use_adaln_lora=self.use_adaln_lora,
                                adaln_lora_dim=self.adaln_lora_dim,
                                backend=self.atten_backend,
                                image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                                block_id=i,
                                hint_dim=hint_nf[-1] if use_input_hint_block else None,
                                use_after_proj=not use_after_proj_for_multi_branch,
                                use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                            )
                            for i in self.control_layers
                        ]
                    ),
                )
            if use_after_proj_for_multi_branch:
                self.after_proj = nn.ModuleList(
                    [
                        nn.Linear(self.model_channels * self.num_control_branches, self.model_channels)
                        for _ in range(len(self.control_layers))
                    ]
                )
        else:
            self.control_blocks = nn.ModuleList(
                [
                    ControlEncoderDiTBlock(
                        x_dim=self.model_channels,
                        context_dim=self.crossattn_emb_channels,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        use_adaln_lora=self.use_adaln_lora,
                        adaln_lora_dim=self.adaln_lora_dim,
                        backend=self.atten_backend,
                        image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                        block_id=i,
                        hint_dim=hint_nf[-1] if use_input_hint_block else None,
                        use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                    )
                    for i in self.control_layers
                ]
            )

        self.init_weights()
        sac_config = kwargs.get("sac_config", SACConfig())
        self.enable_selective_checkpoint(sac_config, self.blocks)
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                self.enable_selective_checkpoint(sac_config, getattr(self, f"control_blocks_{nc}"))
        else:
            self.enable_selective_checkpoint(sac_config, self.control_blocks)

    def build_patch_embed_dino_vace(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.dino_ctrl_channels,
            1,
            1,
            self.model_channels,
        )
        # breakpoint()
        if self.dino_ctrl_channels < self.vace_in_channels:
            in_channels = self.vace_in_channels
        
        if self.use_anyup_hint_block:
            patch_spatial = self.dino_upfactor
        
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        
        if self.use_dino_merge:
            log.info("enable Dinov3Mergehead")
            if self.dino_merge_type == "v1":
                self.dinov3_mergehead = DinoTokMergehead(1024, 
                                                     self.dino_merge_config.compr_shallow,
                                                     self.dino_merge_config.compr_deep)
            elif self.dino_merge_type == "v2":
                self.dinov3_mergehead = DinoTokMergeheadv2(1024, mid_chans=64, 
                                                           out_chans=self.dino_ctrl_channels,
                                                        compr_shallow=self.dino_merge_config.compr_shallow,
                                                        compr_deep=self.dino_merge_config.compr_deep)
            elif self.dino_merge_type == "v3":
                self.dinov3_mergehead = DinoTokMergeheadv3(self.dino_merge_config.in_feat, mid_chans=256, 
                                                           out_chans=self.dino_ctrl_channels,
                                                        compr_shallow=self.dino_merge_config.compr_shallow,
                                                        compr_deep=self.dino_merge_config.compr_deep)
            elif self.dino_merge_type == "v4":
                self.dinov3_mergehead = DinoTokMergeheadv4(self.dino_merge_config.in_feat, 
                                                           mid_chans=self.dino_merge_config.mid_chans, 
                                                           out_chans=self.dino_ctrl_channels,
                                                        compr_shallow=self.dino_merge_config.compr_shallow,
                                                        compr_deep=self.dino_merge_config.compr_deep)
                
            

        log.info(f"dinopatch Embed in_channels is {in_channels}")
        self.control_embedder = DINOPatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )
        if self.separate_embedders:
            # self.x_embedder_for_control_branch = PatchEmbed(
            #     spatial_patch_size=patch_spatial,
            #     temporal_patch_size=patch_temporal,
            #     in_channels=self.in_channels + 1 if concat_padding_mask else self.in_channels,
            #     out_channels=model_channels,
            # )
            assert NotImplementedError
            

    def init_weights(self):
        super().init_weights()

        if hasattr(self, "input_hint_block"):
            for module in self.input_hint_block.modules():
                if hasattr(module, "weight"):
                    std = 1.0 / math.sqrt(module.weight.shape[0])
                    torch.nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
        
        if hasattr(self, "dinov3_mergehead"):
            self.dinov3_mergehead.init_weights()

        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                if hasattr(self, "control_embedder"):  # control branch initialization
                    self.control_embedder[nc].init_weights()
                    for block in getattr(self, f"control_blocks_{nc}"):
                        block.init_weights()
            if hasattr(self, "after_proj"):
                for cl in range(len(self.control_layers)):
                    nn.init.zeros_(self.after_proj[cl].weight)
                    nn.init.zeros_(self.after_proj[cl].bias)
        else:
            if hasattr(self, "control_embedder"):
                self.control_embedder.init_weights()
            if hasattr(self, "control_blocks"):
                for block in self.control_blocks:
                    block.init_weights()

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        embedder: Optional[PatchEmbed] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        # import pdb; pdb.set_trace()
        x_B_T_H_W_D = embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def disable_context_parallel(self):
        # pos_embedder
        self.pos_embedder.disable_context_parallel()

        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                for block in getattr(self, f"control_blocks_{nc}"):
                    block.self_attn.set_context_parallel_group(
                        process_group=None,
                        ranks=None,
                        stream=torch.cuda.Stream(),
                    )
        else:
            for block in self.control_blocks:
                block.self_attn.set_context_parallel_group(
                    process_group=None,
                    ranks=None,
                    stream=torch.cuda.Stream(),
                )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # pos_embedder: shared between base and control branch
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                for block in getattr(self, f"control_blocks_{nc}"):
                    block.self_attn.set_context_parallel_group(
                        process_group=process_group,
                        ranks=cp_ranks,
                        stream=torch.cuda.Stream(),
                    )
        else:
            for block in self.control_blocks:
                block.self_attn.set_context_parallel_group(
                    process_group=process_group,
                    ranks=cp_ranks,
                    stream=torch.cuda.Stream(),
                )

        self._is_context_parallel_enabled = True

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        latent_control_input: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        control_context_scale: float | torch.Tensor = 1.0,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs
        # control branch forward
        # Get the original shape
        B, C, T, H, W = x_B_C_T_H_W.shape
        def _pad_control_input(control_B_C_T_H_W):
            # Pad control input channels to match the maximum number of modalities.
            B, C, T, H, W = control_B_C_T_H_W.shape
            if control_B_C_T_H_W.shape[1] < self.vace_in_channels:
                pad_C = self.vace_in_channels - control_B_C_T_H_W.shape[1]
                # log.info(f"Input control has {c} channels, but we need {self.vace_in_channels} channels. Padding with zeros.")
                control_B_C_T_H_W = torch.cat(
                    [
                        control_B_C_T_H_W,
                        torch.zeros(
                            (B, pad_C, T, H, W), dtype=control_B_C_T_H_W.dtype, device=control_B_C_T_H_W.device
                        ),
                    ],
                    dim=1,
                )
            return control_B_C_T_H_W
        # breakpoint()
        if self.num_control_branches == 1:
            control_B_C_T_H_W = latent_control_input
            if self.use_dino_merge:
                control_B_C_T_H_W = self.dinov3_mergehead(torch.chunk(control_B_C_T_H_W, dim=1, chunks=2))
            if not self.dino_upfactor or self.dino_upfactor != 8.1:
                control_B_C_T_H_W = _pad_control_input(control_B_C_T_H_W)
        else:
            # control_B_C_T_H_W = latent_control_input.chunk(self.num_control_branches, dim=1)
            # control_B_C_T_H_W = [_pad_control_input(c) for c in control_B_C_T_H_W]
            assert NotImplementedError

        def _prepare_transformer_input(x_B_C_T_H_W, embedder):
            # if data_type == DataType.VIDEO:
            #     H, W = x_B_C_T_H_W.shape[-2:]
            #     x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W[:, :, :,:H, :W].type_as(x_B_C_T_H_W)], dim=1)
            # else:
            #     x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, torch.zeros_like(x_B_C_T_H_W[:, :1])], dim=1)

            x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
                x_B_C_T_H_W,
                fps=fps,
                padding_mask=padding_mask,
                embedder=embedder,
            )
            return x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D

        # Add condition mask to both input video and control signal
        # print(f"the shape of x_B_C_T_H_W is {x_B_C_T_H_W.shape}")
        # print(f"the shape of condition_video_input_mask_B_C_T_H_W is {condition_video_input_mask_B_C_T_H_W.shape}")
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = _prepare_transformer_input(
            x_B_C_T_H_W, embedder=self.x_embedder
        )
        # log.info(f"x_B_T_H_W_D_for_control has shape {x_B_T_H_W_D.shape}")
        if self.separate_embedders:
            x_B_T_H_W_D_for_control, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = (
                _prepare_transformer_input(x_B_C_T_H_W, embedder=self.x_embedder_for_control_branch)
            )
        else:
            x_B_T_H_W_D_for_control = x_B_T_H_W_D

        if self.num_control_branches > 1:
            assert NotImplementedError
            # control_B_T_H_W_D = []
            # for nc in range(self.num_control_branches):
            #     control_B_C_T_H_W_i = control_B_C_T_H_W[nc]
            #     control_B_T_H_W_D_i, rope_emb_L_1_1_D_for_control, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control = (
            #         _prepare_transformer_input(
            #             control_B_C_T_H_W_i,
            #             embedder=self.control_embedder[nc],
            #         )
            #     )
            #     if not control_B_C_T_H_W_i.any():
            #         control_B_T_H_W_D_i = torch.zeros_like(control_B_T_H_W_D_i)
            #     control_B_T_H_W_D.append(control_B_T_H_W_D_i)
        else:
            # log.info(f"control_B_T_H_W_D before embedder shape: {control_B_C_T_H_W.shape}")
            control_B_T_H_W_D, rope_emb_L_1_1_D_for_control, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control = (
                _prepare_transformer_input(
                    control_B_C_T_H_W,
                    embedder=self.control_embedder,
                )
            )
            # log.info(f"control_B_T_H_W_D after embedder shape: {control_B_T_H_W_D.shape}")

        #  If not using T5, project context emb's channel dim to constant shape
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if self.use_input_hint_block:
            control_B_T_H_W_D = self.input_hint_block(control_B_T_H_W_D)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timesteps_B_T = timesteps_B_T * self.timestep_scale

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T) # 疑问：有必要把t_embedder这个本来weight都是bf16的模块转成fp32然后做操作么？？？
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

            if self.separate_embedders:
                t_embedding_B_T_D_for_control, adaln_lora_B_T_3D_for_control = self.t_embedder_for_control_branch(
                    timesteps_B_T
                )
                t_embedding_B_T_D_for_control = self.t_embedding_norm_for_control_branch(t_embedding_B_T_D_for_control)
            else:
                t_embedding_B_T_D_for_control = t_embedding_B_T_D
                adaln_lora_B_T_3D_for_control = adaln_lora_B_T_3D

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        B, T, H, W, D = x_B_T_H_W_D.shape

        # NEW CODE: control branch forward
        def _get_control_weight(control_context_scale):
            if isinstance(control_context_scale, torch.Tensor):
                if control_context_scale.ndim == 0:  # Single scalar tensor
                    control_weight_maps = [float(control_context_scale)] * self.num_control_branches
                elif control_context_scale.ndim == 1:  # List of scalar weights
                    control_weight_maps = [float(w) for w in control_context_scale]
                else:  # Spatial-temporal weight maps
                    control_weight_maps = [w for w in control_context_scale]  # Keep as tensor
            elif isinstance(control_context_scale, (float, int)):
                control_weight_maps = [control_context_scale] * self.num_control_branches
            elif isinstance(control_context_scale, list) and all(isinstance(w, float) for w in control_context_scale):
                control_weight_maps = [float(w) for w in control_context_scale]
            else:
                raise ValueError(
                    f"Invalid control_context_scale type: {type(control_context_scale)} {control_context_scale}"
                )
            return control_weight_maps
        # log.info(f"x_B_T_H_W_D_for_control has shape: {x_B_T_H_W_D_for_control.shape}")
        if self.num_control_branches > 1:
            hints = []
            has_hint_nc = [c.any() for c in control_B_T_H_W_D]
            for i in range(len(self.control_layers)):
                for nc in range(self.num_control_branches):
                    if has_hint_nc[nc] or torch.is_grad_enabled():
                        block = getattr(self, f"control_blocks_{nc}")[i]
                        control_B_T_H_W_D[nc] = block(
                            c=control_B_T_H_W_D[nc],
                            x_B_T_H_W_D=x_B_T_H_W_D_for_control,
                            emb_B_T_D=t_embedding_B_T_D_for_control,
                            crossattn_emb=context_input,
                            rope_emb_L_1_1_D=rope_emb_L_1_1_D_for_control,
                            adaln_lora_B_T_3D=adaln_lora_B_T_3D_for_control,
                            extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control,
                        )
                        control_B_T_H_W_D[nc] = control_B_T_H_W_D[nc] * has_hint_nc[nc]
                if self.use_after_proj_for_multi_branch:
                    # Normalize activations based on number of active branches
                    num_active_branches = sum(has_hint_nc)
                    control_B_T_H_W_D_concat = torch.cat(control_B_T_H_W_D, dim=-1) / num_active_branches
                    hints.append(self.after_proj[i](control_B_T_H_W_D_concat))
            if not self.use_after_proj_for_multi_branch:
                weight_maps_scalar_or_B_T_H_W_D = _get_control_weight(control_context_scale)
                control_B_T_H_W_D_sum = sum([c * w for c, w in zip(control_B_T_H_W_D, weight_maps_scalar_or_B_T_H_W_D)])
                hints = torch.unbind(control_B_T_H_W_D_sum)[:-1]  # list of layerwise control modulations
                control_context_scale = 1.0  # already scaled hints by control_context_scale
        else:
            # breakpoint()
            for block in self.control_blocks:
                control_B_T_H_W_D = block(
                    c=control_B_T_H_W_D,
                    x_B_T_H_W_D=x_B_T_H_W_D_for_control,
                    emb_B_T_D=t_embedding_B_T_D_for_control,
                    crossattn_emb=context_input,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D_for_control,
                    adaln_lora_B_T_3D=adaln_lora_B_T_3D_for_control,
                    extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control,
                )
                
            #DEBUG
            # from safetensors.torch import save_file
            # save_file({"control_B_T_H_W_D": control_B_T_H_W_D.cpu()}, f"./tmp/dino_control_B_T_H_W_D_step_{timesteps_B_T[0,0].cpu().item()}.safetensors")
            hints = torch.unbind(control_B_T_H_W_D)[:-1]  # list of layerwise control modulations
            control_context_scale = control_context_scale[0]

        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D=x_B_T_H_W_D,
                hints=hints,
                control_context_scale=control_context_scale,
                emb_B_T_D=t_embedding_B_T_D,
                crossattn_emb=context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp
