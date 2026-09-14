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
from torch.distributed import ProcessGroup, get_process_group_ranks, broadcast, get_world_size, ReduceOp, all_reduce
from torchvision import transforms

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
import torch.nn.functional as F


def match_distribution(h, h_vae, pg=None, eps=1e-6):
    """
    Match h distribution to h_vae distribution with global synchronization.

    Args:
        h: [B, D1, T, H, W]   (DINO features)
        h_vae: [B, D2, T, H, W] (EDGE features)
        pg: Process group for distributed synchronization
    """
    # Compute local mean and std for DINO features
    mean_h_local = h.mean(dim=(0, 2, 3, 4), keepdim=True)
    std_h_local = h.std(dim=(0, 2, 3, 4), keepdim=True)
    
    # Compute local mean and std for VAE features  
    mean_vae_local = h_vae.mean(dim=(0, 2, 3, 4), keepdim=True)
    std_vae_local = h_vae.std(dim=(0, 2, 3, 4), keepdim=True)
    
    if pg is not None:
        # Synchronize statistics across all ranks in the process group
        world_size = get_world_size(pg)
        
        # All-reduce to get global mean and std
        # For mean: global_mean = sum(local_mean * local_count) / global_count
        # But since we're using scalar values, we can simply average
        
        # For DINO features
        all_reduce(mean_h_local, op=ReduceOp.SUM, group=pg)
        mean_h_global = mean_h_local / world_size
        
        all_reduce(std_h_local, op=ReduceOp.SUM, group=pg) 
        std_h_global = std_h_local / world_size
        
        # For VAE features
        all_reduce(mean_vae_local, op=ReduceOp.SUM, group=pg)
        mean_vae_global = mean_vae_local / world_size
        
        all_reduce(std_vae_local, op=ReduceOp.SUM, group=pg)
        std_vae_global = std_vae_local / world_size
        
        # Use synchronized global statistics
        mean_h_scalar = mean_h_global.mean().detach()
        std_h_scalar = std_h_global.mean().detach()
        mean_vae_scalar = mean_vae_global.mean().detach() 
        std_vae_scalar = std_vae_global.mean().detach()
    else:
        # Fallback to local statistics if no process group
        mean_h_scalar = mean_h_local.mean().detach()
        std_h_scalar = std_h_local.mean().detach()
        mean_vae_scalar = mean_vae_local.mean().detach()
        std_vae_scalar = std_vae_local.mean().detach()
    
    # Apply distribution matching
    h_normed = (h - mean_h_scalar) / (std_h_scalar + eps)
    h_aligned = h_normed * std_vae_scalar + mean_vae_scalar

    return h_aligned

class DINOPatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels: int=1024,
        out_channels: int=4096,
        keep_spatio=False,
        bias=True,
        num_blocks = 3
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
        self.proj = nn.Sequential(
                Rearrange("b c t h w -> b t h w c"),
                nn.Linear(in_channels, out_channels, bias=bias),
            )
        self.dim=in_channels
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)
        
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
        x = self.proj(x)
        return x


class DINOPatchEmbedTimeComp(nn.Module):
    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels: int=1024,
        out_channels: int=4096,
        frame_chunk=4,
        bias=True
    ):
        """
        Forward pass of the PatchEmbed module.

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
        super().__init__()
        # patched_channels = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        
        # self.frame_chunk = frame_chunk
        # merge T chunk
        self.merge = nn.Linear(in_channels, out_channels, bias=bias)
        # downsample HW
        self.proj = nn.Linear(out_channels, out_channels, bias=bias)
        
        # self.merge_in_dim = in_channels*frame_chunk
        self.merge_in_dim = in_channels
        self.proj_in_dim = out_channels
        
        self.init_weights()
        
        
    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.merge_in_dim)
        torch.nn.init.trunc_normal_(self.merge.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.proj_in_dim)
        torch.nn.init.trunc_normal_(self.proj.weight, std=std, a=-3 * std, b=3 * std)
        
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
        # merge x from 121 frames to 16 frames
        # first reshape to chunk of 8 frames, then feed through a linear layer to compress the 8 frames to 1 frame
        x = rearrange(x, "b c t h w -> b t h w c")
        x = self.merge(x)
        x = self.proj(x)
        # x = rearrange(x, "b c t h w -> b t h w c")
        return x
    
class DINOProjector(nn.Module):
    def __init__(
        self,
        in_channels: int=1024,
        out_channels: int=16,
        bias=True,
        dist_align=True,
        random_drop=True
    ):
        """
        """
        super().__init__()
        self.dist_align = dist_align
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        if self.training and random_drop:
            self.drop_rate = 0.2
        else:
            self.drop_rate = 0.0
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.conv2.out_channels)
        torch.nn.init.trunc_normal_(self.conv2.weight, std=std, a=-3 * std, b=3 * std)
        
    def enable_context_parallel(self, process_group: ProcessGroup):
        self._cp_group = process_group

    def disable_context_parallel(self):
        self._cp_group = None
    
    def _get_synchronized_random_drop(self):
        """Get synchronized random value for dropout across context parallel group"""
        if self._cp_group is None:
            # If no context parallel group, use local random
            return torch.rand(1).item()
        
        # Generate random value on rank 0 and broadcast to all processes in group
        if get_process_group_ranks(self._cp_group) == 0:
            random_val = torch.rand(1)
        else:
            random_val = torch.empty(1)
        
        # Broadcast the random value to all processes in the context parallel group
        broadcast(random_val, 0, group=self._cp_group)
        
        return random_val.item()
    
    def forward(self, dino_latent: torch.Tensor=None, edge_latent: torch.Tensor=None):

        """
         Forward pass of the DINOPatchEmbed module.

        Parameters:
        - dino_latent (torch.Tensor): The input tensor of shape (B, C, T, H//2, W//2) 
        - edge_latent (torch.Tensor): The input tensor of shape (B, C, T, H, W) 
        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        dino_latent = self.conv1(dino_latent)
        dino_latent = self.conv2(dino_latent)
        # edge_latent = rearrange(edge_latent, "(b t) c h w -> b t h w c", b=B, t=T)
        if self.dist_align:
            match_distribution(dino_latent, edge_latent, self._cp_group)
        # context parallel synchronized random drop
        if self._get_synchronized_random_drop() < self.drop_rate:
            edge_latent = torch.zeros_like(edge_latent)
            
        return torch.cat([dino_latent, edge_latent], dim=1)
    

class DINOProjectorV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 16,
        hidden_dim: int = None,  # 新增：允许自定义中间层维度
        bias: bool = True,
        dist_align: bool = True,
        random_drop: bool = True,
        drop_rate: float = 0.2   # 将 drop_rate 参数化
    ):
        """
        DINO Projector with enhanced MLP structure (Linear -> LN -> GELU -> Linear).
        """
        super().__init__()
        self.dist_align = dist_align
        self.random_drop = random_drop
        self.drop_rate = drop_rate
        self._cp_group = None
        
        # 默认 hidden_dim 与输入保持一致，这是 Projector 的常见做法
        _hidden_dim = hidden_dim if hidden_dim is not None else in_channels

        # 优化结构：使用 Sequential 构建 MLP Block
        # 1. Linear Project
        # 2. LayerNorm (稳定分布)
        # 3. GELU (比 ReLU 更适合 Vision Transformer 类任务)
        # 4. Linear Out
        self.proj = nn.Sequential(
            nn.Linear(in_channels, _hidden_dim, bias=bias),
            nn.LayerNorm(_hidden_dim),
            nn.GELU(),
            nn.Linear(_hidden_dim, out_channels, bias=bias)
        )
        
        self.init_weights()

    def init_weights(self) -> None:
        # 自动遍历 Sequential 中的所有 Linear 层进行初始化
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                std = 1.0 / math.sqrt(m.out_features)
                torch.nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enable_context_parallel(self, process_group: ProcessGroup):
        self._cp_group = process_group

    def disable_context_parallel(self):
        self._cp_group = None
    
    def _get_synchronized_random_drop(self):
        """Get synchronized random value for dropout across context parallel group"""
        if self._cp_group is None:
            # If no context parallel group, use local random
            return torch.rand(1).item()
        
        # 获取当前设备
        device = torch.cuda.current_device()
        
        # Generate random value on rank 0 and broadcast to all processes in group
        if get_process_group_ranks(self._cp_group) == 0:
            random_val = torch.rand(1, device=device)  # 在GPU上创建
        else:
            random_val = torch.empty(1, device=device)  # 在GPU上创建空tensor
        
        # Broadcast the random value to all processes in the context parallel group
        broadcast(random_val, 0, group=self._cp_group)
        
        return random_val.item()
    
    def forward(self, dino_latent: torch.Tensor, edge_latent: torch.Tensor):
        """
        Forward pass.
        dino_latent: (B, C, T, H//2, W//2)
        edge_latent: (B, C, T, H, W)
        """
        # (B, C, T, H, W) -> (B, T, H, W, C) for Linear layer
        dino_latent = rearrange(dino_latent, "b c t h w -> b t h w c")
        
        # 通过优化后的 MLP 网络
        dino_latent = self.proj(dino_latent)
        
        # 还原维度 (B, T, H, W, C) -> (B, C, T, H, W)
        dino_latent = rearrange(dino_latent, "b t h w c -> b c t h w")

        if self.dist_align:
            # 假设 match_distribution 是外部可用的函数
            try:
                match_distribution(dino_latent, edge_latent, self._cp_group)
            except NameError:
                pass # 这里仅作为占位，如果 match_distribution 未导入会报错

        # Dropout Logic 修复：
        # 必须在 forward 中动态检查 self.training，否则 model.eval() 时 dropout 依然生效
        if self.training and self.random_drop:
            # 获取同步的随机值
            rand_val = self._get_synchronized_random_drop()
            if rand_val < self.drop_rate:
                # 保留原有逻辑：drop 掉 edge_latent (置零)
                edge_latent = torch.zeros_like(edge_latent)
            
        return torch.cat([dino_latent, edge_latent], dim=1)
    

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


class MinimalV4LVGControlVaceDiT(MiniTrainDITImageContext):
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

        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        nf = kwargs["model_channels"]
        hint_nf = kwargs.pop("hint_nf", [nf, nf, nf, nf, nf, nf, nf, nf])
        self.dino_ctrl_channels = kwargs.pop("dino_ctrl_channels", None)
        self.sample_dino_key_frame = kwargs.pop("sample_dino_key_frame", None)
        self.dino_projector_type = kwargs.pop("dino_projector_type", "v2")
        self.dino_out_channels = kwargs.pop("dino_out_channels", 64)
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
        in_channels = self.in_channels - 1  # subtract the condition mask
        self.vace_in_channels = (in_channels + spatial_compression_factor**2) if vace_has_mask else in_channels
        self.vace_in_channels *= num_max_modalities
        self.vace_in_channels += 1  # adding the condition mask back

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
        if self.dino_projector_type == "v1":
            self.dino_projector = DINOProjector(in_channels=self.dino_ctrl_channels,out_channels=self.dino_out_channels)
        elif self.dino_projector_type == "v2":
            self.dino_projector = DINOProjectorV2(in_channels=self.dino_ctrl_channels,out_channels=self.dino_out_channels)
        else:
            raise TypeError(f"Unknown dino_projector_type: {self.dino_projector_type}")

        self.init_weights()
        sac_config = kwargs.get("sac_config", SACConfig())
        self.enable_selective_checkpoint(sac_config, self.blocks)
        if self.num_control_branches > 1:
            for nc in range(self.num_control_branches):
                self.enable_selective_checkpoint(sac_config, getattr(self, f"control_blocks_{nc}"))
        else:
            self.enable_selective_checkpoint(sac_config, self.control_blocks)

    def build_patch_embed_dino_vace(self):
        if self.sample_dino_key_frame:
            (
                concat_padding_mask,
                in_channels,
                patch_spatial,
                patch_temporal,
                model_channels,
            ) = (
                self.concat_padding_mask,
                self.vace_in_channels,
                self.patch_spatial,
                self.patch_temporal,
                self.model_channels,
            )
        else:
            raise NotImplementedError("only support sample_dino_key_frame=True for now")
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        
        self.control_embedder = PatchEmbed(
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
            if hasattr(self, "dino_projector"):
                self.dino_projector.init_weights()
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

        self.dino_projector.disable_context_parallel()

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
        
        self.dino_projector.enable_context_parallel(process_group=process_group)

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
            if control_B_C_T_H_W.shape[1] < self.vace_in_channels - 1:
                pad_C = self.vace_in_channels - 1 - control_B_C_T_H_W.shape[1]
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

        latent_control_edge, latent_control_dino = latent_control_input[:,:16], latent_control_input[:,16:]
        latent_control_input = self.dino_projector(latent_control_dino, latent_control_edge)
        
        control_B_C_T_H_W = latent_control_input
        control_B_C_T_H_W = _pad_control_input(control_B_C_T_H_W)

        def _prepare_transformer_input(x_B_C_T_H_W, embedder):
            if data_type == DataType.VIDEO:
                H, W = x_B_C_T_H_W.shape[-2:]
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W[:, :, :,:H, :W].type_as(x_B_C_T_H_W)], dim=1)
            else:
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, torch.zeros_like(x_B_C_T_H_W[:, :1])], dim=1)

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
        if self.separate_embedders:
            x_B_T_H_W_D_for_control, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = (
                _prepare_transformer_input(x_B_C_T_H_W, embedder=self.x_embedder_for_control_branch)
            )
        else:
            x_B_T_H_W_D_for_control = x_B_T_H_W_D

        control_B_T_H_W_D, rope_emb_L_1_1_D_for_control, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D_for_control = (
            _prepare_transformer_input(
                control_B_C_T_H_W,
                embedder=self.control_embedder,
            )
        )

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
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
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



def test_dino_projector():
    torch.manual_seed(42)
    projector = DINOProjector(in_channels=1024)
    projector.eval()

    # 构造假数据
    B, C, T, H, W = 2, 1024, 4, 32, 32
    dino_latent = torch.randn(B, C, T, H//2, W//2)
    edge_latent = torch.randn(B, 16, T, H, W)

    # 前向传播
    with torch.no_grad():
        out = projector(dino_latent, edge_latent)

    print(f"Input DINO latent shape: {dino_latent.shape}")
    print(f"Input EDGE latent shape: {edge_latent.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")

if __name__ == "__main__":
    test_dino_projector()
