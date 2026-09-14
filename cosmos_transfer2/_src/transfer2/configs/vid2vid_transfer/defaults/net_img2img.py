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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_transfer2._src.transfer2.networks.minimal_v4_t2i_dit_control_vace_dino import MinimalV4ControlVaceDiT as MinimalV4ControlVaceDiTDino
from cosmos_transfer2._src.transfer2.networks.minimal_v4_t2i_dit_control_vace_dino_edge import MinimalV4ControlVaceDiT as MinimalV4ControlVaceDiTDinoEdge

from cosmos_transfer2._src.transfer2.networks.minimal_v4_t2i_dit_control_vace import MinimalV4ControlVaceDiT
from dataclasses import dataclass

# @dataclass
# class DINOMergeConfig:
#     in_features: int=1024
#     compr_shallow: bool = False
#     compr_deep: bool = False
    
#     dino_merge_config=DINOMergeConfig(),

TRANSFER2_CONTROL2IMAGE_NET_2B = L(MinimalV4ControlVaceDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)

TRANSFER2_DINOCONTROL2IMAGE_NET_2B = L(MinimalV4ControlVaceDiTDino)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
    dino_ctrl_channels=1024,
    use_dino_merge=False,
    use_dino_pca=False,
)

TRANSFER2_DINOEDGECONTROL2IMAGE_NET_2B = L(MinimalV4ControlVaceDiTDinoEdge)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
    dino_ctrl_channels=1024,
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="transfer2_dinocontrol2image_net_2B",
        node=TRANSFER2_DINOCONTROL2IMAGE_NET_2B,
    )
    
    cs.store(
        group="net",
        package="model.config.net",
        name="transfer2_control2image_net_2B",
        node=TRANSFER2_CONTROL2IMAGE_NET_2B,
    )
    
    cs.store(
        group="net",
        package="model.config.net",
        name="transfer2_dinoedge_control2image_net_2B",
        node=TRANSFER2_DINOEDGECONTROL2IMAGE_NET_2B,
    )