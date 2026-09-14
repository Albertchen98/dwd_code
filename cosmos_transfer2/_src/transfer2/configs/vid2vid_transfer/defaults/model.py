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

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace import (
    ControlVideo2WorldConfig,
    ControlVideo2WorldModel,
)
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import (
    ControlVideo2WorldModelRectifiedFlow,
    ControlVideo2WorldRectifiedFlowConfig,
)

from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_dino import (
    ControlDino2WorldModelRectifiedFlow,
    ControlDino2WorldRectifiedFlowConfig,
)

from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_dino_prep import (
    ControlDinoPrep2WorldModelRectifiedFlow,
    ControlDinoPrep2WorldRectifiedFlowConfig,
)


from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_dino_edge import (
    ControlDinoEdge2WorldModelRectifiedFlow,
    ControlDinoEdge2WorldRectifiedFlowConfig,
)


DDP_CONFIG_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(ControlVideo2WorldModel)(
        config=ControlVideo2WorldConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlVideo2WorldModel)(
        config=ControlVideo2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlVideo2WorldModelRectifiedFlow)(
        config=ControlVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


FSDP_CONFIG_DINO_EDGE_CONTROL_VACE_RECTIFIED_FLOW = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlDinoEdge2WorldModelRectifiedFlow)(
        config=ControlDinoEdge2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

FSDP_CONFIG_DINO_CONTROL_VACE_RECTIFIED_FLOW = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlDino2WorldModelRectifiedFlow)(
        config=ControlDino2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

FSDP_CONFIG_DINO_PREP_CONTROL_VACE_RECTIFIED_FLOW = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlDinoPrep2WorldModelRectifiedFlow)(
        config=ControlDinoPrep2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_control_vace", node=DDP_CONFIG_CONTROL_VACE)
    cs.store(group="model", package="_global_", name="fsdp_control_vace", node=FSDP_CONFIG_CONTROL_VACE)
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_control_vace_rectified_flow",
        node=FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW,
    )
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_dino_control_vace_rectified_flow",
        node=
        FSDP_CONFIG_DINO_CONTROL_VACE_RECTIFIED_FLOW,
    )
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_dino_prep_control_vace_rectified_flow",
        node=
        FSDP_CONFIG_DINO_PREP_CONTROL_VACE_RECTIFIED_FLOW,
    )
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_dino_edge_control_vace_rectified_flow",
        node=FSDP_CONFIG_DINO_EDGE_CONTROL_VACE_RECTIFIED_FLOW,
    )
    
