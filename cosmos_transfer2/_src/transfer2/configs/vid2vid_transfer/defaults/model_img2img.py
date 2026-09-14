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

from cosmos_transfer2._src.transfer2.models.img2img_model_control_vace import (
    ControlImage2ImageModel,
    ControlImage2ImageConfig,
)

from cosmos_transfer2._src.transfer2.models.img2img_model_control_vace_dino import (
    ControlDino2ImageModel,
    ControlDino2ImageConfig,
)

from cosmos_transfer2._src.transfer2.models.img2img_model_control_vace_dino_edge import (
    ControlDinoEdge2ImageModel,
    ControlDinoEdge2ImageConfig,
)

DDP_CONFIG_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(ControlImage2ImageModel)(
        config=ControlImage2ImageConfig(),
        _recursive_=False,
    ),
)


DDP_CONFIG_DINO_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(ControlDino2ImageModel)(
        config=ControlDino2ImageConfig(),
        _recursive_=False,
    ),
)

DDP_CONFIG_DINO_EDGE_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(ControlDinoEdge2ImageModel)(
        config=ControlDinoEdge2ImageConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG_DINO_CONTROL_VACE = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlDino2ImageModel)(
        config=ControlDino2ImageConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

def register_model():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="ddp_control_vace",
        node=DDP_CONFIG_CONTROL_VACE,
    )
    cs.store(
        group="model",
        package="_global_",
        name="ddp_dino_control_vace",
        node=DDP_CONFIG_DINO_CONTROL_VACE,
    )
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_dino_control_vace",
        node=FSDP_CONFIG_DINO_CONTROL_VACE,
    )
    cs.store(
        group="model",
        package="_global_",
        name="ddp_dino_edge_control_vace",
        node=DDP_CONFIG_DINO_EDGE_CONTROL_VACE,
    )
