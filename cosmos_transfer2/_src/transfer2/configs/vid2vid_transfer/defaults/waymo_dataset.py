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

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from cosmos_transfer2._src.transfer2.datasets.data_sources.waymo_dataset import Dataset as WaymoDataset
from cosmos_transfer2._src.transfer2.datasets.data_sources.nuplan_dataset import Dataset as NuplanDataset
from cosmos_transfer2._src.transfer2.datasets.data_sources.nuplan_dataset_images import Dataset as NuplanImageDataset

from torch.utils.data import DataLoader as _DataLoader

_VIDEO_LOADER = L(
    _DataLoader
)(
    dataset=L(WaymoDataset)(
        resolution="720",
        num_video_frames=93,  # number of video frames, the number needs to agree with tokenizer encoder since tokenizer can not handle arbitrary length
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

WAYMO_DATA_VIDEO_ONLY_CONFIG = _VIDEO_LOADER

_VIDEO_LOADER_NUPLAN = L(
    _DataLoader
)(
    dataset=L(NuplanDataset)(
        resolution="720",
        num_video_frames=93,  # number of video frames, the number needs to agree with tokenizer encoder since tokenizer can not handle arbitrary length
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)


_IMAGE_LOADER_NUPLAN = L(
    _DataLoader
)(
    dataset=L(NuplanImageDataset)(
        resolution="720",
        num_video_frames=1,  # number of video frames, the number needs to agree with tokenizer encoder since tokenizer can not handle arbitrary length
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

WAYMO_DATA_VIDEO_ONLY_CONFIG = _VIDEO_LOADER
NUPLAN_DATA_VIDEO_ONLY_CONFIG = _VIDEO_LOADER_NUPLAN
NUPLAN_DATA_IMAGE_ONLY_CONFIG = _IMAGE_LOADER_NUPLAN

