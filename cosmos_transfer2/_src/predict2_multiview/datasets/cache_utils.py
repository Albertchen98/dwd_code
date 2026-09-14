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

import os
from pathlib import Path

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.object_store import sync_s3_dir_to_local


def get_cam_t5_cache_dir() -> str:
    local_dir = os.path.expanduser("~/.cache/imaginaire")
    cache_dir = os.environ.get("CAM_T5_EMBEDDINGS_CACHE_DIR", local_dir)
    if cache_dir.startswith("s3://"):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"Syncing cam_t5_embeddings_cache from {cache_dir} to {local_dir}", rank0_only=False)
        return sync_s3_dir_to_local(
            cache_dir,
            "credentials/s3_checkpoint.secret",
            local_dir,
            # use local rank sync so you can use it on both aws and lepton
            # since lepton does not have a shared storage across between nodes
            local_rank_sync=True,
            rank_sync=False,
        )
    return cache_dir
