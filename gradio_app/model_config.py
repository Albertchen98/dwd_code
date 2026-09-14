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

from dataclasses import dataclass, field
from typing import Dict
import json


default_request_vis = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": "1.0",
        "vis": {"control_path": "assets/robot_example/vis/robot_vis.mp4"},
    },
    indent=2,
)

default_request_depth = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": "1.0",
        "depth": {"control_path": "assets/robot_example/depth/robot_depth.mp4"},
    },
    indent=2,
)

default_request_edge = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": "1.0",
        "edge": {"control_path": "assets/robot_example/edge/robot_edge.mp4", "preset_edge_threshold": "medium"},
    },
    indent=2,
)

default_request_seg = json.dumps(
    {
        "prompt_path": "assets/robot_example/robot_prompt.txt",
        "video_path": "assets/robot_example/robot_input.mp4",
        "control_weight": "1.0",
        "seg": {"control_path": "assets/robot_example/seg/robot_seg.mp4"},
    },
    indent=2,
)

default_request_mv = json.dumps(
    {
        "input_root": "datasets/multiview_example/input_videos",
        "control_root": "datasets/multiview_example/world_scenario_videos",
        "n_views": 7,
        "num_conditional_frames": 2,
        "control_weight": 1.0,
        "use_apg": 0,
    },
    indent=2,
)

help_text_vis = """
                    ### Generation Parameters:
                    - `prompt` (string): Text description of desired output
                    - `video_path` (string): Path to input video file
                    - `control_weight` (float): Weight for the control signal (default: 1.0)
                    - `sigma_max` (float | None): Max noise level (0–80). Higher = more variation
                    - `num_conditional_frames` (int): Conditional frames passed from one chunk to the next.

                    """
help_text_depth = help_text_vis
help_text_edge = help_text_vis
help_text_seg = help_text_vis
help_text_mv = help_text_vis


@dataclass
class Config:
    header: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": "Cosmos-Transfer2.5 Blur Transfer",
            "depth": "Cosmos-Transfer2.5 Depth Transfer",
            "edge": "Cosmos-Transfer2.5 Edge Transfer",
            "seg": "Cosmos-Transfer2.5 Segmentation Transfer",
            "multiview": "Cosmos-Transfer2.5 Multiview",
        }
    )

    help_text: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": help_text_vis,
            "depth": help_text_depth,
            "edge": help_text_edge,
            "seg": help_text_seg,
            "multiview": "",
        }
    )

    default_request: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": default_request_vis,
            "depth": default_request_depth,
            "edge": default_request_edge,
            "seg": default_request_seg,
            "multiview": default_request_mv,
        }
    )
