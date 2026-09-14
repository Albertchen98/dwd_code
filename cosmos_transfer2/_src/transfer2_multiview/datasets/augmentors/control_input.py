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

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import _maybe_torch_to_numpy


class AddControlInputHdmapBbox(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_hdmap_bbox"],
        args: Optional[dict] = None,
        use_random: Optional[bool] = True,
        preset_strength: Optional[str] = "medium",
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        self.preset_strength = preset_strength

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_hdmap_bbox" in data_dict:
            return data_dict
        key_input = self.input_keys[0]
        key_out = self.output_keys[0]
        data_dict[key_out] = data_dict[key_input]
        if "metas" in data_dict:
            del data_dict["metas"]
        return data_dict


def _batched_canny_rgb_with_threadpool(frames_cthw: np.ndarray, t_lower: int, t_upper: int) -> np.ndarray:
    """Compute Canny edges for a stack of frames in one OpenCV call."""
    frames_thwc = frames_cthw.transpose(1, 2, 3, 0)
    t, h, w, c = frames_thwc.shape
    w2, h2 = w // 2, h // 2

    def _process_frame(frame):
        small_frame = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(small_frame, t_lower, t_upper)
        return cv2.resize(edges, (w, h), interpolation=cv2.INTER_NEAREST)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(_process_frame, frames_thwc))
    return np.stack(results)[None]


class AddControlInputEdgeDownUp2X(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_edge"],
        args: Optional[dict] = None,
        use_random: Optional[bool] = True,
        preset_strength: Optional[str] = "medium",
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        self.preset_strength = preset_strength

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_edge" in data_dict:
            return data_dict
        key_img = self.input_keys[0]
        key_out = self.output_keys[0]
        frames = data_dict[key_img]
        # log.info(f"Adding control input edge. Input key: {key_img}, Output key: {key_out}. Use random: {self.use_random}, Preset strength: {self.preset_strength}")
        # Get lower and upper threshold for canny edge detection.
        if self.use_random:  # always on for training, always off for inference
            t_lower = np.random.randint(20, 100)  # Get a random lower thre within [0, 255]
            t_diff = np.random.randint(50, 150)  # Get a random diff between lower and upper
            t_upper = min(255, t_lower + t_diff)  # The upper thre is lower added by the diff
        else:
            if self.preset_strength == "none" or self.preset_strength == "very_low":
                t_lower, t_upper = 20, 50
            elif self.preset_strength == "low":
                t_lower, t_upper = 50, 100
            elif self.preset_strength == "medium":
                t_lower, t_upper = 100, 200
            elif self.preset_strength == "high":
                t_lower, t_upper = 200, 300
            elif self.preset_strength == "very_high":
                t_lower, t_upper = 300, 400
            else:
                raise ValueError(f"Preset {self.preset_strength} not recognized.")
        frames = _maybe_torch_to_numpy(frames)
        is_image = len(frames.shape) < 4
        # Compute the canny edge map by the two thresholds.
        if is_image:
            edge_maps = cv2.Canny(frames, t_lower, t_upper)[None, None]
        else:
            edge_maps = _batched_canny_rgb_with_threadpool(frames, t_lower, t_upper)
        edge_maps = torch.from_numpy(edge_maps).expand(3, -1, -1, -1)
        if is_image:
            edge_maps = edge_maps[:, 0]
        data_dict[key_out] = edge_maps
        return data_dict
