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
import pickle
import traceback
from pathlib import Path
import warnings
from typing import Any, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T
from safetensors.torch import load_file
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo
from einops import rearrange
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2.datasets.utils import  VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import AddControlInputEdge

"""
Test the dataset with the following command:
python -m cosmos_predict1.diffusion.training.datasets.dataset_repa
"""
def dino_func(dino_path, frame_ids, video_h, video_w):
    dino_feat = load_file(dino_path)["dino"]
    H, W = video_h//16, video_w//16
    dino_feat = dino_feat[frame_ids, 5:]
    dino_feat = rearrange(dino_feat, "T (H W) C -> C T H W", H=H, W=W)
    return dino_feat

def canny_func(video):
    pass

class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        num_video_frames,
        resolution,
        len_t5: int = 512,
        t5_dim: int = 1024,
        hint_keys: str = "edge,dino",
    ):
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        h, w = VIDEO_RES_SIZE_INFO[resolution]["9,16"]
        # log.info(f"h and w are {h} {w}")
        self.dataset_dir = dataset_dir
        assert os.path.exists(self.dataset_dir), f"dataset_dir {self.dataset_dir} does not exist."
        self.sequence_length = num_video_frames

        self.hint_keys =  [f"control_input_{k}" for k in hint_keys.split(",")]

        self.len_t5 = len_t5
        self.t5_dim = t5_dim

        video_dir = os.path.join(self.dataset_dir, "videos", "pinhole_front")
        # self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl", "pinhole_front")
        self.t5_dir = os.path.join(self.dataset_dir, "cosmos_reason_xxl")
        # self.caption_dir = os.path.join(self.dataset_dir, "videos", "captions_qwenvl25_32b_qwen3", "pinhole_front")
         
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_paths = sorted(self.video_paths)

        for key in self.hint_keys:
            if "dino" in key:
                self.dino_dir = os.path.join(self.dataset_dir, "dinov3-vitl16")
                assert os.path.exists(self.dino_dir), f"dino dir {self.dino_dir} does not exist."
                
            if "edge" in key:
                self.edge_func = AddControlInputEdge(input_keys=["video"])

        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess(tuple([h, w]))])

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)
    
    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            log.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_video(self, video_path) -> Tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            # If there are not enough frames, let it fail
            warnings.warn(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )
            raise ValueError(f"Video {video_path} has insufficient frames.")

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache
        fps=16.0
        del vr  # delete the reader to avoid memory leak

        return frame_data, fps, frame_ids

    def _get_frames(self, video_path: str) -> Tuple[torch.Tensor, float]:
        frames, fps, frame_ids = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps, frame_ids

    def __getitem__(self, index) -> dict | Any:
        data = dict()
        video_path = self.video_paths[index]
        dino_path = os.path.join(self.dino_dir, os.path.basename(video_path).replace(".mp4", ".safetensors"))
        video, fps, frame_ids = self._get_frames(video_path)

        video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
        _, _, h, w = video.shape
        #TODO hint_key
        #based on hint_key to do preprocessing
        data["video"] = video

        stem = os.path.splitext(os.path.basename(video_path))[0]
        t5_embedding_path = os.path.join(self.t5_dir, f"{stem}.safetensors")
        
        text_embeddings = load_file(t5_embedding_path)["text_embedding"]
        text_embeddings = text_embeddings.squeeze()
        data["t5_text_embeddings"] = text_embeddings
        data["t5_text_mask"] = torch.ones(text_embeddings.shape[0])
        data["fps"] = fps
        data["image_size"] = torch.tensor([h, w, h, w])
        data["num_frames"] = self.sequence_length
        data["padding_mask"] = torch.zeros(1, h, w) 
        
        for key in self.hint_keys:
            if "edge" in key:
                data = self.edge_func(data)
            if "dino" in key:
                dino_path = os.path.join(self.dino_dir, os.path.basename(video_path).replace(".mp4", ".safetensors"))
                control_input_dino = dino_func(dino_path, frame_ids, h, w)
                data["control_input_dino"] = control_input_dino

        return data

if __name__ == "__main__":
    dataset = Dataset(
        dataset_dir="/cache/waymo",
        num_video_frames=121,
        resolution="720",
        hint_keys="edge"
    )
    print("start debugging")
    from torch.utils.data import DataLoader
    video_dataset_cosmos_waymo_repa_8gpu_48gb = dataset
    video_dataloader_cosmos_waymo_repa_8gpu_48gb = DataLoader(
                    dataset=video_dataset_cosmos_waymo_repa_8gpu_48gb,
                    batch_size=1,
                    drop_last=True,
                    num_workers=8,
                    pin_memory=True)

    data_iter = iter(video_dataloader_cosmos_waymo_repa_8gpu_48gb)
    for idx in range(10000):
        data = next(data_iter)
        log.info(
            (
                f"{idx=} "
                f"{data['video'].sum()=}\n"
                f"{data['video'].shape=}\n"
                f"{data['control_input_dino'].shape=}\n"
                f"{data['control_input_edge'].shape=}\n"
                f"{data['t5_text_embeddings'].shape=}\n"
                f"{data['t5_text_mask'].shape=}\n"
                "---"
            )
        )
