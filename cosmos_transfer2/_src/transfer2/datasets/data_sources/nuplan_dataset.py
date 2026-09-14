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
from cosmos_transfer2._src.transfer2.datasets.augmentors.blur import Blur
from torchvision.transforms import v2
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
ALTERNATIVE_CHANNEL=[4, 8, 12, 16, 20, 24, 28, 32]

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
        use_anyup_processed: bool = False, 
        anyup_pca_name: str = "dino_anyupx8_pca8_norm",
        pca_channel: int = None,
        anyup_factor: int = 8, # 默认是anyupx8（什么都不动），
        use_dino_blur: bool = False,  # 是否对dino输入应用blur
        use_pca_random_drop: int | None = None,  # 是否对time维度的pca通道进行随机截断
        use_random_channel: bool | None = None, # 是否对pca通道进行随机截断
        strict_feature_shape: bool = False,
        reverse_train: bool = False, # 是否对视频和特征进行reverse
    ):
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames
            use_anyup_processed (bool): Whether to load processed anyup features
            use_dino_blur (bool): Whether to apply blur to dino input (when not using anyup)

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
        self.pca_channel = pca_channel
        self.use_random_channel = use_random_channel
        self.anyup_factor = anyup_factor
        self.use_pca_random_drop = use_pca_random_drop
        self.strict_feature_shape = strict_feature_shape
        self.reverse_train = reverse_train
        # NEW: Initialize anyup configuration
        self.use_anyup_processed = use_anyup_processed
        if self.use_anyup_processed:
            # Based on user path: .../dataset/nuplan/videos/dino_anyupx8_pca8_norm/
            self.anyup_dir = os.path.join(self.dataset_dir, anyup_pca_name)

        # Initialize blur for dino
        self.use_dino_blur = use_dino_blur
        if self.use_dino_blur:
            self.blur = Blur(use_random=True)  # 训练时使用随机blur参数

        video_dir = os.path.join(self.dataset_dir, "videos", "pinhole_front")
        # self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl", "pinhole_front")
        self.t5_dir = os.path.join(self.dataset_dir, "cosmos_reason_xxl")
        # self.caption_dir = os.path.join(self.dataset_dir, "videos", "captions_qwenvl25_32b_qwen3", "pinhole_front")
         
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_paths = sorted(self.video_paths)

        for key in self.hint_keys:                
            if "edge" in key:
                self.edge_func = AddControlInputEdge(input_keys=["video"])

        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess(tuple([h, w]))])

        self.normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True), 
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

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
        start_frame = np.random.randint(0, max_start_idx + 1)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache
        fps = float(vr.get_avg_fps())
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
        video, fps, frame_ids = self._get_frames(video_path)
        video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
        if self.reverse_train:
            video = torch.flip(video, [1])  # Flip video Time dimension (dim 1)
            frame_ids = frame_ids[::-1]     # Flip frame indices
        _, _, h, w = video.shape
        #TODO hint_key
        #based on hint_key to do preprocessing
        data["video"] = video
        stem = os.path.splitext(os.path.basename(video_path))[0]
        t5_embedding_path = os.path.join(self.t5_dir, f"{stem}.pkl")
        safe_path = os.path.join(self.t5_dir, f"{stem}.safetensors")
        if os.path.isfile(safe_path):
            text_embeddings = load_file(safe_path)["text_embedding"]
        else:
            with open(t5_embedding_path, "rb") as f:
                text_embeddings = pickle.load(f)[0]
        # text_embeddings = load_file(t5_embedding_path)["text_embedding"]
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
                # Logic for anyup processed data
                if self.use_anyup_processed:
                    anyup_path = os.path.join(self.anyup_dir, f"{stem}.pt")
                    # if os.path.exists(anyup_path):
                        # Shape: [1, 8, 200, 352, 640] -> [B, C, T, H, W]
                        # We map map_location to cpu to save GPU memory during dataloading
                    anyup_tensor_B_C_T_H_W = torch.load(anyup_path, map_location="cpu", weights_only=True)
                    
                    if anyup_tensor_B_C_T_H_W.ndim != 5 or anyup_tensor_B_C_T_H_W.shape[0] != 1:
                        raise ValueError(f"{anyup_path}: expected [1,C,T,H,W]")
                    anyup_tensor_C_T_H_W = anyup_tensor_B_C_T_H_W.squeeze(0)
                    expected_hw = (h // 16 * self.anyup_factor, w // 16 * self.anyup_factor)
                    if self.strict_feature_shape:
                        if tuple(anyup_tensor_C_T_H_W.shape[-2:]) != expected_hw:
                            raise ValueError(f"{anyup_path}: expected spatial shape {expected_hw}")
                        if self.pca_channel and anyup_tensor_C_T_H_W.shape[0] != self.pca_channel:
                            raise ValueError(f"{anyup_path}: expected {self.pca_channel} PCA channels")
                    elif tuple(anyup_tensor_C_T_H_W.shape[-2:]) != expected_hw:
                        anyup_tensor_C_T_H_W = torch.nn.functional.interpolate(
                            anyup_tensor_C_T_H_W, size=expected_hw, mode="bilinear", align_corners=False
                        )
                    if max(frame_ids) >= anyup_tensor_C_T_H_W.shape[1]:
                        raise ValueError(f"{anyup_path}: cache has fewer frames than the selected video interval")
                    # 2. Slice using the SAME frame_ids as the video
                    # Logic: "anyup的采样逻辑和原始视频取帧数的采样逻辑一样" -> strict alignment
                    anyup_feat_C_T_H_W = anyup_tensor_C_T_H_W[:, frame_ids, :, :]
                    # kframes = 4采样, 采样应该放到代码里面去做，不能放到预处理里面去做，因为safetensors动态采样速度偏慢, 为啥不用safetensor呢？我之前代码都是safetensors的啊
                    # anyup_feat_C_T_H_W = anyup_feat_C_T_H_W[:, ::4, :, :]                        
                    if self.pca_channel:
                        anyup_feat_C_T_H_W = anyup_feat_C_T_H_W[:self.pca_channel, :, :, :]
                    if self.use_random_channel:
                        random_channel = random.choice(ALTERNATIVE_CHANNEL)
                        anyup_feat_C_T_H_W[random_channel:, :, :, :] = 0.0
                    data["control_input_dino"] = anyup_feat_C_T_H_W
                    continue  # Skip dino_func if anyup is used
                    # else:
                    #     log.warning(f"Anyup file not found: {anyup_path}")
                
                # 对video应用blur后再normalize
                if self.use_dino_blur:
                    # video: [C, T, H, W], blur expects [C, T, H, W] numpy array
                    video_np = video.numpy()  # [C, T, H, W]
                    blurred_video_np = self.blur(video_np)  # [C, T, H, W]
                    blurred_video = torch.from_numpy(blurred_video_np)
                    data["control_input_dino"] = self.normalize(blurred_video.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                else:
                    data["control_input_dino"] = self.normalize(video.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        return data

if __name__ == "__main__":
    import cv2
    import numpy as np
    import torch
    import os
    from cosmos_transfer2._src.transfer2.datasets.augmentors.blur import Blur
    
    # ============== 配置 ==============
    # 输入视频路径(修改为你的视频路径)
    video_path = "./nuplan_dataset/videos/pinhole_front/ff5114d8cdfd51b3_2021.06.09.18.23.43_veh-35_02680_02868.mp4"
    output_dir = "./results/blur_test_output"
    num_frames = 16  # 测试帧数
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ============== 读取视频 ==============
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_ids = list(range(min(num_frames, len(vr))))
    frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
    print(f"原始视频帧形状: {frames.shape}")
    
    # 转换为 [C, T, H, W] 格式 (blur期望的输入格式)
    frames_CTHW = frames.transpose(3, 0, 1, 2).astype(np.uint8)  # [C, T, H, W]
    print(f"转换后形状: {frames_CTHW.shape}")
    
    # ============== 应用Blur ==============
    blur = Blur(use_random=False)  # use_random=False使用固定参数,便于复现
    blurred_frames_CTHW = blur(frames_CTHW)  # [C, T, H, W]
    print(f"模糊后形状: {blurred_frames_CTHW.shape}")
    
    # ============== 保存对比图像 ==============
    # 转回 [T, H, W, C] 格式用于保存
    frames_THWC = frames_CTHW.transpose(1, 2, 3, 0)  # [T, H, W, C]
    blurred_frames_THWC = blurred_frames_CTHW.transpose(1, 2, 3, 0)  # [T, H, W, C]
    
    for i in range(min(5, len(frame_ids))):  # 保存前5帧对比
        original = frames_THWC[i]  # [H, W, C] RGB
        blurred = blurred_frames_THWC[i]  # [H, W, C] RGB
        
        # 水平拼接: 原图 | 模糊图
        comparison = np.concatenate([original, blurred], axis=1)
        
        # RGB -> BGR for cv2
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        
        save_path = os.path.join(output_dir, f"frame_{i:03d}_comparison.jpg")
        cv2.imwrite(save_path, comparison_bgr)
        print(f"保存对比图: {save_path}")
    
    # ============== 保存对比视频 ==============
    H, W = frames_THWC.shape[1], frames_THWC.shape[2]
    video_path_out = os.path.join(output_dir, "blur_comparison.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_out, fourcc, 8.0, (W * 2, H))  # 宽度x2因为左右拼接
    
    for i in range(len(frame_ids)):
        original = frames_THWC[i]
        blurred = blurred_frames_THWC[i]
        comparison = np.concatenate([original, blurred], axis=1)
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        out.write(comparison_bgr)
    
    out.release()
    print(f"\n保存对比视频: {video_path_out}")
    print(f"\n所有输出保存在: {output_dir}")
    print("左侧=原图, 右侧=模糊后")