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

"""
Usage:
python scripts/distcp_to_pt.py \
    --download_checkpoints \
    --s3_path s3://bucket/cosmos_transfer2/vid2vid_2B_control/edge_vis_720p_t24_oldsde_spaced_layer14_mlp_hqv3_20250711_base100k_64N/checkpoints/iter_000057000 \
    --s3_profile aws \
    --convert_checkpoint \
    --save_path checkpoints/nvidia/Cosmos-Transfer2-2B
"""

import argparse
import os
import subprocess

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="Checkpoint conversion from Distributed Checkpoint to PyTorch format")
    # whether to download checkpoints from S3 (can be skipped if checkpoints are already downloaded)
    parser.add_argument("--download_checkpoint", action="store_true",
                        help="Download checkpoints from S3 before conversion")
    # whether to convert downloaded checkpoints to .pt format
    parser.add_argument("--convert_checkpoint", action="store_true",
                        help="Convert downloaded checkpoints to .pt format")
    parser.add_argument("--s3_path", type=str, default="s3://bucket/key", help="S3 path to the checkpoints directory")
    parser.add_argument("--s3_profile", type=str, default="aws", help="S3 profile to use for accessing the checkpoints")
    parser.add_argument("--save_path", type=str, default="checkpoints/nvidia/Cosmos-Transfer2-2B",
                        help="Local path to save the converted checkpoints")
    # fmt: on
    args = parser.parse_args()

    return args


def download_checkpoint(s3_path: str, s3_profile: str, save_path: str) -> None:
    """
    Download checkpoints from S3 to the local directory.
    This function is a placeholder and should be implemented based on your S3 access method.
    """
    # Implement the logic to download checkpoints from S3
    print(f"Downloading checkpoints from {s3_path} with profile {s3_profile} to {save_path}")
    dcp_checkpoint_s3_path = os.path.join(s3_path, "model")
    dcp_checkpoint_dir = os.path.join(save_path, "model")
    # # configure max_concurrent_requests
    # result = subprocess.run(f"aws configure get profile.{s3_profile}.s3.max_concurrent_requests", shell=True, capture_output=True, text=True)
    # current_max_concurrent_requests = result.stdout.strip()

    # result = subprocess.run(f"aws configure set profile.{s3_profile}.s3.max_concurrent_requests 100", shell=True, capture_output=True, text=True)

    command = f"aws s3 sync --profile {s3_profile} {dcp_checkpoint_s3_path} {dcp_checkpoint_dir}"
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading checkpoints: {result.stderr}")
        raise RuntimeError(f"Failed to download checkpoints from {s3_path} to {save_path}. Error: {result.stderr}")
    else:
        print(f"Checkpoints downloaded successfully to {save_path}")

    return


def convert_checkpoint(save_path: str = "checkpoints/nvidia/Cosmos-Transfer2-2B") -> None:
    dcp_checkpoint_dir = os.path.join(save_path, "model")
    if not os.path.exists(dcp_checkpoint_dir):
        print(f"Checkpoint directory {dcp_checkpoint_dir} does not exist, skipping conversion.")
        return

    print(f"Converting checkpoints in {save_path}...")
    torch_save_path_ema_reg = os.path.join(save_path, "model_ema_reg.pt")
    torch_save_path_ema_only_fp32 = torch_save_path_ema_reg.replace("_ema_reg.pt", "_ema_fp32.pt")
    torch_save_path_ema_only_bf16 = torch_save_path_ema_reg.replace("_ema_reg.pt", ".pt")

    # 1. Convert distributed checkpoint to torch single checkpoint
    if os.path.exists(torch_save_path_ema_reg):
        print(f"{torch_save_path_ema_reg} already exists, skipping.")
    else:
        dcp_to_torch_save(dcp_checkpoint_dir, torch_save_path_ema_reg)
        print(f"Converted {dcp_checkpoint_dir} to {torch_save_path_ema_reg}")

    # 2. Drop Reg keys and save EMA weights only in fp32 precision
    if os.path.exists(torch_save_path_ema_only_fp32):
        print(f"{torch_save_path_ema_only_fp32} already exists, skipping.")
    else:
        state_dict_ema_reg = torch.load(torch_save_path_ema_reg, map_location="cpu", weights_only=False)
        state_dict_ema_only_fp32 = dict()  # ema only
        for key in state_dict_ema_reg:
            if key.startswith("net_ema."):
                key_new = key.replace("net_ema.", "net.")
                state_dict_ema_only_fp32[key_new] = state_dict_ema_reg[key]

        torch.save(state_dict_ema_only_fp32, torch_save_path_ema_only_fp32)
        print(f"Saved EMA fp32 weights from {torch_save_path_ema_reg} to {torch_save_path_ema_only_fp32}")

    # 3. Save EMA weights only in bf16 precision
    if os.path.exists(torch_save_path_ema_only_bf16):
        print(f"{torch_save_path_ema_only_bf16} already exists, skipping.")
    else:
        if "state_dict_ema_only_fp32" not in locals():
            state_dict_ema_only_fp32 = torch.load(torch_save_path_ema_only_fp32, map_location="cpu", weights_only=False)

        state_dict_ema_only_bf16 = dict()  # ema only
        for key in state_dict_ema_only_fp32:
            if (
                isinstance(state_dict_ema_only_fp32[key], torch.Tensor)
                and state_dict_ema_only_fp32[key].dtype == torch.float32
            ):
                state_dict_ema_only_bf16[key] = state_dict_ema_only_fp32[key].bfloat16()
            else:
                state_dict_ema_only_bf16[key] = state_dict_ema_only_fp32[key]

        torch.save(state_dict_ema_only_bf16, torch_save_path_ema_only_bf16)
        print(f"fp32 -> bf16: {torch_save_path_ema_only_fp32} to {torch_save_path_ema_only_bf16}")

# python scripts/distcp_to_pt.py --convert_checkpoint --save_path /home/ma-user/work/x50026424/output_weights/cosmos2_waymo/checkpoints/cosmos_transfer2/vid2vid_2B_control/vid2vid_2B_control_720p_t24_dense_dino_control_layer4_cr1_embedding_rectified_flow_train_keyframe/checkpoints/iter_000010000
# vid2vid_2B_control_720p_t24_dense_dino_control_layer4_cr1_embedding_rectified_flow_train_keyframe
if __name__ == "__main__":
    args = parse_args()
    if args.download_checkpoint:
        download_checkpoint(args.s3_path, args.s3_profile, args.save_path)
    if args.convert_checkpoint:
        convert_checkpoint(args.save_path)
    print("All checkpoints converted successfully.")

# /home/ma-user/work/x50026424/output_weights/cosmos2_waymo/checkpoints/cosmos_transfer2/vid2vid_2B_control/vid2vid_2B_control_720p_t24_dinocontrol_layer4_cr1_embedding_rectified_flow_train_keyframe/checkpoints/iter_000040000
# /home/ma-user/work/x50026424/output_weights/cosmos2_waymo/checkpoints/cosmos_transfer2/vid2vid_2B_control/vid2vid_2B_control_720p_t24_dino_control_layer4_cr1_embedding_rectified_flow_train_keyframe_nuplan/checkpoints/iter_000020000
# /home/ma-user/work/x50026424/output_weights/cosmos2_nuplan/checkpoints/cosmos_transfer2/img2img_2B_control/img2img_2B_control_720p_dinoedge_control_layer4_cr1_embedding/checkpoints/iter_000004000/
# /home/ma-user/work/x50026424/output_weights/cosmos2_nuplan/checkpoints/cosmos_transfer2/img2img_2B_control/img2img_2B_control_720p_dino_control_dinotok_0t_23t_layer4_t5_embedding_color_jitter/checkpoints/iter_000004000
