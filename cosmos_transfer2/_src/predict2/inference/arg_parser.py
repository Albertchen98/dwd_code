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

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Text2Image inference script."""
    parser = argparse.ArgumentParser(description="ControlVideo2World inference script")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs used to run inference in parallel.")
    parser.add_argument("--seed", type=int, default=2025, help="Seed")
    parser.add_argument(
        "--ckpt_paths",
        type=str,
        default="",
        help="Paths to the checkpoints for multicontrol, separated by comma",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--is_av_model", action="store_true", help="Test with the general or Sample-AV transfer2 model.", default=False
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720",
        help="Resolution of the video (720-default, 480, etc)",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        help="Number of frames that later chunks take as condition from the previously-generated chunk when generating long videos in the autoregressive, chunk-wise manner.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=35,
        help="Number of sampling steps for the model.",
    )
    parser.add_argument("--num_video_frames_per_chunk", type=int, default=93, help="Number of video frames per chunk")
    parser.add_argument(
        "--preset_edge_threshold",
        type=str,
        default="medium",
        help="Preset strength for the canny edge detection (very_low, low, medium, high, very_high). Used for edge control.",
    )
    parser.add_argument(
        "--skip_load_model",
        action="store_true",
        help="Whether to skip loading model from checkpoint.",
    )
    parser.add_argument("--prompt_folder", type=str, default="", help="Folder of input prompts")
    parser.add_argument("--prompt_path", type=str, default="", help="Filepath of prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for inference")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for inference")
    parser.add_argument("--save_root", type=str, default="results", help="Save root")
    parser.add_argument("--guidance", type=int, nargs="+", default=[7], help="List of integers")
    parser.add_argument(
        "--sigma_max", type=float, default=None, help="Noise level added to the input video. Max value is 200."
    )
    parser.add_argument(
        "--not_keep_input_resolution",
        action="store_true",
        help="Whether to not keep the exact dimension of the input video. If not provided, will keep the input resolution. Otherwise, will output the default resolution\
        of Cosmos Transfer2/Predict2 according to the input video aspect ratio.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for t5 model")
    parser.add_argument(
        "--base_load_from",
        type=str,
        default=None,
        help="Path to the base model checkpoint.",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--shift", type=int, default=None, help="shift of time schedule strategy during inference")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save image.",
    )
    parser.set_defaults(use_neg_prompt=True)
    
    return parser.parse_args()
