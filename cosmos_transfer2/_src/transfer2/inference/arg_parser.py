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
    """Parses command-line arguments for the Transfer2 ControlVideo2World inference script."""
    parser = argparse.ArgumentParser(description="ControlVideo2World inference script")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="The experiment key name as specified in cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list",
    )
    parser.add_argument(
        "--ckpt_iter",
        type=str,
        required=True,
        help="The iteration info (including the 'iter_' prefix) of the checkpoint to use for inference. E.g. --ckpt_iter iter_000020000",
    )
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
        "--preset_blur_strength",
        type=str,
        default="medium",
        help="Preset strength for the blur strength (none, very_low, low, medium, high, very_high). Used for vis control.",
    )
    parser.add_argument(
        "--skip_load_model",
        action="store_true",
        help="Whether to skip loading model from checkpoint.",
    )
    parser.add_argument("--video_path", type=str, default="", help="Filepath of input video")
    parser.add_argument("--video_folder", type=str, default="", help="Folder of input videos")
    parser.add_argument("--prompt_folder", type=str, default="", help="Folder of input prompts")
    parser.add_argument("--prompt_path", type=str, default="", help="Filepath of prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for inference")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for inference")
    parser.add_argument("--save_root", type=str, default="results", help="Save root")
    parser.add_argument("--guidance", type=int, nargs="+", default=[7], help="List of integers")
    parser.add_argument(
        "--control_weight",
        type=str,
        default="1.0",
        help="Control weight for each hint key, separated by comma. Max value is 1.0",
    )
    parser.add_argument(
        "--sigma_max", type=float, default=None, help="Noise level added to the input video. Max value is 200."
    )
    parser.add_argument("--hint_key", type=str, default="edge", help="Hint key for inference")
    parser.add_argument(
        "--input_control_folder_edge",
        type=str,
        default=None,
        help="Folder containing pre-computed edge control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_edge",
        type=str,
        default=None,
        help="Path to pre-computed edge control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_vis",
        type=str,
        default=None,
        help="Folder containing pre-computed vis (blur) control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_vis",
        type=str,
        default=None,
        help="Path to pre-computed vis (blur) control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_depth",
        type=str,
        default=None,
        help="Folder containing pre-computed depth control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_depth",
        type=str,
        default=None,
        help="Path to pre-computed depth control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_seg",
        type=str,
        default=None,
        help="Path to pre-computed segmentation control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_seg",
        type=str,
        default=None,
        help="Path to pre-computed segmentation control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_dino",
        type=str,
        default=None,
        help="Path to pre-computed dino control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_dino_mask",
        type=str,
        default=None,
        help="Folder containing pre-computed dino mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_dino",
        type=str,
        default=None,
        help="Path to pre-computed dino control feature map video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_dino_mask",
        type=str,
        default=None,
        help="Path to pre-computed dino mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    # use_prep_anyup_feature
    parser.add_argument(
        "--use_prep_anyup_feature",
        action="store_true",
        help="Whether to use the pre-processed anyup dino feature for controlnet.",
        default=False,
    )
    parser.add_argument(
        "--anyup_factor",
        type=int,
        default=None,
        help="The upsampling factor used in the anyup pre-processing. E.g., 8 means the pre-processed anyup feature has 1/8 resolution of the input video.",
    )
    parser.add_argument(
        "--prep_anyup_feature_channel",
        type=int,
        default=None,
        help="Number of channels to select from the pre-processed anyup dino feature for controlnet.",
    )
    parser.add_argument(
        "--input_control_folder_edge_mask",
        type=str,
        default=None,
        help="Folder containing pre-computed edge mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_edge_mask",
        type=str,
        default=None,
        help="Path to pre-computed edge mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_vis_mask",
        type=str,
        default=None,
        help="Folder containing pre-computed vis mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_vis_mask",
        type=str,
        default=None,
        help="Path to pre-computed vis mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_depth_mask",
        type=str,
        default=None,
        help="Folder containing pre-computed depth mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_depth_mask",
        type=str,
        default=None,
        help="Path to pre-computed depth mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_seg_mask",
        type=str,
        default=None,
        help="Folder containing pre-computed segmentation mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_seg_mask",
        type=str,
        default=None,
        help="Path to pre-computed segmentation mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_folder_inpaint_mask",
        type=str,
        default=None,
        help="Folder containing pre-computed inpaint mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--input_control_video_path_inpaint_mask",
        type=str,
        default=None,
        help="Path to pre-computed inpaint mask control video for controlnet. If not provided, will compute on-the-fly using the input RGB video.",
    )
    parser.add_argument(
        "--show_control_condition",
        action="store_true",
        help="Whether to show the control condition concatenated to the output video.",
    )
    parser.add_argument(
        "--show_input",
        action="store_true",
        help="Whether to show the input video concatenated to the output video.",
    )
    parser.add_argument(
        "--not_keep_input_resolution",
        action="store_true",
        help="Whether to not keep the exact dimension of the input video. If not provided, will keep the input resolution. Otherwise, will output the default resolution\
        of Cosmos Transfer2/Predict2 according to the input video aspect ratio.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for t5 model")
    parser.add_argument("--limit_num_videos", type=int, default=None, help="Limit the number of videos to process")
    parser.add_argument(
        "--image_context_path",
        type=str,
        default=None,
        help="Path to single reference image file.",
    )
    parser.add_argument(
        "--sample_n_views",
        type=int,
        default=1,
        help="Number of camera views to generate for multiview output. Set to 1 for single-view (default behavior).",
    )
    parser.add_argument(
        "--cam_view_order",
        type=str,
        default=None,
        help="Order of the camera names in input video for multiview in a comma-separated string. Camera names include front, left, right, back, back_right, and back_left",
    )
    parser.add_argument(
        "--base_load_from",
        type=str,
        default=None,
        help="Path to the base model checkpoint.",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--start_frame", type=int, default=None, help="Start frame index for processing the video")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index for processing the video")
    parser.add_argument("--shift", type=int, default=None, help="shift of time schedule strategy during inference")
    parser.add_argument("--context_frame_idx", type=int, default=None, help="Index of the frame to use as context")
    parser.add_argument('--exp_override_opts',nargs='+',type=str,default=[],help='Experiment override options as list')
    parser.add_argument("--reverse", action="store_true", help="Whether to reverse the input video frames.", default=False)
    parser.set_defaults(use_neg_prompt=True)
    return parser.parse_args()
