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
Script for generating videos from checkpoint in s3, edge/vis/depth/seg to video
Supports with/without single reference image file as additional context.
Supports autoregressive long video generation. Will use a different random seed per chunk.\

Steps:
- Find the training job (experiment) of interest in cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list.py
- Pass the experiment key name (the job_name_for_ckpt) to the `--experiment` argument.
- Pass the iteration number (INCLUDING THE "iter_" prefix) to the `--ckpt_iter` argument. E.g. `--ckpt_iter iter_000020000`
- If inferencing the AV-sample model, add `--is_av_model`.

See inference/README.md for more details and example commands.

# Preset arguments
video_root=/project/cosmos/tingchunw/projects/interactive/
experiment=multicontrol_720p_t24_stage2_maskprob0.5_spaced_layer14_mlp_hqv1_20250625_64N
iter=iter_000012000

# for edge only experiment
video_root=/project/cosmos/tingchunw/projects/interactive/
experiment=edge_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_5k_64N
iter=iter_000030000
NEG_PROMPT="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."

#################### 720p, 93 frames (t=24) #######################
#################### Option 1: single control input (choose one from edge, vis, depth, seg) #######################
# edge (no pre-computed control inputs needed)
experiment=edge_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_5k_64N  # edge only ckpt
iter=iter_000005000
NEG_PROMPT="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."
video_folder=${video_root}/demo_requests/gtc_eu_sample_test_videos2/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/refactor_predict2_compare_0725 \
  --video_folder ${video_folder} \
  --hint_key edge \
  --negative_prompt "${NEG_PROMPT}" \
  --preset_edge_threshold very_low --control_weight 1.0 --seed 1 --show_control_condition

# vis (no pre-computed control inputs needed)
video_folder=${video_root}/demo_requests/gtc_eu_sample_test_videos2/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/demo \
  --video_folder ${video_folder} \
  --hint_key vis \
  --preset_blur_strength very_low --control_weight 1.0 --seed 1 --show_control_condition

# depth (need pre-computed depth videos)
video_folder=${video_root}/assets/depth/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/demo
  --video_folder ${video_folder} \
  --prompt_folder ${video_folder} \
  --input_control_folder_depth ${video_folder}/depth \
  --hint_key depth \
  --control_weight 1.0 --seed 1 --show_control_condition

# seg (need pre-computed segmentation videos)
video_folder=${video_root}/assets/segmentation/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/demo \
  --video_folder ${video_folder} \
  --input_control_folder_seg ${video_folder}/seg \
  --hint_key seg \
  --control_weight 1.0 --seed 1 --show_control_condition

# inpaint (will use video_path/video_folder as input video)
video_folder=${video_root}/assets/depth_seg/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/demo \
  --video_folder ${video_folder} \
  --input_control_folder_inpaint_mask ${video_folder}/mask \
  --hint_key inpaint \
  --control_weight 1.0 --seed 1 --show_control_condition

#################### Option 2: single control input with mask #######################
video_folder=${video_root}/assets/depth_seg/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/mask \
  --video_folder ${video_folder} \
  --hint_key depth \
  --input_control_folder_depth ${video_folder}/depth \
  --input_control_folder_depth_mask ${video_folder}/mask \
  --control_weight 1.0 --seed 1 --show_control_condition

#################### Option 3: multiple control inputs #######################
video_folder=${video_root}/assets/depth_seg/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/multicontrol \
  --video_folder ${video_folder} \
  --hint_key depth,seg \
  --input_control_folder_depth ${video_folder}/depth \
  --input_control_folder_seg ${video_folder}/seg \
  --control_weight 1.0 --seed 1 --show_control_condition

#################### Option 4: multiple control inputs with mask #######################
video_folder=${video_root}/assets/depth_seg/

# edge + vis (with mask)
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/multicontrol \
  --video_folder ${video_folder} \
  --hint_key edge,vis \
  --input_control_folder_vis_mask ${video_folder}/mask \
  --preset_edge_threshold very_low --preset_blur_strength very_low --control_weight 1.0 --seed 1 --show_control_condition

# edge + inpaint (with mask)
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/multicontrol \
  --video_folder ${video_folder} \
  --hint_key edge,inpaint \
  --input_control_folder_inpaint_mask ${video_folder}/mask \
  --preset_edge_threshold very_low --control_weight 1.0 --seed 1 --show_control_condition

# edge + vis + depth + seg (with mask)
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --save_root results/transfer2/multicontrol \
  --video_folder ${video_folder} \
  --hint_key edge,vis,depth,seg \
  --input_control_folder_edge_mask ${video_folder}/mask \
  --input_control_folder_vis_mask ${video_folder}/mask \
  --input_control_folder_depth ${video_folder}/depth --input_control_folder_depth_mask ${video_folder}/inverted_mask \
  --input_control_folder_seg ${video_folder}/seg --input_control_folder_seg_mask ${video_folder}/inverted_mask \
  --preset_edge_threshold very_low --preset_blur_strength very_low --control_weight 1.0 --seed 1

#################### Multicontrol inference #######################
video_folder=${video_root}/assets/depth_seg/
experiment=multibranch_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_inference
iter=iter_000000000
edge_ckpt_path=s3://bucket/cosmos_transfer2/vid2vid_2B_control/edge_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_64N/checkpoints/iter_000030000
vis_ckpt_path=s3://bucket/cosmos_transfer2/vid2vid_2B_control/vis_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_64N/checkpoints/iter_000030000
depth_ckpt_path=s3://bucket/cosmos_transfer2/vid2vid_2B_control/depth_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_64N/checkpoints/iter_000030000
seg_ckpt_path=s3://bucket/cosmos_transfer2/vid2vid_2B_control/seg_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_64N/checkpoints/iter_000030000

# option 1: uniform control weight
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --ckpt_paths ${edge_ckpt_path},${vis_ckpt_path},${depth_ckpt_path},${seg_ckpt_path} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --negative_prompt "${NEG_PROMPT}" \
  --save_root results/transfer2/multicontrol \
  --video_folder ${video_folder} \
  --hint_key edge,vis,depth,seg \
  --input_control_folder_depth ${video_folder}/depth \
  --input_control_folder_seg ${video_folder}/seg \
  --control_weight 1.0,1.0,1.0,1.0 --seed 1

# option 2: spatio-temporal control weight using mask
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --ckpt_paths ${edge_ckpt_path},${vis_ckpt_path},${depth_ckpt_path},${seg_ckpt_path} \
  --num_video_frames_per_chunk 93 \
  --num_gpus 8 \
  --negative_prompt "${NEG_PROMPT}" \
  --save_root results/transfer2/multicontrol \
  --video_folder ${video_folder} \
  --hint_key edge,vis,depth,seg \
  --input_control_folder_edge_mask ${video_folder}/mask \
  --input_control_folder_vis_mask ${video_folder}/mask \
  --input_control_folder_depth ${video_folder}/depth --input_control_folder_depth_mask ${video_folder}/inverted_mask \
  --input_control_folder_seg ${video_folder}/seg --input_control_folder_seg_mask ${video_folder}/inverted_mask \
  --control_weight 1.0 --seed 1

#################### Image only (input is image) #######################
video_path=${video_root}/assets/canny/c3d_beachhouse_001
experiment=edge_720p_t24or1_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_64N
iter=iter_000010000
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 1 \
  --num_gpus 1 \
  --save_root results/transfer2/image_only \
  --video_path ${video_path}.png \
  --prompt_path ${video_path}.pkl \
  --hint_key edge \
  --negative_prompt "${NEG_PROMPT}" \
  --preset_edge_threshold very_low --control_weight 1.0 --seed 1 --show_control_condition

#################### Image only (input is video, use first frame) #######################
video_root=/project/cosmos/fangyinw/data/transfer_bench/v1
video_path=${video_root}/opendrive/videos/02cbf8b8-082c-4ec2-adcc-dfc8fef67d28.mp4
prompt="A scenic drive unfolds along a coastal highway. The video captures a smooth, continuous journey along a multi-lane road, with the camera positioned as if from the perspective of a vehicle traveling in the right lane. The road is bordered by a tall, green mountain on the right, which casts a shadow over part of the highway, while the left side opens up to a view of the ocean, visible in the distance beyond a row of low-lying vegetation and a sidewalk. Several vehicles, including two red vehicles, travel ahead, maintaining a steady pace. The road is well-maintained, with clear white lane markings and a concrete barrier separating the lanes from the mountain covered by trees on the right. Utility poles and power lines run parallel to the road on the left, adding to the infrastructure of the scene. The camera remains static, providing a consistent view of the road and surroundings, emphasizing the serene and uninterrupted nature of the drive."
experiment=edge_720p_t24or1_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_64N
iter=iter_000010000
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12345 cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py \
  --experiment=${experiment} \
  --ckpt_iter ${iter} \
  --num_video_frames_per_chunk 1 \
  --num_gpus 1 \
  --save_root results/transfer2/image_only \
  --video_path ${video_path} \
  --prompt "${prompt}" \
  --hint_key edge \
  --negative_prompt "${NEG_PROMPT}" \
  --preset_edge_threshold very_low --control_weight 1.0 --seed 1 --show_control_condition --max_frames 1
"""

import argparse
import os

import torch
from tqdm import tqdm

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment_av.experiment_list import (
    EXPERIMENTS as EXPERIMENTS_AV,
)
from cosmos_transfer2._src.transfer2.inference.arg_parser import parse_arguments
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_dino import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.inference.utils import (
    _IMAGE_EXTENSIONS,
    _VIDEO_EXTENSIONS,
    color_message,
    get_prompt_from_path,
    get_unique_seed,
    parse_control_input_file_paths,
    parse_control_input_single_file_paths,
    validate_image_context_path,
)

def generate_save_path(
    save_dir: str,
    guidance: int,
    iter_num: str,
    seed: int,
    hint_key: str,
    control_weight: float,
    sigma_max: float,
    preset_edge_threshold: str,
    preset_blur_strength: str,
    video_name: str,
    ref_image_name: str = None,
    num_conditional_frames: int = 1,
    context_frame_idx: int = 0,
) -> str:
    """Generate save path for output video. Without extension suffix like .mp4."""

    # Build the path components
    path_components = [save_dir]

    # Add reference image folder if applicable
    if ref_image_name:
        path_components.append(f"ref_img_{ref_image_name}")
    if context_frame_idx is not None:
        path_components.append(f"ref_img_frame_idx_{context_frame_idx}")

    # Add guidance and iteration info
    path_components.append(f"guidance{guidance}_iter{iter_num}_seed{seed}")

    # Add control settings
    control_info = f"{hint_key.replace(',', '+')}_cw{control_weight}"
    if sigma_max is not None:
        control_info += f"_smax{sigma_max}"
    if num_conditional_frames != 1:
        control_info += f"_overlap{num_conditional_frames}"
    if "edge" in hint_key:
        preset_str = preset_edge_threshold.replace("_", "")
        control_info += f"_edge-{preset_str}"
    if "vis" in hint_key:
        preset_str = preset_blur_strength.replace("_", "")
        control_info += f"_vis-{preset_str}"
    path_components.append(control_info)

    # Add base name
    path_components.append(video_name)

    return os.path.join(*path_components)


def process_single_video(
    video_path: str,
    prompt: str,
    neg_prompt: str | None,
    input_control_video_paths: dict,
    save_dir: str,
    guidance: int,
    ref_image_path: str,
    ref_image_name: str,
    inference_pipeline: "ControlVideo2WorldInference",
    args: argparse.Namespace,
    device_rank: int,
) -> None:
    """Process a single video with a single guidance value and single reference image."""
    # Generate save path
    iter_num = args.ckpt_iter.replace("iter_", "")

    video_name = os.path.basename(video_path).split(".")[0]
    save_path = generate_save_path(
        save_dir=save_dir,
        guidance=guidance,
        iter_num=iter_num,
        seed=args.seed,
        hint_key=args.hint_key,
        control_weight=args.control_weight,
        sigma_max=args.sigma_max,
        preset_edge_threshold=args.preset_edge_threshold,
        preset_blur_strength=args.preset_blur_strength,
        video_name=video_name,
        ref_image_name=ref_image_name,
        num_conditional_frames=args.num_conditional_frames,
        context_frame_idx=args.context_frame_idx,
    )
    log.info(f"save_path is defined as {save_path}")

    # Check if video already exists
    video_exists = os.path.exists(save_path + ".mp4") or os.path.exists(save_path)
    if video_exists:
        log.info(color_message(f"Video already exists at {save_path}. Skipping...", "yellow"))
        return
    # Prepare inference arguments
    inference_kwargs = {
        "video_path": video_path,
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "image_context_path": ref_image_path if ref_image_path is not None else None,
        "guidance": guidance,
        "seed": args.seed,
        "resolution": args.resolution,
        "control_weight": args.control_weight,
        "sigma_max": args.sigma_max,
        "hint_key": args.hint_key.split(","),
        "num_steps": args.num_steps,
        "input_control_video_paths": input_control_video_paths,
        "show_control_condition": args.show_control_condition,
        "show_input": args.show_input,
        "keep_input_resolution": not args.not_keep_input_resolution,
        "context_frame_idx": args.context_frame_idx,
        "max_frames": args.max_frames,
        "shift": args.shift,
    }
    # Run model inference
    output_video, control_video_dict, fps, original_hw = inference_pipeline.generate_img2world(**inference_kwargs)

    if device_rank == 0:
        output_video = (1.0 + output_video[0]) / 2
        for key in control_video_dict:
            control_video_dict[key] = (1.0 + control_video_dict[key][0]) / 2
            save_img_or_video(control_video_dict[key], f"{save_path}_control_{key}", fps=fps)
            log.info(f"{key} control video saved to {save_path}_control_{key}.mp4")
        # Remove batch dimension and normalize to [0, 1] range
        save_img_or_video(output_video, save_path, fps=fps)
        # save prompt
        prompt_save_path = f"{save_path}.txt"
        with open(prompt_save_path, "w") as f:
            f.write(prompt)
        log.success(f"Generated video saved to {save_path}.mp4")
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_arguments()
    torch.manual_seed(args.seed)

    # if not args.is_av_model:
    #     ckpt_prefix = "s3://bucket/cosmos_transfer2/vid2vid_2B_control"
    #     registered_exp_name = EXPERIMENTS[args.experiment].registered_exp_name
    #     exp_override_opts = EXPERIMENTS[args.experiment].command_args
    #     job_name_for_ckpt = EXPERIMENTS[args.experiment].job_name_for_ckpt
    # else:
    #     ckpt_prefix = "s3://bucket/cosmos_transfer2/vid2vid_2B_control_av"
    #     registered_exp_name = EXPERIMENTS_AV[args.experiment].registered_exp_name
    #     exp_override_opts = EXPERIMENTS_AV[args.experiment].command_args
    #     job_name_for_ckpt = EXPERIMENTS_AV[args.experiment].job_name_for_ckpt
    registered_exp_name = args.experiment
    # registered_exp_name="vid2vid_2B_control_720p_t24_dinoedgecontrol_layer4_cr1_embedding_rectified_flow_train_keyframe"
    exp_override_opts=["model.config.text_encoder_config.compute_online=True",
                       "model.config.tokenizer.compile_encode=False",
                       "model.config.net.dino_projector_type=\"v2\"",
                       "model.config.net.dino_out_channels=64"]
    
    # ckpt_path = os.path.join(ckpt_prefix, job_name_for_ckpt, "checkpoints", args.ckpt_iter)

    device_rank = 0
    process_group = None
    if args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_transfer2._src.imaginaire.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()
        device_rank = distributed.get_rank(process_group)

    # Initialize the inference class
    inference_pipeline = ControlVideo2WorldInference(
        registered_exp_name=registered_exp_name,
        s3_credential_path=args.s3_cred,
        exp_override_opts=exp_override_opts,
        process_group=process_group,
        cache_dir=args.cache_dir,
        checkpoint_paths=args.ckpt_paths.split(",") if args.ckpt_paths else None,
        skip_load_model=args.skip_load_model,
        base_load_from=args.base_load_from,
    )

    # Create save directory structure
    save_dir = os.path.join(args.save_root, args.experiment)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare reference image info if available
    ref_image_path, ref_image_name = validate_image_context_path(args.image_context_path)

    # Process all videos from folder if specified
    if args.video_folder:
        prompt_folder = args.prompt_folder if args.prompt_folder != "" else args.video_folder

        video_files = [
            f
            for f in sorted(os.listdir(args.video_folder))
            if os.path.splitext(f)[1] in _VIDEO_EXTENSIONS + _IMAGE_EXTENSIONS
        ]
        video_files_to_process = video_files[: args.limit_num_videos] if args.limit_num_videos else video_files
        for video_file in tqdm(video_files_to_process, desc="Processing videos"):
            video_path = os.path.join(args.video_folder, video_file)

            # assuming prompt has same base name as video
            prompt_path = os.path.join(prompt_folder, os.path.splitext(video_file)[0])
            prompt, neg_prompt = get_prompt_from_path(prompt_path, args.prompt)
            if not neg_prompt:
                neg_prompt = args.negative_prompt
            if device_rank == 0:
                log.info(color_message(f"Prompt: {prompt}", "grey"))

            input_control_video_paths = parse_control_input_file_paths(
                args.input_control_folder_edge,
                args.input_control_folder_vis,
                args.input_control_folder_depth,
                args.input_control_folder_seg,
                args.input_control_folder_dino,
                args.input_control_folder_edge_mask,
                args.input_control_folder_vis_mask,
                args.input_control_folder_depth_mask,
                args.input_control_folder_seg_mask,
                args.input_control_folder_dino_mask,
                args.input_control_folder_inpaint_mask,
                video_file,
            )

            if args.seed is None:
                args.seed = get_unique_seed(
                    video_path, args.save_root, args.experiment, args.ckpt_iter, args.num_conditional_frames
                )

            # Process the video with all guidance values
            for guidance in args.guidance:
                process_single_video(
                    video_path=video_path,
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    input_control_video_paths=input_control_video_paths,
                    save_dir=save_dir,
                    guidance=guidance,
                    ref_image_path=ref_image_path,
                    ref_image_name=ref_image_name,
                    inference_pipeline=inference_pipeline,
                    args=args,
                    device_rank=device_rank,
                )

    # Process a single video if specified
    elif args.video_path:
        prompt, neg_prompt = get_prompt_from_path(args.prompt_path, args.prompt)
        if not neg_prompt:
            neg_prompt = args.negative_prompt
        if device_rank == 0:
            log.info(color_message(f"Prompt: {prompt}", "grey"))

        input_control_video_paths = parse_control_input_single_file_paths(
            input_control_video_path_edge=args.input_control_video_path_edge,
            input_control_video_path_vis=args.input_control_video_path_vis,
            input_control_video_path_depth=args.input_control_video_path_depth,
            input_control_video_path_seg=args.input_control_video_path_seg,
            input_control_video_path_dino=args.input_control_video_path_dino,
            input_control_video_path_edge_mask=args.input_control_video_path_edge_mask,
            input_control_video_path_vis_mask=args.input_control_video_path_vis_mask,
            input_control_video_path_depth_mask=args.input_control_video_path_depth_mask,
            input_control_video_path_seg_mask=args.input_control_video_path_seg_mask,
            input_control_video_path_dino_mask=args.input_control_video_path_dino_mask,
            input_control_video_path_inpaint_mask=args.input_control_video_path_inpaint_mask,
        )
        log.info(f"input_control_video_paths are {input_control_video_paths}")
        # assert False
        for guidance in args.guidance:
            process_single_video(
                video_path=args.video_path,
                prompt=prompt,
                neg_prompt=neg_prompt,
                input_control_video_paths=input_control_video_paths,
                save_dir=save_dir,
                guidance=guidance,
                ref_image_path=ref_image_path,
                ref_image_name=ref_image_name,
                inference_pipeline=inference_pipeline,
                args=args,
                device_rank=device_rank,
            )

    else:
        raise ValueError("Either --video_folder or --video_path must be specified")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
