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
from cosmos_transfer2._src.predict2.inference.arg_parser import parse_arguments
from cosmos_transfer2._src.predict2.inference.text2image import Text2ImageInference
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

def process_single_image(
    prompt: str,
    neg_prompt: str | None,
    guidance: int,
    inference_pipeline: "Text2ImageInference",
    args: argparse.Namespace,
    device_rank: int,
) -> None:
    """Process a single video with a single guidance value and single reference image."""
    # Generate save path
   
    save_path = args.save_path
    log.info(f"save_path is defined as {save_path}")

    # Check if video already exists
    video_exists = os.path.exists(save_path + ".mp4") or os.path.exists(save_path)
    if video_exists:
        log.info(color_message(f"Video already exists at {save_path}. Skipping...", "yellow"))
        return
    # Prepare inference arguments
    inference_kwargs = {
        "prompt": prompt,
        "neg_prompt": neg_prompt,
        "guidance": guidance,
        "num_steps": args.num_steps,
        "shift": args.shift,
    }
    
    # Run model inference
    output_image = inference_pipeline.generate_image(**inference_kwargs)
    log.info(f"saving video to {save_path}.mp4")

    output_image_ = output_image.permute(1,0,2,3)
    save_img_or_video(output_image_, save_path)
    # save prompt
    prompt_save_path = f"{save_path}.txt"
    if not isinstance(prompt, list):  
        with open(prompt_save_path, "w") as f:
            f.write(prompt)
    else:
        with open(prompt_save_path, "w") as f:
            for prompt_i in prompt:
                f.write(prompt_i + "\n")
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
    registered_exp_name="txt2img_2B_720p_t5_embedding_rectified_flow_train_keyframe"
    # exp_override_opts=["model.config.text_encoder_config.compute_online=True",
    #                    "model.config.tokenizer.compile_encode=False"] 
    
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
    inference_pipeline = Text2ImageInference(
        experiment_name=registered_exp_name,
        s3_credential_path=args.s3_cred,
        ckpt_path=args.ckpt_paths,
    )

    # Create save directory structure
    save_dir = os.path.join("args.save_root")
    os.makedirs(save_dir, exist_ok=True)

    # Prepare reference image info if available

    # Process all videos from folder if specified
    # Process a single video if specified
    prompt, neg_prompt = get_prompt_from_path(args.prompt_path, args.prompt)
    if not neg_prompt:
        neg_prompt = args.negative_prompt
    if device_rank == 0:
        log.info(color_message(f"Prompt: {prompt}", "grey"))

    # assert False
    for guidance in args.guidance:
        process_single_image(
            prompt=prompt,
            neg_prompt=neg_prompt,
            guidance=guidance,
            inference_pipeline=inference_pipeline,
            args=args,
            device_rank=device_rank,
        )

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
