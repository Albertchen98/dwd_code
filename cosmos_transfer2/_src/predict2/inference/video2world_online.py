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
Folder monitoring system for Text-to-World generation using Video2World model.

Usage:
    # Run the folder monitoring system (single GPU):
    PYTHONPATH=. python cosmos_transfer2/_src/predict2/inference/video2world_online.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000032500

    # Run with 8 GPU context parallel (recommended for faster generation):
    PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_transfer2/_src/predict2/inference/video2world_online.py --experiment=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted --ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000032500 --context_parallel_size=8

Features:
    - Monitors /project/cosmos/yunhaog/code/cosmos_wan_compare/imaginaire4/projects/cosmos/predict2/inference/prompt_list for new .txt files
    - Loads model once and keeps it in GPU memory
    - Automatically generates videos when new prompts are added
    - Saves results to /project/cosmos/yunhaog/code/cosmos_wan_compare/imaginaire4/projects/cosmos/predict2/inference/result
    - Each video gets its own folder named after the prompt file

# use 8 GPUs with context parallel
export EXPERIMENT=Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted
PYTHONPATH=. torchrun --nproc_per_node=8 cosmos_transfer2/_src/predict2/inference/video2world.py \
--experiment=${EXPERIMENT} \
--ckpt_path s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/${EXPERIMENT}/checkpoints/iter_000032500 \
--save_root results/base_model/${EXPERIMENT}_025k_seed0_t2w \
--num_latent_conditional_frames=0 --seed=0 \
--input_root /project/cosmos/fangyinw/data/pbench/v0 \
--context_parallel_size=8
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.inference.video2world import (
    _DEFAULT_NEGATIVE_PROMPT,
    _IMAGE_EXTENSIONS,
    _VIDEO_EXTENSIONS,
    Video2WorldInference,
    read_and_process_image,
    read_and_process_video,
)

# Simple approach: just suppress the specific errors without complex patching


# Default configuration for text-to-video web interface
DEFAULT_POSITIVE_PROMPT = "A beautiful cinematic scene with smooth motion and high quality visuals"
DEFAULT_EXPERIMENT_NAME = (
    "Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted"
)
DEFAULT_S3_CHECKPOINT_DIR = "s3://bucket/cosmos_diffusion_v2/official_runs_vid2vid/Stage-c_pt_4-reason_embeddings-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000032500"


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Video2World folder monitoring script."""
    parser = argparse.ArgumentParser(description="Text-to-Video folder monitoring system")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")

    # Video generation parameters
    parser.add_argument("--num_video_frames", type=int, default=77, help="Number of video frames to generate")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (will be incremented for each video)")
    parser.add_argument("--resolution", type=str, default="none", help="Resolution of the video (H,W)")
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT, help="Custom negative prompt")

    # Folder paths
    parser.add_argument(
        "--prompt_folder",
        type=str,
        default="/project/cosmos/yunhaog/code/cosmos_wan_compare/imaginaire4/projects/cosmos/predict2/inference/prompt_list",
        help="Folder to monitor for new prompt files",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="/project/cosmos/yunhaog/code/cosmos_wan_compare/imaginaire4/projects/cosmos/predict2/inference/result",
        help="Folder to save generated videos",
    )

    # Monitoring parameters
    parser.add_argument("--poll_interval", type=float, default=2.0, help="Polling interval in seconds")

    # Context parallel arguments
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=8,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )

    return parser.parse_args()


class OnlineVideo2WorldInference(Video2WorldInference):
    """
    Extended Video2WorldInference class for online monitoring system.

    Inherits from the base Video2WorldInference class and overrides generate_vid2world
    to support optional input_path and input_tensor parameters needed for the online system.
    """

    def generate_vid2world(
        self,
        prompt: str,
        input_path: Optional[str] = None,
        input_tensor: Optional[torch.Tensor] = None,
        guidance: int = 7,
        num_video_frames: int = 77,
        num_latent_conditional_frames: int = 1,
        resolution: str = "192,320",
        seed: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ):
        """
        Generates a video based on an input image or video and text prompt.

        Processes the input, prepares the data batch, runs the diffusion
        model sampling, and decodes the result into a video tensor.

        Args:
            prompt (str): The text prompt describing the desired video content/style.
            input_path (Optional[str]): Path to the input image or video file.
            input_tensor (Optional[torch.Tensor]): Pre-processed input tensor (alternative to input_path).
            guidance (int, optional): Classifier-free guidance scale. Defaults to 7.
            num_video_frames (int, optional): Number of video frames to generate. Defaults to 77.
            num_latent_conditional_frames (int, optional): Number of latent conditional frames. Defaults to 1.
            resolution (str, optional): Target video resolution in "H,W" format. Defaults to "192,320".
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            negative_prompt (str, optional): Custom negative prompt. Defaults to the predefined default negative prompt.

        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
        """
        # Parse resolution string into tuple of integers
        if resolution == "none":
            h, w = self.model.get_video_height_width()
            video_resolution = (h, w)
        else:
            video_resolution = resolution.split(",")
            video_resolution = tuple([int(x) for x in video_resolution])

        print(f"Video resolution: {video_resolution}")

        # For text-to-video generation (num_latent_conditional_frames == 0), create empty input
        if num_latent_conditional_frames == 0:
            # Text-to-video mode: create minimal dummy input
            vid_input = torch.zeros(1, 3, 1, video_resolution[0], video_resolution[1]).to(torch.uint8)
            logger.info("Text-to-video mode: Using dummy input tensor")
        else:
            # Image/Video-to-video mode (kept for CLI compatibility)
            if input_tensor is not None:
                # Use provided tensor directly
                vid_input = input_tensor
            elif input_path is not None:
                ext = os.path.splitext(input_path)[1].lower()
                if ext in _IMAGE_EXTENSIONS:
                    logger.info(f"Processing image input: {input_path}")
                    vid_input = read_and_process_image(input_path, video_resolution, num_video_frames, resize=True)
                elif ext in _VIDEO_EXTENSIONS:
                    logger.info(f"Processing video input: {input_path}")
                    # Get the correct number of frames needed by the model
                    model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)
                    vid_input = read_and_process_video(
                        input_path, video_resolution, model_required_frames, num_latent_conditional_frames, resize=True
                    )
                else:
                    raise ValueError(
                        f"Unsupported file extension: {ext}. Supported extensions: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}"
                    )
            else:
                raise ValueError(
                    "Either input_path or input_tensor must be provided when num_latent_conditional_frames > 0"
                )

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            prompt,
            num_conditional_frames=num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
        )

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"GPU memory usage after getting data_batch: {mem_bytes / (1024**3):.2f} GB")

        # Generate latent samples using the diffusion model
        # Video should be of shape torch.Size([1, 3, 93, 192, 320]) # Note: Shape check comment
        sample = self.model.generate_samples_from_batch(
            data_batch,
            n_sample=1,  # Generate one sample
            guidance=guidance,
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,  # Use classifier-free guidance
        )

        # Decode the latent sample into a video tensor
        video = self.model.decode(sample)

        return video


class FolderMonitor:
    """Monitors a folder for new text files and processes them for video generation."""

    def __init__(self, model_instance: OnlineVideo2WorldInference, args):
        self.model = model_instance
        self.args = args
        self.prompt_folder = Path(args.prompt_folder)
        self.result_folder = Path(args.result_folder)
        self.processed_files = set()

        # Create folders if they don't exist
        self.prompt_folder.mkdir(parents=True, exist_ok=True)
        self.result_folder.mkdir(parents=True, exist_ok=True)

        # Initialize seed counter
        self.seed_counter = args.seed

        logger.info(f"📁 Monitoring folder: {self.prompt_folder}")
        logger.info(f"💾 Results will be saved to: {self.result_folder}")

    def get_new_prompt_files(self):
        """Get list of new .txt files that haven't been processed."""
        current_files = set()
        if self.prompt_folder.exists():
            for file_path in self.prompt_folder.glob("*.txt"):
                if file_path.is_file():
                    current_files.add(file_path)

        new_files = current_files - self.processed_files
        return sorted(new_files)  # Sort for consistent processing order

    def process_prompt_file(self, file_path: Path):
        """Process a single prompt file and generate video."""
        try:
            # Read the prompt
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            if not prompt:
                logger.warning(f"⚠️ Empty prompt file: {file_path}")
                return

            logger.info(f"🎬 Processing: {file_path.name}")
            logger.info(f"📝 Prompt: {prompt}")

            # Create output folder name from file name (without extension)
            output_folder_name = file_path.stem
            output_folder = self.result_folder / output_folder_name

            # Check if video already exists
            video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
            existing_video = None
            if output_folder.exists():
                for ext in video_extensions:
                    potential_video = output_folder / f"generated_video{ext}"
                    if potential_video.exists():
                        existing_video = potential_video
                        break

            if existing_video:
                logger.info(f"⏭️ Video already exists: {existing_video}")
                logger.info(f"🔄 Skipping generation for: {file_path.name}")
                return

            # Create output folder if it doesn't exist
            output_folder.mkdir(parents=True, exist_ok=True)

            # Generate video
            start_time = time.time()
            video_tensor = self.model.generate_vid2world(
                prompt=prompt,
                input_path=None,  # Text-to-video mode
                input_tensor=None,
                guidance=self.args.guidance,
                num_video_frames=self.args.num_video_frames,
                num_latent_conditional_frames=0,  # Always 0 for text-to-video
                resolution=self.args.resolution,
                seed=self.seed_counter,
                negative_prompt=self.args.negative_prompt,
            )

            # Convert and save video
            video_np = (1.0 + video_tensor[0]) / 2  # Convert from [-1, 1] to [0, 1]
            output_path = str(output_folder / "generated_video")
            save_img_or_video(video_np, output_path, fps=16)

            # Save prompt for reference
            prompt_file_path = output_folder / "prompt.txt"
            with open(prompt_file_path, "w", encoding="utf-8") as f:
                f.write(prompt)

            # Save generation info
            info_file_path = output_folder / "generation_info.txt"
            generation_time = time.time() - start_time
            with open(info_file_path, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Seed: {self.seed_counter}\n")
                f.write(f"Guidance: {self.args.guidance}\n")
                f.write(f"Frames: {self.args.num_video_frames}\n")
                f.write(f"Resolution: {self.args.resolution}\n")
                f.write(f"Generation time: {generation_time:.2f} seconds\n")
                f.write(f"Source file: {file_path.name}\n")
                f.write(f"Context parallel size: {self.args.context_parallel_size}\n")

            logger.info(f"✅ Video saved to: {output_folder}")
            logger.info(f"⏱️ Generation took: {generation_time:.2f} seconds")

            # Increment seed for next generation
            self.seed_counter += 1

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            import traceback

            traceback.print_exc()

    def run(self):
        """Main monitoring loop."""
        logger.info("🚀 Starting folder monitoring system...")
        logger.info(
            f"📊 Model parameters: frames={self.args.num_video_frames}, guidance={self.args.guidance}, resolution={self.args.resolution}"
        )
        logger.info(f"🔄 Polling every {self.args.poll_interval} seconds")
        if self.args.context_parallel_size > 1:
            logger.info(f"🌐 Using {self.args.context_parallel_size} GPUs for context parallel processing")
        logger.info("💡 Add .txt files to the prompt folder to generate videos!")

        try:
            while True:
                new_files = self.get_new_prompt_files()

                for file_path in new_files:
                    self.process_prompt_file(file_path)
                    self.processed_files.add(file_path)

                # Sleep before next check
                time.sleep(self.args.poll_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"💥 Fatal error in monitoring loop: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main function for the folder monitoring system."""
    torch.enable_grad(False)  # Disable gradient calculations for inference
    args = parse_arguments()

    print("🎬 Video2World Folder Monitoring System")
    print("=" * 50)

    # Setup S3 backend
    try:
        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "s3_credential_path": args.s3_cred,
            }
        )
        print("S3 backend configured")
    except ImportError:
        print(" Warning: easy_io not found, S3 backend setup skipped")
    except Exception as e:
        print(f"Error setting S3 backend: {e}")
        return

    # Load the model
    print(f"Loading model from {args.ckpt_path}...")
    print(f"Using context parallel size: {args.context_parallel_size}")
    try:
        model_instance = OnlineVideo2WorldInference(
            experiment_name=args.experiment,
            ckpt_path=args.ckpt_path,
            s3_credential_path=args.s3_cred,
            context_parallel_size=args.context_parallel_size,
        )
        print("Model loaded successfully and ready for inference!")
        if args.context_parallel_size > 1:
            print(f"Context parallel enabled with {args.context_parallel_size} GPUs")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Start monitoring
    monitor = FolderMonitor(model_instance, args)
    monitor.run()


if __name__ == "__main__":
    main()
