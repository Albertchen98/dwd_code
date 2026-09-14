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
This script is based on projects/cosmos/diffusion/v2/inference/vid2vid.py

To run inference on the training data (as visualization/debugging), use:
```bash
EXP=buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromStageC10k_alpamayo2tar_noviewprefix
ckpt_path=s3://bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromStageC10k_alpamayo2tar_noviewprefix-0/checkpoints/iter_000013500
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.predict2_multiview.scripts.local_inference --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 8 --input_is_train_data --use_apg --max_samples 10 --save_root results/apg
```

To run inference on validation data, first format the to
```
./ # root
    prompts/
        xyz.json # camera front caption
        abc.json # {"prompt": "..."}
        ...
    videos/
        xyz/ # corresponding multi-view videos
            xyz.camera_cross_left_120fov.mp4
            xyz.camera_cross_right_120fov.mp4
            xyz.camera_front_wide_120fov.mp4
            xyz.camera_rear_tele_30fov.mp4
            xyz.camera_rear_left_70fov.mp4
            xyz.camera_rear_right_70fov.mp4
            xyz.camera_front_tele_30fov.mp4
        abc/
            ...
```
then run the command:
```bash
EXP=buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPre32k_alpamayo2tar_2p83s_noviewprefix_1cap_cond012
ckpt_path=s3://bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPre32k_alpamayo2tar_2p83s_noviewprefix_1cap_cond012-0/checkpoints/iter_000022000
torchrun --nproc_per_node=8 --master_port=13345 -m cosmos_transfer2._src.predict2_multiview.scripts.local_inference --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 8 --input_root ./benchmark_20250714/internal --max_samples 10 --save_root results_buttercup
```

**NEW: Local Checkpoint Support**
The script now automatically supports local checkpoint directories! The `load_model_from_checkpoint` function has been
enhanced to detect local paths and handle them properly without any S3 operations. Simply provide the local path:

```bash
# Use your downloaded local checkpoint directly:
EXP=buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPre32k_alpamayo2tar_2p83s_noviewprefix_1cap_cond012
ckpt_path=/home/arslana/codes/.cache/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPre32k_alpamayo2tar_2p83s_noviewprefix_1cap_cond012-0/checkpoints/iter_000022000
torchrun --nproc_per_node=8 --master_port=13345 -m cosmos_transfer2._src.predict2_multiview.scripts.local_inference --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 8 --input_root ./benchmark_20250714/internal --max_samples 1 --save_root results_buttercup_local
```

**How it works:**
- The function automatically detects if the path exists locally AND doesn't start with 's3://'
- For local paths: temporarily disables object store loading and uses FileSystemReader
- For S3 paths: uses the original S3StorageReader behavior
- No code changes needed - just pass your local or S3 path as before!
"""

import argparse
import json
import os
import random

import torch as th
from einops import rearrange
from megatron.core import parallel_state

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_transfer2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_transfer2._src.predict2_multiview.datasets.data_sources.data_registration import (
    _get_contiguous_view_indices_options,
)
from cosmos_transfer2._src.predict2_multiview.models.multiview_vid2vid_model import USE_APG_KEY


def to_model_input(data_batch, model):
    """
    Similar to misc.to, but avoid converting uint8 "video" to float
    """
    for k, v in data_batch.items():
        _v = v
        if isinstance(v, th.Tensor):
            _v = _v.cuda()
            if th.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch


class Vid2VidInference:
    """
    Handles the Vid2Vid inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(self, experiment_name: str, ckpt_path: str, s3_credential_path: str, context_parallel_size: int = 1):
        """
        Initializes the Vid2VidInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            s3_credential_path (str): Path to S3 credentials file (if loading from S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        # Initialize distributed processing if context parallel size > 1
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load the model and config
        # The modified load_model_from_checkpoint now automatically handles both local and S3 paths
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_transfer2/_src/predict2_multiview/configs/vid2vid/config.py",
            load_ema_to_reg=True,
            experiment_opts=["+model.config.tokenizer.vae_pth=checkpoints/i4/tok_wan_vae/Wan2.1_VAE.pth"],
        )

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""

        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def generate_from_batch(
        self,
        data_batch,
        guidance: int = 7,
        seed: int = 1,
        use_apg: bool = False,
        num_conditional_frames: int = 1,
    ):
        data_batch = to_model_input(data_batch, self.model)
        data_batch[NUM_CONDITIONAL_FRAMES_KEY] = num_conditional_frames
        if use_apg:
            data_batch[USE_APG_KEY] = True
        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=False,
        )
        # (bsz = 1, c = 3, t = n_camera * t, h, w)
        video = self.model.decode(sample)
        # stack n_camera on the height dimension
        video = rearrange(video, "b c (v t) h w -> b c t (v h) w", v=data_batch["sample_n_views"].item())
        return video

    def generate_from_input(
        self,
        prompt: str,
        input_path: str,
        input_id: str,
        guidance: int = 7,
        seed: int = 1,
        use_apg: bool = False,
        num_conditional_frames: int = 1,
    ):
        """
        Generate video from input prompt and video.
        Assume the multiview video is structured as:
        <input_path>/
            <input_id>/
                <input_id>.<cam_key>.mp4
                ...
            ...
        """
        # Get the driving dataloader config from the training config
        # This contains all the camera mapping and view configuration
        dataloader_config = self.config.dataloader_train
        # The driving_dataloader_config is nested inside the dataset config
        driving_dataloader_config = dataloader_config.dataset.driving_dataloader_config
        assert not driving_dataloader_config.sample_noncontiguous_views
        view_id_to_camera_key = {v: k for k, v in driving_dataloader_config.camera_to_view_id.items()}
        sample_n_views = driving_dataloader_config.sample_n_views
        ref_cam_view_idx = driving_dataloader_config.ref_cam_view_idx
        n_cameras = len(driving_dataloader_config.camera_to_view_id)
        view_indices_options = _get_contiguous_view_indices_options(sample_n_views, ref_cam_view_idx, n_cameras)
        view_indices_selection = random.choice(view_indices_options)
        camera_keys_selection = [view_id_to_camera_key[view_idx] for view_idx in view_indices_selection]
        front_cam_key = driving_dataloader_config.front_tele_and_front_cam_keys[1]

        t5_embedding = get_text_embedding(prompt)[0]  # Shape: (1, seq_len, embed_dim) -> (seq_len, embed_dim)

        # Build initial data_dict with raw data
        data_dict = {
            "__key__": "sample0",  # fixed __key__ for single instance
            "selection_data.json": {
                "view_indices_selection": view_indices_selection,
                "camera_keys_selection": camera_keys_selection,
            },
        }

        caption_data = {"": {}}
        # Add video files and embeddings
        for cam_key in camera_keys_selection:
            view_id = driving_dataloader_config.camera_to_view_id[cam_key]
            view_caption_id = driving_dataloader_config.view_id_to_caption_id[view_id]
            if cam_key == front_cam_key:
                caption_data[""][str(view_caption_id)] = [1, [prompt]]
                data_dict[f"{cam_key}.bin"] = t5_embedding.cpu().numpy()
            else:
                caption_data[""][str(view_caption_id)] = [1, [""]]  # empty caption, not used anyway
                data_dict[f"{cam_key}.bin"] = th.zeros_like(t5_embedding).cpu().numpy()

            # Read video file
            with open(os.path.join(input_path, input_id, f"{input_id}.{cam_key}.mp4"), "rb") as video_file:
                data_dict[f"{cam_key}.mp4"] = video_file.read()

        data_dict["caption.json"] = caption_data

        augmentor = AUGMENTOR_OPTIONS[self.config.dataloader_train.dataset.augmentor_name](
            resolution="720",
            driving_dataloader_config=driving_dataloader_config,
            caption_type="t2w_qwen2p5_7b",
            embedding_type="t5_xxl",
            min_fps=10,
            max_fps=60,
        )
        for aug_name, aug_config in augmentor.items():
            aug = instantiate(aug_config)
            data_dict = aug(data_dict)
        # single instance convert to batch
        data_batch = {
            "__key__": [data_dict["__key__"]],
            "view_indices": data_dict["view_indices"].unsqueeze(0),
            "sample_n_views": th.tensor([data_dict["sample_n_views"]]),
            "ref_cam_view_idx_sample_position": th.tensor([data_dict["ref_cam_view_idx_sample_position"]]),
            "video": data_dict["video"].unsqueeze(0),
            "fps": th.tensor([data_dict["fps"]], dtype=th.float64),
            "camera_keys_selection": [data_dict["camera_keys_selection"]],
            "view_indices_selection": [th.tensor(v) for v in data_dict["view_indices_selection"]],
            "n_orig_video_frames_per_view": [th.tensor(v) for v in data_dict["n_orig_video_frames_per_view"]],
            "aspect_ratio": [data_dict["aspect_ratio"]],
            "num_video_frames_per_view": th.tensor([data_dict["num_video_frames_per_view"]]),
            "padding_mask": data_dict["padding_mask"].unsqueeze(0),
            "image_size": data_dict["image_size"].unsqueeze(0),
            "ai_caption": [data_dict["ai_caption"]],
            "t5_text_embeddings": data_dict["t5_text_embeddings"].unsqueeze(0),
            "t5_text_mask": data_dict["t5_text_mask"].unsqueeze(0),
        }
        # Generate video
        video = self.generate_from_batch(
            data_batch, guidance=guidance, seed=seed, use_apg=use_apg, num_conditional_frames=num_conditional_frames
        )
        return video

    def cleanup(self):
        """Clean up distributed resources."""
        if self.context_parallel_size > 1:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Vid2Vid inference script."""
    parser = argparse.ArgumentParser(description="Image2World/Video2World inference script")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    # generation
    parser.add_argument("--guidance", type=int, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=1, help="Guidance value")
    parser.add_argument("--use_apg", action="store_true", help="Use APG")
    parser.add_argument("--num_conditional_frames", type=int, default=1, help="Number of conditional frames")
    # input
    parser.add_argument(
        "--input_is_train_data",
        action="store_true",
        help="Inference on the training data, the input_root will be ignored if this is set",
    )
    parser.add_argument("--input_root", type=str, default="assets/image2world", help="Input root")
    parser.add_argument("--save_root", type=str, default="results/image2world", help="Save root")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to generate")
    return parser.parse_args()


if __name__ == "__main__":
    th.enable_grad(False)
    args = parse_arguments()
    # Initialize the inference handler with context parallel support
    vid2vid_cli = Vid2VidInference(
        args.experiment, args.ckpt_path, args.s3_cred, context_parallel_size=args.context_parallel_size
    )
    mem_bytes = th.cuda.memory_allocated(device=th.device("cuda" if th.cuda.is_available() else "cpu"))
    log.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    os.makedirs(args.save_root, exist_ok=True)
    if args.input_is_train_data:
        dataloader = instantiate(vid2vid_cli.config.dataloader_train)
        for i, batch in enumerate(dataloader):
            """
            save the prompt and video to the debug/ folder
            the strucutre is same as the input_root structure
            """
            if i > args.max_samples:
                break
            video = vid2vid_cli.generate_from_batch(
                batch,
                guidance=args.guidance,
                seed=args.seed,
                use_apg=args.use_apg,
                num_conditional_frames=args.num_conditional_frames,
            )
            if rank0:
                save_img_or_video((1.0 + video[0]) / 2, f"{args.save_root}/infer_from_train_{i}", fps=10)
    else:
        # Get all prompt files
        prompts_dir = os.path.join(args.input_root, "prompts")
        videos_dir = os.path.join(args.input_root, "videos")

        if not os.path.exists(prompts_dir) or not os.path.exists(videos_dir):
            raise ValueError(f"Expected 'prompts' and 'videos' directories in {args.input_root}")

        prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith(".json")]
        log.info(f"Found {len(prompt_files)} prompt files")

        for i, prompt_file in enumerate(sorted(prompt_files)):
            if i >= args.max_samples:
                break
            sample_name = os.path.splitext(prompt_file)[0]
            log.info(f"Processing sample {i}: {sample_name}")

            with open(os.path.join(prompts_dir, prompt_file), "r") as f:
                prompt_data = json.load(f)
            caption: str = prompt_data["prompt"]
            video = vid2vid_cli.generate_from_input(
                caption,
                videos_dir,
                sample_name,
                guidance=args.guidance,
                seed=args.seed,
                use_apg=args.use_apg,
                num_conditional_frames=args.num_conditional_frames,
            )
            if rank0:
                save_img_or_video((1.0 + video[0]) / 2, f"{args.save_root}/{sample_name}", fps=10)
    vid2vid_cli.cleanup()
