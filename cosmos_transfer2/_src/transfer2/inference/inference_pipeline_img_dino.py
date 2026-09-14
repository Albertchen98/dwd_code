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
import random
import time
from typing import Optional, Union

import torch

from cosmos_transfer2._src.imaginaire.flags import INTERNAL
from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos_transfer2._src.transfer2.inference.utils import (
    get_t5_from_prompt,
    normalized_float_to_uint8,
    read_and_process_control_input,
    read_and_process_image_context,
    read_and_process_video,
    reshape_output_video_to_input_resolution,
    uint8_to_normalized_float,
    read_and_resize_input
)
from cosmos_transfer2._src.predict2.inference.utils import read_and_process_image
import torch.nn.functional as F

class Control2ImageInference:
    """
    Handles the Control2Video inference process, including model loading, data preparation,
    and video transfer from an input video and text prompt.
    """

    def __init__(
        self,
        registered_exp_name: str,
        checkpoint_paths: Union[str, list[str]],
        s3_credential_path: str,
        exp_override_opts: Optional[list[str]] = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        cache_dir: Optional[str] = None,
        skip_load_model: bool = False,
        base_load_from: Optional[str] = None,
    ):
        """
        Initializes the ControlVideo2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path.

        Args:
            registered_exp_name (str): Name of the experiment configuration.
            checkpoint_paths (Union[str, list[str]]): Single checkpoint path or List of checkpoint paths for multi-branch models.
            s3_credential_path (str): Path to S3 credentials file for ckpt & negative embedding (if loading from S3).
            exp_override_opts (list[str]): List of experiment override options.
            process_group (torch.distributed.ProcessGroup): Process group for distributed training.
            cache_dir (str): Cache directory for storing pre-computed embeddings.
            skip_load_model (bool): Whether to skip loading model from checkpoint for multi-control models.
        """
        self.registered_exp_name = registered_exp_name
        self.checkpoint_path = checkpoint_paths if isinstance(checkpoint_paths, str) else checkpoint_paths[0]
        self.s3_credential_path = s3_credential_path
        self.cache_dir = cache_dir

        if exp_override_opts is None:
            exp_override_opts = []
        # no need to load base model separately at inference
        exp_override_opts.append("model.config.base_load_from=null")
        if not INTERNAL:
            exp_override_opts.append("~data_train")
        # Load the model and config. Each trained model's config is composed by
        # loading a pre-registered experiment config, and then (optionally) overriding with some command-line
        # arguments. That is done in experiment_list.py. Here we simply replicate that process.
        model, config = load_model_from_checkpoint(
            experiment_name=self.registered_exp_name,
            s3_checkpoint_dir=self.checkpoint_path,
            config_file="cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config_img2img.py",
            load_ema_to_reg=True,
            local_cache_dir=(
                cache_dir if not checkpoint_paths else None
            ),  # for multi-control models, need to load other branches before caching
            experiment_opts=exp_override_opts,
        )
        if (
            isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1 and not skip_load_model
        ):  # load other branches for multi-control models
            load_from_local = False
            if cache_dir is not None:
                # build a unique path for s3checkpoint dir
                local_s3_ckpt_fp = os.path.join(
                    cache_dir,
                    self.checkpoint_path.split("s3://")[1],
                    "torch_model",
                    f"_rank_{distributed.get_rank()}.pt",
                )
                if os.path.exists(local_s3_ckpt_fp):
                    load_from_local = True

            if load_from_local:
                log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
                model.load_state_dict(easy_io.load(local_s3_ckpt_fp))
            else:
                model.load_multi_branch_checkpoints(checkpoint_paths=checkpoint_paths)
                if cache_dir is not None:
                    log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
                    easy_io.dump(model.state_dict(), local_s3_ckpt_fp)

        if base_load_from is not None:
            log.info(f"Loading base model from {base_load_from}")
            model.config.base_load_from = {
                "load_path": base_load_from,
                "credentials": s3_credential_path,
            }
            model.load_base_model()

        self.text_encoder_class = model.text_encoder_class

        if process_group is not None:
            log.info("Enabling CP in base model\n")
            model.net.enable_context_parallel(process_group)

        self.model = model
        self.config = config
        self.batch_size = 1

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prev_output: torch.Tensor,
        text_embedding: torch.Tensor,
        negative_prompt: str = None,
        control_weight: str = "1.0",
        image_context: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.

            image_context (torch.Tensor, optional): Image context tensor for conditioning. Can be (B, C, H, W).

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = prev_output.shape
        input_key = "video" if T > 1 else "images"

        data_batch = {
            "dataset_name": "video_data",
            "video": video[None],
            "t5_text_embeddings": text_embedding,  # positive prompt embedding. Name has t5 but also supports Reason1.
            "padding_mask": torch.zeros(self.batch_size, 1, H, W).cuda(),  # Padding mask (assumed no padding here)
            "num_conditional_frames": 1,  # Specify that the first frame is conditional
            "control_weight": [float(w) for w in control_weight.split(",")],
            "input_video": video,
        }

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        # Add image context
        if image_context is not None:
            data_batch["image_context"] = image_context.cuda().to(dtype=torch.bfloat16).contiguous()

        # Handle negative prompts for classifier-free guidance
        if negative_prompt is not None:
            assert self.neg_t5_embeddings is not None, "Negative prompt embedding is not computed."
            data_batch["neg_t5_text_embeddings"] = self.neg_t5_embeddings

        return data_batch

    def _pad_input_frames(
        self,
        input_frames: torch.Tensor,
        num_total_frames: int,
        num_video_frames_per_chunk: int,
        padding_mode: str = "reflect",
    ) -> torch.Tensor:
        """
        Pad input frames if total frames is less than chunk size
        """
        if num_total_frames < num_video_frames_per_chunk:
            # Check whether the input_frames is empty. If so, there is nothing to pad.
            if num_total_frames == 0:
                raise ValueError("No input frames; cannot pad. Verify that video frame counts match.")
            if padding_mode == "repeat":
                last_frame = input_frames[:, -1:, :, :]  # Get the last frame
                padding = last_frame.repeat(1, num_video_frames_per_chunk - num_total_frames, 1, 1)
                input_frames = torch.cat([input_frames, padding], dim=1)
            elif padding_mode == "reflect":
                while input_frames.shape[1] < num_video_frames_per_chunk:
                    padding = min(input_frames.shape[1] - 1, num_video_frames_per_chunk - input_frames.shape[1])
                    padding_frames = input_frames.flip(dims=[1])[:, :padding, :, :]
                    input_frames = torch.cat([input_frames, padding_frames], dim=1)
            else:
                raise ValueError(f"Invalid padding mode: {padding_mode}")
        return input_frames

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str | torch.Tensor | list[str] | dict[str, str],
        image_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        num_steps: int = 35,
        control_weight: str = "1.0",
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        preset_edge_threshold: str = "medium",
        preset_blur_strength: str = "medium",
        seg_control_prompt: str | None = None,
        input_control_image_paths: dict[str, str] | None = None,
        negative_prompt: str | None = None,
        shift: int = 5,
        device_rank: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, tuple[int, int]]:
        """
        Generates a video based on an input video and text prompt.
        Supports chunk-wise long video generation.

        Args:
            prompt (str): The text prompt describing the desired video content/style.
            video_path (str): Path to the input conditional video.
            guidance (int, optional): Classifier-free guidance scale. Defaults to 7.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            resolution (str, optional): Resolution of the video (720-default, 480, etc). Defaults to 720.
            image_context_path (str, optional): Path to image file to use as image context. If None, uses random frame from video. Will be ignored and use input video if context_frame_idx is provided.
            keep_input_resolution (bool, optional): Whether to keep the exact dimension of the. Defaults to True.
            negative_prompt (str, optional): Negative prompt for classifier-free guidance. Defaults to None.
            max_frames (int, optional): Maximum number of frames to read from the video. Defaults to None. 1 for image.
            context_frame_idx (int, optional): Frame index of the input video to use as image context. Defaults to None. In this case, can still use image_context_path to provide image context.
        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
            dict[str, torch.Tensor]: Dictionary mapping hint key to the corresponding control input video tensor.
            int: Frames per second of the original input video.
            tuple[int, int]: Original height and width of the input video.

        Raises:
            ValueError: If the input video is empty or invalid.
        """
        
        assert device_rank is not None
        # --------Input processing--------
        # Process input video and get meta info.
        log.info("Loading input image...")
        # aspect_ratio is width / height
        # input_frames is (C, T, H, W)
        
        input_frames, _, _, _ = read_and_resize_input(image_path, resolution=resolution)
        
        if input_frames.shape[1] == 0:
            raise ValueError("Input video is empty")

        # Get text context embeddings
        log.info("Computing prompt text embeddings...")
        # modify the batch if prompt is provided
        if self.model.text_encoder is not None:
            # Text encoder is defined in the model class. Use it
            if prompt:
                # data_batch["ai_caption"] = [prompt]
                text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [prompt], "images": None},
                    input_caption_key="ai_caption",
                )
            if negative_prompt:
                neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [negative_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
        else:
            if prompt:
                text_emb = get_t5_from_prompt(prompt)
                text_embeddings = text_emb.to(dtype=torch.bfloat16).cuda()
            if negative_prompt:
                text_emb = get_t5_from_prompt(negative_prompt)
                neg_text_embeddings = text_emb.to(dtype=torch.bfloat16).cuda()
        
        self.neg_t5_embeddings = neg_text_embeddings
        log.info("Processing image context if available...")

        # Load control inputs from paths, or optionally compute on-the-fly, and add to data batch.
        log.info("Loading  control inputs...")
        control_input_dict = read_and_process_control_input(
            video_path=image_path,
            input_control_paths=input_control_image_paths,
            hint_key=hint_key,
            resolution=resolution,
            seg_control_prompt=seg_control_prompt,
        )
        prev_output = torch.zeros_like(input_frames).to(torch.uint8).cuda()[None]

        x_sigma_max = None
        if input_frames is not None:
            cur_input_frames = input_frames
            if sigma_max is not None:
                x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None].cuda()
                x0 = self.model.encode(x0).contiguous()
                x_sigma_max = self.model.get_x_from_clean(x0, sigma_max, seed=seed)

        # Prepare the data batch with current input. Note: this doesn't include control inputs yet.
        log.info(f"shape of cur_input_frames is {cur_input_frames.shape}")
        data_batch = self._get_data_batch_input(
            cur_input_frames,
            prev_output,
            text_embeddings,
            negative_prompt=negative_prompt,
            control_weight=control_weight
        )

        for k,v in data_batch.items():
            if torch.is_tensor(v):
                log.info(f"data_batch key: {k}, shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
            
            # Process control inputs as specified in the hint_key list.
            # If pre-computed control inputs are provided, load them into the data batch.
        for k, v in control_input_dict.items():
            cur_control_input = v.to("cuda")
            data_batch[k] = cur_control_input
            if k == "control_input_inpaint_mask":
                data_batch["control_input_inpaint"] = cur_input_frames
        # Otherwise, compute control inputs on-the-fly via the augmentor（applicable to edge and vis).
        data_batch = get_augmentor_for_eval(
            data_dict=data_batch,
            input_keys=["input_video"],
            output_keys=hint_key,
            preset_edge_threshold=preset_edge_threshold,
            preset_blur_strength=preset_blur_strength,
        )

        random.seed(seed)
        seed = random.randint(0, 1000000)
        log.info(f"Seed: {seed}")
        for k,v in data_batch.items():
            if torch.is_tensor(v):
                log.info(f"data_batch key: {k}, shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
        # Generate and decode video
        sample = self.model.generate_samples_from_batch(
            data_batch,
            n_sample=1,
            guidance=guidance,
            seed=seed,
            is_negative_prompt=negative_prompt is not None,
            x_sigma_max=x_sigma_max,
            sigma_max=sigma_max,
            num_steps=num_steps,
            shift=shift,
        )
        # sample = torch.rand(1, 16, 24, 88, 160).to("cuda").bfloat16()  # dummy output for testing
        out_samples = self.model.decode(sample).cpu()  # Shape: (1, C, T, H, W)

        out_samples = (1.0 + out_samples) / 2  # Convert from [-1, 1] to [0, 1]
        out_samples = out_samples.clamp(0, 1)  # Clamp values
        # out_samples = out_samples.squeeze(2)  # Convert the video 
       
        return out_samples


def visualize_dino_pca_video(feats: torch.Tensor):
    """
    feats: [1024, 190, 44, 80] torch.Tensor on GPU
    out_path: output mp4 path
    """
    if feats.dtype != torch.float32:
        feats = feats.to(torch.float32)
    B, C, T, H, W = feats.shape
    X = feats.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [T*H*W, C]
    X = X - X.mean(0, keepdim=True)
    
    # PCA via SVD (on GPU)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    pcs = Vh[:3].T  # [C, 3]
    proj = (X @ pcs).reshape(B, T, H, W, 3)
    
    # Normalize to [0,1]
    proj = (proj - proj.min()) / (proj.max() - proj.min())

    # Resize and convert to uint8
    proj = proj.permute(0, 4, 1, 2, 3)  # [B,3,T,H,W]
    proj = F.interpolate(proj, size=(T, 704, 1280), mode='nearest')
    # # Write video
    # frames = [vutils.make_grid(f, normalize=False).permute(1,2,0).numpy() for f in proj]
    # imageio.mimwrite(out_path, frames, fps=25, codec='libx264')

    return proj