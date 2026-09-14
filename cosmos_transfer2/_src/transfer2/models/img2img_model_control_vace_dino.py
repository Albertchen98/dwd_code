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
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import tqdm
import attrs
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from cosmos_transfer2._src.imaginaire.flags import INTERNAL
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.common.modules.res_sampler import COMMON_SOLVER_OPTIONS, Sampler
from cosmos_transfer2._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.imaginaire.utils import log, misc
from cosmos_transfer2._src.predict2.checkpointer.dcp import ModelWrapper
from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_transfer2._src.predict2.models.text2world_model import (
    Text2WorldCondition,
    Text2WorldModelConfig,
    DiffusionModel,
)
from cosmos_transfer2._src.transfer2.datasets.data_sources.nuplan_dataset_images import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import ControlVideo2WorldCondition
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import CTRL_HINT_KEYS
from cosmos_transfer2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_transfer2._src.predict2.utils.context_parallel import broadcast, broadcast_split_tensor, cat_outputs_cp
from dataclasses import replace
from torchvision.transforms.v2 import functional as F

IS_PREPROCESSED_KEY = "is_preprocessed"

@attrs.define(slots=False)
class ControlDino2ImageConfig(Text2WorldModelConfig):
    base_load_from: LazyDict = None
    copy_weight_strategy: str = (
        "first_n"  # How to copy weights from base model to control branch. "first_n" or "spaced_n"
    )
    hint_keys: str = "_".join([key.replace("control_input_", "") for key in CTRL_HINT_KEYS.keys()])
    dinov3_encoder: LazyDict = None  # DINO V3 encoder configuration


class ControlDino2ImageModel(DiffusionModel):
    """
    ImaginaireModel instance of the VACE-styled controlnet for training.
    """

    def __init__(self, config: ControlDino2ImageConfig, *args, **kwargs):
        self.is_new_training = True
        self.copy_weight_strategy = config.copy_weight_strategy
        self.hint_keys = ["control_input_dino"]         
        super().__init__(config, *args, **kwargs)
        # Dino encoder
        # assert len(config.dinov3_encoder.out_layers) == 1 or config.net.use_dino_merge, "when use_dino_merge is False, dinov3_encoder.out_layers must be 1"
        with misc.timer("DiffusionModel: set_up_dinov3_encoder"):
            self.dinov3_encoder = lazy_instantiate(config.dinov3_encoder)
            
        log.info(self.net, rank0_only=True)

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, ControlVideo2WorldCondition]:
        # Get base data and condition
        # input_data_key: str = "video" by default
        # log.info(f"data_batch[{self.input_data_key}].shape is {data_batch[self.input_data_key].shape}")
        if self.input_data_key in data_batch and data_batch[self.input_data_key].shape[2] == 1:
            # log.info("run the weird code")
            data_batch[self.input_image_key] = data_batch[self.input_data_key].squeeze(2)
            assert data_batch[self.input_image_key].dtype == torch.uint8, "Image data is not in uint8 format."
            data_batch[self.input_image_key] = data_batch[self.input_image_key].to(**self.tensor_kwargs) / 127.5 - 1.0
            del data_batch[self.input_data_key]
            # log.info(f"data_batch[{self.input_image_key}].shape is {data_batch[self.input_image_key].shape}")
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        # Add control conditioning
        latent_control_input = []
        control_weight = data_batch.get("control_weight", [1.0] * len(self.hint_keys))
        if len(control_weight) == 1:
            control_weight = control_weight * len(self.hint_keys)
        control_weight_maps = [None] * len(self.hint_keys)  # spatio-temporal control weight
        for hi, hint_key in enumerate(self.hint_keys):
            control_input = getattr(condition, hint_key, None)
            # log.info(f"hint_key is {hint_key}")
            control_input_mask = getattr(condition, hint_key + "_mask", None)
            latent_control_input += self.get_control_latent(latent_state, control_input, control_input_mask)
            if not torch.is_grad_enabled() and not self.net.vace_has_mask:  # inference mode
                if control_input is None:  # set control weight to 0 if no control input
                    if len(control_weight) == len(self.hint_keys):
                        control_weight[hi] = 0.0
                    else:
                        control_weight.insert(hi, 0.0)
                if (
                    control_input_mask is not None and (control_input_mask != 1).any()
                ):  # use control weight to implement masking operation
                    assert control_input_mask.shape[1] == 1, (
                        f"control_input_mask.shape[1] != 1: {control_input_mask.shape[1]}"
                    )
                    control_weight_maps[hi] = control_input_mask * control_weight[hi]
        # If any control mask exists, use spatio-temporal control weight instead of scalar control weight.
        if any(c is not None for c in control_weight_maps):
            for hi in range(len(self.hint_keys)):
                if control_weight_maps[hi] is None:  # convert scalar control weight to spatio-temporal control weight
                    control_weight_maps[hi] = control_weight[hi] * torch.ones_like(
                        next(c for c in control_weight_maps if c is not None)
                    )
            control_weight_maps = torch.stack(control_weight_maps)
            # resize spatio-temporal control weight to match latent_state shape
            control_weight = self.resize_control_weight(control_weight_maps, latent_state)

        # assert num_modalities > 0, "No control input found"
        latent_control_input = torch.cat(latent_control_input, dim=1)
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input,
            control_weight=control_weight,
        )

        return raw_state, latent_state, condition    

    def resize_control_weight(self, control_context_scale: Tensor, latent_state: Tensor) -> Tensor:
        temporal_compression_factor = self.tokenizer.temporal_compression_factor
        control_weight_maps = [w for w in control_context_scale]  # Keep as tensor
        _, _, T, H, W = latent_state.shape
        H = H // self.net.patch_spatial  # spatial patch size
        W = W // self.net.patch_spatial  # spatial patch size
        weight_maps = []
        for weight_map in control_weight_maps:  # [B, 1, T, H, W]
            if weight_map.shape[2:5] != (T, H, W):
                assert weight_map.shape[2] == temporal_compression_factor * (T - 1) + 1, (
                    f"{weight_map.shape[2]} != {temporal_compression_factor * (T - 1) + 1}"
                )
                weight_map_i = [
                    torch.nn.functional.interpolate(
                        weight_map[:, :, :1, :, :],
                        size=(1, H, W),
                        mode="trilinear",
                        align_corners=False,
                    )
                ]
                weight_map_i += [
                    torch.nn.functional.interpolate(
                        weight_map[:, :, 1:],
                        size=(T - 1, H, W),
                        mode="trilinear",
                        align_corners=False,
                    )
                ]
                weight_map = torch.cat(weight_map_i, dim=2)

            # Reshape to match BTHWD format
            weight_map = weight_map.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 1]
            weight_maps.append(weight_map)
        control_weight_maps = weight_maps
        control_weight_maps = torch.stack(control_weight_maps)
        # Cap the sum over dim0 at each T,H,W position to be at most 1.0
        # control_weight_maps shape: [num_modalities, B, T, H, W, 1]
        max_control_weight_sum = 1.0
        sum_over_modalities = control_weight_maps.sum(dim=0)  # [B, T, H, W, 1]
        max_values = torch.clamp_min(sum_over_modalities, max_control_weight_sum)  # [B, T, H, W, 1]
        scale_factors = max_control_weight_sum / max_values  # [B, T, H, W, 1]
        control_weight_maps = control_weight_maps * scale_factors[None]  # [num_modalities, B, T, H, W, 1]
        return control_weight_maps

    def get_control_latent(self, latent_state: Tensor, control_input: Tensor, control_input_mask: Tensor) -> Tensor:
        # even it's named as latent_control_input, they are still rgb frames, do nothing just sampling
        latent_control_input = []
        latent_control_input.append(control_input.to(**self.tensor_kwargs))
        return latent_control_input
    
    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        self._update_train_stats(data_batch)
        # Obtain text embeddings online
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )

        with torch.no_grad():
            # log.info(f"condition.latent_control_input has shape: {condition.latent_control_input.shape}")
            # log.info(f"condition.latent_control_input has range from {condition.latent_control_input.min()} to {condition.latent_control_input.max()}")
            _latent_control_input = self.dinov3_encoder(condition.latent_control_input)

        condition = replace(condition, latent_control_input=_latent_control_input.clone().detach())

        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
            )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """
        for k,v in data_batch.items():
            if torch.is_tensor(v):
                log.info(f"data_batch key: {k}, shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
        # self._normalize_video_databatch_inplace(data_batch)
        # self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]
        for k,v in data_batch.items():
            if torch.is_tensor(v):
                log.info(f"data_batch key: {k}, shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        if self.config.use_flowunipc_scheduler:
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            noise = misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )

            seed_g = torch.Generator(device=self.tensor_kwargs["device"])
            seed_g.manual_seed(seed)

            sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=5)

            timesteps = sample_scheduler.timesteps
            with torch.no_grad():
                x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
                latents = noise

                if self.net.is_context_parallel_enabled:
                    latents = broadcast_split_tensor(
                        latents, seq_dim=2, process_group=self.get_context_parallel_group()
                    )

                if INTERNAL:
                    timesteps_iter = timesteps
                else:
                    timesteps_iter = tqdm.tqdm(timesteps, desc="Generating samples", total=len(timesteps))
                for _, t in enumerate(timesteps_iter):
                    latent_model_input = latents
                    timestep = [t]

                    # our model supports 0-1 while the t is 0-1000
                    timestep = torch.stack(timestep) / 1000
                    noise_pred = x0_fn(latent_model_input, timestep.unsqueeze(0))
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
                    )[0]
                    latents = temp_x0.squeeze(0)

                if self.net.is_context_parallel_enabled:
                    latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())
                return latents

        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.sde.sigma_max
            )

        if self.net.is_context_parallel_enabled:
            x_sigma_max = broadcast_split_tensor(
                x_sigma_max, seq_dim=2, process_group=self.get_context_parallel_group()
            )

        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        samples = self.sampler(
            x0_fn,
            x_sigma_max,
            num_steps=num_steps,
            sigma_max=sigma_max,
            sigma_min=self.sde.sigma_min,
            solver_option=solver_option,
        )
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        return samples

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        is_image_batch = self.is_image_batch(data_batch)

        for k,v in data_batch.items():
            if torch.is_tensor(v):
                log.info(f"data_batch key: {k}, shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
        log.info(self.conditioner)
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, control_condition = self.get_data_and_condition(data_batch)
        log.info(f"x0 has shape {x0.shape}")
        # Set control condition
        latent_control_input = control_condition.latent_control_input

        control_weight = control_condition.control_context_scale
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input, control_weight=control_weight
        )
        uncondition = uncondition.set_control_condition(
            latent_control_input=latent_control_input, control_weight=control_weight
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(None, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(None, uncondition, None, None)
       

        with torch.no_grad():       
            log.info(f"condition.latent_control_input has shape: {condition.control_input_dino.shape}")
            log.info(f"condition.latent_control_input has range from {condition.control_input_dino.min()} to {condition.control_input_dino.max()}")
            _latent_control_input = condition.control_input_dino / 255.0
            B, _, _, _, _ = _latent_control_input.shape
            _latent_control_input = rearrange(_latent_control_input, "B C T H W -> (B T) C H W")
            _latent_control_input = F.normalize(
                                _latent_control_input, 
                                mean=IMAGENET_DEFAULT_MEAN, 
                                std=IMAGENET_DEFAULT_STD
                            )
            _latent_control_input = rearrange(_latent_control_input, "(B T) C H W -> B C T H W", B=B)
            _latent_control_input = self.dinov3_encoder(_latent_control_input.to(torch.bfloat16))
            # _latent_control_input = self.dinov3_encoder(condition.control_input_dino)
        
        log.info(f"_latent_control_input has shape: {_latent_control_input.shape}")
        condition = replace(condition, latent_control_input=_latent_control_input)
        uncondition = replace(uncondition, latent_control_input=_latent_control_input)
        log.info(f"after dino encoder, condition.latent_control_input has shape: {condition.latent_control_input.shape}")
        log.info(f"after dino encoder, uncondition.latent_control_input has shape: {uncondition.latent_control_input.shape}")

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn
    
    def denoise(self, xt_B_C_T_H_W: torch.Tensor, sigma_B_T: torch.Tensor, condition):
        """
        Override denoise method for control branch support in rectified flow.
        """
        # Handle control conditioning
        if hasattr(condition, "latent_control_input"):
            # The control conditioning is already set in the condition object
            pass

        # Call parent's denoise method
        return super().denoise(xt_B_C_T_H_W, sigma_B_T, condition)


    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str | None = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        super()._normalize_video_databatch_inplace(data_batch, input_key)

        # Handle control_input if it exists
        for key in data_batch.keys():
            if "dino" in key:
                # log.info(f"skipping normalizing {key}")
                continue    
            if key.startswith("control_input_") and data_batch[key] is not None:
                hint_key = key
                # Normalize control_input if not already normalized
                # log.info(f"normalizing {hint_key} in-place")
                if data_batch[hint_key].dtype == torch.uint8:
                    data_batch[hint_key] = data_batch[hint_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                elif data_batch[hint_key].dtype == torch.bool:
                    data_batch[hint_key] = data_batch[hint_key].to(**self.tensor_kwargs)

                if data_batch[hint_key].dim() == 5 and data_batch[hint_key].shape[2] > 1:
                    expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
                    original_length = data_batch[hint_key].shape[2]
                    assert original_length == expected_length, (
                        "Input control_input length doesn't match expected length specified by state_t."
                    )

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        super()._augment_image_dim_inplace(data_batch, input_key)
        # Handle control_input if it exists
        for key in data_batch.keys():
            if key.startswith("control_input_") and data_batch[key] is not None and data_batch[key].dim() == 4:
                data_batch[key] = rearrange(data_batch[key], "b c h w -> b c 1 h w").contiguous()

    def copy_weights_to_control_branch(self) -> None:
        """
        VACE has the skip design of control blocks: control block i output modulates base block 2i
        In ControlNet training beginning, we copy base model weights to control branch. There are two strategies:
        1. copy base model's i-th block weight to control net's i-th block (more intuitive, the control blocks is a trainable
         copy of the first N layers of the base model)
        2. copy base model's 2i-th block weight to control net's i-th block (follow the correspondence of skip connection, \
           but the block-to-block connection in the control branch is weird.)
        Here we adopt the first strategy.
        """
        if self.is_new_training:
            control_blocks = (
                self.net.control_blocks if self.net.num_control_branches == 1 else self.net.control_blocks_0
            )
            if self.copy_weight_strategy == "first_n":
                # copy base model's i-th block weight to control net's i-th block
                control_to_base_layer_maping = {i: i for i in range(len(control_blocks))}
                assert len(control_to_base_layer_maping) == len(control_blocks)
            elif self.copy_weight_strategy == "spaced_n":
                # copy base model's 2i-th block weight to control net's i-th block
                control_to_base_layer_maping = {v: k for k, v in self.net.control_layers_mapping.items()}
                assert len(control_to_base_layer_maping) == len(control_blocks)
            else:
                raise ValueError("Other copy weight strategy doesn't seem to make sense.")

            # 1. First copy weights from base model to control net
            for control_layer_idx, base_layer_idx in control_to_base_layer_maping.items():
                log.info(
                    f"======Copying base model's {base_layer_idx}-th block weight to control net's {control_layer_idx}-th block"
                )

                if self.net.num_control_branches > 1:
                    for nc in range(self.net.num_control_branches):
                        missing_keys, unexpected_keys = getattr(self.net, f"control_blocks_{nc}")[
                            control_layer_idx
                        ].load_state_dict(self.net.blocks[base_layer_idx].state_dict(), strict=False)
                else:
                    missing_keys, unexpected_keys = self.net.control_blocks[control_layer_idx].load_state_dict(
                        self.net.blocks[base_layer_idx].state_dict(), strict=False
                    )
                assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
                assert set(missing_keys).issubset(
                    {
                        "before_proj.weight",
                        "before_proj.bias",
                        "after_proj.weight",
                        "after_proj.bias",
                        "_checkpoint_wrapped_module.before_proj.weight",
                        "_checkpoint_wrapped_module.before_proj.bias",
                        "_checkpoint_wrapped_module.after_proj.weight",
                        "_checkpoint_wrapped_module.after_proj.bias",
                    }
                ), f"missing_keys: {missing_keys}"

            if self.net.separate_embedders:
                self.net.t_embedder_for_control_branch.load_state_dict(self.net.t_embedder.state_dict(), strict=True)
                self.net.t_embedding_norm_for_control_branch.load_state_dict(
                    self.net.t_embedding_norm.state_dict(), strict=True
                )
                self.net.x_embedder_for_control_branch.load_state_dict(self.net.x_embedder.state_dict(), strict=True)

            self.is_new_training = False

    def freeze_base_model(self):
        log.info("\nFreezing base model\n")
        # 1. freeze everything
        for param in self.net.parameters():
            param.requires_grad = False

        # 2. unfreeze control-specific parameters: the blocks and patch embedding
        if self.net.num_control_branches > 1:
            for nc in range(self.net.num_control_branches):
                for param in getattr(self.net, f"control_blocks_{nc}").parameters():
                    param.requires_grad = True
            if hasattr(self.net, "after_proj"):
                for param in self.net.after_proj.parameters():
                    param.requires_grad = True
        else:
            for block in self.net.control_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.net.control_embedder.parameters():
            param.requires_grad = True
        
        if self.net.use_dino_merge:
            for param in self.net.dinov3_mergehead.parameters():
                param.requires_grad = True

        if self.net.separate_embedders:
            for param in self.net.t_embedder_for_control_branch.parameters():
                param.requires_grad = True
            for param in self.net.t_embedding_norm_for_control_branch.parameters():
                param.requires_grad = True
            for param in self.net.x_embedder_for_control_branch.parameters():
                param.requires_grad = True

        if self.net.use_input_hint_block:
            for param in self.net.input_hint_block.parameters():
                param.requires_grad = True

        # 3. unfreeze reference image weights if we use reference image control

            # 3.1 Unfreeze reference image weights in each ControlAwareDiTBlock
            if hasattr(self.net, "blocks"):
                for i, block in enumerate(self.net.blocks):
                    # Access the actual block inside CheckpointWrapper
                    actual_block = block._checkpoint_wrapped_module
                    cross_attn = actual_block.cross_attn

                    # Unfreeze k_img, v_img, k_img_norm
                    for param_key in ["k_img", "v_img", "k_img_norm", "q_img", "q_img_norm"]:
                        if hasattr(cross_attn, param_key):
                            for param in getattr(cross_attn, param_key).parameters():
                                param.requires_grad = True

                    log.info(f"✓ Unfroze reference image weights in ControlAwareDiTBlock {i}")

            # 3.2 Unfreeze reference image weights in each ControlEncoderDiTBlock
            if hasattr(self.net, "control_blocks"):
                for i, block in enumerate(self.net.control_blocks):
                    # Access the actual block inside CheckpointWrapper
                    actual_block = block._checkpoint_wrapped_module
                    cross_attn = actual_block.cross_attn

                    # Unfreeze k_img, v_img, k_img_norm
                    for param_key in ["k_img", "v_img", "k_img_norm", "q_img", "q_img_norm"]:
                        if hasattr(cross_attn, param_key):
                            for param in getattr(cross_attn, param_key).parameters():
                                param.requires_grad = True

                    log.info(f"✓ Unfroze reference image weights in ControlEncoderDiTBlock {i}")

    def set_up_model(self):
        super().set_up_model()
        self.freeze_base_model()
        self.load_base_model()
        self.copy_weights_to_control_branch()

    def load_multi_branch_checkpoints(self, checkpoint_paths: list[str]):
        """
        Load control blocks from multiple checkpoint paths into control_blocks_0, control_blocks_1, etc.

        Args:
            checkpoint_paths (list[str]): List of checkpoint paths containing control blocks
        """
        if not checkpoint_paths:
            log.warning("No checkpoint paths provided for control branches")
            return

        # Use the same credentials as base model if available
        credential_path = "credentials/s3_checkpoint.secret"
        if hasattr(self.config, "base_load_from") and self.config.base_load_from is not None:
            credential_path = self.config.base_load_from.credentials

        load_planner = DefaultLoadPlanner(allow_partial_load=False)
        _model_wrapper = ModelWrapper(self)
        _state_dict = _model_wrapper.state_dict()

        # Filter out _extra_state entries to avoid metadata mismatch
        checkpoint_state_dict = {k: v for k, v in _state_dict.items() if "_extra_state" not in k}
        # Replace control_blocks_{nc} with control_blocks in the state dict
        for k in list(checkpoint_state_dict.keys()):
            for nc in range(self.net.num_control_branches):
                if f"control_blocks_{nc}" in k:
                    new_key = k.replace(f"control_blocks_{nc}", "control_blocks")
                    checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(k)
                elif f"control_embedder.{nc}" in k:
                    new_key = k.replace(f"control_embedder.{nc}", "control_embedder")
                    checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(k)

        for nc, checkpoint_path in enumerate(checkpoint_paths):
            if checkpoint_path is None:
                log.warning(f"No checkpoint path provided for control branch {nc}")
                continue

            checkpoint_format = "pt" if checkpoint_path.endswith(".pt") else "dcp"
            # Handle checkpoint path with or without "model" suffix
            cur_key_ckpt_full_path = (
                checkpoint_path
                if checkpoint_path.endswith("model") or checkpoint_format == "pt"
                else os.path.join(checkpoint_path, "model")
            )
            log.critical(f"Start loading checkpoint for control branch {nc} from {checkpoint_path}")

            if "s3://" in checkpoint_path:
                storage_reader = S3StorageReader(
                    credential_path=credential_path,
                    path=cur_key_ckpt_full_path,
                )
            else:
                storage_reader = FileSystemReader(cur_key_ckpt_full_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if checkpoint_format == "dcp":  # load dcp checkpoint
                dcp.load(
                    checkpoint_state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                )
            else:
                # load pytorch checkpoint appending all keys to checkpoint_to_model_keys
                checkpoint_state_dict = torch.load(checkpoint_path)

            # Create mapping from checkpoint keys to model keys
            # Checkpoint has "control_blocks" but we want to load into "control_blocks_{nc}"
            checkpoint_to_model_keys = {}
            for k, v in checkpoint_state_dict.items():
                if "control_blocks." in k:
                    # Replace "control_blocks" with "control_blocks_{nc}" in the key
                    new_key = k.replace("control_blocks", f"control_blocks_{nc}")
                    checkpoint_to_model_keys[new_key] = v
                elif "control_embedder" in k:
                    new_key = k.replace("control_embedder", f"control_embedder.{nc}")
                    checkpoint_to_model_keys[new_key] = v
                else:
                    checkpoint_to_model_keys[k] = v

            assert checkpoint_to_model_keys, f"No control_blocks keys found in checkpoint for branch {nc}"

            log.info(f"Checkpoint to model keys: {checkpoint_to_model_keys}")
            _model_wrapper.load_state_dict(checkpoint_to_model_keys)
            log.info(f"Done loading the control branch {nc} checkpoint.")

    def load_base_model(self) -> None:
        config = self.config
        if config.base_load_from is not None:
            checkpoint_path = config.base_load_from["load_path"]
        else:
            checkpoint_path = None
        # breakpoint()
        if checkpoint_path is not None:
            load_planner = DefaultLoadPlanner(allow_partial_load=True)
            if config.base_load_from.get("credentials", None):
                cur_key_ckpt_full_path = os.path.join("s3://", checkpoint_path, "model")
                if INTERNAL:
                    storage_reader = S3StorageReader(
                        credential_path=config.base_load_from.credentials,
                        path=cur_key_ckpt_full_path,
                    )
                else:
                    from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

                    checkpoint_path = get_checkpoint_path(cur_key_ckpt_full_path)
            else:
                storage_reader = FileSystemReader(checkpoint_path)

            log.critical(f"Start loading checkpoint for base model from {checkpoint_path}")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            _model_wrapper = ModelWrapper(self)
            _state_dict = _model_wrapper.state_dict()

            # Filter out _extra_state entries to avoid metadata mismatch
            filtered_state_dict = {k: v for k, v in _state_dict.items() if "_extra_state" not in k}

            # Copy EMA weights to regular weights
            all_keys = list(filtered_state_dict.keys())
            # log.info(f"All keys: {all_keys}")
            for k in all_keys:
                if k.startswith("net.") and k.replace("net.", "net_ema.") in filtered_state_dict:
                    filtered_state_dict[k] = filtered_state_dict[k.replace("net.", "net_ema.")]
            self.load_state_dict(easy_io.load(checkpoint_path), strict=False)
        log.info("Done loading the base model checkpoint.")

    def get_x_from_clean(
        self,
        in_clean_img: torch.Tensor,
        sigma_max: float | None,
        seed: int = 1,
    ) -> Tensor:
        """
        in_clean_img (torch.Tensor): input clean image for image-to-image/video-to-video by adding noise then denoising
        sigma_max (float): maximum sigma applied to in_clean_image for image-to-image/video-to-video
        """
        if in_clean_img is None:
            return None
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        noise = torch.randn(*in_clean_img.shape, **self.tensor_kwargs, generator=generator)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        x_sigma_max = in_clean_img + noise * sigma_max
        return x_sigma_max
