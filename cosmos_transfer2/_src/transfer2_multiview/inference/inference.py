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
EXP=buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_world_scenario
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_world_scenario-0/checkpoints/iter_000021000
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference --seed 0 --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 8 --max_samples 1 --save_root results/
```
"""

import pickle
import random
import argparse
import os

import torch
import numpy as np
from loguru import logger
from megatron.core import parallel_state

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint


def set_seeds(seed: int, deterministic: bool = False):
    """
    Set all random seeds for maximum reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable all deterministic settings
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # CuDNN settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True)

        # Set environment variables for additional reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic CUBLAS
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info(f"All random seeds set to {seed}, deterministic mode: {deterministic}")


def to_model_input(data_batch, model):
    """
    Similar to misc.to, but avoid converting uint8 "video" to float
    """
    for k, v in data_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch


class ControlVideo2WorldInference:
    """
    Handles the Vid2Vid inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        context_parallel_size: int = 1,
        experiment_opts: tuple[str, ...] = (),
    ):
        """
        Initializes the Vid2VidInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
            experiment_opts (tuple[str, ...]): Experiment options overrides.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        # Initialize distributed processing if context parallel size > 1
        if self.context_parallel_size > 1:
            self._init_distributed()

        # Load the model and config
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py",
            load_ema_to_reg=True,
            experiment_opts=list(experiment_opts),
        )

        # Enable context parallel on the model if using context parallelism
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.config = config
        self.batch_size = 1

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

        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def generate_from_batch(
        self,
        data_batch,
        guidance: float = 7.0,
        seed: int = 1,
    ):
        data_batch = to_model_input(data_batch, self.model)
        if self.model.config.text_encoder_config is not None and self.model.config.text_encoder_config.compute_online:
            self.model.inplace_compute_text_embeddings_online(data_batch)

        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            seed=seed,  # Fixed seed for reproducibility
            is_negative_prompt=True,
        )
        # (bsz = 1, c = 3, t = n_camera * t, h, w)
        return self.model.decode(sample).cpu()

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
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    # generation
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=1, help="Guidance value")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    # input
    parser.add_argument("--save_root", type=str, default="results/image2world", help="Save root")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to generate")
    parser.add_argument("--control_weight", type=float, default=1.0, help="Control weight")
    # reproducibility
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms for reproducibility (may impact performance)",
    )
    # output
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Additionally save results as numpy files",
    )
    parser.add_argument(
        "--save_data_batch",
        action="store_true",
        help="Additionally save data batch as pkl files",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the inference."""
    torch.enable_grad(False)
    args = parse_arguments()

    # Set all random seeds and enable deterministic mode if requested
    set_seeds(args.seed, deterministic=args.deterministic)
    experiment_opts = ()
    if args.deterministic:
        experiment_opts = ("dataloader_train.dataset.is_train=False",)

    # Initialize the inference handler with context parallel support
    vid2vid_cli = ControlVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        context_parallel_size=args.context_parallel_size,
        experiment_opts=experiment_opts,
    )
    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    os.makedirs(args.save_root, exist_ok=True)
    dataloader = instantiate(vid2vid_cli.config.dataloader_train)
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break
        batch["control_weight"] = args.control_weight
        if args.save_data_batch and rank0:
            with open(f"{args.save_root}/data_batch_{i}.pkl", "wb") as f:
                pickle.dump(batch, f)

        video = vid2vid_cli.generate_from_batch(batch, guidance=args.guidance, seed=args.seed)
        if rank0:
            rescaled = (1.0 + video[0]) / 2.0
            save_img_or_video(rescaled, f"{args.save_root}/infer_from_train_{i}", fps=args.fps)

            if args.save_npy:
                rescaled_npy = rescaled.float().cpu().numpy()
                np.save(f"{args.save_root}/infer_from_train_{i}.npy", rescaled_npy)


if __name__ == "__main__":
    main()
