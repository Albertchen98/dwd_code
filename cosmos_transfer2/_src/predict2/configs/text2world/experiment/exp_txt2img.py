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
Configs for submitting large scale job using ./_submit.py from local workstation.

Recommended usage: the config here serve as a base config for large scale job, don't modify this script; instead, override specific fields in the config
by creating a new experiment in ./experiment_list.py. See examples there for how to add new experiments and how to submit a job.
"""

import functools
import math

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.predict2.datasets.cached_replay_dataloader import duplicate_batches_random
from cosmos_transfer2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.reason1.configs.default.model_config_qwen import QwenModelConfig, QwenVisionConfig
from cosmos_transfer2._src.reason1.models.vlm_qwen_omni import QwenVLBaseModel
from cosmos_transfer2._src.reason1.tokenizer.processor import build_tokenizer


txt2img_2B_720p_t5_embedding_rectified_flow_train_keyframe = LazyDict(
    dict(
        defaults=[
            {"override /model": "fsdp"},
            {"override /net": "cosmos_v2_2B"},
            {"override /conditioner": "add_fps_padding_mask"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="txt2img_2B_control",
            name="txt2img_2B_720p_t5_embedding_rectified_flow_train_keyframe",
        ),
        optimizer=dict(
            lr=8.63e-5,  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=1e-3,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[100],
            cycle_lengths=[100_000],
        ),
        model=dict(
            config=dict(
                fsdp_shard_size=1,
                resolution="720",
                state_t=1,
                sigma_data=1.0,
                scaling="rectified_flow",
                net=dict(
                    max_img_h=240,
                    max_img_w=240,
                    max_frames=128,
                    in_channels=16,
                    out_channels=16,
                    patch_spatial=2,
                    patch_temporal=1,
                    concat_padding_mask=True,
                    # attention settings
                    model_channels=2048,
                    num_blocks=28,
                    num_heads=16,
                    mlp_ratio=4.0,
                    atten_backend="minimal_a2a",
                    # cross attention settings
                    crossattn_emb_channels=1024,
                    # positional embedding settings
                    pos_emb_cls="rope3d",
                    pos_emb_learnable=True,
                    pos_emb_interpolation="crop",
                    min_fps=1,
                    max_fps=30,
                    use_adaln_lora=True,
                    adaln_lora_dim=256,
                    rope_h_extrapolation_ratio=4.0,
                    rope_w_extrapolation_ratio=4.0,
                    rope_t_extrapolation_ratio=1.0,
                    extra_per_block_abs_pos_emb=False,
                    extra_h_extrapolation_ratio=1.0,
                    extra_w_extrapolation_ratio=1.0,
                    extra_t_extrapolation_ratio=1.0,
                    rope_enable_fps_modulation=False,
                ),
                conditioner=dict(
                    text=dict(
                        dropout_rate=0,
                        use_empty_string=False,  # (TODO: hanzim): check
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                    compile_encode=False,
                    vae_pth="./model_hubs/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
                ),
                text_encoder_class="T5",
                text_encoder_config=dict(
                    compute_online=True,
                    ckpt_path="/cache/checkpoints/google-t5/t5-11b",
                ),
            )
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=False,
            load_path="cosmos_transfer2/vid2vid_2B_control/edge_720p_t24_spaced_layer4_cr1pt1_sdev2_lowsigma0.05_nonuniform_hqv3p1_20250714_64N/checkpoints/iter_000060000",
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=200,
            straggler_detection=dict(
                enabled=True,
                max_diff=1.5,
            ),
            callbacks=dict(
                iter_speed=dict(hit_thres=10000, every_n=100),
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=100000,
                    guidance=[0, 3, 7],
                ),
                every_n_sample_ema=dict(
                    every_n=100000,
                    guidance=[0, 3, 7],
                ),
            ),
        ),
        dataloader_train=dict(
            num_workers=4,
            batch_size=4,
            dataset=dict(
                dataset_dir="/cache/waymo",
                hint_keys="dino,",
                resolution="720",
                num_video_frames=1,
            )
        ),
        dataloader_val=dict(
            num_workers=4,
            dataset=dict(
                dataset_dir="/cache/waymo",
                hint_keys="dino,",
                resolution="720",
                num_video_frames=1,
            )
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()
cs.store(
            group="experiment",
            package="_global_",
            name=f"{txt2img_2B_720p_t5_embedding_rectified_flow_train_keyframe['job']['name']}",
            node=txt2img_2B_720p_t5_embedding_rectified_flow_train_keyframe,
        )