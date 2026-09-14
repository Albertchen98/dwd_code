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
import os
import pickle
from typing import Tuple
from safetensors.torch import save_file

import numpy as np
import torch
import tqdm
import ray

from cosmos_transfer2._src.predict2.text_encoders.text_encoder import TextEncoderConfig, TextEncoder

"""example command
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings.py --dataset_path datasets/hdvila
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute cosmos_reason embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="/cache/waymo", help="Root path to the dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="./nvidia/Cosmos-Reason1-7B", help="cosmos_reason model name or the local path"
    )
    parser.add_argument("--num_gpus", type=float, default=1.0, help="Number of GPUs per worker")
    parser.add_argument("--num_cpus", type=float, default=4.0, help="Number of CPUs per worker")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: number of GPUs available)")
    return parser.parse_args()


@ray.remote(num_gpus=1, num_cpus=4)
class TextEncoderWorker:
    def __init__(self, pretrained_model_name_or_path: str):
        config = TextEncoderConfig(
            embedding_concat_strategy="full_concat",
            ckpt_path=pretrained_model_name_or_path
        )
        self.text_encoder = TextEncoder(config)
        print(f"TextEncoderWorker initialized with model: {pretrained_model_name_or_path}")
    
    def process_single_file(self, meta_filename: str, t5_xxl_dir: str) -> str:
        """Process a single file and save embeddings"""
        t5_xxl_filename = os.path.join(t5_xxl_dir, os.path.basename(meta_filename).replace(".txt", ".safetensors"))
        
        # Skip if the file already exists
        # if os.path.exists(t5_xxl_filename):
        #     return f"Skipped (already exists): {meta_filename}"
        
        try:
            # Read prompt from file
            with open(meta_filename, "r") as fp:
                prompt = fp.read().strip()
            
            # Compute T5 embeddings
            encoded_text = self.text_encoder.compute_text_embeddings_online(
                {
                    "text": [prompt],
                },
                "text",
            )
            encoded_text = encoded_text.cpu().contiguous()
            # Save T5 embeddings as pickle file
            # with open(t5_xxl_filename, "wb") as fp:
            #     pickle.dump(encoded_text, fp)
            save_file({"text_embedding": encoded_text}, t5_xxl_filename)
            
            return f"Successfully processed: {meta_filename}"
        
        except Exception as e:
            return f"Error processing {meta_filename}: {str(e)}"


def main(args) -> None:
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    metas_dir = os.path.join(args.dataset_path, "metas")
    metas_list = [
        os.path.join(metas_dir, filename) for filename in sorted(os.listdir(metas_dir)) if filename.endswith(".txt")
    ]

    t5_xxl_dir = os.path.join(args.dataset_path, "cosmos_reason_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)

    # Determine number of workers
    if args.num_workers is None:
        num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        num_workers = args.num_workers
    
    if not metas_list:
        raise ValueError(f"No caption .txt files found in {metas_dir}")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    print(f"Using {num_workers} workers for parallel processing")
    print(f"Found {len(metas_list)} text files to process")
    
    # Create text encoder workers
    text_encoder_workers = [
        TextEncoderWorker.remote(args.pretrained_model_name_or_path) 
        for _ in range(num_workers)
    ]
    
    # Distribute tasks to workers in a round-robin fashion
    futures = []
    for i, meta_filename in enumerate(metas_list):
        worker_idx = i % num_workers
        # 这里调用的是 worker 实例的远程方法，不是普通函数
        future = text_encoder_workers[worker_idx].process_single_file.remote(meta_filename, t5_xxl_dir)
        futures.append(future)
    
    # Collect results with progress bar
    results = []
    for future in tqdm.tqdm(futures, total=len(futures), desc="Processing files"):
        result = ray.get(future)
        results.append(result)
    
    # Print summary
    successful = sum(1 for r in results if "Successfully" in r)
    errors = sum(1 for r in results if "Error" in r)
    skipped = sum(1 for r in results if "Skipped" in r)
    
    print(f"\nProcessing completed:")
    print(f"  Successful: {successful}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    
    # Print errors if any
    if errors > 0:
        print("\nError details:")
        for result in results:
            if "Error" in result:
                print(f"  {result}")
        raise RuntimeError(f"Text embedding extraction failed for {errors} files")

# python scripts/get_cr1_embeddings.py --dataset_path /cache/waymo
if __name__ == "__main__":
    args = parse_args()
    main(args)