#!/bin/bash

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

# Set up the environment
export PYTHONPATH=.
export MODEL_NAME="sample"
export OUTPUT_DIR="data/outputs"
export UPLOADS_DIR="data/uploads"
export LOG_FILE="data/output.log"
export NUM_GPUS=2

python3 sample/bootstrapper.py 2>&1 | tee $LOG_FILE
