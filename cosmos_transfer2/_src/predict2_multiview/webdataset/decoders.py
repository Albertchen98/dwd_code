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

import json
import pickle
import re
from typing import Optional

import numpy as np

from cosmos_transfer2._src.imaginaire.utils import log


def bin_decoder(key: str, data: bytes) -> Optional[dict]:
    r"""
    Function to decode a pkl file.
    Args:
        key: Data key.
        data: Data dict.
    """
    extension = re.sub(r".*[.]", "", key)
    try:
        if extension == "bin":
            data_dict = np.array(pickle.loads(data))
            return data_dict
        else:
            return None
    except Exception as e:
        log.error(f"Error decoding {key}: {e}")
        return None


def json_decoder(key: str, data: bytes) -> Optional[dict]:
    r"""
    Function to decode a json file.
    Args:
        key: Data key.
        data: Data dict.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension == "json":
        data_dict = json.loads(data)
        return data_dict
    else:
        return None
