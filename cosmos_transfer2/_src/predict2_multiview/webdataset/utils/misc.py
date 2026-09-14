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
from typing import Iterator


def update_url(data: Iterator[dict]) -> Iterator[dict]:
    r"""Function to update the URLs so that the TarSample is removed from data.
    Instead, we replace the URL with a string.
    Args:
        data (dict): Input data dict
    Returns:
        data dict with URL replaced with a string
    """
    for data_dict in data:
        if isinstance(data_dict["__url__"].path, tuple):
            data_dict["__t5_url__"] = data_dict["__url__"].path[1][0]
            data_dict["__url__"] = os.path.join(data_dict["__url__"].root, data_dict["__url__"].path[0])

        else:
            data_dict["__url__"] = os.path.join(data_dict["__url__"].root, data_dict["__url__"].path)
        yield data_dict
