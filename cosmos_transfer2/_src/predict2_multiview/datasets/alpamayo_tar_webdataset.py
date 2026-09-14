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

import gzip
import os
import pickle

from cosmos_transfer2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo, TarSample
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2_multiview.datasets.alpamayo_raw_webdataset import AlpamayoRawWebdataset
from cosmos_transfer2._src.predict2_multiview.datasets.cache_utils import get_cam_t5_cache_dir
from cosmos_transfer2._src.predict2_multiview.utils.object_store import ObjectStore


class AlpamayoTarWebdataset(AlpamayoRawWebdataset):
    def parse_dataset_info(
        self,
        dataset_info: list[DatasetInfo],
        use_multithread: bool = False,
    ):
        r"""Parse metadata about the list of tar files.

        Args:
            dataset_info (list): List of dictionaries containing paths to metadata files.
        """
        if use_multithread:
            log.warning("use_multithread not supported for AlpamayoTarWebdataset. Ignoring it.")
        cam_t5_cache_dir = get_cam_t5_cache_dir()
        t5_object_store_initialized = False
        t5_prefix = None
        skip_files_without_t5 = False
        t5_dset_id = "t5_dset: 0"
        for dset_num, dset_info in enumerate(dataset_info):
            # For each dataset, we parse the file paths and store them as a list of TarSample.
            # TarSample will then be used by each worker to load the data.
            use_object_store = dset_info.object_store_config.enabled
            self.use_object_store = use_object_store
            dset_id = "dset: {}".format(dset_num)
            sampling_data = (dset_info.opts["view_indices_options"], dset_info.opts["view_id_to_camera_key"])
            download_t5_tar = dset_info.opts["download_t5_tar"]
            overfit_firstn = dset_info.opts["overfit_firstn"]
            if use_object_store:
                data_object_store_reader = ObjectStore(config_object_storage=dset_info.object_store_config)
                # Create PBSS config if data is loaded from PBSS
                bucket_dset = dset_info.object_store_config.bucket
                s3_client_dset = data_object_store_reader.client
                self.s3_client[dset_id] = s3_client_dset
                self.bucket[dset_id] = bucket_dset
                if not t5_object_store_initialized:
                    t5_store_config = dset_info.opts["t5_store_config"]
                    t5_object_store_reader = ObjectStore(config_object_storage=t5_store_config["object_store"])
                    t5_object_store_initialized = True

                    self.s3_client[t5_dset_id] = t5_object_store_reader.client
                    self.bucket[t5_dset_id] = t5_store_config["object_store"].bucket
                    t5_prefix = t5_store_config["prefix"]
                    video_prefix = t5_store_config["video_prefix"]
                    skip_files_without_t5 = t5_store_config["skip_files_without_t5"]
                    assert skip_files_without_t5, "skip_files_without_t5 must be True for tar webdataset"

            # Read all wdinfo files and obtain the DataSample list
            for wdinfo_path, t5_mappings_path in dset_info.wdinfo:
                local_urllist_path = os.path.join(cam_t5_cache_dir, wdinfo_path)
                log.debug(f"Loading {wdinfo_path} locally")
                url_list = pickle.loads(gzip.open(local_urllist_path, "rb").read())
                local_t5_mappings_path = os.path.join(cam_t5_cache_dir, t5_mappings_path)
                t5_mappings_data = pickle.loads(gzip.open(local_t5_mappings_path, "rb").read())
                data_list = []
                for url in url_list:
                    key = url.split("/")[-1]
                    if key not in t5_mappings_data:
                        continue
                    t5_tar_path_val = t5_mappings_data[key]
                    t5_tar_path = f"{t5_prefix}/{t5_tar_path_val}" if key in t5_mappings_data else None
                    chunk_id = t5_tar_path_val.split("/")[-3]
                    tar_url = f"{video_prefix}/{chunk_id}/{key}"
                    data_list.append((tar_url, (t5_tar_path, t5_dset_id, sampling_data, download_t5_tar)))
                log.info(f"Skip loading sample_keys_full_list_paths for {wdinfo_path}")
                if overfit_firstn > 0:
                    data_list = sorted(data_list, key=lambda x: x[0])[:overfit_firstn]
                sample_keys_full_list_per_tar = [None] * len(data_list)
                tar_files_list = data_list
                tar_files = [
                    TarSample(
                        path=tar_file,
                        root=bucket_dset,
                        keys=(
                            dset_info.per_dataset_keys if dset_info.per_dataset_keys else self.data_keys
                        ),  # use per dataset keys if available
                        meta=dset_info,
                        dset_id=dset_id,
                        sample_keys_full_list=sample_keys_full_list,
                    )
                    for tar_file, sample_keys_full_list in zip(
                        tar_files_list, sample_keys_full_list_per_tar, strict=True
                    )
                ]

                # Update the master winfo
                self.wdinfo.tar_files.extend(tar_files)
                self.wdinfo.total_key_count += len(data_list)
                self.wdinfo.chunk_size = 1
        log.info(f"Total number of samples: {self.wdinfo.total_key_count}")
