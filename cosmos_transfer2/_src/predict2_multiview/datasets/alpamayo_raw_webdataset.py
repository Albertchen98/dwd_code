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

import collections.abc as cabc
import gzip
import os
import pickle
import re
from typing import Iterator

import omegaconf
import webdataset as wds
from webdataset import filters

import cosmos_transfer2._src.predict2.datasets.webdataset as webdataset
from cosmos_transfer2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo, TarSample
from cosmos_transfer2._src.imaginaire.datasets.webdataset.utils.misc import remove_extensions_from_keys, skip_keys
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2_multiview.datasets.cache_utils import get_cam_t5_cache_dir
from cosmos_transfer2._src.predict2_multiview.utils.object_store import ObjectStore
from cosmos_transfer2._src.predict2_multiview.webdataset.utils.iterators import RawWebDataset
from cosmos_transfer2._src.predict2_multiview.webdataset.utils.misc import update_url


class AlpamayoRawWebdataset(webdataset.Dataset):
    def parse_dataset_info(
        self,
        dataset_info: list[DatasetInfo],
        use_multithread: bool = True,
    ):
        r"""Parse metadata about the list of tar files.

        Args:
            dataset_info (list): List of dictionaries containing paths to metadata files.
        """
        if use_multithread:
            log.warning("use_multithread not supported for AlpamayoRawWebdataset. Ignoring it.")
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
                    skip_files_without_t5 = t5_store_config["skip_files_without_t5"]

            # Read all wdinfo files and obtain the DataSample list
            for wdinfo_path, t5_mappings_path in dset_info.wdinfo:
                local_t5_mappings_path = os.path.join(cam_t5_cache_dir, t5_mappings_path)
                t5_mappings_data = pickle.loads(gzip.open(local_t5_mappings_path, "rb").read())
                if wdinfo_path.endswith(".jsonl"):
                    if use_object_store:
                        if not data_object_store_reader.object_exists(wdinfo_path):
                            raise FileNotFoundError(f"{wdinfo_path} not found")
                        data_list = data_object_store_reader.load_object(key=wdinfo_path, type="jsonl")["data"]  # type: ignore
                        cur_dset_info = {}
                        for k in data_list:
                            cur_dset_info[k["key"]] = k["pdx_url"].replace("s3://", "")
                    else:
                        with open(wdinfo_path, "rb") as fp:
                            cur_dset_info = pickle.load(fp)

                    if not hasattr(self.config, "sample_keys_full_list_path"):
                        # Remind the user to add the sample_keys_full_list_path to the config
                        log.warning("sample_keys_full_list_path not found in config;")
                    data_list = []
                    for key, url in cur_dset_info.items():
                        url = "/".join(url.split("/")[1:-1])
                        t5_tar_path = f"{t5_prefix}/{t5_mappings_data[key]}" if key in t5_mappings_data else None
                        data_list.append((url, (t5_tar_path, t5_dset_id, sampling_data)))
                    data_list = list(set(data_list))
                elif wdinfo_path.endswith(".pkl.gz"):
                    local_urllist_path = os.path.join(cam_t5_cache_dir, wdinfo_path)
                    log.debug(f"Loading {wdinfo_path} locally")
                    url_list = pickle.loads(gzip.open(local_urllist_path, "rb").read())
                    data_list = []
                    urls_without_t5 = []
                    for url in url_list:
                        key = url.split("/")[-1]
                        t5_tar_path = f"{t5_prefix}/{t5_mappings_data[key]}" if key in t5_mappings_data else None
                        if t5_tar_path is None:
                            urls_without_t5.append((url, (None, t5_dset_id, sampling_data)))
                        data_list.append((url, (t5_tar_path, t5_dset_id, sampling_data)))
                    if skip_files_without_t5:
                        data_list = [k for k in data_list if k[1][0] is not None]
                        log.warning(
                            f"Skipping {len(urls_without_t5)} out of {len(url_list)} files without t5 tar path. They will not be included in the dataset."
                        )
                    else:
                        log.warning(
                            f"Found {len(urls_without_t5)} out of {len(url_list)} urls without t5 tar path. These will be included in the dataset."
                        )
                else:
                    raise ValueError(f"Unsupported file type: {wdinfo_path}")
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

    def build_dataset(self, **kwargs) -> RawWebDataset:
        r"""
        Overrides parent function by updating the WebDataset class to RawWebDataset, and update_url function
        """
        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(self.config, "buffer_size", self.wdinfo.chunk_size)

        # update distributor urls and chunk size
        distributor_fn = self.config.distributor

        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)

        dataset = RawWebDataset(
            distributor_fn,
            load_from_object_store=self.use_object_store,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket,
            streaming_download=self.streaming_download,
            handler=self.handler,
        )

        # Creating a shuffle buffer
        if self.detshuffle:
            dataset.append(filters.detshuffle(shuffle_buffer_size))
        else:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        def filter_out_key_extensions(data: Iterator[dict]) -> Iterator[dict]:
            for data_dict in data:
                output_data_dict = {}
                allowed_extensions = ["mp4", "json", "bin"]
                for key, value in data_dict.items():
                    if "." in key:
                        extension = re.sub(r".*[.]", "", key)
                        if extension in allowed_extensions:
                            output_data_dict[key] = value
                    else:
                        output_data_dict[key] = value
                yield output_data_dict

        dataset.append(filter_out_key_extensions)

        # Adding decoders
        # Decoders are functions that decode the input IO stream
        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = []
        for decoder in decoder_list:
            # If the specified decoder is a string, use the webdataset decoder
            # If its a callable function, use the defined function to decode data
            assert isinstance(decoder, str) or callable(decoder), "Decoder should either be callable or a str"
            decoder_functions.append(decoder)
        dataset.append(wds.decode(*decoder_functions, handler=self.decoder_handler))

        # After the decoders are added, remove extension from the keys
        # Extensions in the data keys are needed for auto-detection of decoders in webdataset.
        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)

        # Function to skip keys
        dataset.append(skip_keys)

        # dataset.append(dbg("after_skip_keys"))

        # Building augmentors
        augmentor_cfg = getattr(self.config, "augmentation", None)
        assert isinstance(augmentor_cfg, (dict, omegaconf.dictconfig.DictConfig)), (
            f"getting type: {type(augmentor_cfg)}"
        )
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        # Updates URL names so that the collate function can handle
        dataset.append(update_url)

        dataset.total_images = self.wdinfo.total_key_count  # type: ignore
        log.info("Total number of training shards: %d" % num_tars)
        log.info("Total training key count: %d" % dataset.total_images)  # type: ignore

        return dataset


def dbg(tag: str):
    """
    Pipeline tap: prints every element that flows past.
    Works whether that element is a generator (early stages) or
    an individual sample dict (later stages).

    Usage:
        dataset.append(dbg("after_expander"))
    """

    def _dbg(element):
        # 1) If the element is already a sample dict ⇒ just log & yield it.
        if isinstance(element, dict):
            log.info(f"[{tag}] dict  {element.get('__key__', '<no key>')} {list(element.keys())}")
            yield element
            return

        # 2) Otherwise the element is an iterator/generator.
        if isinstance(element, cabc.Iterable):
            for sample in element:
                if isinstance(sample, dict):
                    log.info(f"[{tag}] sample {sample.get('__key__', '<no key>')} {list(sample.keys())}")
                else:
                    log.info(f"[{tag}] item of type {type(sample)}")
                yield sample
            return

        # 3) Fallback: single non-dict item (rare)
        log.info(f"[{tag}] single item of type {type(element)}")
        yield element

    return _dbg
