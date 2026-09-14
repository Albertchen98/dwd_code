# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Configuration for rendering HD maps.
"""

from collections.abc import Mapping

# Settings aligned with dataset_rds_hq_mv.json
# From https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits
SETTINGS = {
    "INPUT_POSE_FPS": 30,  # Target pose FPS after interpolation
    "INPUT_LIDAR_FPS": 10,
    "GT_VIDEO_FPS": 30,
    "SOURCE_POSE_FPS": 10,  # Original ego pose FPS in clipGT
    "SOURCE_VIDEO_FPS": 30,  # Original video FPS in clipGT
    "SOURCE_OBSTACLE_FPS": 10,  # Original obstacle FPS in clipGT
    "COSMOS_RESOLUTION": [1280, 720],
    "RESIZE_RESOLUTION": [1280, 720],
    "TARGET_CHUNK_FRAME": 121,
    "OVERLAP_FRAME": 0,
    "TARGET_RENDER_FPS": 30,
    "MAX_CHUNK": 10,
}

CAMERA_NAME_MAPPING: Mapping = {
    "camera:front:wide:120fov": "camera_front_wide_120fov",
    "camera:front:tele:sat:30fov": "camera_front_tele_30fov",
    "camera:cross:right:120fov": "camera_cross_right_120fov",
    "camera:cross:left:120fov": "camera_cross_left_120fov",
    "camera:rear:left:70fov": "camera_rear_left_70fov",
    "camera:rear:right:70fov": "camera_rear_right_70fov",
    "camera:rear:tele:30fov": "camera_rear_tele_30fov",
}
