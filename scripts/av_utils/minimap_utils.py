# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
File modified from https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits
"""

import json
from pathlib import Path
from typing import Final

import numpy as np

from av_utils.camera.base import CameraBase


def load_hdmap_colors(version: str = "v3") -> dict:
    """
    Load hdmap colors based on the specified version.

    Args:
        version: str, version key from config_color_hdmap.json
                Available versions: 'v3'

    Returns:
        dict: Color configuration for the specified version
    """
    # Try to find the config file in the local color_configs directory
    config_path = Path(__file__).parent / "color_configs" / "config_color_hdmap.json"
    with open(config_path) as f:
        hdmap_config = json.load(f)

    if version not in hdmap_config:
        available_versions = list(hdmap_config.keys())
        raise ValueError(f"Version '{version}' not found in hdmap config. Available versions: {available_versions}")

    return hdmap_config[version]


def load_laneline_colors() -> dict:
    """
    Load laneline type-specific colors.

    Returns:
        dict: Color configuration for laneline types
    """
    config_path = Path(__file__).parent / "color_configs" / "config_color_laneline.json"
    with open(config_path) as f:
        laneline_config = json.load(f)
    return laneline_config


def load_traffic_light_colors(version: str = "v2") -> dict:
    """
    Load traffic light colors based on the specified version.

    Args:
        version: str, version key from config_color_traffic_light.json
                Available versions: 'v2'

    Returns:
        dict: Color configuration for the specified version
    """
    config_path = Path(__file__).parent / "color_configs" / "config_color_traffic_light.json"
    with open(config_path) as f:
        tl_config = json.load(f)

    if version not in tl_config:
        available_versions = list(tl_config.keys())
        raise ValueError(
            f"Version '{version}' not found in traffic light config. Available versions: {available_versions}"
        )

    return tl_config[version]


MINIMAP_TO_TYPE: Final = {
    "lanelines": "polyline",
    "lanes": "polyline",
    "poles": "polyline",
    "road_boundaries": "polyline",
    "wait_lines": "polyline",
    "crosswalks": "polygon",
    "road_markings": "polygon",
    "traffic_signs": "cuboid3d",
    "traffic_lights": "cuboid3d",
    "intersection_areas": "polygon",
    "road_islands": "polygon",
}

MINIMAP_TO_SEMANTIC_LABEL: Final = {
    "lanelines": 5,
    "lanes": 5,
    "poles": 9,
    "road_boundaries": 5,
    "wait_lines": 10,
    "crosswalks": 5,
    "road_markings": 10,
}


def extract_vertices(minimap_data: dict | list, vertices_list: list | None = None) -> list:
    if vertices_list is None:
        vertices_list = []

    if isinstance(minimap_data, dict):
        for key, value in minimap_data.items():
            if key == "vertices":
                vertices_list.append(value)
            else:
                extract_vertices(value, vertices_list)
    elif isinstance(minimap_data, list):
        for item in minimap_data:
            extract_vertices(item, vertices_list)
    else:
        raise ValueError(f"Invalid minimap data type: {type(minimap_data)}")

    return vertices_list


def get_type_from_name(minimap_name: str) -> str:
    """
    Args:
        minimap_name: str, name of the minimap

    Returns:
        minimap_type: str, type of the minimap
    """
    if minimap_name in MINIMAP_TO_TYPE:
        return MINIMAP_TO_TYPE[minimap_name]
    else:
        raise ValueError(f"Invalid minimap name: {minimap_name}")


def cuboid3d_to_polyline(cuboid3d_eight_vertices: np.ndarray) -> np.ndarray:
    """
    Convert cuboid3d to polyline

    Args:
        cuboid3d_eight_vertices: np.ndarray, shape (8, 3), dtype=np.float32,
            eight vertices of the cuboid3d

    Returns:
        polyline: np.ndarray, shape (N, 3), dtype=np.float32,
            polyline vertices
    """
    if isinstance(cuboid3d_eight_vertices, list):
        cuboid3d_eight_vertices = np.array(cuboid3d_eight_vertices)

    connected_vertices_indices = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
    connected_polyline = np.array(cuboid3d_eight_vertices)[connected_vertices_indices]

    return connected_polyline


def create_minimap_projection(
    minimap_name: str,
    minimap_data_wo_meta_info: list,
    camera_poses: np.ndarray,
    camera_model: CameraBase,
    hdmap_color_version: str = "v3",
) -> np.ndarray:
    """
    Args:
        minimap_name: str, name of the minimap
        minimap_data_wo_meta_info: list of np.ndarray
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera poses of N frames
        camera_model: CameraModel, camera model
        hdmap_color_version: str, version key for hdmap colors ('v3')

    Returns:
        minimaps_projection: np.ndarray,
            shape (N, H, W, 3), dtype=np.uint8, projected minimap data across N frames
    """

    # Load colors based on the specified version
    minimap_to_rgb = load_hdmap_colors(hdmap_color_version)

    minimap_type = get_type_from_name(minimap_name)

    if minimap_type == "polygon":
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
        )
    elif minimap_type == "polyline":
        if minimap_name == "lanelines" or minimap_name == "road_boundaries":
            segment_interval = 0.8
        else:
            segment_interval = 0

        projection_images = camera_model.draw_line_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
            segment_interval=segment_interval,
        )
    elif minimap_type == "cuboid3d":
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
        )
    else:
        raise ValueError(f"Invalid minimap type: {minimap_type}")

    return projection_images


def create_laneline_type_projection(
    laneline_data_with_types: list,
    camera_poses: np.ndarray,
    camera_model: CameraBase,
) -> np.ndarray:
    """
    Create laneline projection with type-specific colors.

    Args:
        laneline_data_with_types: list of tuples (polyline, type_string)
            where type_string is like "WHITE SOLID_SINGLE", "YELLOW DASHED_SINGLE", etc.
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera poses of N frames
        camera_model: CameraModel, camera model

    Returns:
        minimaps_projection: np.ndarray,
            shape (N, H, W, 3), dtype=np.uint8, projected laneline data across N frames
    """
    laneline_type_to_rgb = load_laneline_colors()

    # Group lanelines by type for efficient rendering
    type_to_polylines = {}
    for polyline, type_string in laneline_data_with_types:
        if type_string not in type_to_polylines:
            type_to_polylines[type_string] = []
        type_to_polylines[type_string].append(polyline)

    # Initialize output array
    num_frames = len(camera_poses) if camera_poses.ndim == 3 else 1
    projection_images = np.zeros((num_frames, camera_model.height, camera_model.width, 3), dtype=np.uint8)

    # Render each type with its specific color
    for type_string, polylines in type_to_polylines.items():
        # Get color for this type, default to "OTHER" if not found
        color = laneline_type_to_rgb.get(type_string, laneline_type_to_rgb.get("OTHER", [100, 100, 100]))

        # Render this type's lanelines
        type_projection = camera_model.draw_line_depth(
            camera_poses,
            polylines,
            colors=np.array(color),
            segment_interval=0.8,  # Smooth the polyline
        )

        # Merge with existing projection
        projection_images = np.maximum(projection_images, type_projection)

    return projection_images
