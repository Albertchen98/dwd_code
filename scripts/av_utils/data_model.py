# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Task data model of the transfer2-multiview pipeline."""

from pathlib import Path

import attrs
import numpy as np

from av_utils.camera.ftheta import FThetaCamera


@attrs.define
class SceneFiles:
    """Container for all ClipGT scene file paths (local paths only)."""

    # Required files
    obstacle_file: Path
    calibration_estimate_file: Path
    egomotion_estimate_file: Path

    # Optional map files
    lane_file: Path | None = None
    lane_line_file: Path | None = None
    road_boundary_file: Path | None = None
    crosswalk_file: Path | None = None
    pole_file: Path | None = None
    road_marking_file: Path | None = None
    wait_line_file: Path | None = None
    traffic_light_file: Path | None = None
    traffic_sign_file: Path | None = None


@attrs.define
class LocalControlTask:
    """Simplified local processing task - no S3 dependencies."""

    # File paths (all local)
    scene_files: SceneFiles
    clip_id: str
    camera_names: list[str]

    # rendering parameters
    resize_resolution: tuple[int, int] = attrs.field(default=(1280, 720))
    input_pose_fps: int = attrs.field(default=30)
    max_frames: int = -1

    # processed data (filled during processing)
    camera_models: dict[str, FThetaCamera] | None = None
    camera_poses: dict[str, np.ndarray] | None = None
    all_object_info: dict | None = None
    map_data: dict | None = None
    rendered_frames: dict[str, np.ndarray] | None = None
