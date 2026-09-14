# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
HD map rendering utilities for generating control videos.
"""

from typing import Literal

import numpy as np
from av_utils.bbox_utils import (
    build_cuboid_bounding_box,
    load_bbox_colors,
    simplify_type_in_object_info,
)
from av_utils.camera.ftheta import FThetaCamera
from av_utils.clip_gt_io import (
    load_camera_from_calibration_estimate,
    load_ego_poses_interpolated,
    load_map_data,
    load_obstacle_data_interpolated,
)
from av_utils.data_model import SceneFiles
from av_utils.graphics_utils import (
    BoundingBox2D,
    LineSegment2D,
    Polygon2D,
    TriangleList2D,
    render_geometries,
)
from av_utils.laneline_utils import prepare_laneline_geometry_data
from av_utils.minimap_utils import (
    cuboid3d_to_polyline,
    get_type_from_name,
    load_hdmap_colors,
)
from av_utils.pcd_utils import (
    filter_by_height_relative_to_ego,
    interpolate_polyline_to_points,
    triangulate_polygon_3d,
)
from av_utils.traffic_light_utils import (
    create_traffic_light_status_geometry_objects_from_data,
    prepare_traffic_light_status_data_clipgt,
)
from loguru import logger
from scipy.spatial.transform import Rotation


def load_and_interpolate(
    scene_files: SceneFiles,
    camera_names: list[str],
    resize_resolution: tuple[int, int],
    input_pose_fps: int,
) -> tuple[dict[str, FThetaCamera], dict[str, np.ndarray], dict[str, dict], dict[str, list[dict]]]:
    """Render HD map visualization with full map elements and proper interpolation."""

    logger.info("Rendering with full map elements and fixed obstacle interpolation")

    # Load camera calibration
    camera_models = load_camera_from_calibration_estimate(
        scene_files.calibration_estimate_file, camera_names, resize_resolution
    )

    # Load and interpolate ego poses to 30Hz
    logger.info("Loading ego poses and interpolating from source to 30Hz...")
    ego_positions_30hz, ego_quaternions_30hz, ego_timestamps_30hz = load_ego_poses_interpolated(
        scene_files.egomotion_estimate_file, input_pose_fps
    )

    all_camera_poses = dict()
    all_camera_models = dict()
    for camera_name, (camera_model, camera_extrinsics) in camera_models.items():
        # Parse camera extrinsics
        cam_pos = camera_extrinsics["t"]
        cam_rpy = camera_extrinsics["roll-pitch-yaw"]
        cam_rotation = Rotation.from_euler("xyz", np.radians(cam_rpy))

        # Build camera poses for all frames at 30Hz
        camera_poses = []
        for i in range(len(ego_positions_30hz)):
            ego_to_world = np.eye(4)
            ego_rotation = Rotation.from_quat(ego_quaternions_30hz[i])
            ego_to_world[:3, :3] = ego_rotation.as_matrix()
            ego_to_world[:3, 3] = ego_positions_30hz[i]

            camera_to_ego = np.eye(4)
            camera_to_ego[:3, :3] = cam_rotation.as_matrix()
            camera_to_ego[:3, 3] = cam_pos

            camera_to_world_flu = ego_to_world @ camera_to_ego

            # Convert to OpenCV convention
            camera_to_world_opencv = np.concatenate(
                [
                    -camera_to_world_flu[:, 1:2],
                    -camera_to_world_flu[:, 2:3],
                    camera_to_world_flu[:, 0:1],
                    camera_to_world_flu[:, 3:4],
                ],
                axis=1,
            )

            camera_poses.append(camera_to_world_opencv)

        all_camera_poses[camera_name] = np.array(camera_poses)
        all_camera_models[camera_name] = camera_model

    # Determine frames to render
    render_frame_ids = list(range(len(ego_positions_30hz)))
    logger.info(f"Processing {len(render_frame_ids)} frames at 30Hz...")

    # Load obstacle data with interpolation
    logger.info("Loading and interpolating obstacle data from 10Hz to 30Hz...")
    all_object_info = load_obstacle_data_interpolated(scene_files.obstacle_file, ego_timestamps_30hz)

    # Load map data (now loads ALL elements)
    logger.info("Loading map data...")
    map_data = load_map_data(
        lane_file=scene_files.lane_file,
        lane_line_file=scene_files.lane_line_file,
        road_boundary_file=scene_files.road_boundary_file,
        crosswalk_file=scene_files.crosswalk_file,
        pole_file=scene_files.pole_file,
        road_marking_file=scene_files.road_marking_file,
        wait_line_file=scene_files.wait_line_file,
        traffic_light_file=scene_files.traffic_light_file,
        traffic_sign_file=scene_files.traffic_sign_file,
    )

    return all_camera_models, all_camera_poses, all_object_info, map_data


def create_minimap_geometry_objects_from_data(
    map_data: dict,
    camera_pose: np.ndarray,
    camera_model: FThetaCamera,
    hdmap_color_version: str = "v3",
    camera_pose_init: np.ndarray | None = None,
) -> list:
    """
    Build geometry objects for minimap layers for a single frame.

    Args:
        map_data: dict[name -> list[np.ndarray]], map data
        camera_pose: np.ndarray (4,4), camera pose
        camera_model: FThetaCamera, camera model
        hdmap_color_version: str, HD map color version
        camera_pose_init: Optional[np.ndarray], initial camera pose for height filtering

    Returns:
        list: geometry objects (LineSegment2D/Polygon2D/TriangleList2D)
    """
    minimap_to_rgb = load_hdmap_colors(hdmap_color_version)
    all_geometry_objects = []

    for minimap_name, elements in map_data.items():
        if not elements:
            continue

        # Skip 'lanes' if not in color config (it's often just for compatibility)
        if minimap_name == "lanes" and minimap_name not in minimap_to_rgb:
            continue

        # Special handling for lanelines with geometric patterns in V3
        if minimap_name == "lanelines" and elements and isinstance(elements[0], tuple):
            # Prepare laneline data with geometric patterns
            processed_lanelines = prepare_laneline_geometry_data(elements)

            # Render each laneline with its pattern
            for laneline_info in processed_lanelines:
                # Apply height filtering if enabled (check the original polyline)
                if camera_pose_init is not None:
                    if filter_by_height_relative_to_ego(
                        laneline_info["original_polyline"], camera_model, camera_pose, camera_pose_init
                    ):
                        continue  # Skip this laneline - it's on an underpass/overpass
                # Each laneline can have multiple segment lists (e.g., dual lines)
                for segments in laneline_info["pattern_segments_list"]:
                    if len(segments) == 0:
                        continue

                    # Project segments to camera space
                    xy_and_depth = camera_model.get_xy_and_depth(segments.reshape(-1, 3), camera_pose)
                    # Convert tensor to numpy if needed
                    if not isinstance(xy_and_depth, np.ndarray):
                        xy_and_depth = xy_and_depth.numpy()
                    xy_and_depth = xy_and_depth.reshape(-1, 2, 3)

                    # Filter valid line segments (both vertices in front of camera)
                    valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
                    valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
                    valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]

                    if len(valid_xy_and_depth) > 0:
                        all_geometry_objects.append(
                            LineSegment2D(
                                valid_xy_and_depth,
                                base_color=laneline_info["rgb_float"],
                                line_width=laneline_info["line_width"],
                            )
                        )
            continue  # Skip the normal processing for lanelines

        # Extract polylines from tuples if needed (for backward compatibility)
        if minimap_name == "lanelines" and elements and isinstance(elements[0], tuple):
            polylines = [polyline for polyline, _ in elements]
        else:
            polylines = elements

        # Skip if element type not defined
        try:
            minimap_type = get_type_from_name(minimap_name)
        except ValueError:
            # Skip unknown minimap types
            continue

        if minimap_type == "polyline":
            line_segment_list = []
            for polyline in polylines:
                # Apply height filtering if enabled
                if camera_pose_init is not None:
                    if filter_by_height_relative_to_ego(polyline, camera_model, camera_pose, camera_pose_init):
                        continue  # Skip this polyline - it's on an underpass/overpass

                # Subdivide the polyline for smooth rendering
                if minimap_name in ["lanelines", "road_boundaries"]:
                    polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.8)
                else:
                    polyline_subdivided = polyline

                if len(polyline_subdivided) < 2:
                    continue

                # Create line segments
                line_segment = np.stack([polyline_subdivided[:-1], polyline_subdivided[1:]], axis=1)
                line_segment_list.append(line_segment)

            if len(line_segment_list) == 0:
                continue

            all_line_segments = np.concatenate(line_segment_list, axis=0)
            xy_and_depth = camera_model.get_xy_and_depth(all_line_segments.reshape(-1, 3), camera_pose)
            # Convert tensor to numpy if needed
            if not isinstance(xy_and_depth, np.ndarray):
                xy_and_depth = xy_and_depth.numpy()
            xy_and_depth = xy_and_depth.reshape(-1, 2, 3)

            # Filter valid line segments
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]

            if len(valid_xy_and_depth) > 0:
                color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
                all_geometry_objects.append(
                    LineSegment2D(
                        valid_xy_and_depth,
                        base_color=color_float,
                        line_width=5 if minimap_name == "poles" else 12,
                    )
                )

        elif minimap_type == "polygon" or minimap_type == "cuboid3d":
            for polygon in polylines:
                if minimap_type == "cuboid3d":
                    # Convert cuboid to polyline for rendering
                    polygon_converted = cuboid3d_to_polyline(polygon)
                else:
                    polygon_converted = polygon

                # Apply height filtering if enabled
                if camera_pose_init is not None:
                    if filter_by_height_relative_to_ego(polygon_converted, camera_model, camera_pose, camera_pose_init):
                        continue  # Skip this polygon - it's on an underpass/overpass

                if minimap_name == "crosswalks":
                    # Subdivide and triangulate crosswalks
                    polygon_subdivided = interpolate_polyline_to_points(polygon_converted, segment_interval=0.8)
                    triangles_3d = triangulate_polygon_3d(polygon_subdivided)

                    if len(triangles_3d) == 0 or triangles_3d.size == 0:
                        continue

                    triangles_proj = camera_model.get_xy_and_depth(triangles_3d.reshape(-1, 3), camera_pose)
                    # Convert tensor to numpy if needed
                    if not isinstance(triangles_proj, np.ndarray):
                        triangles_proj = triangles_proj.numpy()
                    triangles_proj = triangles_proj.reshape(-1, 3, 3)
                    # Filter out triangles behind camera
                    invalid_triangles_indices = np.all(triangles_proj[:, :, 2] < 0, axis=1)
                    valid_triangles_indices = ~invalid_triangles_indices

                    if valid_triangles_indices.sum() > 0:
                        color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
                        all_geometry_objects.append(
                            TriangleList2D(
                                triangles_proj[valid_triangles_indices],
                                base_color=color_float,
                            )
                        )
                else:
                    # Regular polygon rendering
                    polygon_xy_and_depth = camera_model.get_xy_and_depth(polygon_converted, camera_pose)
                    # Convert tensor to numpy if needed
                    if not isinstance(polygon_xy_and_depth, np.ndarray):
                        polygon_xy_and_depth = polygon_xy_and_depth.numpy()

                    if not np.all(polygon_xy_and_depth[:, 2] < 0):
                        color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
                        all_geometry_objects.append(
                            Polygon2D(
                                polygon_xy_and_depth,
                                base_color=color_float,
                            )
                        )

    return all_geometry_objects


def create_bbox_geometry_objects_for_frame(
    current_object_info: dict,
    camera_pose: np.ndarray,
    camera_model: FThetaCamera,
    bbox_color_version: str = "v3",
    fill_face: str = "all",
    fill_face_style: str = "solid",
    line_width: int = 4,
    edge_color: list | None = None,
) -> list:
    """
    Build BoundingBox2D geometry objects for a single frame.

    Args:
        current_object_info: dict, object info for current frame
        camera_pose: np.ndarray (4,4), camera pose
        camera_model: FThetaCamera, camera model
        bbox_color_version: str, bbox color version
        fill_face: str, which faces to fill
        fill_face_style: str, style of face filling
        line_width: int, line width for rendering
        edge_color: list, optional edge color

    Returns:
        list[BoundingBox2D]: geometry objects for the current frame
    """
    # Build per-vertex color map
    gradient_class_colors = load_bbox_colors(bbox_color_version)
    object_type_to_per_vertex_color = {}

    for object_type, colors in gradient_class_colors.items():
        if isinstance(colors, list) and len(colors) == 2 and isinstance(colors[0], list):
            # Gradient color
            per_vertex_color = np.zeros((8, 3))
            per_vertex_color[[0, 1, 4, 5]] = np.array(colors[0]) / 255.0
            per_vertex_color[[2, 3, 6, 7]] = np.array(colors[1]) / 255.0
        else:
            # Uniform color
            per_vertex_color = np.tile(np.array(colors) / 255.0, (8, 1))
        object_type_to_per_vertex_color[object_type] = per_vertex_color

    edge_color_array = None
    if edge_color is not None:
        edge_color_array = np.array(edge_color) / 255.0

    # Store the 8 corner vertices of each object type
    object_type_to_corner_vertices = {"Car": [], "Truck": [], "Pedestrian": [], "Cyclist": [], "Others": []}

    tracking_ids = list(current_object_info.keys())
    tracking_ids.sort()

    for tracking_id in tracking_ids:
        object_info = current_object_info[tracking_id]
        object_info = simplify_type_in_object_info(object_info)

        object_to_world = np.array(object_info["object_to_world"])
        object_lwh = np.array(object_info["object_lwh"])
        cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)

        # Cull objects entirely behind camera
        if np.all(np.dot(cuboid_eight_vertices - camera_pose[:3, 3], camera_pose[:3, 2]) < 0):
            continue

        if object_info["object_type"] in ["Car", "Truck", "Pedestrian", "Cyclist"]:
            object_type_to_corner_vertices[object_info["object_type"]].append(cuboid_eight_vertices)
        else:
            object_type_to_corner_vertices["Others"].append(cuboid_eight_vertices)

    # Draw the bbox projection
    geometry_objects = []
    for object_type, all_corner_vertices in object_type_to_corner_vertices.items():
        if len(all_corner_vertices) == 0:
            continue

        n_objects = len(all_corner_vertices)
        all_corner_vertices_flatten = np.array(all_corner_vertices).reshape(-1, 3)
        all_points_in_cam = camera_model.transform_points(all_corner_vertices_flatten, np.linalg.inv(camera_pose))
        all_depth = all_points_in_cam[:, 2:3]
        all_xy = camera_model.ray2pixel(all_points_in_cam)
        all_xy_and_depth = np.hstack([all_xy, all_depth]).reshape(n_objects, 8, 3)

        # Valid corner: (1) 0 <= x <= width, (2) 0 <= y <= height, (3) depth > 0
        valid_x_mask = (all_xy_and_depth[:, :, 0] >= 0) & (all_xy_and_depth[:, :, 0] < camera_model.width)
        valid_y_mask = (all_xy_and_depth[:, :, 1] >= 0) & (all_xy_and_depth[:, :, 1] < camera_model.height)
        valid_depth_mask = all_xy_and_depth[:, :, 2] > 0
        not_valid_vertex_mask = ~valid_x_mask | ~valid_y_mask | ~valid_depth_mask
        not_valid_object_mask = np.all(not_valid_vertex_mask, axis=1)
        valid_object_mask = ~not_valid_object_mask

        all_xy_and_depth = all_xy_and_depth[valid_object_mask]

        for xy_and_depth in all_xy_and_depth:
            geometry_objects.append(
                BoundingBox2D(
                    xy_and_depth=xy_and_depth,
                    base_color_or_per_vertex_color=object_type_to_per_vertex_color[object_type],
                    fill_face=fill_face,
                    fill_face_style=fill_face_style,
                    line_width=line_width,
                    edge_color=edge_color_array,
                )
            )

    return geometry_objects


def render_hdmap_v3(
    camera_poses: np.ndarray,
    all_object_info: dict,
    map_data: dict,
    camera_model: FThetaCamera,
    render_frame_ids: list[int] | None = None,
    hdmap_color_version: str = "v3",
    bbox_color_version: str = "v3",
    traffic_light_color_version: str = "v2",
    enable_height_filter: bool = False,
    device_index: int = 0,
) -> np.ndarray:
    """
    Render HD map using V3 moderngl-based rendering with proper depth occlusion.

    Args:
        camera_poses: np.ndarray, shape (N, 4, 4), camera poses
        all_object_info: dict, containing all object info
        map_data: dict, map data
        camera_model: FThetaCamera, camera model
        render_frame_ids: list[int], frame ids to render
        hdmap_color_version: str, HD map color version
        bbox_color_version: str, bbox color version
        traffic_light_color_version: str, traffic light color version
        enable_height_filter: bool, whether to filter elements by height
        device_index: int, device index

    Returns:
        np.ndarray, shape (N, H, W, 3), rendered frames
    """
    if render_frame_ids is None:
        render_frame_ids = list(range(len(camera_poses)))

    combined_frames = []

    # Get initial camera pose for height filtering
    camera_pose_init = camera_poses[render_frame_ids[0]] if enable_height_filter else None

    # Prepare traffic light status data if enabled
    tl_position_list = None
    tl_status_dict = None
    tl_status_to_rgb = None
    if map_data.get("traffic_lights"):
        tl_position_list, tl_status_dict, tl_status_to_rgb = prepare_traffic_light_status_data_clipgt(
            map_data["traffic_lights"],
            traffic_light_color_version=traffic_light_color_version,
        )
        logger.info(f"Traffic light rendering enabled with {traffic_light_color_version} colors")

    logger.info(
        f"Rendering {len(render_frame_ids)} frames with V3 renderer "
        f"(hdmap: {hdmap_color_version}, bbox: {bbox_color_version})"
    )

    for frame_id in render_frame_ids:
        camera_pose = camera_poses[frame_id]

        # Build all geometry objects for this frame
        geometry_objects = []

        # Add minimap layers (excluding traffic lights if status rendering is enabled)
        map_data_filtered = map_data.copy()
        if tl_position_list is not None:
            # Remove traffic lights from regular minimap rendering
            map_data_filtered = {k: v for k, v in map_data.items() if k != "traffic_lights"}

        geometry_objects.extend(
            create_minimap_geometry_objects_from_data(
                map_data_filtered,
                camera_pose,
                camera_model,
                hdmap_color_version,
                camera_pose_init=camera_pose_init,
            )
        )

        # Add traffic lights with proper colors
        if tl_position_list is not None and tl_status_to_rgb is not None:
            tl_objects = create_traffic_light_status_geometry_objects_from_data(
                tl_position_list,
                tl_status_dict,
                frame_id,
                camera_pose,
                camera_model,
                tl_status_to_rgb,
            )
            geometry_objects.extend(tl_objects)

        # Add bounding boxes
        current_object_info = all_object_info[f"{frame_id:06d}.all_object_info.json"]
        geometry_objects.extend(
            create_bbox_geometry_objects_for_frame(
                current_object_info,
                camera_pose,
                camera_model,
                bbox_color_version,
                fill_face="all",
                fill_face_style="solid",
                line_width=4,
                edge_color=[200, 200, 200],
            )
        )

        # Render all geometries with proper depth ordering
        combined_frame = render_geometries(
            geometry_objects,
            camera_model.height,
            camera_model.width,
            depth_max=200,
            depth_gradient=True,
            device_index=device_index,
        )
        combined_frames.append(combined_frame)

    return np.stack(combined_frames, axis=0)


def render_hdmap(
    camera_poses: np.ndarray,
    all_object_info: dict,
    map_data: dict,
    camera_model: FThetaCamera,
    render_frame_ids: list[int] | None = None,
    render_version: Literal["v3"] = "v3",
    device_index: int = 0,
) -> np.ndarray:
    return render_hdmap_v3(
        camera_poses,
        all_object_info,
        map_data,
        camera_model,
        render_frame_ids,
        "v3",
        "v3",
        "v2",
        False,
        device_index,
    )
