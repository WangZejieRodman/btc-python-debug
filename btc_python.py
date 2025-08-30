#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import yaml
import os
import struct
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


@dataclass
class ConfigSetting:
    # Binary descriptor parameters
    useful_corner_num: int = 500
    plane_detection_thre: float = 0.01
    plane_merge_normal_thre: float = 0.1
    plane_merge_dis_thre: float = 0.3
    voxel_size: float = 1.0
    voxel_init_num: int = 10
    proj_plane_num: int = 1
    proj_image_resolution: float = 0.5
    proj_image_high_inc: float = 0.1
    proj_dis_min: float = -1.0
    proj_dis_max: float = 4.0
    summary_min_thre: int = 10
    line_filter_enable: int = 0

    # Triangle descriptor parameters
    descriptor_near_num: int = 10
    descriptor_min_len: float = 1.0
    descriptor_max_len: float = 10.0
    non_max_suppression_radius: float = 3.0
    std_side_resolution: float = 0.2

    # Candidate search parameters
    skip_near_num: int = 20
    candidate_num: int = 50
    rough_dis_threshold: float = 0.03
    similarity_threshold: float = 0.7
    icp_threshold: float = 0.5
    normal_threshold: float = 0.1
    dis_threshold: float = 0.3


@dataclass
class BinaryDescriptor:
    occupy_array: List[bool]
    summary: int
    location: np.ndarray


@dataclass
class BTC:
    triangle: np.ndarray
    angle: np.ndarray
    center: np.ndarray
    frame_number: int
    binary_A: BinaryDescriptor
    binary_B: BinaryDescriptor
    binary_C: BinaryDescriptor


@dataclass
class Plane:
    center: np.ndarray
    normal: np.ndarray
    covariance: np.ndarray
    radius: float = 0.0
    min_eigen_value: float = 1.0
    d: float = 0.0
    id: int = 0
    sub_plane_num: int = 0
    points_size: int = 0
    is_plane: bool = False


class BTCDescManager:
    def __init__(self, config_setting: ConfigSetting):
        self.config_setting = config_setting
        self.print_debug_info = False

        # Hash table for descriptors - using dict with tuple keys
        self.data_base = {}

        # Save all key clouds (optional)
        self.key_cloud_vec = []

        # Save all binary descriptors of key frame
        self.history_binary_list = []

        # Save all planes of key frame
        self.plane_cloud_vec = []

    def load_config_from_yaml(self, config_file: str) -> ConfigSetting:
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        config = ConfigSetting()

        # Binary descriptor
        config.useful_corner_num = config_data.get('useful_corner_num', 500)
        config.plane_detection_thre = config_data.get('plane_detection_thre', 0.01)
        config.plane_merge_normal_thre = config_data.get('plane_merge_normal_thre', 0.1)
        config.plane_merge_dis_thre = config_data.get('plane_merge_dis_thre', 0.3)
        config.voxel_size = config_data.get('voxel_size', 1.0)
        config.voxel_init_num = config_data.get('voxel_init_num', 10)
        config.proj_plane_num = config_data.get('proj_plane_num', 1)
        config.proj_image_resolution = config_data.get('proj_image_resolution', 0.5)
        config.proj_image_high_inc = config_data.get('proj_image_high_inc', 0.1)
        config.proj_dis_min = config_data.get('proj_dis_min', -1.0)
        config.proj_dis_max = config_data.get('proj_dis_max', 4.0)
        config.summary_min_thre = config_data.get('summary_min_thre', 10)
        config.line_filter_enable = config_data.get('line_filter_enable', 0)

        # Triangle descriptor
        config.descriptor_near_num = config_data.get('descriptor_near_num', 10)
        config.descriptor_min_len = config_data.get('descriptor_min_len', 1.0)
        config.descriptor_max_len = config_data.get('descriptor_max_len', 10.0)
        config.non_max_suppression_radius = config_data.get('max_constrait_dis', 2.0)
        config.std_side_resolution = config_data.get('triangle_resolution', 0.2)

        # Candidate search
        config.skip_near_num = config_data.get('skip_near_num', 20)
        config.candidate_num = config_data.get('candidate_num', 50)
        config.rough_dis_threshold = config_data.get('rough_dis_threshold', 0.03)
        config.similarity_threshold = config_data.get('similarity_threshold', 0.7)
        config.icp_threshold = config_data.get('icp_threshold', 0.5)
        config.normal_threshold = config_data.get('normal_threshold', 0.1)
        config.dis_threshold = config_data.get('dis_threshold', 0.3)

        return config

    def read_kitti_bin(self, bin_path: str) -> np.ndarray:
        """Read KITTI binary point cloud file"""
        with open(bin_path, 'rb') as f:
            data = f.read()

        points = []
        for i in range(0, len(data), 16):  # 4 floats * 4 bytes = 16 bytes
            if i + 16 <= len(data):
                x, y, z, intensity = struct.unpack('ffff', data[i:i + 16])
                points.append([x, y, z, intensity])

        return np.array(points)

    def load_pose_file(self, pose_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load pose file (EVO format: time x y z qx qy qz qw)"""
        translations = []
        rotations = []

        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    # Skip timestamp
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                    # Convert quaternion to rotation matrix
                    from scipy.spatial.transform import Rotation as R
                    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

                    translations.append(np.array([tx, ty, tz]))
                    rotations.append(rotation)

        return translations, rotations

    def down_sampling_voxel(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Voxel down sampling"""
        if voxel_size < 0.01 or len(points) == 0:
            return points

        # Create voxel map
        voxel_map = {}

        for point in points:
            # Calculate voxel coordinates (matching C++ logic exactly)
            voxel_coords = np.floor(point[:3] / voxel_size).astype(int)
            voxel_key = tuple(voxel_coords)

            if voxel_key not in voxel_map:
                voxel_map[voxel_key] = {'sum': np.zeros(4), 'count': 0}

            voxel_map[voxel_key]['sum'] += point[:4]
            voxel_map[voxel_key]['count'] += 1

        # Calculate average points for each voxel
        downsampled_points = []
        for voxel_data in voxel_map.values():
            avg_point = voxel_data['sum'] / voxel_data['count']
            downsampled_points.append(avg_point)

        return np.array(downsampled_points)

    def init_voxel_map(self, input_cloud: np.ndarray) -> Dict:
        """Initialize voxel map for plane detection"""
        voxel_map = {}

        for point in input_cloud:
            # Calculate voxel coordinates (exact C++ logic)
            loc_xyz = point[:3] / self.config_setting.voxel_size
            # C++ logic: if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
            for j in range(3):
                if loc_xyz[j] < 0:
                    loc_xyz[j] -= 1.0

            voxel_coords = np.floor(loc_xyz).astype(np.int64)
            voxel_key = tuple(voxel_coords)

            if voxel_key not in voxel_map:
                voxel_map[voxel_key] = {'points': [], 'plane': None}

            voxel_map[voxel_key]['points'].append(point[:3])

        # Initialize planes for each voxel with consistent ordering
        sorted_keys = sorted(voxel_map.keys())
        for voxel_key in sorted_keys:
            voxel_data = voxel_map[voxel_key]
            if len(voxel_data['points']) >= self.config_setting.voxel_init_num:
                plane = self.init_plane(np.array(voxel_data['points']))
                voxel_data['plane'] = plane

        return voxel_map

    def init_plane(self, voxel_points: np.ndarray) -> Optional[Plane]:
        """Initialize plane from voxel points"""
        if len(voxel_points) <= self.config_setting.voxel_init_num:
            return None

        # Calculate center
        center = np.mean(voxel_points, axis=0)

        # Calculate covariance matrix (exact C++ logic)
        covariance = np.zeros((3, 3))
        for point in voxel_points:
            covariance += np.outer(point, point)

        covariance = covariance / len(voxel_points) - np.outer(center, center)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Sort eigenvalues and eigenvectors (ascending order)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        plane = Plane(
            center=center,
            normal=eigenvectors[:, 0],  # Smallest eigenvalue -> normal
            covariance=covariance,
            min_eigen_value=eigenvalues[0],
            radius=np.sqrt(eigenvalues[2]),  # Largest eigenvalue -> radius
            points_size=len(voxel_points),
            sub_plane_num=0,  # 修复：原始平面sub_plane_num为0
            is_plane=eigenvalues[0] < self.config_setting.plane_detection_thre
        )

        if plane.is_plane:
            plane.d = -np.dot(plane.normal, plane.center)

        return plane

    def get_planes_from_voxel_map(self, voxel_map: Dict) -> List[np.ndarray]:
        """Extract planes from voxel map"""
        planes = []

        for voxel_data in voxel_map.values():
            plane = voxel_data['plane']
            if plane is not None and plane.is_plane:
                # Store as [x, y, z, nx, ny, nz] format
                plane_point = np.concatenate([plane.center, plane.normal])
                planes.append(plane_point)

        return planes

    def get_projection_planes(self, voxel_map: Dict) -> List[Plane]:
        """Get projection planes from voxel map"""
        origin_list = []

        for voxel_data in voxel_map.values():
            plane = voxel_data['plane']
            if plane is not None and plane.is_plane:
                origin_list.append(plane)

        if not origin_list:
            return []

        # Merge similar planes (exact C++ logic)
        merged_planes = self.merge_planes(origin_list)

        # Sort by points size (largest first)
        merged_planes.sort(key=lambda x: x.points_size, reverse=True)

        return merged_planes

    def merge_planes(self, origin_list: List[Plane]) -> List[Plane]:
        """Merge similar planes with exact C++ logic"""
        print(f"[DEBUG] merge_planes called with {len(origin_list)} planes")
        if len(origin_list) <= 1:
            return origin_list

        # Initialize plane IDs (exact C++ logic)
        for plane in origin_list:
            plane.id = 0

        current_id = 1

        # CRITICAL: Use exact C++ reverse iteration pattern
        # C++: for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--)
        for i in range(len(origin_list) - 1, 0, -1):  # 从 len-1 到 1
            for j in range(i):  # 从 0 到 i-1
                plane1, plane2 = origin_list[i], origin_list[j]

                # Exact C++ normal comparison logic
                normal_diff = np.linalg.norm(plane1.normal - plane2.normal)
                normal_add = np.linalg.norm(plane1.normal + plane2.normal)

                # Exact C++ distance calculation
                dis1 = abs(np.dot(plane1.normal, plane2.center) + plane1.d)
                dis2 = abs(np.dot(plane2.normal, plane1.center) + plane2.d)

                # Exact C++ merge conditions
                if ((normal_diff < self.config_setting.plane_merge_normal_thre or
                     normal_add < self.config_setting.plane_merge_normal_thre) and
                        dis1 < self.config_setting.plane_merge_dis_thre and
                        dis2 < self.config_setting.plane_merge_dis_thre):

                    # CRITICAL: Exact C++ ID assignment logic
                    if plane1.id == 0 and plane2.id == 0:
                        plane1.id = current_id
                        plane2.id = current_id
                        current_id += 1
                    elif plane1.id == 0 and plane2.id != 0:
                        plane1.id = plane2.id
                    elif plane1.id != 0 and plane2.id == 0:
                        plane2.id = plane1.id

        # Merge planes with same ID (exact C++ logic)
        merged_planes = []
        processed_ids = set()

        for plane in origin_list:
            if plane.id == 0:
                merged_planes.append(plane)
            elif plane.id not in processed_ids:
                processed_ids.add(plane.id)

                # Find all planes with same ID and merge them
                same_id_planes = [p for p in origin_list if p.id == plane.id]
                if len(same_id_planes) > 1:
                    merged_plane = self.merge_plane_group_exact_cpp(same_id_planes)
                    merged_planes.append(merged_plane)
                else:
                    merged_planes.append(plane)

        print(f"[DEBUG] merge_planes result: {len(merged_planes)} planes")
        return merged_planes

    def merge_plane_group_exact_cpp(self, planes: List[Plane]) -> Plane:
        """Merge plane group using exact C++ covariance reconstruction"""
        total_points = sum(p.points_size for p in planes)

        # Weighted center (exact C++ formula)
        weighted_center = sum(p.center * p.points_size for p in planes) / total_points

        # CRITICAL: Exact C++ covariance merging formula
        merged_covariance = np.zeros((3, 3))
        for p in planes:
            # Exact C++: P_PT = (covariance + center*center^T) * points_size
            P_PT = (p.covariance + np.outer(p.center, p.center)) * p.points_size
            merged_covariance += P_PT

        # Final covariance calculation
        merged_covariance = merged_covariance / total_points - np.outer(weighted_center, weighted_center)

        # CRITICAL: Exact C++ eigenvalue sorting
        eigenvalues, eigenvectors = np.linalg.eigh(merged_covariance)

        # C++ uses: evalsReal.rowwise().sum().minCoeff(&evalsMin)
        # This is equivalent to finding the minimum eigenvalue index
        evals_min_idx = np.argmin(eigenvalues)
        evals_max_idx = np.argmax(eigenvalues)

        # Correct sub_plane_num accumulation
        total_sub_plane_num = sum(p.sub_plane_num if p.sub_plane_num > 0 else 1 for p in planes)

        merged_plane = Plane(
            center=weighted_center,
            normal=eigenvectors[:, evals_min_idx],  # Use minimum eigenvalue eigenvector
            covariance=merged_covariance,
            radius=np.sqrt(eigenvalues[evals_max_idx]),
            points_size=total_points,
            sub_plane_num=total_sub_plane_num,
            is_plane=True,
            id=planes[0].id  # Keep the same ID
        )

        merged_plane.d = -np.dot(merged_plane.normal, merged_plane.center)

        return merged_plane

    def binary_similarity(self, b1: BinaryDescriptor, b2: BinaryDescriptor) -> float:
        """Calculate binary descriptor similarity"""
        common_bits = sum(1 for a, b in zip(b1.occupy_array, b2.occupy_array) if a and b)
        return 2 * common_bits / (b1.summary + b2.summary) if (b1.summary + b2.summary) > 0 else 0

    # 临时调试修改 - 在 extract_binary_descriptors 函数开头添加：
    def extract_binary_descriptors_debug_override(self, proj_planes: List[Plane], input_cloud: np.ndarray) -> List[
        BinaryDescriptor]:
        """临时调试版本：强制使用第一个合并平面"""
        print(f"[DEBUG] DEBUG OVERRIDE: extract_binary_descriptors with {len(proj_planes)} planes")

        if len(proj_planes) == 0:
            print(f"[DEBUG] No projection planes available!")
            return []

        # 强制使用第一个合并平面
        first_plane = proj_planes[0]
        print(f"[DEBUG] FORCED: Using first plane: normal={first_plane.normal}, center={first_plane.center}")
        print(f"[DEBUG] FORCED: First plane points_size={first_plane.points_size}")

        binary_list = self.extract_binary_from_plane(first_plane.center, first_plane.normal, input_cloud)

        print(f"[DEBUG] FORCED: First plane produced {len(binary_list)} binary descriptors")

        # Apply non-maximum suppression
        binary_list = self.non_max_suppression(binary_list)

        # Keep only the best corners
        if len(binary_list) > self.config_setting.useful_corner_num:
            binary_list.sort(key=lambda x: x.summary, reverse=True)
            binary_list = binary_list[:self.config_setting.useful_corner_num]

        print(f"[DEBUG] FORCED: Final result: {len(binary_list)} binary descriptors")
        return binary_list

    def extract_binary_descriptors(self, proj_planes: List[Plane], input_cloud: np.ndarray) -> List[BinaryDescriptor]:
        """
        Extract binary descriptors with EXACT C++ logic and detailed debugging
        """
        print(f"[DEBUG] extract_binary_descriptors called with {len(proj_planes)} planes")
        print(f"[DEBUG] Input cloud size: {len(input_cloud)}")

        binary_list = []
        last_normal = np.zeros(3)  # C++ uses this to track previous normal
        useful_proj_num = 0

        # DEBUG: Show all available planes
        for i, plane in enumerate(proj_planes):
            print(f"[DEBUG] Plane {i}: center={plane.center}, normal={plane.normal}, points={plane.points_size}")

        # C++ logic: iterate through sorted projection planes
        for i, plane in enumerate(proj_planes):
            proj_center = plane.center
            proj_normal = plane.normal

            # C++ logic: ensure normal points upward (positive Z)
            if proj_normal[2] < 0:
                proj_normal = -proj_normal

            print(f"[DEBUG] Processing plane {i}: normal={proj_normal}")

            # C++ key filtering logic: check normal difference with last used normal
            normal_diff = np.linalg.norm(proj_normal - last_normal)
            normal_add = np.linalg.norm(proj_normal + last_normal)

            print(f"[DEBUG] last_normal={last_normal}")
            print(f"[DEBUG] normal_diff={normal_diff:.6f}, normal_add={normal_add:.6f}")

            # C++ condition for accepting this projection plane
            # NOTE: This condition might be wrong - let's test different interpretations
            condition1 = normal_diff < 0.3 or normal_add > 0.3
            condition2 = normal_diff >= 0.3 and normal_add <= 0.3  # Alternative interpretation

            print(f"[DEBUG] condition1 (current): {condition1}")
            print(f"[DEBUG] condition2 (alternative): {condition2}")

            if condition1:  # Using current logic
                last_normal = proj_normal.copy()

                if self.print_debug_info:
                    print(f"[Description] reference plane normal: {proj_normal}, center: {proj_center}")

                useful_proj_num += 1
                temp_binary_list = self.extract_binary_from_plane(proj_center, proj_normal, input_cloud)

                print(f"[DEBUG] Plane {i} produced {len(temp_binary_list)} binary descriptors")
                binary_list.extend(temp_binary_list)

                # C++ logic: stop when we have enough projection planes
                if useful_proj_num >= self.config_setting.proj_plane_num:
                    print(f"[DEBUG] Reached proj_plane_num limit: {self.config_setting.proj_plane_num}")
                    break
            else:
                print(f"[DEBUG] Plane {i} skipped due to normal filtering")

        print(f"[DEBUG] Total useful_proj_num: {useful_proj_num}")
        print(f"[DEBUG] Total binary descriptors before NMS: {len(binary_list)}")

        # Non-maximum suppression
        binary_list = self.non_max_suppression(binary_list)

        # Keep only the best corners (sorted by summary)
        if len(binary_list) > self.config_setting.useful_corner_num:
            binary_list.sort(key=lambda x: x.summary, reverse=True)
            binary_list = binary_list[:self.config_setting.useful_corner_num]

        print(f"[DEBUG] Final binary descriptors: {len(binary_list)}")
        return binary_list

    def extract_binary_from_plane(self, proj_center: np.ndarray, proj_normal: np.ndarray,
                                  input_cloud: np.ndarray) -> List[BinaryDescriptor]:
        """Extract binary descriptors from a single projection plane with detailed debugging"""

        print(f"[DEBUG] extract_binary_from_plane called")
        print(f"[DEBUG] proj_center: {proj_center}")
        print(f"[DEBUG] proj_normal: {proj_normal}")
        print(f"[DEBUG] input_cloud size: {len(input_cloud)}")

        resolution = self.config_setting.proj_image_resolution
        dis_threshold_min = self.config_setting.proj_dis_min
        dis_threshold_max = self.config_setting.proj_dis_max
        high_inc = self.config_setting.proj_image_high_inc
        summary_min_thre = self.config_setting.summary_min_thre
        line_filter_enable = self.config_setting.line_filter_enable

        print(f"[DEBUG] Parameters: resolution={resolution}, dis_range=[{dis_threshold_min}, {dis_threshold_max}]")
        print(f"[DEBUG] summary_min_thre={summary_min_thre}, line_filter={line_filter_enable}")

        A, B, C = proj_normal
        D = -np.dot(proj_normal, proj_center)

        # FIXED: Create coordinate system exactly matching C++ logic
        x_axis = np.array([1.0, 0.0, 0.0])
        if C != 0:
            x_axis[2] = -(A + B) / C
        elif B != 0:
            x_axis[1] = -A / B
        else:
            # FIXED: Match C++ logic exactly
            x_axis[0] = 0.0
            x_axis[1] = 1.0
            # x_axis[2] remains 0.0

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(proj_normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # FIXED: Calculate coefficients exactly like C++
        ax, bx, cx = x_axis
        dx = -(ax * proj_center[0] + bx * proj_center[1] + cx * proj_center[2])
        ay, by, cy = y_axis
        dy = -(ay * proj_center[0] + by * proj_center[1] + cy * proj_center[2])

        # DEBUG: 投影坐标系
        if self.print_debug_info:
            print(f"[DEBUG] Projection coordinate system:")
            print(f"  proj_normal: {proj_normal}")
            print(f"  proj_center: {proj_center}")
            print(f"  x_axis: {x_axis}")
            print(f"  y_axis: {y_axis}")
            print(f"  A,B,C,D: {A}, {B}, {C}, {D}")

        # Project points
        point_list_2d = []
        dis_list = []

        debug_count = 0
        points_in_range = 0

        for point in input_cloud:
            x, y, z = point[:3]
            dis = x * A + y * B + z * C + D

            if dis_threshold_min < dis <= dis_threshold_max:
                points_in_range += 1

                # FIXED: Project point onto plane using exact C++ method
                cur_project = np.zeros(3)
                cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) / (A * A + B * B + C * C)
                cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) / (A * A + B * B + C * C)
                cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) / (A * A + B * B + C * C)

                # FIXED: Calculate 2D coordinates exactly matching C++ logic
                project_x = cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy
                project_y = cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx

                point_list_2d.append([project_x, project_y])
                dis_list.append(dis)

                # DEBUG: 前10个投影点的详细信息
                if debug_count < 10 and self.print_debug_info:
                    print(f"[DEBUG] Point {debug_count}: 3D({x:.3f}, {y:.3f}, {z:.3f}) -> "
                          f"dis={dis:.3f} -> 2D({project_x:.3f}, {project_y:.3f})")
                debug_count += 1

        print(f"[DEBUG] Points in distance range: {points_in_range}/{len(input_cloud)}")

        if len(point_list_2d) <= 5:
            print(f"[DEBUG] Too few points for projection: {len(point_list_2d)}")
            return []

        point_list_2d = np.array(point_list_2d)

        # Calculate image bounds
        min_x, max_x = np.min(point_list_2d[:, 0]), np.max(point_list_2d[:, 0])
        min_y, max_y = np.min(point_list_2d[:, 1]), np.max(point_list_2d[:, 1])

        # Create image grid (exact C++ logic)
        segmen_base_num = 5
        segmen_len = segmen_base_num * resolution
        x_segment_num = int((max_x - min_x) / segmen_len) + 1
        y_segment_num = int((max_y - min_y) / segmen_len) + 1
        x_axis_len = int((max_x - min_x) / resolution) + segmen_base_num
        y_axis_len = int((max_y - min_y) / resolution) + segmen_base_num

        # DEBUG: 图像网格参数
        if self.print_debug_info:
            print(f"[DEBUG] Image grid parameters:")
            print(f"  2D bounds: x[{min_x:.3f}, {max_x:.3f}], y[{min_y:.3f}, {max_y:.3f}]")
            print(f"  resolution: {resolution}, segmen_base_num: {segmen_base_num}")
            print(f"  grid size: x_axis_len={x_axis_len}, y_axis_len={y_axis_len}")
            print(f"  segments: x_segment_num={x_segment_num}, y_segment_num={y_segment_num}")
            print(f"  projected points count: {len(point_list_2d)}")

        # Initialize grids
        img_count = np.zeros((x_axis_len, y_axis_len))
        dis_array = np.zeros((x_axis_len, y_axis_len))
        mean_x_list = np.zeros((x_axis_len, y_axis_len))
        mean_y_list = np.zeros((x_axis_len, y_axis_len))
        dis_containers = [[[] for _ in range(y_axis_len)] for _ in range(x_axis_len)]

        # Fill grids
        valid_grid_count = 0
        for i, (px, py) in enumerate(point_list_2d):
            x_index = int((px - min_x) / resolution)
            y_index = int((py - min_y) / resolution)

            if 0 <= x_index < x_axis_len and 0 <= y_index < y_axis_len:
                mean_x_list[x_index, y_index] += px
                mean_y_list[x_index, y_index] += py
                img_count[x_index, y_index] += 1
                dis_containers[x_index][y_index].append(dis_list[i])
                valid_grid_count += 1

        print(f"[DEBUG] Valid grid assignments: {valid_grid_count}/{len(point_list_2d)}")
        non_empty_grids = np.sum(img_count > 0)
        print(f"[DEBUG] Non-empty grids: {non_empty_grids}")

        # Calculate occupancy arrays
        cut_num = int((dis_threshold_max - dis_threshold_min) / high_inc)
        binary_containers = {}

        grids_with_data = 0
        for x in range(x_axis_len):
            for y in range(y_axis_len):
                if img_count[x, y] > 0:
                    grids_with_data += 1
                    occupy_list = [False] * cut_num
                    cnt_list = [0] * cut_num

                    for dis in dis_containers[x][y]:
                        cnt_index = int((dis - dis_threshold_min) / high_inc)
                        if 0 <= cnt_index < cut_num:
                            cnt_list[cnt_index] += 1

                    segment_dis = 0
                    for i in range(cut_num):
                        if cnt_list[i] >= 1:
                            segment_dis += 1
                            occupy_list[i] = True

                    dis_array[x, y] = segment_dis
                    binary_containers[(x, y)] = BinaryDescriptor(
                        occupy_array=occupy_list,
                        summary=int(segment_dis),
                        location=np.zeros(3)  # Will be set later
                    )

        print(f"[DEBUG] Grids with occupancy data: {grids_with_data}")

        # Extract maximum points in each segment
        binary_list = []

        segment_candidates = 0
        segment_accepted = 0

        # FIXED: Use exact C++ segment selection logic
        for x_seg in range(x_segment_num):
            for y_seg in range(y_segment_num):
                max_dis = 0
                max_x_idx = max_y_idx = -1

                # Find maximum in this segment
                for x_idx in range(x_seg * segmen_base_num, min((x_seg + 1) * segmen_base_num, x_axis_len)):
                    for y_idx in range(y_seg * segmen_base_num, min((y_seg + 1) * segmen_base_num, y_axis_len)):
                        if dis_array[x_idx, y_idx] > max_dis:
                            max_dis = dis_array[x_idx, y_idx]
                            max_x_idx, max_y_idx = x_idx, y_idx

                if max_dis >= summary_min_thre and max_x_idx >= 0:
                    segment_candidates += 1

                    # FIXED: Check if it's a line (exact C++ logic)
                    is_add = True
                    if line_filter_enable and self._is_line_point_fixed(dis_array, max_x_idx, max_y_idx, max_dis,
                                                                        x_axis_len, y_axis_len):
                        is_add = False

                    if is_add:
                        segment_accepted += 1
                        px = mean_x_list[max_x_idx, max_y_idx] / img_count[max_x_idx, max_y_idx]
                        py = mean_y_list[max_x_idx, max_y_idx] / img_count[max_x_idx, max_y_idx]

                        # FIXED: Convert back to 3D coordinates using exact C++ method
                        coord = py * x_axis + px * y_axis + proj_center

                        binary_desc = binary_containers[(max_x_idx, max_y_idx)]
                        binary_desc.location = coord
                        binary_list.append(binary_desc)

        # DEBUG: 段落选择统计
        if self.print_debug_info:
            print(f"[DEBUG] Segment selection:")
            print(f"  summary_min_thre: {summary_min_thre}")
            print(f"  segment_candidates: {segment_candidates}")
            print(f"  segment_accepted: {segment_accepted}")
            print(f"  line_filter_enable: {line_filter_enable}")

        print(f"[DEBUG] extract_binary_from_plane returning {len(binary_list)} descriptors")
        return binary_list

    def _is_line_point_fixed(self, dis_array: np.ndarray, x: int, y: int, max_dis: float, x_axis_len: int,
                             y_axis_len: int) -> bool:
        """Check if a point is part of a line structure using exact C++ logic"""
        if (x <= 0 or x >= x_axis_len - 1 or y <= 0 or y >= y_axis_len - 1):
            return False

        # FIXED: Use exact C++ direction logic
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        threshold = max_dis - 3

        for dx, dy in directions:
            p1_x, p1_y = x + dx, y + dy
            p2_x, p2_y = x - dx, y - dy

            # Ensure indices are within bounds
            if (0 <= p1_x < x_axis_len and 0 <= p1_y < y_axis_len and
                    0 <= p2_x < x_axis_len and 0 <= p2_y < y_axis_len):

                p1_val = dis_array[p1_x, p1_y]
                p2_val = dis_array[p2_x, p2_y]

                # FIXED: Use exact C++ filtering conditions
                if p1_val >= threshold and p2_val >= 0.5 * max_dis:
                    return True
                if p2_val >= threshold and p1_val >= 0.5 * max_dis:
                    return True
                if p1_val >= threshold and p2_val >= threshold:
                    return True

        return False

    def non_max_suppression(self, binary_list: List[BinaryDescriptor]) -> List[BinaryDescriptor]:
        """Apply non-maximum suppression using exact C++ logic"""
        if not binary_list:
            return binary_list

        # Build point cloud and preprocessing (matching C++ exactly)
        locations = np.array([bd.location for bd in binary_list])
        tree = KDTree(locations)
        radius = self.config_setting.non_max_suppression_radius

        # Pre-compute summary list (matching C++ pre_count_list)
        pre_count_list = [bd.summary for bd in binary_list]
        is_add_list = [True] * len(binary_list)

        # FIXED: Use exact C++ suppression logic
        for i in range(len(binary_list)):
            search_point = binary_list[i].location

            # Find all neighbors within radius
            indices = tree.query_ball_point(search_point, radius)

            if len(indices) > 0:
                # FIXED: Check ALL neighbors, don't skip based on current status
                for j in indices:
                    if j == i:
                        continue  # Skip self

                    # FIXED: Use exact C++ comparison logic
                    # If current point summary <= neighbor summary, mark current for removal
                    if pre_count_list[i] <= pre_count_list[j]:
                        is_add_list[i] = False
                        # FIXED: Don't break, continue checking all neighbors
                        # This matches C++ behavior exactly

        # Collect surviving points
        result_list = []
        for i, is_add in enumerate(is_add_list):
            if is_add:
                result_list.append(binary_list[i])

        return result_list

    def generate_btc_descriptors(self, binary_list: List[BinaryDescriptor], frame_id: int) -> List[BTC]:
        """Generate BTC descriptors from binary descriptors"""
        if not binary_list:
            return []

        locations = np.array([bd.location for bd in binary_list])
        tree = KDTree(locations)

        btc_list = []
        scale = 1.0 / self.config_setting.std_side_resolution
        K = self.config_setting.descriptor_near_num

        feat_map = set()  # To avoid duplicate triangles

        for i, bd_i in enumerate(binary_list):
            distances, indices = tree.query(bd_i.location, k=min(K, len(binary_list)))

            if len(indices) < 3:
                continue

            for m in range(1, len(indices) - 1):
                for n in range(m + 1, len(indices)):
                    p1 = bd_i.location
                    p2 = binary_list[indices[m]].location
                    p3 = binary_list[indices[n]].location

                    # Calculate side lengths
                    a = np.linalg.norm(p1 - p2)
                    b = np.linalg.norm(p1 - p3)
                    c = np.linalg.norm(p2 - p3)

                    # Check length constraints
                    if (a > self.config_setting.descriptor_max_len or
                            b > self.config_setting.descriptor_max_len or
                            c > self.config_setting.descriptor_max_len or
                            a < self.config_setting.descriptor_min_len or
                            b < self.config_setting.descriptor_min_len or
                            c < self.config_setting.descriptor_min_len):
                        continue

                    # Sort side lengths (a <= b <= c)
                    sides = [a, b, c]
                    sides.sort()
                    a, b, c = sides

                    # Check if triangle is too flat
                    if abs(c - (a + b)) < 0.2:
                        continue

                    # Create unique key for this triangle
                    triangle_key = (int(a * 1000), int(b * 1000), int(c * 1000))
                    if triangle_key in feat_map:
                        continue
                    feat_map.add(triangle_key)

                    # Determine which points correspond to which sides
                    points = [p1, p2, p3]
                    binaries = [bd_i, binary_list[indices[m]], binary_list[indices[n]]]

                    # Assign binary descriptors to triangle vertices
                    binary_A = binaries[0]
                    binary_B = binaries[1]
                    binary_C = binaries[2]

                    btc = BTC(
                        triangle=np.array([scale * a, scale * b, scale * c]),
                        angle=np.zeros(3),  # Angles not used in this implementation
                        center=(p1 + p2 + p3) / 3,
                        frame_number=frame_id,
                        binary_A=binary_A,
                        binary_B=binary_B,
                        binary_C=binary_C
                    )

                    btc_list.append(btc)

        return btc_list

    def generate_btc_descs(self, input_cloud: np.ndarray, frame_id: int) -> List[BTC]:
        """Main function to generate BTC descriptors from point cloud"""
        # Step 1: Voxelization and plane detection
        voxel_map = self.init_voxel_map(input_cloud)

        # Step 2: Get original planes for storage (matching C++ behavior)
        original_plane_points = self.get_planes_from_voxel_map(voxel_map)
        self.plane_cloud_vec.append(np.array(original_plane_points))

        # Step 3: Get projection planes and merge them
        proj_planes = self.get_projection_planes(voxel_map)

        if self.print_debug_info:
            print(f"[DEBUG] Original planes count: {len(original_plane_points)}")
            print(f"[DEBUG] Merged planes count: {len(proj_planes)}")
            print(f"[Description] planes size: {len(original_plane_points)}")  # 保持与C++一致的输出

        # Handle case when no valid planes found
        if not proj_planes:
            # Create default projection plane if no planes found
            default_plane = Plane(
                center=np.array([input_cloud[0, 0], input_cloud[0, 1], input_cloud[0, 2]]),
                normal=np.array([0, 0, 1]),
                covariance=np.eye(3),
                is_plane=True,
                points_size=1,
                sub_plane_num=0
            )
            proj_planes = [default_plane]
        else:
            # C++ logic: sort by points_size in descending order
            proj_planes.sort(key=lambda x: x.points_size, reverse=True)

        # Step 4: Extract binary descriptors using merged planes
        binary_list = self.extract_binary_descriptors(proj_planes, input_cloud)
        # binary_list = self.extract_binary_descriptors_debug_override(proj_planes, input_cloud)
        self.history_binary_list.append(binary_list)

        if self.print_debug_info:
            print(f"[Description] binary size: {len(binary_list)}")

        # Step 5: Generate BTC descriptors
        btc_list = self.generate_btc_descriptors(binary_list, frame_id)

        if self.print_debug_info:
            print(f"[Description] btcs size: {len(btc_list)}")

        return btc_list

    def add_btc_descs(self, btc_list: List[BTC]):
        """Add BTC descriptors to database"""
        for btc in btc_list:
            # Calculate hash key from triangle
            position_key = (
                int(btc.triangle[0] + 0.5),
                int(btc.triangle[1] + 0.5),
                int(btc.triangle[2] + 0.5)
            )

            if position_key not in self.data_base:
                self.data_base[position_key] = []

            self.data_base[position_key].append(btc)

    def candidate_selector(self, current_btc_list: List[BTC]) -> List[Dict]:
        """Select candidate frames for loop closure"""
        if not current_btc_list:
            return []

        current_frame_id = current_btc_list[0].frame_number
        match_array = np.zeros(20000)  # Assume max 20000 frames
        match_results = []

        # Search around each voxel
        voxel_rounds = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    voxel_rounds.append((x, y, z))

        for btc in current_btc_list:
            dis_threshold = np.linalg.norm(btc.triangle) * self.config_setting.rough_dis_threshold

            for vx, vy, vz in voxel_rounds:
                search_key = (
                    int(btc.triangle[0] + vx),
                    int(btc.triangle[1] + vy),
                    int(btc.triangle[2] + vz)
                )

                # Check if search key is close to btc triangle
                voxel_center = np.array([search_key[0] + 0.5, search_key[1] + 0.5, search_key[2] + 0.5])
                if np.linalg.norm(btc.triangle - voxel_center) < 1.5:

                    if search_key in self.data_base:
                        for candidate_btc in self.data_base[search_key]:
                            if (current_frame_id - candidate_btc.frame_number) > self.config_setting.skip_near_num:
                                triangle_dis = np.linalg.norm(btc.triangle - candidate_btc.triangle)

                                if triangle_dis < dis_threshold:
                                    # Calculate binary similarity
                                    similarity = (
                                                         self.binary_similarity(btc.binary_A, candidate_btc.binary_A) +
                                                         self.binary_similarity(btc.binary_B, candidate_btc.binary_B) +
                                                         self.binary_similarity(btc.binary_C, candidate_btc.binary_C)
                                                 ) / 3

                                    if similarity > self.config_setting.similarity_threshold:
                                        match_array[candidate_btc.frame_number] += 1
                                        match_results.append({
                                            'current_btc': btc,
                                            'candidate_btc': candidate_btc,
                                            'frame_id': candidate_btc.frame_number
                                        })

        # Select best candidates
        candidate_matches = []

        for _ in range(self.config_setting.candidate_num):
            max_vote_idx = np.argmax(match_array)
            max_vote = match_array[max_vote_idx]

            if max_vote >= 5:  # Minimum vote threshold
                match_array[max_vote_idx] = 0  # Reset to avoid selecting again

                # Collect all matches for this frame
                frame_matches = [m for m in match_results if m['frame_id'] == max_vote_idx]

                candidate_match = {
                    'match_frame': max_vote_idx,
                    'current_frame': current_frame_id,
                    'matches': frame_matches,
                    'vote_count': len(frame_matches)
                }

                candidate_matches.append(candidate_match)
            else:
                break

        return candidate_matches

    def candidate_verify(self, candidate_match: Dict) -> Tuple[float, np.ndarray, np.ndarray, List]:
        """Verify candidate using geometric consistency"""
        matches = candidate_match['matches']
        if not matches:
            return -1, np.eye(3), np.zeros(3), []

        # Try different match pairs to find best transformation
        skip_len = max(1, len(matches) // 50)
        use_size = len(matches) // skip_len

        vote_results = []

        for i in range(0, min(use_size * skip_len, len(matches)), skip_len):
            single_match = matches[i]

            # Solve transformation from this match
            try:
                R, t = self.triangle_solver(single_match['current_btc'], single_match['candidate_btc'])

                # Count votes for this transformation
                vote_count = 0
                valid_matches = []

                for match in matches:
                    curr_btc = match['current_btc']
                    cand_btc = match['candidate_btc']

                    # Transform current triangle vertices
                    A_transform = R @ curr_btc.binary_A.location + t
                    B_transform = R @ curr_btc.binary_B.location + t
                    C_transform = R @ curr_btc.binary_C.location + t

                    # Check distances to candidate vertices
                    dis_A = np.linalg.norm(A_transform - cand_btc.binary_A.location)
                    dis_B = np.linalg.norm(B_transform - cand_btc.binary_B.location)
                    dis_C = np.linalg.norm(C_transform - cand_btc.binary_C.location)

                    if dis_A < 3 and dis_B < 3 and dis_C < 3:  # Distance threshold
                        vote_count += 1
                        valid_matches.append(match)

                vote_results.append({
                    'votes': vote_count,
                    'R': R,
                    't': t,
                    'matches': valid_matches
                })

            except:
                continue

        if not vote_results:
            return -1, np.eye(3), np.zeros(3), []

        # Find best result
        best_result = max(vote_results, key=lambda x: x['votes'])

        if best_result['votes'] >= 4:  # Minimum vote threshold
            # Verify using plane geometric consistency
            verify_score = self.plane_geometric_verify(
                self.plane_cloud_vec[-1],  # Current frame planes
                self.plane_cloud_vec[candidate_match['match_frame']],  # Candidate frame planes
                best_result['R'], best_result['t']
            )

            return verify_score, best_result['R'], best_result['t'], best_result['matches']
        else:
            return -1, np.eye(3), np.zeros(3), []

    def triangle_solver(self, btc1: BTC, btc2: BTC) -> Tuple[np.ndarray, np.ndarray]:
        """Solve transformation between two triangles"""
        # Create source and reference matrices
        src = np.column_stack([
            btc1.binary_A.location - btc1.center,
            btc1.binary_B.location - btc1.center,
            btc1.binary_C.location - btc1.center
        ])

        ref = np.column_stack([
            btc2.binary_A.location - btc2.center,
            btc2.binary_B.location - btc2.center,
            btc2.binary_C.location - btc2.center
        ])

        # Compute covariance matrix
        H = src @ ref.T

        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = btc2.center - R @ btc1.center

        return R, t

    def plane_geometric_verify(self, source_planes: np.ndarray, target_planes: np.ndarray,
                               R: np.ndarray, t: np.ndarray) -> float:
        """Verify transformation using plane-to-plane geometric consistency"""
        if len(source_planes) == 0 or len(target_planes) == 0:
            return 0.0

        target_points = target_planes[:, :3]
        target_normals = target_planes[:, 3:]

        tree = KDTree(target_points)

        useful_match = 0
        normal_threshold = self.config_setting.normal_threshold
        dis_threshold = self.config_setting.dis_threshold

        for source_plane in source_planes:
            source_point = source_plane[:3]
            source_normal = source_plane[3:]

            # Transform source plane
            transformed_point = R @ source_point + t
            transformed_normal = R @ source_normal

            # Find nearest target plane
            distance, idx = tree.query(transformed_point, k=1)

            if idx < len(target_planes):
                target_point = target_points[idx]
                target_normal = target_normals[idx]

                # Check normal consistency
                normal_diff = np.linalg.norm(transformed_normal - target_normal)
                normal_add = np.linalg.norm(transformed_normal + target_normal)

                # Check point-to-plane distance
                point_to_plane = abs(np.dot(target_normal, transformed_point - target_point))

                if ((normal_diff < normal_threshold or normal_add < normal_threshold) and
                        point_to_plane < dis_threshold):
                    useful_match += 1

        return useful_match / len(source_planes)

    def search_loop(self, btc_list: List[BTC]) -> Tuple[int, float, np.ndarray, np.ndarray, List]:
        """Search for loop closure"""
        if not btc_list:
            print("No BTCs!")
            return -1, 0, np.eye(3), np.zeros(3), []

        # Step 1: Select candidates
        candidate_matches = self.candidate_selector(btc_list)

        # Step 2: Verify candidates
        best_score = 0
        best_candidate_id = -1
        best_R = np.eye(3)
        best_t = np.zeros(3)
        best_matches = []

        for candidate_match in candidate_matches:
            verify_score, R, t, valid_matches = self.candidate_verify(candidate_match)

            if self.print_debug_info:
                print(f"[Retrieval] try frame: {candidate_match['match_frame']}, "
                      f"rough size: {len(candidate_match['matches'])}, score: {verify_score}")

            if verify_score > best_score:
                best_score = verify_score
                best_candidate_id = candidate_match['match_frame']
                best_R = R
                best_t = t
                best_matches = valid_matches

        if self.print_debug_info:
            print(f"[Retrieval] best candidate: {best_candidate_id}, score: {best_score}")

        if best_score > self.config_setting.icp_threshold:
            return best_candidate_id, best_score, best_R, best_t, best_matches
        else:
            return -1, 0, np.eye(3), np.zeros(3), []

    def calc_overlap(self, cloud1: np.ndarray, cloud2: np.ndarray,
                     dis_threshold: float = 0.5, skip_num: int = 2) -> float:
        """Calculate overlap between two point clouds"""
        if len(cloud1) == 0 or len(cloud2) == 0:
            return 0.0

        tree = KDTree(cloud2[:, :3])
        match_num = 0

        for i in range(0, len(cloud1), skip_num):
            distance, _ = tree.query(cloud1[i, :3], k=1)
            if distance < dis_threshold:
                match_num += 1

        overlap = (2 * match_num * skip_num) / (len(cloud1) + len(cloud2))
        return overlap


def main():
    """Main function for place recognition example"""
    # Configuration
    config_file = "config_outdoor.yaml"  # or config_indoor.yaml
    pcds_dir = "/path/to/kitti/sequences/00/velodyne"
    pose_file = "/path/to/kitti/pose/file.txt"
    cloud_overlap_thr = 0.5
    read_bin = True

    # Create config (you can also load from YAML file)
    config = ConfigSetting()

    # Initialize BTC manager
    btc_manager = BTCDescManager(config)
    btc_manager.print_debug_info = True

    # Load poses for visualization and overlap calculation
    if os.path.exists(pose_file):
        translations, rotations = btc_manager.load_pose_file(pose_file)
        print(f"Successfully loaded pose file: {pose_file}. Pose size: {len(translations)}")
    else:
        print(f"Pose file not found: {pose_file}")
        translations, rotations = [], []

    # Processing statistics
    descriptor_times = []
    query_times = []
    update_times = []
    trigger_loop_num = 0
    true_loop_num = 0

    # Process each frame
    for submap_id in range(len(translations) if translations else 100):  # Adjust range as needed
        print(f"\n[Description] submap id: {submap_id}")

        # Load point cloud
        if read_bin:
            bin_file = os.path.join(pcds_dir, f"{submap_id:06d}.bin")
            if not os.path.exists(bin_file):
                print(f"File not found: {bin_file}")
                break

            points = btc_manager.read_kitti_bin(bin_file)
        else:
            pcd_file = os.path.join(pcds_dir, f"{submap_id:06d}.pcd")
            if not os.path.exists(pcd_file):
                print(f"File not found: {pcd_file}")
                break

            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            # Add intensity column (zeros if not available)
            if points.shape[1] == 3:
                points = np.column_stack([points, np.zeros(len(points))])

        # Apply pose transformation if available
        if submap_id < len(translations):
            translation = translations[submap_id]
            rotation = rotations[submap_id]

            transformed_points = points.copy()
            transformed_points[:, :3] = (rotation @ points[:, :3].T).T + translation
        else:
            transformed_points = points

        # Step 1: Descriptor Extraction
        start_time = time.time()
        btc_list = btc_manager.generate_btc_descs(transformed_points, submap_id)
        descriptor_time = (time.time() - start_time) * 1000
        descriptor_times.append(descriptor_time)

        # Step 2: Search Loop
        start_time = time.time()
        if submap_id > config.skip_near_num:
            loop_id, loop_score, R, t, matches = btc_manager.search_loop(btc_list)

            if loop_id >= 0:
                print(f"[Loop Detection] trigger loop: {submap_id} -- {loop_id}, score: {loop_score}")
                trigger_loop_num += 1

                # Calculate ground truth overlap if poses available
                if submap_id < len(translations):
                    current_cloud = btc_manager.down_sampling_voxel(transformed_points, 0.5)
                    matched_cloud = btc_manager.key_cloud_vec[loop_id]
                    overlap = btc_manager.calc_overlap(current_cloud, matched_cloud, cloud_overlap_thr)

                    if overlap >= cloud_overlap_thr:
                        true_loop_num += 1
                        print(f"[Loop Detection] True positive! Overlap: {overlap:.3f}")
                    else:
                        print(f"[Loop Detection] False positive. Overlap: {overlap:.3f}")
        else:
            loop_id = -1

        query_time = (time.time() - start_time) * 1000
        query_times.append(query_time)

        # Step 3: Add descriptors to database
        start_time = time.time()
        btc_manager.add_btc_descs(btc_list)
        update_time = (time.time() - start_time) * 1000
        update_times.append(update_time)

        # Save downsampled cloud for overlap calculation
        downsampled_cloud = btc_manager.down_sampling_voxel(transformed_points, 0.5)
        btc_manager.key_cloud_vec.append(downsampled_cloud)

        print(f"[Time] descriptor extraction: {descriptor_time:.2f}ms, "
              f"query: {query_time:.2f}ms, update map: {update_time:.2f}ms")

    # Print final statistics
    total_frames = len(descriptor_times)
    mean_descriptor_time = np.mean(descriptor_times)
    mean_query_time = np.mean(query_times)
    mean_update_time = np.mean(update_times)

    print(f"\nTotal submap number: {total_frames}")
    print(f"Trigger loop number: {trigger_loop_num}")
    print(f"True loop number: {true_loop_num}")
    print(f"Mean time for descriptor extraction: {mean_descriptor_time:.2f}ms, "
          f"query: {mean_query_time:.2f}ms, update: {mean_update_time:.2f}ms, "
          f"total: {mean_descriptor_time + mean_query_time + mean_update_time:.2f}ms")


if __name__ == "__main__":
    main()
