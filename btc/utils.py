import numpy as np
import open3d as o3d
import struct
from typing import List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from .data_structures import BinaryDescriptor, VoxelLoc, MPoint


def load_pose_with_time(pose_file: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """Load pose with timestamp from file"""
    pose_list = []
    time_list = []

    try:
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    time_stamp = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                    # Convert quaternion to rotation matrix
                    from scipy.spatial.transform import Rotation as R
                    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    translation = np.array([tx, ty, tz])

                    pose_list.append((translation, rotation))
                    time_list.append(time_stamp)
    except Exception as e:
        print(f"Error loading pose file: {e}")

    return pose_list, time_list


def load_evo_pose_with_time(pose_file: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """Load EVO format pose with timestamp"""
    pose_list = []
    time_list = []

    try:
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    time_stamp = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                    # Convert quaternion to rotation matrix (EVO format: qx, qy, qz, qw)
                    from scipy.spatial.transform import Rotation as R
                    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    translation = np.array([tx, ty, tz])

                    pose_list.append((translation, rotation))
                    time_list.append(time_stamp)
    except Exception as e:
        print(f"Error loading EVO pose file: {e}")

    return pose_list, time_list


def read_lidar_data(lidar_data_path: str) -> np.ndarray:
    """Read KITTI binary lidar data"""
    try:
        with open(lidar_data_path, 'rb') as f:
            data = f.read()

        # Each point has 4 float32 values (x, y, z, intensity)
        points_data = struct.unpack('f' * (len(data) // 4), data)
        points = np.array(points_data).reshape(-1, 4)
        return points
    except Exception as e:
        print(f"Error reading lidar data: {e}")
        return np.empty((0, 4))


def load_point_cloud_from_pcd(pcd_file: str) -> np.ndarray:
    """Load point cloud from PCD file"""
    try:
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)

        # Add intensity channel (set to zeros if not available)
        if not hasattr(pcd, 'colors') or len(pcd.colors) == 0:
            intensity = np.zeros((points.shape[0], 1))
        else:
            # Use the first color channel as intensity
            colors = np.asarray(pcd.colors)
            intensity = colors[:, 0:1]

        points_with_intensity = np.hstack([points, intensity])
        return points_with_intensity
    except Exception as e:
        print(f"Error loading PCD file: {e}")
        return np.empty((0, 4))


def down_sampling_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel downsampling of point cloud"""
    if voxel_size < 0.01 or len(points) == 0:
        return points

    voxel_map = {}

    for i, point in enumerate(points):
        # Calculate voxel coordinates
        loc_xyz = point[:3] / voxel_size
        loc_xyz = np.floor(loc_xyz).astype(int)

        # Adjust for negative coordinates
        for j in range(3):
            if loc_xyz[j] < 0:
                loc_xyz[j] -= 1

        voxel_key = tuple(loc_xyz)

        if voxel_key in voxel_map:
            voxel_map[voxel_key].xyz += point[:3]
            voxel_map[voxel_key].intensity += point[3] if len(point) > 3 else 0
            voxel_map[voxel_key].count += 1
        else:
            m_point = MPoint()
            m_point.xyz = point[:3].copy()
            m_point.intensity = point[3] if len(point) > 3 else 0
            m_point.count = 1
            voxel_map[voxel_key] = m_point

    # Create downsampled point cloud
    downsampled_points = []
    for m_point in voxel_map.values():
        avg_point = np.zeros(4)
        avg_point[:3] = m_point.xyz / m_point.count
        avg_point[3] = m_point.intensity / m_point.count
        downsampled_points.append(avg_point)

    return np.array(downsampled_points)


def binary_similarity(b1: BinaryDescriptor, b2: BinaryDescriptor) -> float:
    """Calculate binary descriptor similarity"""
    if len(b1.occupy_array) == 0 or len(b2.occupy_array) == 0:
        return 0.0

    common_count = 0
    min_len = min(len(b1.occupy_array), len(b2.occupy_array))

    for i in range(min_len):
        if b1.occupy_array[i] and b2.occupy_array[i]:
            common_count += 1

    total_summary = b1.summary + b2.summary
    if total_summary == 0:
        return 0.0

    return 2.0 * common_count / total_summary


def calc_overlap(cloud1: np.ndarray, cloud2: np.ndarray,
                 dis_threshold: float = 0.5, skip_num: int = 2) -> float:
    """Calculate overlap between two point clouds"""
    if len(cloud1) == 0 or len(cloud2) == 0:
        return 0.0

    # Build KD-tree for cloud2
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nbrs.fit(cloud2[:, :3])

    match_count = 0
    sampled_points = cloud1[::skip_num]

    for point in sampled_points:
        distances, _ = nbrs.kneighbors([point[:3]])
        if distances[0][0] < dis_threshold:
            match_count += 1

    overlap = (2.0 * match_count * skip_num) / (len(cloud1) + len(cloud2))
    return overlap


def non_max_suppression(binary_list: List[BinaryDescriptor], radius: float) -> List[BinaryDescriptor]:
    """Non-maximum suppression for binary descriptors"""
    if len(binary_list) == 0:
        return []

    # Extract locations and summaries
    locations = np.array([bd.location for bd in binary_list])
    summaries = np.array([bd.summary for bd in binary_list])

    # Build KD-tree
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree')
    nbrs.fit(locations)

    keep_flags = np.ones(len(binary_list), dtype=bool)

    for i, location in enumerate(locations):
        if not keep_flags[i]:
            continue

        # Find neighbors within radius
        indices = nbrs.radius_neighbors([location], return_distance=False)[0]

        for j in indices:
            if i != j and summaries[i] <= summaries[j]:
                keep_flags[i] = False
                break

    return [binary_list[i] for i in range(len(binary_list)) if keep_flags[i]]


def transform_point_cloud(points: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Transform point cloud with rotation and translation"""
    if len(points) == 0:
        return points

    transformed_points = points.copy()
    transformed_points[:, :3] = (rotation @ points[:, :3].T).T + translation
    return transformed_points


def point_to_plane_distance(point: np.ndarray, plane_center: np.ndarray, plane_normal: np.ndarray) -> float:
    """Calculate point to plane distance"""
    return abs(np.dot(plane_normal, point - plane_center))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm