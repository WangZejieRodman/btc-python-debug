import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import open3d as o3d


@dataclass
class BinaryDescriptor:
    """Binary descriptor for keypoints"""
    occupy_array: List[bool]
    summary: float
    location: np.ndarray  # 3D position

    def __init__(self, occupy_array: List[bool] = None, summary: float = 0.0, location: np.ndarray = None):
        self.occupy_array = occupy_array if occupy_array is not None else []
        self.summary = summary
        self.location = location if location is not None else np.zeros(3)


@dataclass
class Plane:
    """Plane structure for geometric processing"""
    p_center: np.ndarray  # Point center (x, y, z, normal_x, normal_y, normal_z)
    center: np.ndarray  # 3D center
    normal: np.ndarray  # 3D normal vector
    covariance: np.ndarray  # 3x3 covariance matrix
    radius: float = 0.0
    min_eigen_value: float = 1.0
    d: float = 0.0  # Plane equation parameter
    id: int = 0
    sub_plane_num: int = 0
    points_size: int = 0
    is_plane: bool = False

    def __init__(self):
        self.p_center = np.zeros(6)  # [x, y, z, nx, ny, nz]
        self.center = np.zeros(3)
        self.normal = np.zeros(3)
        self.covariance = np.zeros((3, 3))
        self.radius = 0.0
        self.min_eigen_value = 1.0
        self.d = 0.0
        self.id = 0
        self.sub_plane_num = 0
        self.points_size = 0
        self.is_plane = False


@dataclass
class BTC:
    """Binary Triangle Combined descriptor"""
    triangle: np.ndarray  # Triangle side lengths (3D)
    angle: np.ndarray  # Triangle angles (3D)
    center: np.ndarray  # Triangle center (3D)
    frame_number: int  # Frame ID
    binary_A: BinaryDescriptor
    binary_B: BinaryDescriptor
    binary_C: BinaryDescriptor

    def __init__(self, frame_number: int = 0):
        self.triangle = np.zeros(3)
        self.angle = np.zeros(3)
        self.center = np.zeros(3)
        self.frame_number = frame_number
        self.binary_A = BinaryDescriptor()
        self.binary_B = BinaryDescriptor()
        self.binary_C = BinaryDescriptor()


@dataclass
class BTCMatchList:
    """Match list for BTC descriptors"""
    match_list: List[Tuple[BTC, BTC]]
    match_id: Tuple[int, int]  # (current_frame, matched_frame)
    match_frame: int
    mean_dis: float

    def __init__(self):
        self.match_list = []
        self.match_id = (-1, -1)
        self.match_frame = -1
        self.mean_dis = 0.0


class VoxelLoc:
    """Voxel location for hash mapping"""

    def __init__(self, x: int = 0, y: int = 0, z: int = 0):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    def __hash__(self):
        HASH_P = 116101
        MAX_N = 10000000000
        return int(((((self.z) * HASH_P) % MAX_N + (self.y)) * HASH_P) % MAX_N + (self.x))

    def __str__(self):
        return f"VoxelLoc({self.x}, {self.y}, {self.z})"


class BTCLoc:
    """BTC location for hash mapping"""

    def __init__(self, x: int = 0, y: int = 0, z: int = 0,
                 a: int = 0, b: int = 0, c: int = 0):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    def __hash__(self):
        HASH_P = 116101
        MAX_N = 10000000000
        return int(((((self.z) * HASH_P) % MAX_N + (self.y)) * HASH_P) % MAX_N + (self.x))

    def __str__(self):
        return f"BTCLoc({self.x}, {self.y}, {self.z})"


class MPoint:
    """Point structure for voxel processing"""

    def __init__(self):
        self.xyz = np.zeros(3)
        self.intensity = 0.0
        self.count = 0


class OctoTree:
    """Octree structure for voxel-based plane detection"""

    def __init__(self, config_setting):
        self.config_setting = config_setting
        self.voxel_points = []  # List of 3D points
        self.plane_ptr = Plane()
        self.layer = 0
        self.octo_state = 0  # 0 is end of tree, 1 is not
        self.merge_num = 0
        self.is_project = False
        self.project_normal = []
        self.is_publish = False
        self.leaves = [None] * 8
        self.voxel_center = np.zeros(3)
        self.quater_length = 0.0
        self.init_octo = False

        # For plot
        self.is_check_connect = [False] * 6
        self.connect = [False] * 6
        self.connect_tree = [None] * 6

    def init_plane(self):
        """Initialize plane from voxel points"""
        if len(self.voxel_points) == 0:
            return

        points = np.array(self.voxel_points)
        self.plane_ptr.points_size = len(points)

        # Calculate center
        self.plane_ptr.center = np.mean(points, axis=0)

        # Calculate covariance
        centered_points = points - self.plane_ptr.center
        self.plane_ptr.covariance = np.cov(centered_points.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(self.plane_ptr.covariance)

        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Check if it's a plane
        if eigenvalues[0] < self.config_setting.plane_detection_thre:
            self.plane_ptr.normal = eigenvectors[:, 0]
            self.plane_ptr.min_eigen_value = eigenvalues[0]
            self.plane_ptr.radius = np.sqrt(eigenvalues[2])
            self.plane_ptr.is_plane = True

            # Calculate plane equation parameter d
            self.plane_ptr.d = -(np.dot(self.plane_ptr.normal, self.plane_ptr.center))

            # Update p_center
            self.plane_ptr.p_center[:3] = self.plane_ptr.center
            self.plane_ptr.p_center[3:] = self.plane_ptr.normal
        else:
            self.plane_ptr.is_plane = False

    def init_octo_tree(self):
        """Initialize octree"""
        if len(self.voxel_points) > self.config_setting.voxel_init_num:
            self.init_plane()