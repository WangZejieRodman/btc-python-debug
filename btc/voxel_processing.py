import numpy as np
from typing import Dict, List, Tuple
from .data_structures import VoxelLoc, OctoTree, Plane
from .config import ConfigSetting


class VoxelProcessor:
    """Handle voxelization and plane detection"""

    def __init__(self, config: ConfigSetting):
        self.config = config

    def init_voxel_map(self, input_cloud: np.ndarray) -> Dict[VoxelLoc, OctoTree]:
        """Initialize voxel map from input point cloud"""
        voxel_map = {}

        for i, point in enumerate(input_cloud):
            # Calculate voxel location
            loc_xyz = point[:3] / self.config.voxel_size

            # Handle negative coordinates
            for j in range(3):
                if loc_xyz[j] < 0:
                    loc_xyz[j] -= 1.0

            position = VoxelLoc(int(loc_xyz[0]), int(loc_xyz[1]), int(loc_xyz[2]))

            if position in voxel_map:
                voxel_map[position].voxel_points.append(point[:3])
            else:
                octo_tree = OctoTree(self.config)
                voxel_map[position] = octo_tree
                voxel_map[position].voxel_points.append(point[:3])

        # Initialize octrees
        for octo_tree in voxel_map.values():
            octo_tree.init_octo_tree()

        return voxel_map

    def get_planes(self, voxel_map: Dict[VoxelLoc, OctoTree]) -> List[np.ndarray]:
        """Extract planes from voxel map"""
        plane_cloud = []

        for octo_tree in voxel_map.values():
            if octo_tree.plane_ptr.is_plane:
                # Create plane point with normal
                plane_point = np.zeros(6)  # [x, y, z, nx, ny, nz]
                plane_point[:3] = octo_tree.plane_ptr.center
                plane_point[3:] = octo_tree.plane_ptr.normal
                plane_cloud.append(plane_point)

        return plane_cloud

    def get_project_planes(self, voxel_map: Dict[VoxelLoc, OctoTree]) -> List[Plane]:
        """Get projection planes for binary descriptor extraction"""
        origin_planes = []

        # Collect all valid planes
        for octo_tree in voxel_map.values():
            if octo_tree.plane_ptr.is_plane:
                origin_planes.append(octo_tree.plane_ptr)

        # Reset plane IDs
        for plane in origin_planes:
            plane.id = 0

        # Merge similar planes
        merged_planes = self._merge_planes(origin_planes)

        return merged_planes

    def _merge_planes(self, origin_planes: List[Plane]) -> List[Plane]:
        """Merge similar planes based on normal and distance thresholds"""
        if len(origin_planes) <= 1:
            return origin_planes

        # Assign IDs to similar planes
        current_id = 1
        for i in range(len(origin_planes) - 1, 0, -1):
            for j in range(i):
                plane1 = origin_planes[i]
                plane2 = origin_planes[j]

                # Calculate normal difference
                normal_diff = np.linalg.norm(plane1.normal - plane2.normal)
                normal_add = np.linalg.norm(plane1.normal + plane2.normal)

                # Calculate distance between planes
                dis1 = abs(np.dot(plane1.normal, plane2.center) + plane1.d)
                dis2 = abs(np.dot(plane2.normal, plane1.center) + plane2.d)

                # Check if planes should be merged
                if (normal_diff < self.config.plane_merge_normal_thre or
                        normal_add < self.config.plane_merge_normal_thre):
                    if (dis1 < self.config.plane_merge_dis_thre and
                            dis2 < self.config.plane_merge_dis_thre):
                        if plane1.id == 0 and plane2.id == 0:
                            plane1.id = current_id
                            plane2.id = current_id
                            current_id += 1
                        elif plane1.id == 0 and plane2.id != 0:
                            plane1.id = plane2.id
                        elif plane1.id != 0 and plane2.id == 0:
                            plane2.id = plane1.id

        # Merge planes with same ID
        merged_planes = []
        processed_ids = set()

        for i, plane in enumerate(origin_planes):
            if plane.id in processed_ids:
                continue

            if plane.id == 0:
                merged_planes.append(plane)
                continue

            # Create merged plane
            merged_plane = Plane()
            merged_plane.id = plane.id
            merged_plane.points_size = plane.points_size
            merged_plane.center = plane.center.copy()
            merged_plane.covariance = plane.covariance.copy()
            merged_plane.normal = plane.normal.copy()
            merged_plane.sub_plane_num = 1
            merged_plane.is_plane = True

            # Find all planes with same ID and merge
            for j, other_plane in enumerate(origin_planes):
                if i != j and other_plane.id == plane.id:
                    # Merge statistics
                    total_points = merged_plane.points_size + other_plane.points_size

                    # Weighted average of centers
                    merged_center = (merged_plane.center * merged_plane.points_size +
                                     other_plane.center * other_plane.points_size) / total_points

                    # Merge covariances
                    P_PT1 = (merged_plane.covariance +
                             np.outer(merged_plane.center, merged_plane.center)) * merged_plane.points_size
                    P_PT2 = (other_plane.covariance +
                             np.outer(other_plane.center, other_plane.center)) * other_plane.points_size

                    merged_covariance = (P_PT1 + P_PT2) / total_points - np.outer(merged_center, merged_center)

                    # Update merged plane
                    merged_plane.center = merged_center
                    merged_plane.covariance = merged_covariance
                    merged_plane.points_size = total_points
                    merged_plane.sub_plane_num += 1

                    # Recompute normal from merged covariance
                    eigenvalues, eigenvectors = np.linalg.eig(merged_covariance)
                    min_idx = np.argmin(eigenvalues)
                    merged_plane.normal = eigenvectors[:, min_idx]
                    merged_plane.radius = np.sqrt(np.max(eigenvalues))

                    # Update plane equation
                    merged_plane.d = -np.dot(merged_plane.normal, merged_plane.center)

                    # Update p_center
                    merged_plane.p_center[:3] = merged_plane.center
                    merged_plane.p_center[3:] = merged_plane.normal

            merged_planes.append(merged_plane)
            processed_ids.add(plane.id)

        return merged_planes