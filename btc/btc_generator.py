import numpy as np
from typing import List, Set
from sklearn.neighbors import NearestNeighbors
from .data_structures import BinaryDescriptor, BTC
from .config import ConfigSetting


class BTCGenerator:
    """Generate BTC (Binary Triangle Combined) descriptors"""

    def __init__(self, config: ConfigSetting):
        self.config = config

    def generate_btc_descriptors(self, binary_list: List[BinaryDescriptor],
                                 frame_id: int) -> List[BTC]:
        """Generate BTC descriptors from binary descriptors"""
        if len(binary_list) == 0:
            return []

        btc_list = []
        scale = 1.0 / self.config.std_side_resolution

        # Extract locations for KD-tree
        locations = np.array([bd.location for bd in binary_list])

        # Build KD-tree
        nbrs = NearestNeighbors(n_neighbors=min(self.config.descriptor_near_num, len(binary_list)),
                                algorithm='kd_tree')
        nbrs.fit(locations)

        # Track used triangles to avoid duplicates
        used_triangles: Set[str] = set()

        for i, search_point in enumerate(locations):
            # Find K nearest neighbors
            distances, indices = nbrs.kneighbors([search_point])
            indices = indices[0]  # Get first row

            if len(indices) < 3:
                continue

            # Generate triangles from combinations of neighbors
            for m in range(1, len(indices) - 1):
                for n in range(m + 1, len(indices)):
                    idx1, idx2, idx3 = i, indices[m], indices[n]

                    p1 = locations[idx1]
                    p2 = locations[idx2]
                    p3 = locations[idx3]

                    # Calculate triangle side lengths
                    a = np.linalg.norm(p1 - p2)
                    b = np.linalg.norm(p1 - p3)
                    c = np.linalg.norm(p2 - p3)

                    # Filter by length constraints
                    if (a > self.config.descriptor_max_len or
                            b > self.config.descriptor_max_len or
                            c > self.config.descriptor_max_len or
                            a < self.config.descriptor_min_len or
                            b < self.config.descriptor_min_len or
                            c < self.config.descriptor_min_len):
                        continue

                    # Sort sides to ensure consistency (a <= b <= c)
                    sides = [a, b, c]
                    points = [p1, p2, p3]
                    binaries = [binary_list[idx1], binary_list[idx2], binary_list[idx3]]

                    # Sort by side length
                    sorted_indices = np.argsort(sides)
                    sorted_sides = [sides[i] for i in sorted_indices]
                    sorted_points = [points[i] for i in sorted_indices]
                    sorted_binaries = [binaries[i] for i in sorted_indices]

                    a_sorted, b_sorted, c_sorted = sorted_sides

                    # Skip degenerate triangles
                    if abs(c_sorted - (a_sorted + b_sorted)) < 0.2:
                        continue

                    # Create unique triangle identifier
                    triangle_key = f"{int(a_sorted * 1000):06d}_{int(b_sorted * 1000):06d}_{int(c_sorted * 1000):06d}"

                    if triangle_key in used_triangles:
                        continue

                    used_triangles.add(triangle_key)

                    # Create BTC descriptor
                    btc = BTC(frame_id)
                    btc.triangle = np.array([scale * a_sorted, scale * b_sorted, scale * c_sorted])
                    btc.center = (sorted_points[0] + sorted_points[1] + sorted_points[2]) / 3

                    # Assign binary descriptors (maintaining order)
                    btc.binary_A = sorted_binaries[0]
                    btc.binary_B = sorted_binaries[1]
                    btc.binary_C = sorted_binaries[2]

                    # Calculate angles (simplified - could be improved)
                    btc.angle = np.array([0.0, 0.0, 0.0])  # Placeholder for angles

                    btc_list.append(btc)

        return btc_list


def triangle_solver(btc_pair: tuple) -> tuple:
    """Solve for rotation and translation between two BTC descriptors"""
    btc1, btc2 = btc_pair

    # Create source and reference matrices
    src = np.zeros((3, 3))
    ref = np.zeros((3, 3))

    src[:, 0] = btc1.binary_A.location - btc1.center
    src[:, 1] = btc1.binary_B.location - btc1.center
    src[:, 2] = btc1.binary_C.location - btc1.center

    ref[:, 0] = btc2.binary_A.location - btc2.center
    ref[:, 1] = btc2.binary_B.location - btc2.center
    ref[:, 2] = btc2.binary_C.location - btc2.center

    # Compute rotation using SVD
    covariance = src @ ref.T
    U, _, Vt = np.linalg.svd(covariance)

    rot = Vt.T @ U.T

    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(rot) < 0:
        K = np.eye(3)
        K[2, 2] = -1
        rot = Vt.T @ K @ U.T

    # Compute translation
    t = -rot @ btc1.center + btc2.center

    return t, rot


def calc_triangle_distance(match_list: List[tuple]) -> float:
    """Calculate mean triangle distance for matched BTC pairs"""
    if len(match_list) == 0:
        return -1.0

    total_distance = 0.0
    for btc1, btc2 in match_list:
        triangle_diff = btc1.triangle - btc2.triangle
        triangle_norm = np.linalg.norm(btc1.triangle)
        if triangle_norm > 0:
            total_distance += np.linalg.norm(triangle_diff) / triangle_norm

    return total_distance / len(match_list)


def calc_binary_similarity(match_list: List[tuple]) -> float:
    """Calculate mean binary similarity for matched BTC pairs"""
    if len(match_list) == 0:
        return -1.0

    from .utils import binary_similarity

    total_similarity = 0.0
    for btc1, btc2 in match_list:
        sim_A = binary_similarity(btc1.binary_A, btc2.binary_A)
        sim_B = binary_similarity(btc1.binary_B, btc2.binary_B)
        sim_C = binary_similarity(btc1.binary_C, btc2.binary_C)
        total_similarity += (sim_A + sim_B + sim_C) / 3.0

    return total_similarity / len(match_list)