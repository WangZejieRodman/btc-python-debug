import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import time

from .config import ConfigSetting
from .data_structures import BTC, BTCMatchList, BTCLoc, BinaryDescriptor
from .voxel_processing import VoxelProcessor
from .binary_extractor import BinaryExtractor
from .btc_generator import BTCGenerator, triangle_solver, calc_binary_similarity
from .utils import binary_similarity, down_sampling_voxel


class BTCDescManager:
    """Main manager class for BTC descriptor processing"""

    def __init__(self, config: ConfigSetting):
        self.config = config
        self.print_debug_info = False

        # Initialize processors
        self.voxel_processor = VoxelProcessor(config)
        self.binary_extractor = BinaryExtractor(config)
        self.btc_generator = BTCGenerator(config)

        # Database and history storage
        self.database: Dict[BTCLoc, List[BTC]] = {}
        self.key_cloud_list: List[np.ndarray] = []
        self.history_binary_list: List[List[BinaryDescriptor]] = []
        self.plane_cloud_list: List[np.ndarray] = []

    def generate_btc_descriptors(self, input_cloud: np.ndarray, frame_id: int) -> List[BTC]:
        """Generate BTC descriptors from input point cloud"""
        if self.print_debug_info:
            print(f"[Description] Processing frame {frame_id}")

        # Step 1: Voxelization and plane detection
        voxel_map = self.voxel_processor.init_voxel_map(input_cloud)
        plane_cloud = self.voxel_processor.get_planes(voxel_map)

        if self.print_debug_info:
            print(f"[Description] Detected {len(plane_cloud)} planes")

        self.plane_cloud_list.append(np.array(plane_cloud))

        # Step 2: Get projection planes
        proj_plane_list = self.voxel_processor.get_project_planes(voxel_map)

        # Handle case with no planes
        if len(proj_plane_list) == 0:
            from .data_structures import Plane
            single_plane = Plane()
            single_plane.normal = np.array([0, 0, 1])
            single_plane.center = np.array([input_cloud[0, 0], input_cloud[0, 1], input_cloud[0, 2]])
            single_plane.points_size = 1
            proj_plane_list = [single_plane]

        # Step 3: Extract binary descriptors
        binary_list = self.binary_extractor.extract_binary_descriptors(proj_plane_list, input_cloud)
        self.history_binary_list.append(binary_list)

        if self.print_debug_info:
            print(f"[Description] Extracted {len(binary_list)} binary descriptors")

        # Step 4: Generate BTC descriptors
        btc_list = self.btc_generator.generate_btc_descriptors(binary_list, frame_id)

        if self.print_debug_info:
            print(f"[Description] Generated {len(btc_list)} BTC descriptors")

        return btc_list

    def search_loop(self, btc_list: List[BTC]) -> Tuple[Tuple[int, float],
    Tuple[np.ndarray, np.ndarray],
    List[Tuple[BTC, BTC]]]:
        """Search for loop closure candidates"""
        if len(btc_list) == 0:
            print("No BTC descriptors!")
            return (-1, 0.0), (np.zeros(3), np.eye(3)), []

        # Step 1: Select candidates
        candidate_list = self._candidate_selector(btc_list)

        # Step 2: Verify candidates
        best_score = 0.0
        best_candidate_id = -1
        best_transform = (np.zeros(3), np.eye(3))
        best_match_pairs = []

        for candidate in candidate_list:
            score, transform, match_pairs = self._candidate_verify(candidate)

            if self.print_debug_info:
                print(f"[Retrieval] Candidate frame {candidate.match_id[1]}, "
                      f"matches: {len(candidate.match_list)}, score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_candidate_id = candidate.match_id[1]
                best_transform = transform
                best_match_pairs = match_pairs

        if self.print_debug_info:
            print(f"[Retrieval] Best candidate: {best_candidate_id}, score: {best_score:.3f}")

        if best_score > self.config.icp_threshold:
            return (best_candidate_id, best_score), best_transform, best_match_pairs
        else:
            return (-1, 0.0), (np.zeros(3), np.eye(3)), []

    def add_btc_descriptors(self, btc_list: List[BTC]):
        """Add BTC descriptors to database"""
        for btc in btc_list:
            # Calculate hash position
            position = BTCLoc(
                int(btc.triangle[0] + 0.5),
                int(btc.triangle[1] + 0.5),
                int(btc.triangle[2] + 0.5)
            )

            if position in self.database:
                self.database[position].append(btc)
            else:
                self.database[position] = [btc]

    def _candidate_selector(self, current_btc_list: List[BTC]) -> List[BTCMatchList]:
        """Select candidate frames based on rough matching"""
        if len(current_btc_list) == 0:
            return []

        current_frame_id = current_btc_list[0].frame_number
        match_array = np.zeros(20000)
        match_pairs = []
        match_frame_ids = []

        # Define voxel search neighborhood
        voxel_offsets = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    voxel_offsets.append((x, y, z))

        # Search for matches
        for btc in current_btc_list:
            dis_threshold = np.linalg.norm(btc.triangle) * self.config.rough_dis_threshold

            for dx, dy, dz in voxel_offsets:
                position = BTCLoc(
                    int(btc.triangle[0] + dx),
                    int(btc.triangle[1] + dy),
                    int(btc.triangle[2] + dz)
                )

                # Check voxel center distance
                voxel_center = np.array([position.x + 0.5, position.y + 0.5, position.z + 0.5])
                if np.linalg.norm(btc.triangle - voxel_center) < 1.5:
                    if position in self.database:
                        for stored_btc in self.database[position]:
                            # Skip if too close in time
                            if (btc.frame_number - stored_btc.frame_number) <= self.config.skip_near_num:
                                continue

                            # Check triangle distance
                            triangle_dis = np.linalg.norm(btc.triangle - stored_btc.triangle)
                            if triangle_dis < dis_threshold:
                                # Check binary similarity
                                sim_A = binary_similarity(btc.binary_A, stored_btc.binary_A)
                                sim_B = binary_similarity(btc.binary_B, stored_btc.binary_B)
                                sim_C = binary_similarity(btc.binary_C, stored_btc.binary_C)
                                avg_similarity = (sim_A + sim_B + sim_C) / 3.0

                                if avg_similarity > self.config.similarity_threshold:
                                    match_array[stored_btc.frame_number] += 1
                                    match_pairs.append((btc, stored_btc))
                                    match_frame_ids.append(stored_btc.frame_number)

        # Select top candidates
        candidate_list = []
        for _ in range(self.config.candidate_num):
            max_votes = np.max(match_array)
            if max_votes < 5:
                break

            max_frame_id = np.argmax(match_array)
            match_array[max_frame_id] = 0

            # Create candidate match list
            candidate = BTCMatchList()
            candidate.match_frame = max_frame_id
            candidate.match_id = (current_frame_id, max_frame_id)

            # Collect all matches for this frame
            for i, frame_id in enumerate(match_frame_ids):
                if frame_id == max_frame_id:
                    candidate.match_list.append(match_pairs[i])

            candidate_list.append(candidate)

        return candidate_list

    def _candidate_verify(self, candidate: BTCMatchList) -> Tuple[float,
    Tuple[np.ndarray, np.ndarray],
    List[Tuple[BTC, BTC]]]:
        """Verify candidate using geometric consistency"""
        if len(candidate.match_list) < 4:
            return -1.0, (np.zeros(3), np.eye(3)), []

        # Sample matches for efficiency
        skip_len = max(1, len(candidate.match_list) // 50)
        sampled_matches = candidate.match_list[::skip_len]

        best_vote = 0
        best_transform = (np.zeros(3), np.eye(3))
        success_matches = []

        # Test each match as a potential transformation seed
        for seed_match in sampled_matches:
            # Solve for transformation
            t, rot = triangle_solver((seed_match[0], seed_match[1]))

            # Count consistent matches
            vote_count = 0
            temp_matches = []

            for match_pair in candidate.match_list:
                btc1, btc2 = match_pair

                # Transform first BTC points
                A_transformed = rot @ btc1.binary_A.location + t
                B_transformed = rot @ btc1.binary_B.location + t
                C_transformed = rot @ btc1.binary_C.location + t

                # Check distances to second BTC points
                dis_A = np.linalg.norm(A_transformed - btc2.binary_A.location)
                dis_B = np.linalg.norm(B_transformed - btc2.binary_B.location)
                dis_C = np.linalg.norm(C_transformed - btc2.binary_C.location)

                if dis_A < 3.0 and dis_B < 3.0 and dis_C < 3.0:
                    vote_count += 1
                    temp_matches.append(match_pair)

            if vote_count > best_vote:
                best_vote = vote_count
                best_transform = (t, rot)
                success_matches = temp_matches

        if best_vote >= 4:
            # Perform plane geometric verification
            score = self._plane_geometric_verify(
                self.plane_cloud_list[-1],
                self.plane_cloud_list[candidate.match_id[1]],
                best_transform
            )
            return score, best_transform, success_matches
        else:
            return -1.0, (np.zeros(3), np.eye(3)), []

    def _plane_geometric_verify(self, source_planes: np.ndarray,
                                target_planes: np.ndarray,
                                transform: Tuple[np.ndarray, np.ndarray]) -> float:
        """Verify transformation using plane-to-plane matching"""
        if len(source_planes) == 0 or len(target_planes) == 0:
            return 0.0

        t, rot = transform

        # Build KD-tree for target planes
        target_positions = target_planes[:, :3]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nbrs.fit(target_positions)

        useful_matches = 0
        total_planes = len(source_planes)

        for source_plane in source_planes:
            source_pos = source_plane[:3]
            source_normal = source_plane[3:6]

            # Transform source plane
            transformed_pos = rot @ source_pos + t
            transformed_normal = rot @ source_normal

            # Find nearest target plane
            distances, indices = nbrs.kneighbors([transformed_pos])
            nearest_idx = indices[0][0]

            target_pos = target_planes[nearest_idx, :3]
            target_normal = target_planes[nearest_idx, 3:6]

            # Check normal similarity
            normal_diff = np.linalg.norm(transformed_normal - target_normal)
            normal_add = np.linalg.norm(transformed_normal + target_normal)

            # Check point-to-plane distance
            point_to_plane_dist = abs(np.dot(target_normal, transformed_pos - target_pos))

            if ((normal_diff < self.config.normal_threshold or
                 normal_add < self.config.normal_threshold) and
                    point_to_plane_dist < self.config.dis_threshold):
                useful_matches += 1

        if total_planes > 0:
            return useful_matches / total_planes
        else:
            return 0.0

    def plane_geometric_icp(self, source_planes: np.ndarray,
                            target_planes: np.ndarray,
                            initial_transform: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Refine transformation using plane-to-plane ICP"""
        t, rot = initial_transform

        # Convert rotation matrix to quaternion for optimization
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rot)
        quat = r.as_quat()  # [x, y, z, w]

        # Initial parameters [tx, ty, tz, qx, qy, qz, qw]
        initial_params = np.concatenate([t, quat])

        def objective_function(params):
            t_opt = params[:3]
            quat_opt = params[3:]
            quat_opt = quat_opt / np.linalg.norm(quat_opt)  # Normalize quaternion

            r_opt = R.from_quat(quat_opt)
            rot_opt = r_opt.as_matrix()

            total_error = 0.0
            valid_matches = 0

            # Build KD-tree for target planes
            target_positions = target_planes[:, :3]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
            nbrs.fit(target_positions)

            for source_plane in source_planes:
                source_pos = source_plane[:3]
                source_normal = source_plane[3:6]

                # Transform source
                transformed_pos = rot_opt @ source_pos + t_opt
                transformed_normal = rot_opt @ source_normal

                # Find nearest target
                distances, indices = nbrs.kneighbors([transformed_pos])
                nearest_idx = indices[0][0]

                target_pos = target_planes[nearest_idx, :3]
                target_normal = target_planes[nearest_idx, 3:6]

                # Check if it's a valid correspondence
                normal_diff = np.linalg.norm(transformed_normal - target_normal)
                normal_add = np.linalg.norm(transformed_normal + target_normal)
                point_to_plane_dist = abs(np.dot(target_normal, transformed_pos - target_pos))

                if ((normal_diff < self.config.normal_threshold or
                     normal_add < self.config.normal_threshold) and
                        point_to_plane_dist < self.config.dis_threshold and
                        distances[0][0] < 3.0):
                    # Point-to-plane error
                    error = np.dot(target_normal, transformed_pos - target_pos) ** 2
                    total_error += error
                    valid_matches += 1

            return total_error / max(valid_matches, 1)

        # Optimize
        try:
            result = minimize(objective_function, initial_params, method='BFGS',
                              options={'maxiter': 100})

            if result.success:
                optimized_t = result.x[:3]
                optimized_quat = result.x[3:]
                optimized_quat = optimized_quat / np.linalg.norm(optimized_quat)

                r_opt = R.from_quat(optimized_quat)
                optimized_rot = r_opt.as_matrix()

                return optimized_t, optimized_rot
            else:
                return t, rot
        except:
            return t, rot