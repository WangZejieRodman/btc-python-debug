import numpy as np
from typing import List, Tuple
from .data_structures import BinaryDescriptor, Plane
from .config import ConfigSetting
from .utils import non_max_suppression


class BinaryExtractor:
    """Extract binary descriptors from point clouds"""

    def __init__(self, config: ConfigSetting):
        self.config = config

    def extract_binary_descriptors(self, proj_plane_list: List[Plane],
                                   input_cloud: np.ndarray) -> List[BinaryDescriptor]:
        """Extract binary descriptors using projection planes"""
        binary_list = []
        temp_binary_list = []

        last_normal = np.zeros(3)
        useful_proj_num = 0

        # Sort planes by point size (descending)
        proj_plane_list.sort(key=lambda x: x.points_size, reverse=True)

        for plane in proj_plane_list:
            prepare_binary_list = []
            proj_center = plane.center
            proj_normal = plane.normal.copy()

            # Ensure normal points upward (z > 0)
            if proj_normal[2] < 0:
                proj_normal = -proj_normal

            # Check if this normal is different enough from the last one
            if (np.linalg.norm(proj_normal - last_normal) < 0.3 or
                    np.linalg.norm(proj_normal + last_normal) > 0.3):
                last_normal = proj_normal
                print(f"[Description] reference plane normal: {proj_normal}, center: {proj_center}")
                useful_proj_num += 1

                self._extract_binary_from_projection(proj_center, proj_normal,
                                                     input_cloud, prepare_binary_list)

                temp_binary_list.extend(prepare_binary_list)

                if useful_proj_num >= self.config.proj_plane_num:
                    break

        # Apply non-maximum suppression
        temp_binary_list = non_max_suppression(temp_binary_list,
                                               self.config.non_max_suppression_radius)

        # Select top descriptors
        if self.config.useful_corner_num > len(temp_binary_list):
            binary_list = temp_binary_list
        else:
            # Sort by summary (descending) and take top N
            temp_binary_list.sort(key=lambda x: x.summary, reverse=True)
            binary_list = temp_binary_list[:self.config.useful_corner_num]

        return binary_list

    def _extract_binary_from_projection(self, project_center: np.ndarray,
                                        project_normal: np.ndarray,
                                        input_cloud: np.ndarray,
                                        binary_list: List[BinaryDescriptor]):
        """Extract binary descriptors from a single projection plane"""
        binary_list.clear()

        resolution = self.config.proj_image_resolution
        dis_threshold_min = self.config.proj_dis_min
        dis_threshold_max = self.config.proj_dis_max
        high_inc = self.config.proj_image_high_inc
        binary_min_dis = self.config.summary_min_thre
        line_filter_enable = bool(self.config.line_filter_enable)

        # Plane equation: Ax + By + Cz + D = 0
        A, B, C = project_normal
        D = -np.dot(project_normal, project_center)

        # Create coordinate system on the projection plane
        # x_axis is perpendicular to the normal
        x_axis = np.array([1.0, 0.0, 0.0])
        if C != 0:
            x_axis[2] = -(A + B) / C
        elif B != 0:
            x_axis[1] = -A / B
        else:
            x_axis = np.array([0.0, 1.0, 0.0])

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(project_normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Project points onto the plane and convert to 2D coordinates
        point_list_2d = []
        point_list_3d = []
        dis_list_2d = []

        for point in input_cloud:
            x, y, z = point[:3]

            # Calculate distance to plane
            dis = x * A + y * B + z * C + D

            if dis < dis_threshold_min or dis > dis_threshold_max:
                continue

            # Project point onto plane
            proj_point = np.array([x, y, z]) - dis * project_normal

            # Convert to 2D coordinates on the plane
            project_x = np.dot(proj_point - project_center, y_axis)
            project_y = np.dot(proj_point - project_center, x_axis)

            point_list_2d.append([project_x, project_y])
            point_list_3d.append(point[:3])
            dis_list_2d.append(dis)

        if len(point_list_2d) <= 5:
            return

        point_list_2d = np.array(point_list_2d)

        # Find bounding box
        min_x, max_x = np.min(point_list_2d[:, 0]), np.max(point_list_2d[:, 0])
        min_y, max_y = np.min(point_list_2d[:, 1]), np.max(point_list_2d[:, 1])

        # Create grid
        segment_base_num = 5
        segment_len = segment_base_num * resolution
        x_segment_num = int((max_x - min_x) / segment_len) + 1
        y_segment_num = int((max_y - min_y) / segment_len) + 1
        x_axis_len = int((max_x - min_x) / resolution) + segment_base_num
        y_axis_len = int((max_y - min_y) / resolution) + segment_base_num

        # Initialize grids
        img_count = np.zeros((x_axis_len, y_axis_len))
        mean_x_list = np.zeros((x_axis_len, y_axis_len))
        mean_y_list = np.zeros((x_axis_len, y_axis_len))
        dis_container = [[[] for _ in range(y_axis_len)] for _ in range(x_axis_len)]

        # Fill grids
        for i, (point_2d, dis) in enumerate(zip(point_list_2d, dis_list_2d)):
            x_index = int((point_2d[0] - min_x) / resolution)
            y_index = int((point_2d[1] - min_y) / resolution)

            if 0 <= x_index < x_axis_len and 0 <= y_index < y_axis_len:
                mean_x_list[x_index, y_index] += point_2d[0]
                mean_y_list[x_index, y_index] += point_2d[1]
                img_count[x_index, y_index] += 1
                dis_container[x_index][y_index].append(dis)

        # Calculate binary descriptors for each grid cell
        dis_array = np.zeros((x_axis_len, y_axis_len))
        binary_container = [[BinaryDescriptor() for _ in range(y_axis_len)]
                            for _ in range(x_axis_len)]

        cut_num = int((dis_threshold_max - dis_threshold_min) / high_inc)

        for x in range(x_axis_len):
            for y in range(y_axis_len):
                if img_count[x, y] > 0:
                    # Create occupancy array
                    occup_list = [False] * cut_num
                    cnt_list = [0] * cut_num

                    for dis in dis_container[x][y]:
                        cnt_index = int((dis - dis_threshold_min) / high_inc)
                        if 0 <= cnt_index < cut_num:
                            cnt_list[cnt_index] += 1

                    segment_dis = 0
                    for i in range(cut_num):
                        if cnt_list[i] >= 1:
                            segment_dis += 1
                            occup_list[i] = True

                    dis_array[x, y] = segment_dis

                    # Create binary descriptor
                    single_binary = BinaryDescriptor()
                    single_binary.occupy_array = occup_list
                    single_binary.summary = segment_dis
                    binary_container[x][y] = single_binary

        # Extract maxima from segments
        max_dis_list = []
        max_dis_x_indices = []
        max_dis_y_indices = []

        for x_seg in range(x_segment_num):
            for y_seg in range(y_segment_num):
                max_dis = 0
                max_x_idx = -1
                max_y_idx = -1

                x_start = x_seg * segment_base_num
                x_end = min((x_seg + 1) * segment_base_num, x_axis_len)
                y_start = y_seg * segment_base_num
                y_end = min((y_seg + 1) * segment_base_num, y_axis_len)

                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        if dis_array[x, y] > max_dis:
                            max_dis = dis_array[x, y]
                            max_x_idx = x
                            max_y_idx = y

                if max_dis >= binary_min_dis:
                    max_dis_list.append(max_dis)
                    max_dis_x_indices.append(max_x_idx)
                    max_dis_y_indices.append(max_y_idx)

        # Apply line filtering and create final binary descriptors
        directions = np.array([[0, 1], [1, 0], [1, 1], [1, -1]])

        for i, (max_dis, max_x, max_y) in enumerate(zip(max_dis_list,
                                                        max_dis_x_indices,
                                                        max_dis_y_indices)):
            if max_x <= 0 or max_x >= x_axis_len - 1 or max_y <= 0 or max_y >= y_axis_len - 1:
                continue

            is_add = True

            # Apply line filtering
            if line_filter_enable:
                threshold = max_dis - 3
                for direction in directions:
                    p1_x, p1_y = max_x + direction[0], max_y + direction[1]
                    p2_x, p2_y = max_x - direction[0], max_y - direction[1]

                    if (0 <= p1_x < x_axis_len and 0 <= p1_y < y_axis_len and
                            0 <= p2_x < x_axis_len and 0 <= p2_y < y_axis_len):

                        if (dis_array[p1_x, p1_y] >= threshold and
                                dis_array[p2_x, p2_y] >= 0.5 * max_dis):
                            is_add = False
                            break
                        if (dis_array[p2_x, p2_y] >= threshold and
                                dis_array[p1_x, p1_y] >= 0.5 * max_dis):
                            is_add = False
                            break

            if is_add:
                # Calculate 3D position
                mean_x = mean_x_list[max_x, max_y] / img_count[max_x, max_y]
                mean_y = mean_y_list[max_x, max_y] / img_count[max_x, max_y]

                coord_3d = project_center + mean_y * x_axis + mean_x * y_axis

                single_binary = binary_container[max_x][max_y]
                single_binary.location = coord_3d
                binary_list.append(single_binary)