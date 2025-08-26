import yaml
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigSetting:
    """Configuration settings for BTC descriptor"""

    # For submap process
    cloud_ds_size: float = 0.25

    # For binary descriptor
    useful_corner_num: int = 30
    plane_merge_normal_thre: float = 0.1
    plane_merge_dis_thre: float = 0.3
    plane_detection_thre: float = 0.01
    voxel_size: float = 1.0
    voxel_init_num: int = 10
    proj_plane_num: int = 1
    proj_image_resolution: float = 0.5
    proj_image_high_inc: float = 0.5
    proj_dis_min: float = 0.0
    proj_dis_max: float = 5.0
    summary_min_thre: float = 10.0
    line_filter_enable: int = 0

    # For triangle descriptor
    descriptor_near_num: float = 10
    descriptor_min_len: float = 1
    descriptor_max_len: float = 10
    non_max_suppression_radius: float = 3.0
    std_side_resolution: float = 0.2

    # For place recognition
    skip_near_num: int = 20
    candidate_num: int = 50
    sub_frame_num: int = 10
    rough_dis_threshold: float = 0.03
    similarity_threshold: float = 0.7
    icp_threshold: float = 0.5
    normal_threshold: float = 0.1
    dis_threshold: float = 0.3

    # Extrinsic for lidar to vehicle
    rot_lidar_to_vehicle: Optional[np.ndarray] = None
    t_lidar_to_vehicle: Optional[np.ndarray] = None

    # For gt file style
    gt_file_style: int = 0

    def __post_init__(self):
        if self.rot_lidar_to_vehicle is None:
            self.rot_lidar_to_vehicle = np.eye(3)
        if self.t_lidar_to_vehicle is None:
            self.t_lidar_to_vehicle = np.zeros(3)


def load_config_setting(config_file: str) -> ConfigSetting:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = ConfigSetting()

        # Binary descriptor parameters
        config.useful_corner_num = config_dict.get('useful_corner_num', 30)
        config.plane_merge_normal_thre = config_dict.get('plane_merge_normal_thre', 0.1)
        config.plane_merge_dis_thre = config_dict.get('plane_merge_dis_thre', 0.3)
        config.plane_detection_thre = config_dict.get('plane_detection_thre', 0.01)
        config.voxel_size = config_dict.get('voxel_size', 1.0)
        config.voxel_init_num = config_dict.get('voxel_init_num', 10)
        config.proj_plane_num = config_dict.get('proj_plane_num', 1)
        config.proj_image_resolution = config_dict.get('proj_image_resolution', 0.5)
        config.proj_image_high_inc = config_dict.get('proj_image_high_inc', 0.5)
        config.proj_dis_min = config_dict.get('proj_dis_min', 0.0)
        config.proj_dis_max = config_dict.get('proj_dis_max', 5.0)
        config.summary_min_thre = config_dict.get('summary_min_thre', 10.0)
        config.line_filter_enable = config_dict.get('line_filter_enable', 0)

        # Triangle descriptor parameters
        config.descriptor_near_num = config_dict.get('descriptor_near_num', 10)
        config.descriptor_min_len = config_dict.get('descriptor_min_len', 1)
        config.descriptor_max_len = config_dict.get('descriptor_max_len', 10)
        config.non_max_suppression_radius = config_dict.get('max_constrait_dis', 3.0)
        config.std_side_resolution = config_dict.get('triangle_resolution', 0.2)

        # Candidate search parameters
        config.skip_near_num = config_dict.get('skip_near_num', 20)
        config.candidate_num = config_dict.get('candidate_num', 50)
        config.rough_dis_threshold = config_dict.get('rough_dis_threshold', 0.03)
        config.similarity_threshold = config_dict.get('similarity_threshold', 0.7)
        config.icp_threshold = config_dict.get('icp_threshold', 0.5)
        config.normal_threshold = config_dict.get('normal_threshold', 0.1)
        config.dis_threshold = config_dict.get('dis_threshold', 0.3)

        print(f"Successfully loaded config file: {config_file}")
        return config

    except Exception as e:
        print(f"Failed to load config file: {config_file}, error: {e}")
        return ConfigSetting()