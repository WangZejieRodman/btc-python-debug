#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage example for BTC Place Recognition
"""

import numpy as np
import os
import sys
from btc_python import BTCDescManager, ConfigSetting
import time


def run_kitti_example():
    """Run BTC place recognition on KITTI dataset"""

    # Configuration paths - MODIFY THESE PATHS
    config_file = "config_outdoor.yaml"
    pcds_dir = "/home/wzj/pan1/Data/KITTI/00/velodyne"  # KITTI .bin files
    pose_file = "/home/wzj/pan1/btc_ws/src/btc_descriptor/poses/kitti00.txt"  # Ground truth poses

    # Parameters
    cloud_overlap_thr = 0.5
    read_bin = True  # True for KITTI .bin format, False for .pcd

    # Create BTC manager
    config = ConfigSetting()

    # Load config from YAML if exists
    if os.path.exists(config_file):
        btc_manager = BTCDescManager(config)
        config = btc_manager.load_config_from_yaml(config_file)
        btc_manager.config_setting = config
    else:
        btc_manager = BTCDescManager(config)

    btc_manager.print_debug_info = True

    # Load ground truth poses
    translations, rotations = [], []
    if os.path.exists(pose_file):
        translations, rotations = btc_manager.load_pose_file(pose_file)
        print(f"Successfully loaded {len(translations)} poses")
    else:
        print(f"Warning: Pose file not found: {pose_file}")

    # Statistics
    stats = {
        'descriptor_times': [],
        'query_times': [],
        'update_times': [],
        'trigger_loops': [],
        'true_loops': [],
        'false_loops': []
    }

    # Process frames
    max_frames = min(len(translations), 1000) if translations else 100

    for frame_id in range(max_frames):
        print(f"\n=== Processing Frame {frame_id} ===")

        # Load point cloud
        if read_bin:
            bin_file = os.path.join(pcds_dir, f"{frame_id:06d}.bin")
            if not os.path.exists(bin_file):
                print(f"File not found: {bin_file}")
                break
            points = btc_manager.read_kitti_bin(bin_file)
        else:
            # For PCD files
            try:
                import open3d as o3d
                pcd_file = os.path.join(pcds_dir, f"{frame_id:06d}.pcd")
                if not os.path.exists(pcd_file):
                    print(f"File not found: {pcd_file}")
                    break
                pcd = o3d.io.read_point_cloud(pcd_file)
                points = np.asarray(pcd.points)
                # Add intensity column
                if points.shape[1] == 3:
                    points = np.column_stack([points, np.zeros(len(points))])
            except ImportError:
                print("Open3D not available for PCD reading")
                break

        print(f"Loaded {len(points)} points")

        # Apply pose transformation if available
        if frame_id < len(translations):
            translation = translations[frame_id]
            rotation = rotations[frame_id]
            transformed_points = points.copy()
            transformed_points[:, :3] = (rotation @ points[:, :3].T).T + translation
        else:
            transformed_points = points

        # 1. Descriptor extraction
        start_time = time.time()
        btc_list = btc_manager.generate_btc_descs(transformed_points, frame_id)
        desc_time = (time.time() - start_time) * 1000
        stats['descriptor_times'].append(desc_time)

        print(f"Generated {len(btc_list)} BTC descriptors")

        # 2. Loop detection
        start_time = time.time()
        loop_result = (-1, 0, np.eye(3), np.zeros(3), [])

        if frame_id > config.skip_near_num and len(btc_list) > 0:
            loop_result = btc_manager.search_loop(btc_list)
            loop_id, loop_score, R, t, matches = loop_result

            if loop_id >= 0:
                print(f"Loop detected: {frame_id} -> {loop_id}, score: {loop_score:.3f}")
                stats['trigger_loops'].append((frame_id, loop_id, loop_score))

                # Verify with ground truth overlap
                if frame_id < len(translations):
                    current_cloud = btc_manager.down_sampling_voxel(transformed_points, 0.5)
                    if loop_id < len(btc_manager.key_cloud_vec):
                        matched_cloud = btc_manager.key_cloud_vec[loop_id]
                        overlap = btc_manager.calc_overlap(current_cloud, matched_cloud, cloud_overlap_thr)

                        if overlap >= cloud_overlap_thr:
                            stats['true_loops'].append((frame_id, loop_id, overlap))
                            print(f"TRUE POSITIVE: overlap = {overlap:.3f}")
                        else:
                            stats['false_loops'].append((frame_id, loop_id, overlap))
                            print(f"FALSE POSITIVE: overlap = {overlap:.3f}")
                    else:
                        print("Warning: matched frame not in key cloud vector")

        query_time = (time.time() - start_time) * 1000
        stats['query_times'].append(query_time)

        # 3. Update database
        start_time = time.time()
        btc_manager.add_btc_descs(btc_list)
        update_time = (time.time() - start_time) * 1000
        stats['update_times'].append(update_time)

        # Store downsampled cloud
        downsampled_cloud = btc_manager.down_sampling_voxel(transformed_points, 0.5)
        btc_manager.key_cloud_vec.append(downsampled_cloud)

        print(f"Times - Desc: {desc_time:.1f}ms, Query: {query_time:.1f}ms, Update: {update_time:.1f}ms")

        # Progress indicator
        if (frame_id + 1) % 50 == 0:
            print(f"\nProgress: {frame_id + 1}/{max_frames} frames processed")
            print(f"Loops detected so far: {len(stats['trigger_loops'])}")

    # Print final statistics
    print_final_statistics(stats)


def run_custom_dataset_example():
    """Example for custom dataset with PCD files"""

    config = ConfigSetting()
    # Adjust config for indoor/custom environment
    config.voxel_size = 0.5
    config.proj_plane_num = 2
    config.useful_corner_num = 300
    config.skip_near_num = 50

    btc_manager = BTCDescManager(config)
    btc_manager.print_debug_info = True

    # Your custom data paths
    pcd_files = [
        "frame_000.pcd",
        "frame_001.pcd",
        # ... add your PCD files
    ]

    try:
        import open3d as o3d
    except ImportError:
        print("Open3D required for PCD files. Install with: pip install open3d")
        return

    for frame_id, pcd_file in enumerate(pcd_files):
        if not os.path.exists(pcd_file):
            print(f"File not found: {pcd_file}")
            continue

        # Load point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)

        # Add intensity if not present
        if points.shape[1] == 3:
            points = np.column_stack([points, np.ones(len(points))])

        print(f"Processing frame {frame_id}: {len(points)} points")

        # Generate descriptors
        btc_list = btc_manager.generate_btc_descs(points, frame_id)
        print(f"Generated {len(btc_list)} BTC descriptors")

        # Search for loops
        if frame_id > config.skip_near_num:
            loop_id, loop_score, R, t, matches = btc_manager.search_loop(btc_list)

            if loop_id >= 0:
                print(f"Loop closure detected: frame {frame_id} <-> frame {loop_id}")
                print(f"Loop score: {loop_score:.3f}")
                print(f"Transformation:")
                print(f"  Rotation:\n{R}")
                print(f"  Translation: {t}")

        # Add to database
        btc_manager.add_btc_descs(btc_list)

        # Store cloud for overlap calculation
        downsampled = btc_manager.down_sampling_voxel(points, 0.3)
        btc_manager.key_cloud_vec.append(downsampled)


def print_final_statistics(stats):
    """Print comprehensive statistics"""
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    total_frames = len(stats['descriptor_times'])
    trigger_loops = len(stats['trigger_loops'])
    true_loops = len(stats['true_loops'])
    false_loops = len(stats['false_loops'])

    print(f"Total frames processed: {total_frames}")
    print(f"Total loop closures detected: {trigger_loops}")
    print(f"True positives: {true_loops}")
    print(f"False positives: {false_loops}")

    if trigger_loops > 0:
        precision = true_loops / trigger_loops
        print(f"Precision: {precision:.3f}")

    # Timing statistics
    if stats['descriptor_times']:
        mean_desc_time = np.mean(stats['descriptor_times'])
        mean_query_time = np.mean(stats['query_times'])
        mean_update_time = np.mean(stats['update_times'])
        total_time = mean_desc_time + mean_query_time + mean_update_time

        print(f"\nTiming Statistics (ms):")
        print(f"  Descriptor extraction: {mean_desc_time:.2f} ± {np.std(stats['descriptor_times']):.2f}")
        print(f"  Loop query: {mean_query_time:.2f} ± {np.std(stats['query_times']):.2f}")
        print(f"  Database update: {mean_update_time:.2f} ± {np.std(stats['update_times']):.2f}")
        print(f"  Total per frame: {total_time:.2f}")

    # Loop details
    if stats['trigger_loops']:
        print(f"\nDetected Loops:")
        for i, (curr, matched, score) in enumerate(stats['trigger_loops'][:10]):  # Show first 10
            status = "TP" if any(curr == tp[0] for tp in stats['true_loops']) else "FP"
            print(f"  {i + 1}: Frame {curr} -> {matched} (score: {score:.3f}) [{status}]")

        if len(stats['trigger_loops']) > 10:
            print(f"  ... and {len(stats['trigger_loops']) - 10} more")


def create_sample_config():
    """Create sample configuration files"""

    # Outdoor config
    outdoor_config = """# BTC Configuration for Outdoor Environments
useful_corner_num: 500
plane_detection_thre: 0.01
plane_merge_normal_thre: 0.1
plane_merge_dis_thre: 0.3
voxel_size: 2.0
voxel_init_num: 10
proj_plane_num: 1
proj_image_resolution: 0.5
proj_image_high_inc: 0.1
proj_dis_min: -1.0
proj_dis_max: 4.0
summary_min_thre: 10
line_filter_enable: 1

descriptor_near_num: 10
descriptor_min_len: 2.0
descriptor_max_len: 50.0
max_constrait_dis: 2.0
triangle_resolution: 0.2

skip_near_num: 100
candidate_num: 50
similarity_threshold: 0.7
rough_dis_threshold: 0.01
normal_threshold: 0.2
dis_threshold: 0.5
icp_threshold: 0.2
"""

    # Indoor config
    indoor_config = """# BTC Configuration for Indoor Environments  
useful_corner_num: 500
plane_detection_thre: 0.01
plane_merge_normal_thre: 0.1
plane_merge_dis_thre: 0.3
voxel_size: 0.5
voxel_init_num: 10
proj_plane_num: 2
proj_image_resolution: 0.2
proj_image_high_inc: 0.1
proj_dis_min: -1.0
proj_dis_max: 4.0
summary_min_thre: 6
line_filter_enable: 0

descriptor_near_num: 15
descriptor_min_len: 1.0
descriptor_max_len: 30.0
max_constrait_dis: 1.0
triangle_resolution: 0.2

skip_near_num: 100
candidate_num: 50
similarity_threshold: 0.7
rough_dis_threshold: 0.01
normal_threshold: 0.2
dis_threshold: 0.5
icp_threshold: 0.2
"""

    with open("config_outdoor.yaml", "w") as f:
        f.write(outdoor_config)

    with open("config_indoor.yaml", "w") as f:
        f.write(indoor_config)

    print("Created config_outdoor.yaml and config_indoor.yaml")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BTC Place Recognition Example")
    parser.add_argument("--mode", choices=["kitti", "custom", "config"],
                        default="kitti", help="Running mode")
    parser.add_argument("--data_dir", type=str,
                        help="Path to data directory")
    parser.add_argument("--pose_file", type=str,
                        help="Path to pose file")
    parser.add_argument("--config", type=str, default="config_outdoor.yaml",
                        help="Path to config file")

    args = parser.parse_args()

    if args.mode == "config":
        create_sample_config()
    elif args.mode == "kitti":
        if args.data_dir:
            # Update paths in the example
            print(f"Running KITTI example with data from: {args.data_dir}")
        run_kitti_example()
    elif args.mode == "custom":
        run_custom_dataset_example()
    else:
        print("Invalid mode. Use --help for options.")

# Requirements and Installation Instructions:
"""
Required packages:
pip install numpy scipy open3d pyyaml

Optional for visualization:
pip install matplotlib

File structure:
├── btc_python.py          # Main BTC implementation
├── usage_example.py       # This usage example
├── config_outdoor.yaml    # Configuration for outdoor scenes
├── config_indoor.yaml     # Configuration for indoor scenes
├── data/
│   ├── sequences/00/velodyne/  # KITTI bin files
│   └── poses/00.txt            # Ground truth poses
└── results/               # Output directory (optional)

Usage:
1. Basic KITTI example:
   python usage_example.py --mode kitti --data_dir /path/to/kitti

2. Custom dataset:
   python usage_example.py --mode custom

3. Create config files:
   python usage_example.py --mode config

Data format:
- Point clouds: KITTI .bin format or PCD files
- Poses: EVO format (timestamp x y z qx qy qz qw)
- Config: YAML format with all parameters

Performance notes:
- Indoor scenes: Use smaller voxel_size (0.5), more proj_plane_num (2)
- Outdoor scenes: Use larger voxel_size (2.0), fewer proj_plane_num (1)
- Adjust skip_near_num based on frame rate and vehicle speed
- For real-time: reduce useful_corner_num and candidate_num
"""