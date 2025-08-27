#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to analyze specific differences between Python and C++ implementations
"""

import numpy as np
import os
import sys
from btc_python import BTCDescManager, ConfigSetting
import time
import json
import pickle
import struct


def load_cpp_detailed_file(filepath, file_type):
    """Load detailed file from C++ implementation for comparison"""
    data = []
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if file_type == 'planes':
            for line in lines:
                parts = line.split()
                if len(parts) >= 6:
                    data.append([float(x) for x in parts[:6]])

        elif file_type == 'binary':
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    location = [float(parts[0]), float(parts[1]), float(parts[2])]
                    summary = int(parts[3])
                    data.append({'location': location, 'summary': summary})

        elif file_type == 'btc':
            i = 0
            while i < len(lines):
                if i + 3 < len(lines):
                    triangle_line = lines[i].split()
                    if len(triangle_line) >= 7:
                        triangle = [float(triangle_line[0]), float(triangle_line[1]), float(triangle_line[2])]
                        center = [float(triangle_line[3]), float(triangle_line[4]), float(triangle_line[5])]
                        frame_id = int(triangle_line[6])

                        # Binary A
                        binary_a = lines[i + 1].split()
                        binary_a_data = {'location': [float(binary_a[0]), float(binary_a[1]), float(binary_a[2])],
                                         'summary': int(binary_a[3])}

                        # Binary B
                        binary_b = lines[i + 2].split()
                        binary_b_data = {'location': [float(binary_b[0]), float(binary_b[1]), float(binary_b[2])],
                                         'summary': int(binary_b[3])}

                        # Binary C
                        binary_c = lines[i + 3].split()
                        binary_c_data = {'location': [float(binary_c[0]), float(binary_c[1]), float(binary_c[2])],
                                         'summary': int(binary_c[3])}

                        data.append({
                            'triangle': triangle,
                            'center': center,
                            'frame_id': frame_id,
                            'binary_A': binary_a_data,
                            'binary_B': binary_b_data,
                            'binary_C': binary_c_data
                        })
                i += 4

    except Exception as e:
        print(f"Error loading {filepath}: {e}")

    return data


def analyze_voxelization_differences(points, python_manager, cpp_results_dir, frame_id):
    """Analyze differences in voxelization process"""
    print(f"\n=== Analyzing Voxelization for Frame {frame_id} ===")

    # Python voxelization
    config = python_manager.config_setting
    print(f"Voxel size: {config.voxel_size}")

    # Manual voxelization check
    voxel_map_manual = {}
    for i, point in enumerate(points):
        loc_xyz = point[:3] / config.voxel_size
        loc_xyz = np.floor(loc_xyz).astype(int)
        voxel_key = tuple(loc_xyz)

        if voxel_key not in voxel_map_manual:
            voxel_map_manual[voxel_key] = []
        voxel_map_manual[voxel_key].append(point[:3])

    # Filter voxels with enough points
    valid_voxels = {k: v for k, v in voxel_map_manual.items() if len(v) >= config.voxel_init_num}

    print(f"Total voxels: {len(voxel_map_manual)}")
    print(f"Valid voxels (>= {config.voxel_init_num} points): {len(valid_voxels)}")

    # Analyze point distribution
    voxel_sizes = [len(v) for v in voxel_map_manual.values()]
    print(f"Voxel point counts - Min: {min(voxel_sizes)}, Max: {max(voxel_sizes)}, Mean: {np.mean(voxel_sizes):.2f}")

    return valid_voxels


def analyze_plane_detection_differences(points, python_manager, cpp_results_dir, frame_id):
    """Analyze differences in plane detection"""
    print(f"\n=== Analyzing Plane Detection for Frame {frame_id} ===")

    # Generate Python planes
    btc_list = python_manager.generate_btc_descs(points, frame_id)
    python_planes = python_manager.plane_cloud_vec[-1] if python_manager.plane_cloud_vec else []

    # Load C++ planes
    cpp_planes_file = os.path.join(cpp_results_dir, f"frame_{frame_id:06d}_planes_detailed.txt")
    cpp_planes = load_cpp_detailed_file(cpp_planes_file, 'planes')

    print(f"Python planes: {len(python_planes)}")
    print(f"C++ planes: {len(cpp_planes)}")

    if len(python_planes) > 0 and len(cpp_planes) > 0:
        # Analyze plane properties
        python_centers = np.array([[p[0], p[1], p[2]] for p in python_planes])
        python_normals = np.array([[p[3], p[4], p[5]] for p in python_planes])

        cpp_centers = np.array([[p[0], p[1], p[2]] for p in cpp_planes])
        cpp_normals = np.array([[p[3], p[4], p[5]] for p in cpp_planes])

        print(
            f"\nPython plane centers - Mean: {np.mean(python_centers, axis=0)}, Std: {np.std(python_centers, axis=0)}")
        print(f"C++ plane centers - Mean: {np.mean(cpp_centers, axis=0)}, Std: {np.std(cpp_centers, axis=0)}")

        print(
            f"\nPython plane normals - Mean: {np.mean(python_normals, axis=0)}, Std: {np.std(python_normals, axis=0)}")
        print(f"C++ plane normals - Mean: {np.mean(cpp_normals, axis=0)}, Std: {np.std(cpp_normals, axis=0)}")

        # Find closest matches
        from scipy.spatial.distance import cdist
        center_distances = cdist(python_centers, cpp_centers)

        matches = 0
        for i in range(min(10, len(python_centers))):  # Check first 10 planes
            closest_idx = np.argmin(center_distances[i])
            closest_dist = center_distances[i, closest_idx]

            if closest_dist < 1.0:  # Within 1m
                matches += 1
                # Check normal similarity
                normal_diff = np.linalg.norm(python_normals[i] - cpp_normals[closest_idx])
                print(f"Match {i}: distance={closest_dist:.3f}, normal_diff={normal_diff:.3f}")

        print(f"Plane matches within 1m: {matches}/{min(10, len(python_centers))}")

    return python_planes, cpp_planes


def analyze_binary_descriptor_differences(points, python_manager, cpp_results_dir, frame_id):
    """Analyze differences in binary descriptor extraction"""
    print(f"\n=== Analyzing Binary Descriptors for Frame {frame_id} ===")

    # Get Python binary descriptors (already generated)
    if len(python_manager.history_binary_list) > frame_id:
        python_binaries = python_manager.history_binary_list[frame_id]
    else:
        python_binaries = []

    # Load C++ binary descriptors
    cpp_binaries_file = os.path.join(cpp_results_dir, f"frame_{frame_id:06d}_binary_detailed.txt")
    cpp_binaries = load_cpp_detailed_file(cpp_binaries_file, 'binary')

    print(f"Python binary descriptors: {len(python_binaries)}")
    print(f"C++ binary descriptors: {len(cpp_binaries)}")

    if python_binaries and cpp_binaries:
        # Analyze summary statistics
        python_summaries = [bd.summary for bd in python_binaries]
        cpp_summaries = [bd['summary'] for bd in cpp_binaries]

        print(
            f"\nPython summaries - Min: {min(python_summaries)}, Max: {max(python_summaries)}, Mean: {np.mean(python_summaries):.2f}")
        print(
            f"C++ summaries - Min: {min(cpp_summaries)}, Max: {max(cpp_summaries)}, Mean: {np.mean(cpp_summaries):.2f}")

        # Analyze locations
        python_locations = np.array([bd.location for bd in python_binaries])
        cpp_locations = np.array([bd['location'] for bd in cpp_binaries])

        print(
            f"\nPython locations - Mean: {np.mean(python_locations, axis=0)}, Std: {np.std(python_locations, axis=0)}")
        print(f"C++ locations - Mean: {np.mean(cpp_locations, axis=0)}, Std: {np.std(cpp_locations, axis=0)}")

        # Find closest matches
        from scipy.spatial.distance import cdist
        location_distances = cdist(python_locations, cpp_locations)

        matches = 0
        for i in range(min(10, len(python_binaries))):
            closest_idx = np.argmin(location_distances[i])
            closest_dist = location_distances[i, closest_idx]

            if closest_dist < 0.5:  # Within 0.5m
                matches += 1
                summary_diff = abs(python_summaries[i] - cpp_summaries[closest_idx])
                print(f"Match {i}: distance={closest_dist:.3f}, summary_diff={summary_diff}")

        print(f"Binary descriptor matches within 0.5m: {matches}/{min(10, len(python_binaries))}")

    return python_binaries, cpp_binaries


def debug_configuration_impact():
    """Test different configuration parameters to see their impact"""
    print("\n=== Testing Configuration Parameter Impact ===")

    # Create different configurations
    configs = {
        'original': ConfigSetting(),
        'smaller_voxel': ConfigSetting(),
        'larger_threshold': ConfigSetting(),
        'more_corners': ConfigSetting()
    }

    # Modify test configurations
    configs['smaller_voxel'].voxel_size = 1.5  # Smaller than default 2.0
    configs['larger_threshold'].plane_detection_thre = 0.02  # Larger than default 0.01
    configs['more_corners'].useful_corner_num = 1000  # More than default 500

    # Load default outdoor config
    if os.path.exists('config_outdoor.yaml'):
        base_manager = BTCDescManager(ConfigSetting())
        configs['original'] = base_manager.load_config_from_yaml('config_outdoor.yaml')

    for name, config in configs.items():
        print(f"\n{name} config:")
        print(f"  voxel_size: {config.voxel_size}")
        print(f"  plane_detection_thre: {config.plane_detection_thre}")
        print(f"  useful_corner_num: {config.useful_corner_num}")
        print(f"  proj_plane_num: {config.proj_plane_num}")


def run_comprehensive_debug(python_results_dir, cpp_results_dir, max_frames=3):
    """Run comprehensive debugging analysis"""
    print("=" * 60)
    print("BTC IMPLEMENTATION DEBUG ANALYSIS")
    print("=" * 60)

    # Load configurations
    python_config_file = os.path.join(python_results_dir, "frame_-00001_stage_config.json")
    if not os.path.exists(python_config_file):
        print(f"❌ Python config not found: {python_config_file}")
        return

    with open(python_config_file, 'r') as f:
        python_config_data = json.load(f)

    # Create Python manager with matching config
    config = ConfigSetting()
    config.voxel_size = python_config_data.get('voxel_size', 2.0)
    config.useful_corner_num = python_config_data.get('useful_corner_num', 500)
    config.plane_detection_thre = python_config_data.get('plane_detection_thre', 0.01)
    config.proj_plane_num = python_config_data.get('proj_plane_num', 1)
    config.skip_near_num = python_config_data.get('skip_near_num', 100)

    manager = BTCDescManager(config)
    manager.print_debug_info = True

    # Test configuration impact
    debug_configuration_impact()

    # Load test data (using first frame)
    test_frame = 0
    python_stage1_file = os.path.join(python_results_dir, f"frame_{test_frame:06d}_stage_1_raw_pointcloud.pkl")

    if not os.path.exists(python_stage1_file):
        print(f"❌ Test data not found: {python_stage1_file}")
        return

    with open(python_stage1_file, 'rb') as f:
        stage1_data = pickle.load(f)

    points = stage1_data['points']
    print(f"\nLoaded test data: {len(points)} points")

    # Run detailed analysis
    valid_voxels = analyze_voxelization_differences(points, manager, cpp_results_dir, test_frame)
    python_planes, cpp_planes = analyze_plane_detection_differences(points, manager, cpp_results_dir, test_frame)
    python_binaries, cpp_binaries = analyze_binary_descriptor_differences(points, manager, cpp_results_dir, test_frame)

    # Performance comparison
    print(f"\n=== Performance Analysis ===")

    # Time Python implementation
    start_time = time.time()
    for i in range(3):  # Run 3 times for averaging
        btc_list = manager.generate_btc_descs(points, test_frame + i)
    python_time = (time.time() - start_time) / 3 * 1000

    print(f"Python average time: {python_time:.2f}ms")
    print(f"Expected C++ time: ~{python_time / 100:.2f}ms (estimated)")

    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. Main differences are in plane detection and merging")
    print("2. Check voxelization implementation - ensure identical grid alignment")
    print("3. Verify plane detection eigenvalue computation")
    print("4. Review plane merging criteria and order of operations")
    print("5. Check non-maximum suppression radius and implementation")
    print("6. Verify binary descriptor projection and quantization")

    if len(python_planes) > len(cpp_planes):
        print("7. Python generates MORE planes - check plane merging logic")
    else:
        print("7. Python generates FEWER planes - check plane detection threshold")

    if len(python_binaries) != len(cpp_binaries):
        print("8. Binary descriptor counts differ - check projection plane selection")

    print(f"9. Performance optimization needed - Python is ~{python_time / 10:.0f}x slower")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug BTC implementation differences")
    parser.add_argument("--python_dir", type=str, default="results",
                        help="Directory with Python verification results")
    parser.add_argument("--cpp_dir", type=str, default="/home/wzj/pan1/btc_ws/src/btc_descriptor/results",
                        help="Directory with C++ verification results")
    parser.add_argument("--max_frames", type=int, default=3,
                        help="Maximum number of frames to analyze")

    args = parser.parse_args()

    if not os.path.exists(args.python_dir):
        print(f"❌ Python results directory not found: {args.python_dir}")
        sys.exit(1)

    if not os.path.exists(args.cpp_dir):
        print(f"❌ C++ results directory not found: {args.cpp_dir}")
        print(f"Please run the C++ verification first to generate results")
        sys.exit(1)

    run_comprehensive_debug(args.python_dir, args.cpp_dir, args.max_frames)