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
        # CRITICAL: Match C++ logic exactly
        for j in range(3):
            if loc_xyz[j] < 0:
                loc_xyz[j] -= 1.0
        loc_xyz = np.floor(loc_xyz).astype(np.int64)
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
    """Analyze differences in plane detection with both original and merged planes"""
    print(f"\n=== Analyzing Plane Detection for Frame {frame_id} ===")

    # Generate Python planes and get internal data
    btc_list = python_manager.generate_btc_descs(points, frame_id)

    # Get original planes (stored in plane_cloud_vec)
    python_original_planes = python_manager.plane_cloud_vec[-1] if python_manager.plane_cloud_vec else []

    # Get merged planes by re-running the process
    voxel_map = python_manager.init_voxel_map(points)
    python_merged_planes_objs = python_manager.get_projection_planes(voxel_map)

    # Convert merged plane objects to array format
    python_merged_planes = []
    for plane in python_merged_planes_objs:
        plane_point = np.concatenate([plane.center, plane.normal])
        python_merged_planes.append(plane_point)

    # Load C++ planes - try both original and merged files
    cpp_original_file = os.path.join(cpp_results_dir, f"frame_{frame_id:06d}_planes_original.txt")
    cpp_merged_file = os.path.join(cpp_results_dir, f"frame_{frame_id:06d}_planes_merged.txt")

    # Try to load original planes first, fallback to detailed file
    if os.path.exists(cpp_original_file):
        cpp_original_planes = load_cpp_detailed_file(cpp_original_file, 'planes')
    else:
        # Fallback to the detailed file (which should be original planes based on C++ code)
        cpp_detailed_file = os.path.join(cpp_results_dir, f"frame_{frame_id:06d}_planes_detailed.txt")
        cpp_original_planes = load_cpp_detailed_file(cpp_detailed_file, 'planes')

    # Load merged planes if available
    cpp_merged_planes = []
    if os.path.exists(cpp_merged_file):
        cpp_merged_planes = load_cpp_detailed_file(cpp_merged_file, 'planes')

    print(f"Python original planes: {len(python_original_planes)}")
    print(f"Python merged planes: {len(python_merged_planes)}")
    print(f"C++ original planes: {len(cpp_original_planes)}")
    if cpp_merged_planes:
        print(f"C++ merged planes: {len(cpp_merged_planes)}")
    else:
        print("C++ merged planes: Not available")

    # Analyze original planes comparison
    if len(python_original_planes) > 0 and len(cpp_original_planes) > 0:
        python_orig_centers = np.array([[p[0], p[1], p[2]] for p in python_original_planes])
        python_orig_normals = np.array([[p[3], p[4], p[5]] for p in python_original_planes])

        cpp_orig_centers = np.array([[p[0], p[1], p[2]] for p in cpp_original_planes])
        cpp_orig_normals = np.array([[p[3], p[4], p[5]] for p in cpp_original_planes])

        print(f"\n=== ORIGINAL PLANES COMPARISON ===")
        print(
            f"Python original centers - Mean: {np.mean(python_orig_centers, axis=0)}, Std: {np.std(python_orig_centers, axis=0)}")
        print(
            f"C++ original centers - Mean: {np.mean(cpp_orig_centers, axis=0)}, Std: {np.std(cpp_orig_centers, axis=0)}")

        print(
            f"Python original normals - Mean: {np.mean(python_orig_normals, axis=0)}, Std: {np.std(python_orig_normals, axis=0)}")
        print(
            f"C++ original normals - Mean: {np.mean(cpp_orig_normals, axis=0)}, Std: {np.std(cpp_orig_normals, axis=0)}")

        # Find matches for original planes
        from scipy.spatial.distance import cdist
        orig_distances = cdist(python_orig_centers, cpp_orig_centers)

        orig_matches = 0
        for i in range(min(10, len(python_orig_centers))):
            closest_idx = np.argmin(orig_distances[i])
            closest_dist = orig_distances[i, closest_idx]

            if closest_dist < 1.0:
                orig_matches += 1
                normal_diff = np.linalg.norm(python_orig_normals[i] - cpp_orig_normals[closest_idx])
                print(f"Original Match {i}: distance={closest_dist:.3f}, normal_diff={normal_diff:.3f}")

        print(f"Original plane matches within 1m: {orig_matches}/{min(10, len(python_orig_centers))}")

    # Analyze merged planes comparison if available
    if len(python_merged_planes) > 0 and len(cpp_merged_planes) > 0:
        python_merged_centers = np.array([[p[0], p[1], p[2]] for p in python_merged_planes])
        python_merged_normals = np.array([[p[3], p[4], p[5]] for p in python_merged_planes])

        cpp_merged_centers = np.array([[p[0], p[1], p[2]] for p in cpp_merged_planes])
        cpp_merged_normals = np.array([[p[3], p[4], p[5]] for p in cpp_merged_planes])

        print(f"\n=== MERGED PLANES COMPARISON ===")
        print(
            f"Python merged centers - Mean: {np.mean(python_merged_centers, axis=0)}, Std: {np.std(python_merged_centers, axis=0)}")
        print(
            f"C++ merged centers - Mean: {np.mean(cpp_merged_centers, axis=0)}, Std: {np.std(cpp_merged_centers, axis=0)}")

        print(
            f"Python merged normals - Mean: {np.mean(python_merged_normals, axis=0)}, Std: {np.std(python_merged_normals, axis=0)}")
        print(
            f"C++ merged normals - Mean: {np.mean(cpp_merged_normals, axis=0)}, Std: {np.std(cpp_merged_normals, axis=0)}")

        # Find matches for merged planes
        merged_distances = cdist(python_merged_centers, cpp_merged_centers)

        merged_matches = 0
        for i in range(min(10, len(python_merged_centers))):
            closest_idx = np.argmin(merged_distances[i])
            closest_dist = merged_distances[i, closest_idx]

            if closest_dist < 1.0:
                merged_matches += 1
                normal_diff = np.linalg.norm(python_merged_normals[i] - cpp_merged_normals[closest_idx])
                print(f"Merged Match {i}: distance={closest_dist:.3f}, normal_diff={normal_diff:.3f}")

        print(f"Merged plane matches within 1m: {merged_matches}/{min(10, len(python_merged_centers))}")

    return python_original_planes, cpp_original_planes, python_merged_planes, cpp_merged_planes


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
    configs['smaller_voxel'].voxel_size = 1.5
    configs['larger_threshold'].plane_detection_thre = 0.02
    configs['more_corners'].useful_corner_num = 1000

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
    py_orig_planes, cpp_orig_planes, py_merged_planes, cpp_merged_planes = analyze_plane_detection_differences(points,
                                                                                                               manager,
                                                                                                               cpp_results_dir,
                                                                                                               test_frame)
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

    # Updated recommendations based on new analysis
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. Main differences are in plane detection and merging")
    print("2. Check voxelization implementation - ensure identical grid alignment")
    print("3. Verify plane detection eigenvalue computation")
    print("4. Review plane merging criteria and order of operations")
    print("5. Check non-maximum suppression radius and implementation")
    print("6. Verify binary descriptor projection and quantization")

    # Updated plane analysis recommendations
    if len(py_orig_planes) > len(cpp_orig_planes):
        print("7. Python generates MORE original planes - check plane detection threshold")
    elif len(py_orig_planes) < len(cpp_orig_planes):
        print("7. Python generates FEWER original planes - check plane detection threshold")
    else:
        print("7. Original plane counts match - focus on merging differences")

    if len(py_merged_planes) != len(cpp_merged_planes):
        print("8. Merged plane counts differ - check plane merging logic")

    if len(python_binaries) != len(cpp_binaries):
        print("9. Binary descriptor counts differ - check projection plane selection")

    print(f"10. Performance optimization needed - Python is ~{python_time / 10:.0f}x slower")


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
