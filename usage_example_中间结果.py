#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage example for BTC Place Recognition with intermediate results saving for verification
"""

import numpy as np
import os
import sys
from btc_python import BTCDescManager, ConfigSetting
import time
import json
import pickle


def save_intermediate_results(frame_id, stage, data, output_dir):
    """Save intermediate results for comparison"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"frame_{frame_id:06d}_stage_{stage}"

    def convert_to_json_serializable(obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return {
                'type': 'numpy_array',
                'shape': list(obj.shape),
                'data': obj.tolist()
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Handle list of numpy arrays or other objects
            if len(obj) > 0:
                if isinstance(obj[0], np.ndarray):
                    return {
                        'type': 'numpy_array_list',
                        'count': len(obj),
                        'data': [arr.tolist() for arr in obj[:10]]  # Save first 10 for JSON
                    }
                elif hasattr(obj[0], '__dict__'):
                    return {
                        'type': 'object_list',
                        'count': len(obj),
                        'sample': str(obj[0] if len(obj) > 0 else None)
                    }
                else:
                    return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        else:
            return obj

    if isinstance(data, dict):
        # Save dict as JSON and pickle
        try:
            with open(os.path.join(output_dir, f"{filename}.json"), 'w') as f:
                json_data = convert_to_json_serializable(data)
                json.dump(json_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save JSON for {filename}: {e}")
            # Just save the basic info
            basic_info = {}
            for key, value in data.items():
                try:
                    if isinstance(value, (int, float, str, bool)):
                        basic_info[key] = value
                    elif isinstance(value, np.ndarray):
                        basic_info[key] = f"numpy_array_shape_{value.shape}"
                    elif isinstance(value, list):
                        basic_info[key] = f"list_length_{len(value)}"
                    else:
                        basic_info[key] = str(type(value))
                except:
                    basic_info[key] = "conversion_failed"

            with open(os.path.join(output_dir, f"{filename}.json"), 'w') as f:
                json.dump(basic_info, f, indent=2)

        # Save full data as pickle
        with open(os.path.join(output_dir, f"{filename}.pkl"), 'wb') as f:
            pickle.dump(data, f)
    else:
        # Save as pickle for complex data structures
        with open(os.path.join(output_dir, f"{filename}.pkl"), 'wb') as f:
            pickle.dump(data, f)


def save_point_cloud(cloud, filename):
    """Save point cloud as text file"""
    with open(filename, 'w') as f:
        f.write("# x y z intensity\n")
        for point in cloud:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f}\n")


def save_binary_descriptors(binary_list, filename):
    """Save binary descriptors as text file"""
    with open(filename, 'w') as f:
        f.write("# Binary Descriptors\n")
        f.write("# location_x location_y location_z summary occupy_array_length\n")
        for i, bd in enumerate(binary_list):
            occupy_str = ''.join(['1' if x else '0' for x in bd.occupy_array])
            f.write(f"{bd.location[0]:.6f} {bd.location[1]:.6f} {bd.location[2]:.6f} "
                    f"{bd.summary} {len(bd.occupy_array)} {occupy_str}\n")


def save_btc_descriptors(btc_list, filename):
    """Save BTC descriptors as text file"""
    with open(filename, 'w') as f:
        f.write("# BTC Descriptors\n")
        f.write("# triangle_x triangle_y triangle_z center_x center_y center_z frame_number\n")
        f.write("# binary_A_loc_x binary_A_loc_y binary_A_loc_z binary_A_summary\n")
        f.write("# binary_B_loc_x binary_B_loc_y binary_B_loc_z binary_B_summary\n")
        f.write("# binary_C_loc_x binary_C_loc_y binary_C_loc_z binary_C_summary\n")
        for btc in btc_list:
            f.write(f"{btc.triangle[0]:.6f} {btc.triangle[1]:.6f} {btc.triangle[2]:.6f} "
                    f"{btc.center[0]:.6f} {btc.center[1]:.6f} {btc.center[2]:.6f} "
                    f"{btc.frame_number}\n")
            f.write(f"{btc.binary_A.location[0]:.6f} {btc.binary_A.location[1]:.6f} "
                    f"{btc.binary_A.location[2]:.6f} {btc.binary_A.summary}\n")
            f.write(f"{btc.binary_B.location[0]:.6f} {btc.binary_B.location[1]:.6f} "
                    f"{btc.binary_B.location[2]:.6f} {btc.binary_B.summary}\n")
            f.write(f"{btc.binary_C.location[0]:.6f} {btc.binary_C.location[1]:.6f} "
                    f"{btc.binary_C.location[2]:.6f} {btc.binary_C.summary}\n")


def save_planes(planes, filename):
    """Save plane information as text file"""
    with open(filename, 'w') as f:
        f.write("# Planes\n")
        f.write("# center_x center_y center_z normal_x normal_y normal_z\n")
        for plane in planes:
            f.write(f"{plane[0]:.6f} {plane[1]:.6f} {plane[2]:.6f} "
                    f"{plane[3]:.6f} {plane[4]:.6f} {plane[5]:.6f}\n")


def run_kitti_verification():
    """Run BTC place recognition on KITTI dataset with verification output"""

    # Configuration paths - UPDATE THESE PATHS TO YOUR ACTUAL DATA
    config_file = "config_outdoor.yaml"
    pcds_dir = "/home/wzj/pan1/Data/KITTI/00/velodyne"  # UPDATE THIS PATH
    pose_file = "/home/wzj/pan1/btc_ws/src/btc_descriptor/poses/kitti00.txt"  # UPDATE THIS PATH
    output_dir = "results"  # Output directory for intermediate results

    # Parameters
    cloud_overlap_thr = 0.5
    read_bin = True
    max_frames = 50  # Limit frames for verification

    print("Starting BTC verification with intermediate result saving...")
    print(f"Output directory: {output_dir}")

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

    # Save configuration
    config_data = {
        'voxel_size': config.voxel_size,
        'useful_corner_num': config.useful_corner_num,
        'plane_detection_thre': config.plane_detection_thre,
        'proj_plane_num': config.proj_plane_num,
        'proj_image_resolution': config.proj_image_resolution,
        'similarity_threshold': config.similarity_threshold,
        'skip_near_num': config.skip_near_num,
        'descriptor_near_num': config.descriptor_near_num,
        'descriptor_min_len': config.descriptor_min_len,
        'descriptor_max_len': config.descriptor_max_len,
    }
    save_intermediate_results(-1, "config", config_data, output_dir)

    # Load ground truth poses
    translations, rotations = [], []
    if os.path.exists(pose_file):
        translations, rotations = btc_manager.load_pose_file(pose_file)
        print(f"Successfully loaded {len(translations)} poses")

        # Save poses for verification
        poses_data = []
        for i, (t, r) in enumerate(zip(translations[:max_frames], rotations[:max_frames])):
            poses_data.append({
                'frame_id': i,
                'translation': t.tolist(),
                'rotation': r.tolist()
            })
        save_intermediate_results(-1, "poses", poses_data, output_dir)
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
    process_frames = min(len(translations) if translations else 100, max_frames)

    for frame_id in range(process_frames):
        print(f"\n=== Processing Frame {frame_id} ===")

        # STAGE 1: Load point cloud
        if read_bin:
            bin_file = os.path.join(pcds_dir, f"{frame_id:06d}.bin")
            if not os.path.exists(bin_file):
                print(f"File not found: {bin_file}")
                break
            points = btc_manager.read_kitti_bin(bin_file)
        else:
            try:
                import open3d as o3d
                pcd_file = os.path.join(pcds_dir, f"{frame_id:06d}.pcd")
                if not os.path.exists(pcd_file):
                    print(f"File not found: {pcd_file}")
                    break
                pcd = o3d.io.read_point_cloud(pcd_file)
                points = np.asarray(pcd.points)
                if points.shape[1] == 3:
                    points = np.column_stack([points, np.zeros(len(points))])
            except ImportError:
                print("Open3D not available for PCD reading")
                break

        print(f"Loaded {len(points)} points")

        # Save raw point cloud
        stage1_data = {
            'frame_id': frame_id,
            'points_count': len(points),
            'points': points,
            'bounds': {
                'min_x': float(np.min(points[:, 0])),
                'max_x': float(np.max(points[:, 0])),
                'min_y': float(np.min(points[:, 1])),
                'max_y': float(np.max(points[:, 1])),
                'min_z': float(np.min(points[:, 2])),
                'max_z': float(np.max(points[:, 2]))
            }
        }
        save_intermediate_results(frame_id, "1_raw_pointcloud", stage1_data, output_dir)

        # Apply pose transformation if available
        if frame_id < len(translations):
            translation = translations[frame_id]
            rotation = rotations[frame_id]
            transformed_points = points.copy()
            transformed_points[:, :3] = (rotation @ points[:, :3].T).T + translation

            # Save transformed point cloud
            stage2_data = {
                'frame_id': frame_id,
                'transformation': {
                    'translation': translation.tolist(),
                    'rotation': rotation.tolist()
                },
                'transformed_points': transformed_points,
                'bounds_after_transform': {
                    'min_x': float(np.min(transformed_points[:, 0])),
                    'max_x': float(np.max(transformed_points[:, 0])),
                    'min_y': float(np.min(transformed_points[:, 1])),
                    'max_y': float(np.max(transformed_points[:, 1])),
                    'min_z': float(np.min(transformed_points[:, 2])),
                    'max_z': float(np.max(transformed_points[:, 2]))
                }
            }
            save_intermediate_results(frame_id, "2_transformed_pointcloud", stage2_data, output_dir)
        else:
            transformed_points = points

        # STAGE 3: Descriptor extraction (with internal stages)
        start_time = time.time()

        # Generate BTC descriptors using the main function
        btc_list = btc_manager.generate_btc_descs(transformed_points, frame_id)

        # Get internal data for stage analysis
        # Access the plane data that was stored during generation
        if len(btc_manager.plane_cloud_vec) > 0:
            current_planes = btc_manager.plane_cloud_vec[-1]
            plane_points = [[p[0], p[1], p[2], p[3], p[4], p[5]] for p in current_planes] if len(
                current_planes) > 0 else []
        else:
            plane_points = []

        stage3a_data = {
            'frame_id': frame_id,
            'plane_count': len(plane_points),
            'planes': plane_points
        }
        save_intermediate_results(frame_id, "3a_voxels_and_planes", stage3a_data, output_dir)

        # Get binary descriptors from the history
        if len(btc_manager.history_binary_list) > 0:
            binary_list = btc_manager.history_binary_list[-1]

            stage3b_data = {
                'frame_id': frame_id,
                'binary_descriptors_count': len(binary_list),
                'binary_summary_stats': {
                    'min_summary': min([bd.summary for bd in binary_list]) if binary_list else 0,
                    'max_summary': max([bd.summary for bd in binary_list]) if binary_list else 0,
                    'mean_summary': np.mean([bd.summary for bd in binary_list]) if binary_list else 0
                }
            }
            save_intermediate_results(frame_id, "3b_binary_descriptors", stage3b_data, output_dir)
        else:
            binary_list = []
            stage3b_data = {
                'frame_id': frame_id,
                'binary_descriptors_count': 0,
                'binary_summary_stats': {'min_summary': 0, 'max_summary': 0, 'mean_summary': 0}
            }
            save_intermediate_results(frame_id, "3b_binary_descriptors", stage3b_data, output_dir)

        stage3c_data = {
            'frame_id': frame_id,
            'btc_count': len(btc_list),
            'triangle_stats': {
                'min_side_length': float(min([min(btc.triangle) for btc in btc_list])) if btc_list else 0,
                'max_side_length': float(max([max(btc.triangle) for btc in btc_list])) if btc_list else 0,
                'mean_side_length': float(np.mean([np.mean(btc.triangle) for btc in btc_list])) if btc_list else 0
            }
        }
        save_intermediate_results(frame_id, "3c_btc_descriptors", stage3c_data, output_dir)

        desc_time = (time.time() - start_time) * 1000
        stats['descriptor_times'].append(desc_time)

        print(f"Generated {len(btc_list)} BTC descriptors")

        # STAGE 4: Loop detection
        start_time = time.time()
        loop_result = (-1, 0, np.eye(3), np.zeros(3), [])

        if frame_id > config.skip_near_num and len(btc_list) > 0:
            loop_result = btc_manager.search_loop(btc_list)
            loop_id, loop_score, R, t, matches = loop_result

            stage4_data = {
                'frame_id': frame_id,
                'loop_detected': loop_id >= 0,
                'loop_id': int(loop_id) if loop_id >= 0 else -1,
                'loop_score': float(loop_score),
                'matches_count': len(matches),
                'database_size': len(btc_manager.data_base)
            }

            if loop_id >= 0:
                print(f"Loop detected: {frame_id} -> {loop_id}, score: {loop_score:.3f}")
                stats['trigger_loops'].append((frame_id, loop_id, loop_score))

                # Verify with ground truth overlap
                if frame_id < len(translations):
                    current_cloud = btc_manager.down_sampling_voxel(transformed_points, 0.5)
                    if loop_id < len(btc_manager.key_cloud_vec):
                        matched_cloud = btc_manager.key_cloud_vec[loop_id]
                        overlap = btc_manager.calc_overlap(current_cloud, matched_cloud, cloud_overlap_thr)

                        stage4_data['ground_truth_overlap'] = float(overlap)

                        if overlap >= cloud_overlap_thr:
                            stats['true_loops'].append((frame_id, loop_id, overlap))
                            stage4_data['is_true_positive'] = True
                            print(f"TRUE POSITIVE: overlap = {overlap:.3f}")
                        else:
                            stats['false_loops'].append((frame_id, loop_id, overlap))
                            stage4_data['is_true_positive'] = False
                            print(f"FALSE POSITIVE: overlap = {overlap:.3f}")
        else:
            stage4_data = {
                'frame_id': frame_id,
                'loop_detected': False,
                'skip_reason': 'too_early' if frame_id <= config.skip_near_num else 'no_btc_descriptors'
            }

        save_intermediate_results(frame_id, "4_loop_detection", stage4_data, output_dir)

        query_time = (time.time() - start_time) * 1000
        stats['query_times'].append(query_time)

        # STAGE 5: Update database
        start_time = time.time()
        btc_manager.add_btc_descs(btc_list)
        update_time = (time.time() - start_time) * 1000
        stats['update_times'].append(update_time)

        # Store downsampled cloud
        downsampled_cloud = btc_manager.down_sampling_voxel(transformed_points, 0.5)
        btc_manager.key_cloud_vec.append(downsampled_cloud)

        stage5_data = {
            'frame_id': frame_id,
            'database_entries': len(btc_manager.data_base),
            'key_clouds_count': len(btc_manager.key_cloud_vec),
            'downsampled_cloud_size': len(downsampled_cloud)
        }
        save_intermediate_results(frame_id, "5_database_update", stage5_data, output_dir)

        print(f"Times - Desc: {desc_time:.1f}ms, Query: {query_time:.1f}ms, Update: {update_time:.1f}ms")

        # Save detailed results for key frames
        if frame_id % 10 == 0 or (loop_result[0] >= 0):  # Every 10 frames or when loop detected
            # Save detailed binary descriptors
            binary_file = os.path.join(output_dir, f"frame_{frame_id:06d}_binary_detailed.txt")
            save_binary_descriptors(binary_list, binary_file)

            # Save detailed BTC descriptors
            btc_file = os.path.join(output_dir, f"frame_{frame_id:06d}_btc_detailed.txt")
            save_btc_descriptors(btc_list, btc_file)

            # Save detailed planes
            if plane_points:
                planes_file = os.path.join(output_dir, f"frame_{frame_id:06d}_planes_detailed.txt")
                save_planes(plane_points, planes_file)

        # Progress indicator
        if (frame_id + 1) % 10 == 0:
            print(f"\nProgress: {frame_id + 1}/{process_frames} frames processed")
            print(f"Loops detected so far: {len(stats['trigger_loops'])}")

    # STAGE 6: Save final statistics
    print_final_statistics(stats, output_dir)

    # Save comprehensive final results
    final_results = {
        'processed_frames': process_frames,
        'statistics': stats,
        'configuration': config_data,
        'database_final_size': len(btc_manager.data_base),
        'timing_summary': {
            'mean_descriptor_time': np.mean(stats['descriptor_times']),
            'mean_query_time': np.mean(stats['query_times']),
            'mean_update_time': np.mean(stats['update_times'])
        }
    }
    save_intermediate_results(-1, "final_results", final_results, output_dir)

    print(f"\nVerification complete! Results saved in: {output_dir}")
    return output_dir


def print_final_statistics(stats, output_dir):
    """Print and save comprehensive statistics"""
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    total_frames = len(stats['descriptor_times'])
    trigger_loops = len(stats['trigger_loops'])
    true_loops = len(stats['true_loops'])
    false_loops = len(stats['false_loops'])

    stats_summary = {
        'total_frames': total_frames,
        'trigger_loops': trigger_loops,
        'true_loops': true_loops,
        'false_loops': false_loops,
        'precision': true_loops / trigger_loops if trigger_loops > 0 else 0
    }

    print(f"Total frames processed: {total_frames}")
    print(f"Total loop closures detected: {trigger_loops}")
    print(f"True positives: {true_loops}")
    print(f"False positives: {false_loops}")

    if trigger_loops > 0:
        precision = true_loops / trigger_loops
        print(f"Precision: {precision:.3f}")
        stats_summary['precision'] = precision

    # Timing statistics
    if stats['descriptor_times']:
        timing_stats = {
            'mean_descriptor_time': np.mean(stats['descriptor_times']),
            'std_descriptor_time': np.std(stats['descriptor_times']),
            'mean_query_time': np.mean(stats['query_times']),
            'std_query_time': np.std(stats['query_times']),
            'mean_update_time': np.mean(stats['update_times']),
            'std_update_time': np.std(stats['update_times'])
        }

        print(f"\nTiming Statistics (ms):")
        print(
            f"  Descriptor extraction: {timing_stats['mean_descriptor_time']:.2f} ± {timing_stats['std_descriptor_time']:.2f}")
        print(f"  Loop query: {timing_stats['mean_query_time']:.2f} ± {timing_stats['std_query_time']:.2f}")
        print(f"  Database update: {timing_stats['mean_update_time']:.2f} ± {timing_stats['std_update_time']:.2f}")
        print(
            f"  Total per frame: {timing_stats['mean_descriptor_time'] + timing_stats['mean_query_time'] + timing_stats['mean_update_time']:.2f}")

        stats_summary['timing'] = timing_stats

    # Loop details
    if stats['trigger_loops']:
        print(f"\nDetected Loops:")
        loop_details = []
        for i, (curr, matched, score) in enumerate(stats['trigger_loops'][:10]):  # Show first 10
            status = "TP" if any(curr == tp[0] for tp in stats['true_loops']) else "FP"
            loop_info = f"  {i + 1}: Frame {curr} -> {matched} (score: {score:.3f}) [{status}]"
            print(loop_info)
            loop_details.append({
                'current_frame': curr,
                'matched_frame': matched,
                'score': score,
                'status': status
            })

        if len(stats['trigger_loops']) > 10:
            print(f"  ... and {len(stats['trigger_loops']) - 10} more")

        stats_summary['loop_details'] = loop_details

    # Save statistics to file
    with open(os.path.join(output_dir, "final_statistics.json"), 'w') as f:
        json.dump(stats_summary, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BTC Place Recognition Verification")
    parser.add_argument("--data_dir", type=str, default="/path/to/kitti/sequences/00/velodyne",
                        help="Path to data directory")
    parser.add_argument("--pose_file", type=str, default="/path/to/kitti/poses.txt",
                        help="Path to pose file")
    parser.add_argument("--config", type=str, default="config_outdoor.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="python_verification_results",
                        help="Output directory for intermediate results")
    parser.add_argument("--max_frames", type=int, default=50,
                        help="Maximum number of frames to process")

    args = parser.parse_args()

    print("BTC Python Verification - Intermediate Results Saving")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Pose file: {args.pose_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max frames: {args.max_frames}")

    output_dir = run_kitti_verification()
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Run the C++ verification version")
    print("2. Compare intermediate results using comparison tools")
    print("3. Analyze differences at each processing stage")