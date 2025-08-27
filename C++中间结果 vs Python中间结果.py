#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison analysis script for BTC Python vs C++ implementation
"""

import numpy as np
import json
import pickle
import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


class BTCComparison:
    def __init__(self, python_dir, cpp_dir):
        self.python_dir = Path(python_dir)
        self.cpp_dir = Path(cpp_dir)
        self.comparison_results = {}

    def load_json_safe(self, filepath):
        """Load JSON file safely"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def load_pickle_safe(self, filepath):
        """Load pickle file safely"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def compare_configurations(self):
        """Compare configuration settings"""
        print("Comparing configurations...")

        # Load Python config
        python_config_file = self.python_dir / "frame_-00001_stage_config.json"
        cpp_config_file = self.cpp_dir / "frame_-00001_stage_config.json"

        python_config = self.load_json_safe(python_config_file)
        cpp_config = self.load_json_safe(cpp_config_file)

        if not python_config or not cpp_config:
            print("❌ Could not load configuration files")
            return False

        config_match = True
        config_diffs = {}

        # Compare key configuration parameters
        key_params = [
            'voxel_size', 'useful_corner_num', 'plane_detection_thre',
            'proj_plane_num', 'proj_image_resolution', 'similarity_threshold',
            'skip_near_num', 'descriptor_near_num', 'descriptor_min_len',
            'descriptor_max_len'
        ]

        for param in key_params:
            # Handle different naming conventions
            py_key = param
            cpp_key = param + '_'  # C++ version adds underscore

            if py_key in python_config and cpp_key in cpp_config:
                py_val = python_config[py_key]
                cpp_val = cpp_config[cpp_key]

                if abs(py_val - cpp_val) > 1e-6:
                    config_match = False
                    config_diffs[param] = {'python': py_val, 'cpp': cpp_val}
                    print(f"❌ {param}: Python={py_val}, C++={cpp_val}")
                else:
                    print(f"✅ {param}: {py_val}")

        self.comparison_results['config'] = {
            'match': config_match,
            'differences': config_diffs
        }

        return config_match

    def compare_poses(self):
        """Compare pose loading"""
        print("\nComparing poses...")

        python_poses_file = self.python_dir / "frame_-00001_stage_poses.json"
        cpp_poses_file = self.cpp_dir / "frame_-00001_stage_poses.json"

        python_poses = self.load_json_safe(python_poses_file)
        cpp_poses = self.load_json_safe(cpp_poses_file)

        if not python_poses or not cpp_poses:
            print("❌ Could not load pose files")
            return False

        if len(python_poses) != len(cpp_poses):
            print(f"❌ Pose count mismatch: Python={len(python_poses)}, C++={len(cpp_poses)}")
            return False

        poses_match = True
        max_trans_error = 0
        max_rot_error = 0

        for i, (py_pose, cpp_pose) in enumerate(zip(python_poses, cpp_poses)):
            # Compare translations
            py_trans = np.array(py_pose['translation'])
            cpp_trans = np.array(cpp_pose['translation'])
            trans_error = np.linalg.norm(py_trans - cpp_trans)
            max_trans_error = max(max_trans_error, trans_error)

            # Compare rotations
            py_rot = np.array(py_pose['rotation'])
            cpp_rot = np.array(cpp_pose['rotation'])
            rot_error = np.linalg.norm(py_rot - cpp_rot)
            max_rot_error = max(max_rot_error, rot_error)

            if trans_error > 1e-6 or rot_error > 1e-6:
                poses_match = False
                if i < 3:  # Print first few errors
                    print(f"❌ Frame {i}: trans_error={trans_error:.2e}, rot_error={rot_error:.2e}")

        if poses_match:
            print("✅ All poses match")
        else:
            print(f"❌ Poses differ: max_trans_error={max_trans_error:.2e}, max_rot_error={max_rot_error:.2e}")

        self.comparison_results['poses'] = {
            'match': poses_match,
            'max_translation_error': max_trans_error,
            'max_rotation_error': max_rot_error
        }

        return poses_match

    def compare_frame_processing(self, max_frames=10):
        """Compare frame-by-frame processing results"""
        print(f"\nComparing frame processing (up to {max_frames} frames)...")

        frame_results = []
        stages = [
            '1_raw_pointcloud',
            '2_transformed_pointcloud',
            '3a_voxels_and_planes',
            '3b_binary_descriptors',
            '3c_btc_descriptors',
            '4_loop_detection',
            '5_database_update'
        ]

        for frame_id in range(max_frames):
            print(f"\n--- Frame {frame_id} ---")
            frame_result = {'frame_id': frame_id}

            for stage in stages:
                py_file = self.python_dir / f"frame_{frame_id:06d}_stage_{stage}.json"
                cpp_file = self.cpp_dir / f"frame_{frame_id:06d}_stage_{stage}.json"

                if not py_file.exists() or not cpp_file.exists():
                    print(f"⚠️  {stage}: Missing files")
                    frame_result[stage] = {'status': 'missing'}
                    continue

                py_data = self.load_json_safe(py_file)
                cpp_data = self.load_json_safe(cpp_file)

                if not py_data or not cpp_data:
                    print(f"⚠️  {stage}: Could not load data")
                    frame_result[stage] = {'status': 'load_error'}
                    continue

                # Compare stage-specific metrics
                stage_result = self.compare_stage_data(stage, py_data, cpp_data)
                frame_result[stage] = stage_result

                if stage_result['match']:
                    print(f"✅ {stage}: Match")
                else:
                    print(f"❌ {stage}: Differences found")
                    for key, diff in stage_result.get('differences', {}).items():
                        print(f"   {key}: Python={diff.get('python', 'N/A')}, C++={diff.get('cpp', 'N/A')}")

            frame_results.append(frame_result)

        self.comparison_results['frames'] = frame_results
        return frame_results

    def compare_stage_data(self, stage, py_data, cpp_data):
        """Compare data for a specific processing stage"""
        result = {'match': True, 'differences': {}}
        tolerance = 1e-6

        if stage == '1_raw_pointcloud':
            # Compare point cloud loading
            keys = ['points_count', 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']
            for key in keys:
                if key in py_data and key in cpp_data:
                    py_val = py_data[key]
                    cpp_val = cpp_data[key]
                    if abs(py_val - cpp_val) > tolerance:
                        result['match'] = False
                        result['differences'][key] = {'python': py_val, 'cpp': cpp_val}

        elif stage == '2_transformed_pointcloud':
            # Compare transformation results
            keys = ['min_x_transformed', 'max_x_transformed', 'min_y_transformed',
                    'max_y_transformed', 'min_z_transformed', 'max_z_transformed']
            for key in keys:
                if key in py_data and key in cpp_data:
                    py_val = py_data[key]
                    cpp_val = cpp_data[key]
                    if abs(py_val - cpp_val) > tolerance:
                        result['match'] = False
                        result['differences'][key] = {'python': py_val, 'cpp': cpp_val}

        elif stage == '3a_voxels_and_planes':
            # Compare voxelization and plane detection
            keys = ['plane_count']
            for key in keys:
                if key in py_data and key in cpp_data:
                    py_val = py_data[key]
                    cpp_val = cpp_data[key]
                    if abs(py_val - cpp_val) > tolerance:
                        result['match'] = False
                        result['differences'][key] = {'python': py_val, 'cpp': cpp_val}

        elif stage == '3b_binary_descriptors':
            # Compare binary descriptor extraction
            keys = ['binary_descriptors_count', 'min_summary', 'max_summary', 'mean_summary']
            for key in keys:
                if key in py_data and key in cpp_data:
                    py_val = py_data[key]
                    cpp_val = cpp_data[key]
                    if abs(py_val - cpp_val) > tolerance:
                        result['match'] = False
                        result['differences'][key] = {'python': py_val, 'cpp': cpp_val}

        elif stage == '3c_btc_descriptors':
            # Compare BTC descriptor generation
            keys = ['btc_count', 'min_side_length', 'max_side_length', 'mean_side_length']
            for key in keys:
                if key in py_data and key in cpp_data:
                    py_val = py_data[key]
                    cpp_val = cpp_data[key]
                    if abs(py_val - cpp_val) > tolerance:
                        result['match'] = False
                        result['differences'][key] = {'python': py_val, 'cpp': cpp_val}

        elif stage == '4_loop_detection':
            # Compare loop detection results
            keys = ['loop_detected', 'loop_id', 'loop_score', 'matches_count']
            for key in keys:
                if key in py_data and key in cpp_data:
                    py_val = py_data[key]
                    cpp_val = cpp_data[key]
                    if key == 'loop_detected' or key == 'loop_id':
                        # Exact match for boolean and integer values
                        if py_val != cpp_val:
                            result['match'] = False
                            result['differences'][key] = {'python': py_val, 'cpp': cpp_val}
                    else:
                        # Tolerance for floating point values
                        if abs(py_val - cpp_val) > tolerance:
                            result['match'] = False
                            result['differences'][key] = {'python': py_val, 'cpp': cpp_val}

        elif stage == '5_database_update':
            # Compare database update
            keys = ['database_entries', 'key_clouds_count', 'downsampled_cloud_size']
            for key in keys:
                if key in py_data and key in cpp_data:
                    py_val = py_data[key]
                    cpp_val = cpp_data[key]
                    if abs(py_val - cpp_val) > tolerance:
                        result['match'] = False
                        result['differences'][key] = {'python': py_val, 'cpp': cpp_val}

        return result

    def compare_detailed_files(self, max_frames=5):
        """Compare detailed output files for key frames"""
        print(f"\nComparing detailed files (up to {max_frames} frames)...")

        detailed_results = {}

        for frame_id in range(0, max_frames * 10, 10):  # Check every 10th frame
            print(f"\n--- Detailed comparison for Frame {frame_id} ---")

            # Compare binary descriptors
            py_binary_file = self.python_dir / f"frame_{frame_id:06d}_binary_detailed.txt"
            cpp_binary_file = self.cpp_dir / f"frame_{frame_id:06d}_binary_detailed.txt"

            if py_binary_file.exists() and cpp_binary_file.exists():
                binary_match = self.compare_text_files(py_binary_file, cpp_binary_file, 'binary')
                print(f"Binary descriptors: {'✅ Match' if binary_match else '❌ Differ'}")
            else:
                print("⚠️  Binary descriptor files missing")
                binary_match = None

            # Compare BTC descriptors
            py_btc_file = self.python_dir / f"frame_{frame_id:06d}_btc_detailed.txt"
            cpp_btc_file = self.cpp_dir / f"frame_{frame_id:06d}_btc_detailed.txt"

            if py_btc_file.exists() and cpp_btc_file.exists():
                btc_match = self.compare_text_files(py_btc_file, cpp_btc_file, 'btc')
                print(f"BTC descriptors: {'✅ Match' if btc_match else '❌ Differ'}")
            else:
                print("⚠️  BTC descriptor files missing")
                btc_match = None

            # Compare planes
            py_planes_file = self.python_dir / f"frame_{frame_id:06d}_planes_detailed.txt"
            cpp_planes_file = self.cpp_dir / f"frame_{frame_id:06d}_planes_detailed.txt"

            if py_planes_file.exists() and cpp_planes_file.exists():
                planes_match = self.compare_text_files(py_planes_file, cpp_planes_file, 'planes')
                print(f"Planes: {'✅ Match' if planes_match else '❌ Differ'}")
            else:
                print("⚠️  Planes files missing")
                planes_match = None

            detailed_results[frame_id] = {
                'binary_match': binary_match,
                'btc_match': btc_match,
                'planes_match': planes_match
            }

        self.comparison_results['detailed_files'] = detailed_results
        return detailed_results

    def compare_text_files(self, file1, file2, file_type):
        """Compare two text files with numerical tolerance"""
        try:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                lines1 = [line.strip() for line in f1 if not line.startswith('#')]
                lines2 = [line.strip() for line in f2 if not line.startswith('#')]

            if len(lines1) != len(lines2):
                print(f"   Line count mismatch: {len(lines1)} vs {len(lines2)}")
                return False

            tolerance = 1e-4  # More relaxed tolerance for detailed comparisons
            mismatches = 0

            for i, (line1, line2) in enumerate(zip(lines1[:50], lines2[:50])):  # Compare first 50 lines
                if line1 == line2:
                    continue

                # Try numerical comparison
                try:
                    values1 = [float(x) for x in line1.split()]
                    values2 = [float(x) for x in line2.split()]

                    if len(values1) != len(values2):
                        mismatches += 1
                        continue

                    numerical_match = all(abs(v1 - v2) < tolerance for v1, v2 in zip(values1, values2))
                    if not numerical_match:
                        mismatches += 1
                        if mismatches <= 3:  # Show first few mismatches
                            print(f"   Line {i + 1}: numerical mismatch")
                            print(f"     Python: {line1[:100]}")
                            print(f"     C++:    {line2[:100]}")

                except ValueError:
                    # Non-numerical comparison
                    if line1 != line2:
                        mismatches += 1

            mismatch_rate = mismatches / len(lines1) if lines1 else 0
            print(f"   Mismatch rate: {mismatch_rate:.1%} ({mismatches}/{len(lines1)})")

            return mismatch_rate < 0.1  # Allow up to 10% mismatch due to floating point precision

        except Exception as e:
            print(f"   Error comparing files: {e}")
            return False

    def compare_final_statistics(self):
        """Compare final statistics"""
        print("\nComparing final statistics...")

        py_stats_file = self.python_dir / "final_statistics.json"
        cpp_stats_file = self.cpp_dir / "final_statistics.json"

        py_stats = self.load_json_safe(py_stats_file)
        cpp_stats = self.load_json_safe(cpp_stats_file)

        if not py_stats or not cpp_stats:
            print("❌ Could not load final statistics")
            return False

        stats_match = True

        # Compare key metrics
        key_metrics = ['processed_frames', 'total_loops', 'true_loops', 'false_loops', 'precision']

        for metric in key_metrics:
            if metric in py_stats and metric in cpp_stats:
                py_val = py_stats[metric]
                cpp_val = cpp_stats[metric]

                if isinstance(py_val, (int, float)) and isinstance(cpp_val, (int, float)):
                    if abs(py_val - cpp_val) > 1e-6:
                        stats_match = False
                        print(f"❌ {metric}: Python={py_val}, C++={cpp_val}")
                    else:
                        print(f"✅ {metric}: {py_val}")
                else:
                    if py_val != cpp_val:
                        stats_match = False
                        print(f"❌ {metric}: Python={py_val}, C++={cpp_val}")
                    else:
                        print(f"✅ {metric}: {py_val}")

        # Compare timing statistics
        if 'timing' in py_stats and 'timing' in cpp_stats:
            py_timing = py_stats['timing']
            cpp_timing = cpp_stats['timing']

            timing_keys = ['mean_descriptor_time', 'mean_query_time', 'mean_update_time']
            for key in timing_keys:
                if key in py_timing and key in cpp_timing:
                    py_val = py_timing[key]
                    cpp_val = cpp_timing[key]
                    ratio = py_val / cpp_val if cpp_val > 0 else float('inf')
                    print(f"⏱️  {key}: Python={py_val:.2f}ms, C++={cpp_val:.2f}ms (ratio: {ratio:.2f})")

        self.comparison_results['final_stats'] = {
            'match': stats_match,
            'python': py_stats,
            'cpp': cpp_stats
        }

        return stats_match

    def generate_comparison_report(self, output_file="comparison_report.html"):
        """Generate HTML comparison report"""
        print(f"\nGenerating comparison report: {output_file}")

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BTC Implementation Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .match {{ color: green; font-weight: bold; }}
        .mismatch {{ color: red; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .stats-table {{ width: 100%; border-collapse: collapse; }}
        .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .stats-table th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BTC Implementation Comparison Report</h1>
        <p>Comparison between Python and C++ implementations</p>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

        # Configuration comparison
        config_result = self.comparison_results.get('config', {})
        html_content += f"""
    <div class="section">
        <h2>Configuration Comparison</h2>
        <p class="{'match' if config_result.get('match', False) else 'mismatch'}">
            Status: {'✅ MATCH' if config_result.get('match', False) else '❌ MISMATCH'}
        </p>
"""

        if config_result.get('differences'):
            html_content += "<h3>Configuration Differences:</h3><ul>"
            for param, diff in config_result['differences'].items():
                html_content += f"<li><strong>{param}</strong>: Python={diff['python']}, C++={diff['cpp']}</li>"
            html_content += "</ul>"

        html_content += "</div>"

        # Final statistics comparison
        final_stats = self.comparison_results.get('final_stats', {})
        if final_stats:
            py_stats = final_stats.get('python', {})
            cpp_stats = final_stats.get('cpp', {})

            html_content += f"""
    <div class="section">
        <h2>Final Statistics Comparison</h2>
        <table class="stats-table">
            <tr><th>Metric</th><th>Python</th><th>C++</th><th>Match</th></tr>
            <tr><td>Processed Frames</td><td>{py_stats.get('processed_frames', 'N/A')}</td><td>{cpp_stats.get('processed_frames', 'N/A')}</td><td>{'✅' if py_stats.get('processed_frames') == cpp_stats.get('processed_frames') else '❌'}</td></tr>
            <tr><td>Total Loops</td><td>{py_stats.get('total_loops', 'N/A')}</td><td>{cpp_stats.get('total_loops', 'N/A')}</td><td>{'✅' if py_stats.get('total_loops') == cpp_stats.get('total_loops') else '❌'}</td></tr>
            <tr><td>True Loops</td><td>{py_stats.get('true_loops', 'N/A')}</td><td>{cpp_stats.get('true_loops', 'N/A')}</td><td>{'✅' if py_stats.get('true_loops') == cpp_stats.get('true_loops') else '❌'}</td></tr>
            <tr><td>False Loops</td><td>{py_stats.get('false_loops', 'N/A')}</td><td>{cpp_stats.get('false_loops', 'N/A')}</td><td>{'✅' if py_stats.get('false_loops') == cpp_stats.get('false_loops') else '❌'}</td></tr>
            <tr><td>Precision</td><td>{py_stats.get('precision', 'N/A'):.3f}</td><td>{cpp_stats.get('precision', 'N/A'):.3f}</td><td>{'✅' if abs(py_stats.get('precision', 0) - cpp_stats.get('precision', 0)) < 1e-6 else '❌'}</td></tr>
        </table>
    </div>
"""

        # Frame-by-frame comparison summary
        frame_results = self.comparison_results.get('frames', [])
        if frame_results:
            html_content += """
    <div class="section">
        <h2>Frame Processing Comparison Summary</h2>
        <table class="stats-table">
            <tr><th>Frame</th><th>Raw Cloud</th><th>Transform</th><th>Planes</th><th>Binary</th><th>BTC</th><th>Loop</th><th>DB Update</th></tr>
"""

            for frame in frame_results:
                frame_id = frame['frame_id']
                stages = ['1_raw_pointcloud', '2_transformed_pointcloud', '3a_voxels_and_planes',
                          '3b_binary_descriptors', '3c_btc_descriptors', '4_loop_detection', '5_database_update']

                html_content += f"<tr><td>{frame_id}</td>"

                for stage in stages:
                    if stage in frame:
                        match = frame[stage].get('match', False)
                        status = '✅' if match else '❌'
                        if frame[stage].get('status') == 'missing':
                            status = '⚠️'
                    else:
                        status = '⚠️'
                    html_content += f"<td>{status}</td>"

                html_content += "</tr>"

            html_content += """
        </table>
    </div>
"""

        # Detailed file comparison
        detailed_results = self.comparison_results.get('detailed_files', {})
        if detailed_results:
            html_content += """
    <div class="section">
        <h2>Detailed File Comparison</h2>
        <table class="stats-table">
            <tr><th>Frame</th><th>Binary Descriptors</th><th>BTC Descriptors</th><th>Planes</th></tr>
"""

            for frame_id, results in detailed_results.items():
                html_content += f"""
            <tr>
                <td>{frame_id}</td>
                <td>{'✅' if results['binary_match'] else '❌' if results['binary_match'] is not None else '⚠️'}</td>
                <td>{'✅' if results['btc_match'] else '❌' if results['btc_match'] is not None else '⚠️'}</td>
                <td>{'✅' if results['planes_match'] else '❌' if results['planes_match'] is not None else '⚠️'}</td>
            </tr>
"""

            html_content += """
        </table>
    </div>
"""

        html_content += """
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Report generated: {output_file}")

    def run_full_comparison(self, max_frames=10):
        """Run full comparison analysis"""
        print("=" * 60)
        print("BTC IMPLEMENTATION COMPARISON ANALYSIS")
        print("=" * 60)

        results_summary = {}

        # 1. Compare configurations
        config_match = self.compare_configurations()
        results_summary['config'] = config_match

        # 2. Compare poses
        poses_match = self.compare_poses()
        results_summary['poses'] = poses_match

        # 3. Compare frame processing
        frame_results = self.compare_frame_processing(max_frames)
        frame_matches = sum(1 for frame in frame_results for stage, result in frame.items()
                            if isinstance(result, dict) and result.get('match', False))
        total_comparisons = sum(1 for frame in frame_results for stage, result in frame.items()
                                if isinstance(result, dict) and 'match' in result)
        frame_match_rate = frame_matches / total_comparisons if total_comparisons > 0 else 0
        results_summary['frames'] = frame_match_rate

        # 4. Compare detailed files
        detailed_results = self.compare_detailed_files()
        detailed_matches = sum(1 for results in detailed_results.values()
                               for match in results.values() if match is True)
        detailed_total = sum(1 for results in detailed_results.values()
                             for match in results.values() if match is not None)
        detailed_match_rate = detailed_matches / detailed_total if detailed_total > 0 else 0
        results_summary['detailed'] = detailed_match_rate

        # 5. Compare final statistics
        final_match = self.compare_final_statistics()
        results_summary['final'] = final_match

        # Generate report
        self.generate_comparison_report()

        # Print summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Configuration Match: {'✅' if config_match else '❌'}")
        print(f"Poses Match: {'✅' if poses_match else '❌'}")
        print(f"Frame Processing Match Rate: {frame_match_rate:.1%}")
        print(f"Detailed Files Match Rate: {detailed_match_rate:.1%}")
        print(f"Final Statistics Match: {'✅' if final_match else '❌'}")

        overall_score = np.mean([
            1.0 if config_match else 0.0,
            1.0 if poses_match else 0.0,
            frame_match_rate,
            detailed_match_rate,
            1.0 if final_match else 0.0
        ])

        print(f"\nOverall Match Score: {overall_score:.1%}")

        if overall_score > 0.95:
            print("🎉 Excellent! Python implementation is highly consistent with C++ version.")
        elif overall_score > 0.85:
            print("✅ Good! Python implementation is mostly consistent with minor differences.")
        elif overall_score > 0.70:
            print("⚠️  Fair. Python implementation has some differences that need investigation.")
        else:
            print("❌ Poor. Significant differences found. Implementation needs review.")

        return results_summary


def main():
    parser = argparse.ArgumentParser(description="Compare BTC Python and C++ implementations")
    parser.add_argument("--python_dir", type=str, default="/home/wzj/pan1/btc_python/results",
                        help="Directory with Python verification results")
    parser.add_argument("--cpp_dir", type=str, default="/home/wzj/pan1/btc_ws/src/btc_descriptor/results",
                        help="Directory with C++ verification results")
    parser.add_argument("--max_frames", type=int, default=10,
                        help="Maximum number of frames to compare")
    parser.add_argument("--report", type=str, default="comparison_report.html",
                        help="Output HTML report filename")

    args = parser.parse_args()

    if not os.path.exists(args.python_dir):
        print(f"❌ Python results directory not found: {args.python_dir}")
        sys.exit(1)

    if not os.path.exists(args.cpp_dir):
        print(f"❌ C++ results directory not found: {args.cpp_dir}")
        sys.exit(1)

    # Run comparison
    comparator = BTCComparison(args.python_dir, args.cpp_dir)
    results = comparator.run_full_comparison(args.max_frames)

    print(f"\nDetailed report saved as: {args.report}")
    print("\nRecommended next steps:")
    print("1. Review the HTML report for detailed differences")
    print("2. Focus on stages with mismatches for debugging")
    print("3. Check parameter sensitivity for small numerical differences")
    print("4. Verify algorithm correctness for major differences")


if __name__ == "__main__":
    main()