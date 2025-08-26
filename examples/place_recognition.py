#!/usr/bin/env python3

import os
import sys
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for PyCharm
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from btc import (
        ConfigSetting,
        load_config_setting,
        BTCDescManager,
        load_evo_pose_with_time,
        read_lidar_data,
        load_point_cloud_from_pcd,
        down_sampling_voxel,
        calc_overlap,
        transform_point_cloud
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure the btc package is properly installed or the path is correct")
    sys.exit(1)


class PlaceRecognitionDemo:
    """Demo class for place recognition using BTC descriptors"""

    def __init__(self, config_path: str, pcds_dir: str, pose_file: str,
                 read_bin: bool = True, cloud_overlap_thr: float = 0.5):

        # Validate paths first
        if not self._validate_paths(config_path, pcds_dir, pose_file):
            raise ValueError("Invalid file paths provided")

        # Load configuration
        try:
            self.config = load_config_setting(config_path)
            print(f"✓ Successfully loaded config: {config_path}")
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")

        self.pcds_dir = pcds_dir
        self.pose_file = pose_file
        self.read_bin = read_bin
        self.cloud_overlap_thr = cloud_overlap_thr

        # Initialize BTC manager
        self.btc_manager = BTCDescManager(self.config)
        self.btc_manager.print_debug_info = True

        # Load poses
        try:
            self.poses, self.timestamps = load_evo_pose_with_time(pose_file)
            print(f"✓ Loaded {len(self.poses)} poses from {pose_file}")
        except Exception as e:
            raise ValueError(f"Failed to load pose file: {e}")

        # Statistics
        self.descriptor_times = []
        self.query_times = []
        self.update_times = []
        self.triggered_loops = 0
        self.true_loops = 0

        # Storage for visualization
        self.point_clouds = []
        self.loop_results = []

    def _validate_paths(self, config_path: str, pcds_dir: str, pose_file: str) -> bool:
        """Validate that all required paths exist"""
        issues = []

        if not os.path.exists(config_path):
            issues.append(f"Config file not found: {config_path}")

        if not os.path.exists(pcds_dir):
            issues.append(f"Point clouds directory not found: {pcds_dir}")

        if not os.path.exists(pose_file):
            issues.append(f"Pose file not found: {pose_file}")

        if issues:
            print("❌ Path validation failed:")
            for issue in issues:
                print(f"   {issue}")
            print("\n💡 Please update the file paths in the main() function")
            return False

        return True

    def load_point_cloud(self, submap_id: int) -> np.ndarray:
        """Load point cloud from file"""
        try:
            if self.read_bin:
                # Load from binary file
                filename = f"{submap_id:06d}.bin"
                file_path = os.path.join(self.pcds_dir, filename)

                if not os.path.exists(file_path):
                    print(f"⚠️  File not found: {file_path}")
                    return np.empty((0, 4))

                points_data = read_lidar_data(file_path)
            else:
                # Load from PCD file
                filename = f"{submap_id:06d}.pcd"
                file_path = os.path.join(self.pcds_dir, filename)

                if not os.path.exists(file_path):
                    print(f"⚠️  File not found: {file_path}")
                    return np.empty((0, 4))

                points_data = load_point_cloud_from_pcd(file_path)

            return points_data

        except Exception as e:
            print(f"❌ Error loading point cloud {submap_id}: {e}")
            return np.empty((0, 4))

    def run_place_recognition(self):
        """Main place recognition loop"""
        print("\n🚀 Starting place recognition...")
        print(f"📁 Processing {len(self.poses)} frames")
        print(f"🔧 Config: skip_near_num={self.config.skip_near_num}, "
              f"icp_threshold={self.config.icp_threshold}")

        try:
            for submap_id in tqdm(range(len(self.poses)), desc="Processing frames"):
                # Load point cloud
                cloud = self.load_point_cloud(submap_id)
                if len(cloud) == 0:
                    if submap_id < 10:  # Only warn for first few frames
                        print(f"⚠️  Skipping frame {submap_id} - no point cloud data")
                    continue

                # Transform point cloud to global frame
                translation, rotation = self.poses[submap_id]
                transformed_cloud = transform_point_cloud(cloud, translation, rotation)

                # Downsample for memory efficiency
                transformed_cloud = down_sampling_voxel(transformed_cloud, 0.5)
                self.point_clouds.append(transformed_cloud)

                # Step 1: Extract BTC descriptors
                start_time = time.time()
                btc_descriptors = self.btc_manager.generate_btc_descriptors(transformed_cloud, submap_id)
                descriptor_time = (time.time() - start_time) * 1000
                self.descriptor_times.append(descriptor_time)

                # Step 2: Search for loops
                start_time = time.time()
                if submap_id > self.config.skip_near_num:
                    loop_result, loop_transform, loop_pairs = self.btc_manager.search_loop(btc_descriptors)
                    loop_id, loop_score = loop_result
                else:
                    loop_id, loop_score = -1, 0.0
                    loop_transform = (np.zeros(3), np.eye(3))
                    loop_pairs = []

                query_time = (time.time() - start_time) * 1000
                self.query_times.append(query_time)

                # Step 3: Add descriptors to database
                start_time = time.time()
                self.btc_manager.add_btc_descriptors(btc_descriptors)
                update_time = (time.time() - start_time) * 1000
                self.update_times.append(update_time)

                # Process loop detection result
                if loop_id >= 0:
                    self.triggered_loops += 1
                    print(f"\n🔄 [Loop Detection] Frame {submap_id} -> Frame {loop_id}, Score: {loop_score:.3f}")

                    # Calculate ground truth overlap
                    if loop_id < len(self.point_clouds):
                        overlap = calc_overlap(transformed_cloud, self.point_clouds[loop_id], 0.5)

                        if overlap >= self.cloud_overlap_thr:
                            self.true_loops += 1
                            print(f"✅ [True Positive] Overlap: {overlap:.3f}")
                        else:
                            print(f"❌ [False Positive] Overlap: {overlap:.3f}")

                        self.loop_results.append({
                            'query_id': submap_id,
                            'match_id': loop_id,
                            'score': loop_score,
                            'overlap': overlap,
                            'is_true_positive': overlap >= self.cloud_overlap_thr
                        })

                # Print timing info
                if submap_id % 20 == 0 and submap_id > 0:
                    print(f"\n📊 [Timing] Frame {submap_id}:")
                    print(f"   Descriptor: {descriptor_time:.1f}ms")
                    print(f"   Query: {query_time:.1f}ms")
                    print(f"   Update: {update_time:.1f}ms")
                    print(f"   Total: {descriptor_time + query_time + update_time:.1f}ms")

        except KeyboardInterrupt:
            print(f"\n⚠️  Process interrupted by user at frame {submap_id}")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()

        self.print_final_statistics()

    def print_final_statistics(self):
        """Print final performance statistics"""
        print("\n" + "=" * 60)
        print("📈 FINAL STATISTICS")
        print("=" * 60)

        total_frames = len(self.descriptor_times)
        print(f"📊 Total frames processed: {total_frames}")
        print(f"🔄 Triggered loop closures: {self.triggered_loops}")
        print(f"✅ True positive loops: {self.true_loops}")

        if self.triggered_loops > 0:
            precision = self.true_loops / self.triggered_loops
            print(f"🎯 Precision: {precision:.3f} ({self.true_loops}/{self.triggered_loops})")
        else:
            print("🎯 Precision: N/A (no loops detected)")

        if total_frames > 0:
            # Timing statistics
            mean_descriptor_time = np.mean(self.descriptor_times)
            mean_query_time = np.mean(self.query_times)
            mean_update_time = np.mean(self.update_times)
            mean_total_time = mean_descriptor_time + mean_query_time + mean_update_time

            print(f"\n⏱️  Average Timing (ms):")
            print(f"   Descriptor extraction: {mean_descriptor_time:.1f}")
            print(f"   Query: {mean_query_time:.1f}")
            print(f"   Database update: {mean_update_time:.1f}")
            print(f"   Total per frame: {mean_total_time:.1f}")
            print(f"   Processing rate: {1000 / mean_total_time:.1f} Hz")

            # Database statistics
            print(f"\n💾 Database Statistics:")
            print(f"   Hash buckets: {len(self.btc_manager.database)}")
            total_descriptors = sum(len(bucket) for bucket in self.btc_manager.database.values())
            print(f"   Total descriptors: {total_descriptors}")
            if total_descriptors > 0:
                avg_per_bucket = total_descriptors / len(self.btc_manager.database)
                print(f"   Avg descriptors per bucket: {avg_per_bucket:.1f}")

    def visualize_results(self):
        """Create visualization of results"""
        if not self.loop_results:
            print("📊 No loop closures detected - skipping visualization")
            return

        print(f"📈 Creating visualization with {len(self.loop_results)} loop detections...")

        try:
            # Set up matplotlib for better display
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'BTC Place Recognition Results ({len(self.poses)} frames)', fontsize=16)

            # Plot 1: Trajectory with loop closures
            poses_array = np.array([pose[0] for pose in self.poses])
            ax1.plot(poses_array[:, 0], poses_array[:, 1], 'b-', alpha=0.7, linewidth=1, label='Trajectory')
            ax1.scatter(poses_array[0, 0], poses_array[0, 1], c='g', s=100, marker='^', label='Start', zorder=5)
            ax1.scatter(poses_array[-1, 0], poses_array[-1, 1], c='r', s=100, marker='v', label='End', zorder=5)

            # Draw loop closures
            true_positive_count = 0
            false_positive_count = 0
            for result in self.loop_results:
                query_pos = poses_array[result['query_id']]
                match_pos = poses_array[result['match_id']]
                if result['is_true_positive']:
                    color = 'g'
                    true_positive_count += 1
                else:
                    color = 'r'
                    false_positive_count += 1
                ax1.plot([query_pos[0], match_pos[0]], [query_pos[1], match_pos[1]],
                         color=color, alpha=0.6, linewidth=2)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title(f'Trajectory with Loop Closures\n(TP: {true_positive_count}, FP: {false_positive_count})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')

            # Plot 2: Loop scores vs overlap
            scores = [r['score'] for r in self.loop_results]
            overlaps = [r['overlap'] for r in self.loop_results]
            colors = ['g' if r['is_true_positive'] else 'r' for r in self.loop_results]

            scatter = ax2.scatter(overlaps, scores, c=colors, alpha=0.7, s=50)
            ax2.axvline(x=self.cloud_overlap_thr, color='k', linestyle='--', alpha=0.5,
                        label=f'Overlap Threshold ({self.cloud_overlap_thr})')
            ax2.axhline(y=self.config.icp_threshold, color='k', linestyle='--', alpha=0.5,
                        label=f'Score Threshold ({self.config.icp_threshold})')
            ax2.set_xlabel('Ground Truth Overlap')
            ax2.set_ylabel('Loop Score')
            ax2.set_title('Loop Scores vs Ground Truth Overlap')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Timing analysis
            if len(self.descriptor_times) > 0:
                frame_ids = list(range(len(self.descriptor_times)))
                ax3.plot(frame_ids, self.descriptor_times, label='Descriptor', alpha=0.7, linewidth=1)
                ax3.plot(frame_ids, self.query_times, label='Query', alpha=0.7, linewidth=1)
                ax3.plot(frame_ids, self.update_times, label='Update', alpha=0.7, linewidth=1)
                total_times = [d + q + u for d, q, u in zip(self.descriptor_times, self.query_times, self.update_times)]
                ax3.plot(frame_ids, total_times, 'k-', label='Total', linewidth=2)

                ax3.set_xlabel('Frame ID')
                ax3.set_ylabel('Time (ms)')
                ax3.set_title('Processing Time Analysis')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            # Plot 4: Precision-Recall curve (simplified)
            if len(self.loop_results) > 0:
                query_ids = [r['query_id'] for r in self.loop_results]
                true_positives = [r['is_true_positive'] for r in self.loop_results]

                cumulative_tp = np.cumsum(true_positives)
                cumulative_total = np.arange(1, len(true_positives) + 1)
                precision_curve = cumulative_tp / cumulative_total

                ax4.plot(query_ids, precision_curve, 'b-', linewidth=2, marker='o', markersize=3)
                ax4.set_xlabel('Frame ID')
                ax4.set_ylabel('Cumulative Precision')
                ax4.set_title('Precision over Time')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim([0, 1])

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function - Configure your paths here for PyCharm"""

    # 🔧 CONFIGURATION - UPDATE THESE PATHS FOR YOUR SETUP
    # =====================================================

    # Path to configuration file (relative to project root)
    config_path = "../config/config_outdoor.yaml"

    # Path to point cloud directory (UPDATE THIS!)
    pcds_dir = "/home/wzj/pan1/Data/KITTI/00/velodyne"  # Example KITTI path
    # pcds_dir = "/path/to/your/point/clouds"  # Generic path

    # Path to pose file (UPDATE THIS!)
    pose_file = "kitti00.txt"  # Example KITTI pose file
    # pose_file = "/path/to/your/poses.txt"  # Generic path

    # File format settings
    read_bin = True  # True for KITTI .bin files, False for .pcd files
    cloud_overlap_thr = 0.5  # Overlap threshold for true positive evaluation

    # =====================================================

    print("🚀 BTC Place Recognition Demo")
    print("=" * 40)
    print(f"📁 Config: {config_path}")
    print(f"📁 Point clouds: {pcds_dir}")
    print(f"📁 Poses: {pose_file}")
    print(f"🔧 Read binary: {read_bin}")
    print(f"🎯 Overlap threshold: {cloud_overlap_thr}")

    try:
        # Create and run demo
        demo = PlaceRecognitionDemo(
            config_path=config_path,
            pcds_dir=pcds_dir,
            pose_file=pose_file,
            read_bin=read_bin,
            cloud_overlap_thr=cloud_overlap_thr
        )

        demo.run_place_recognition()
        demo.visualize_results()

        print("\n✅ Demo completed successfully!")

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\n💡 Please update the file paths in the main() function")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()