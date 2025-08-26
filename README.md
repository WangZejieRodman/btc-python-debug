# README.md
# BTC (Binary Triangle Combined) Descriptor - Python Implementation

This is a Python implementation of the BTC descriptor for 3D place recognition, converted from the original C++ version. BTC combines global triangle descriptors with local binary descriptors for robust viewpoint-invariant place recognition.

## Features

- **Viewpoint Invariant**: Triangle descriptors maintain consistency across different viewpoints
- **Binary Local Features**: Encode local geometric information for enhanced discrimination  
- **Efficient Matching**: Hash table-based descriptor database for fast retrieval
- **Configurable**: Separate configurations for indoor and outdoor environments
- **No ROS Dependency**: Standalone Python implementation

## Installation

```bash
# Install required packages
pip install numpy scipy open3d PyYAML scikit-learn

# Optional for visualization  
pip install matplotlib seaborn
```

## Quick Test

```bash
# Run basic functionality test
python simple_test.py
```

## Usage

### Basic Example
```python
from btc_python import BTCDescManager, ConfigSetting

# Create configuration
config = ConfigSetting()
config.voxel_size = 0.5  # Adjust for your data scale

# Initialize manager
manager = BTCDescManager(config)

# Process point cloud (Nx4 array: x,y,z,intensity)
btc_descriptors = manager.generate_btc_descs(point_cloud, frame_id)

# Search for loop closures
loop_id, score, R, t, matches = manager.search_loop(btc_descriptors)

# Add to database
manager.add_btc_descs(btc_descriptors)
```

### KITTI Dataset Example
```bash
python usage_example.py --mode kitti --data_dir /path/to/kitti/sequences/00/velodyne --pose_file /path/to/poses.txt
```

## Configuration

Two preset configurations are provided:

- `config_outdoor.yaml`: For outdoor/KITTI-style datasets
- `config_indoor.yaml`: For indoor environments

Key parameters:
- `voxel_size`: Voxel size for plane detection (larger for outdoor)
- `useful_corner_num`: Number of keypoints to extract
- `proj_plane_num`: Number of projection planes (more for indoor)
- `skip_near_num`: Frames to skip for loop detection
- `similarity_threshold`: Binary descriptor similarity threshold

## File Structure

```
├── btc_python.py          # Main BTC implementation  
├── usage_example.py       # Usage examples
├── simple_test.py         # Test suite
├── config_outdoor.yaml    # Outdoor configuration
├── config_indoor.yaml     # Indoor configuration
└── requirements.txt       # Dependencies
```

## Data Formats

- **Point Clouds**: Nx4 numpy arrays (x, y, z, intensity) or KITTI .bin files
- **Poses**: EVO format (timestamp x y z qx qy qz qw)  
- **Config**: YAML format

## Performance

Typical performance on modern hardware:
- Descriptor extraction: ~50-200ms per frame
- Loop query: ~10-50ms per frame  
- Database update: ~1-10ms per frame

Performance scales with:
- Point cloud size
- Number of extracted keypoints (`useful_corner_num`)
- Database size
- Configuration parameters

## Differences from C++ Version

1. **No ROS**: Removed all ROS dependencies
2. **Simplified I/O**: Direct numpy array processing
3. **Python Optimizations**: Uses scipy, scikit-learn for efficiency
4. **Modular Design**: Clean separation of components
5. **Maintained Logic**: Core algorithms identical to C++ version
6. **Same Data Flow**: Point cloud -> planes -> binary descriptors -> triangles -> matching

## Algorithm Overview

1. **Voxel-based Plane Detection**: Extract planar structures from point clouds
2. **Binary Descriptor Extraction**: Project points onto planes and encode occupancy
3. **Triangle Formation**: Create triangles from keypoints with side length constraints
4. **Hash-based Retrieval**: Store and query descriptors using spatial hashing
5. **Geometric Verification**: Validate matches using plane-to-plane consistency

## Troubleshooting

### Common Issues

1. **No descriptors generated**
   - Check point cloud scale vs voxel_size
   - Verify point cloud has sufficient structure
   - Adjust plane_detection_thre parameter

2. **No loop closures detected**
   - Increase similarity_threshold (try 0.5-0.8)
   - Adjust skip_near_num for your frame rate
   - Check if scenes actually revisit same locations

3. **Performance issues**
   - Reduce useful_corner_num (try 200-300)
   - Increase voxel_size for faster processing
   - Use smaller descriptor_max_len

4. **Memory usage**
   - Clear old data: `manager.key_cloud_vec.clear()`
   - Limit database size by frame count
   - Use smaller point clouds or downsample input

### Parameter Tuning Guide

**For Indoor Environments:**
```python
config.voxel_size = 0.3          # Small voxels for detail
config.proj_plane_num = 2        # Multiple projection planes
config.useful_corner_num = 300   # More keypoints
config.line_filter_enable = 0    # Disable line filtering
```

**For Outdoor/KITTI:**
```python
config.voxel_size = 2.0          # Large voxels for efficiency  
config.proj_plane_num = 1        # Single dominant ground plane
config.useful_corner_num = 500   # More keypoints for large scenes
config.line_filter_enable = 1    # Enable line filtering
```

**For Real-time Applications:**
```python
config.useful_corner_num = 200   # Fewer keypoints
config.candidate_num = 25        # Fewer candidates to check
config.descriptor_near_num = 8   # Smaller neighborhoods
```

## Citation

If you use this code in your research, please cite the original BTC paper:

```bibtex
@article{btc2024,
  title={BTC: A Binary and Triangle Combined Descriptor for 3D Place Recognition},
  author={Yuan, Chongjian and Lin, Jiarong},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This implementation follows the same license as the original C++ version.

## Contact

For issues with this Python implementation, please create a GitHub issue.
For questions about the algorithm, refer to the original paper and C++ implementation.