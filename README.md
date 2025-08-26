# BTC Python Implementation

A Python implementation of the BTC (Binary Triangle Combined) descriptor for 3D place recognition. This is a research-oriented port of the original C++ implementation designed for easier debugging and experimentation.

## Overview

BTC is a novel global and local combined descriptor for 3D place recognition that achieves viewpoint invariance through:

- **Triangle Descriptor**: Global geometric features based on triangle side lengths formed by keypoints
- **Binary Descriptor**: Local geometric information encoded for each keypoint
- **Combined Approach**: Robust place recognition in large-scale unstructured environments

## Features

- Pure Python implementation for easy debugging and modification
- Support for KITTI binary format and PCD files
- Built-in performance evaluation and visualization
- Modular design for easy extension and experimentation
- No ROS dependency (simplified for research use)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd btc_python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from btc import ConfigSetting, load_config_setting, BTCDescManager
import numpy as np

# Load configuration
config = load_config_setting("config/config_outdoor.yaml")

# Initialize BTC manager
btc_manager = BTCDescManager(config)

# Process point cloud (shape: N x 4, columns: x, y, z, intensity)
point_cloud = np.random.rand(1000, 4)  # Replace with your data
frame_id = 0

# Generate BTC descriptors
btc_descriptors = btc_manager.generate_btc_descriptors(point_cloud, frame_id)

# Add to database
btc_manager.add_btc_descriptors(btc_descriptors)

# Search for loop closures (on subsequent frames)
loop_result, transform, matches = btc_manager.search_loop(btc_descriptors)
```

### Running Place Recognition Demo

1. Update the paths in `examples/place_recognition.py`:
```python
config_path = "config/config_outdoor.yaml"
pcds_dir = "/path/to/your/point/clouds"
pose_file = "/path/to/your/poses.txt"
read_bin = True  # True for .bin files, False for .pcd files
```

2. Run the demo:
```bash
python examples/place_recognition.py
```

## Configuration

The system uses YAML configuration files. Two templates are provided:

- `config/config_indoor.yaml`: For indoor environments
- `config/config_outdoor.yaml`: For outdoor environments (like KITTI)

### Key Parameters

**Binary Descriptor Parameters:**
- `useful_corner_num`: Maximum number of keypoints to extract
- `voxel_size`: Voxel size for plane detection
- `proj_plane_num`: Number of projection planes
- `proj_image_resolution`: Resolution for binary descriptor extraction

**Triangle Descriptor Parameters:**
- `descriptor_near_num`: Number of neighboring points for triangle generation
- `descriptor_min_len`: Minimum triangle edge length
- `descriptor_max_len`: Maximum triangle edge length
- `triangle_resolution`: Edge length quantization resolution

**Loop Detection Parameters:**
- `skip_near_num`: Skip frames close in time for loop detection
- `candidate_num`: Maximum number of candidate frames to evaluate
- `similarity_threshold`: Binary similarity threshold
- `icp_threshold`: Final verification threshold

## File Structure

```
btc_python/
├── btc/
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration loading
│   ├── data_structures.py       # Core data structures
│   ├── voxel_processing.py      # Voxelization and plane detection
│   ├── binary_extractor.py      # Binary descriptor extraction
│   ├── btc_generator.py         # BTC descriptor generation
│   ├── btc_manager.py           # Main manager class
│   └── utils.py                 # Utility functions
├── config/
│   ├── config_indoor.yaml       # Indoor parameters
│   └── config_outdoor.yaml      # Outdoor parameters
├── examples/
│   └── place_recognition.py     # Main demo script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Data Formats

### Point Cloud Format
- **Input**: NumPy array of shape (N, 4)
- **Columns**: [x, y, z, intensity]
- **Units**: Meters for coordinates

### Pose Format
The system supports EVO trajectory format:
```
timestamp tx ty tz qx qy qz qw
```
Where:
- `timestamp`: Unix timestamp
- `tx, ty, tz`: Translation in meters
- `qx, qy, qz, qw`: Quaternion rotation

### Supported Point Cloud Files
- **KITTI Binary (.bin)**: Direct binary format
- **PCD Files (.pcd)**: PCL point cloud format

## API Reference

### Core Classes

#### `BTCDescManager`
Main interface for BTC descriptor processing.

```python
class BTCDescManager:
    def __init__(self, config: ConfigSetting)
    
    def generate_btc_descriptors(self, point_cloud: np.ndarray, frame_id: int) -> List[BTC]
    """Generate BTC descriptors from point cloud"""
    
    def search_loop(self, btc_list: List[BTC]) -> Tuple[Tuple[int, float], Tuple[np.ndarray, np.ndarray], List[Tuple[BTC, BTC]]]
    """Search for loop closure candidates"""
    
    def add_btc_descriptors(self, btc_list: List[BTC])
    """Add descriptors to database"""
```

#### `ConfigSetting`
Configuration parameters container.

```python
@dataclass
class ConfigSetting:
    # Binary descriptor parameters
    useful_corner_num: int = 30
    voxel_size: float = 1.0
    proj_plane_num: int = 1
    
    # Triangle descriptor parameters
    descriptor_near_num: float = 10
    descriptor_min_len: float = 1
    descriptor_max_len: float = 10
    
    # Loop detection parameters
    skip_near_num: int = 20
    similarity_threshold: float = 0.7
    icp_threshold: float = 0.5
```

### Utility Functions

```python
# Data loading
load_config_setting(config_file: str) -> ConfigSetting
load_evo_pose_with_time(pose_file: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]
read_lidar_data(lidar_data_path: str) -> np.ndarray
load_point_cloud_from_pcd(pcd_file: str) -> np.ndarray

# Point cloud processing
down_sampling_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray
transform_point_cloud(points: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray

# Evaluation
calc_overlap(cloud1: np.ndarray, cloud2: np.ndarray, dis_threshold: float) -> float
binary_similarity(b1: BinaryDescriptor, b2: BinaryDescriptor) -> float
```

## Performance

### Computational Complexity
- **Descriptor Generation**: O(N log N) where N is number of points
- **Database Query**: O(M log M) where M is number of descriptors
- **Memory**: O(K) where K is number of stored descriptors

### Typical Processing Times (Python)
- **Descriptor Extraction**: 50-200ms per frame
- **Loop Query**: 10-50ms per frame  
- **Database Update**: 1-5ms per frame

*Note: Times are significantly slower than C++ due to Python overhead, but sufficient for research and debugging.*

## Debugging and Visualization

### Enable Debug Output
```python
btc_manager = BTCDescManager(config)
btc_manager.print_debug_info = True
```

### Visualization Features
The demo script includes:
- Trajectory plot with detected loops
- Precision-recall analysis
- Processing time statistics
- Loop score vs ground truth overlap

## Differences from C++ Version

### Simplifications
- No ROS dependencies
- Simplified visualization (matplotlib instead of RViz)
- No multi-threading (for easier debugging)
- Reduced optimization for readability

### Maintained Features
- Core BTC algorithm unchanged
- Same parameter interface
- Compatible configuration files
- Equivalent loop detection logic

## Known Limitations

1. **Performance**: ~10x slower than optimized C++ version
2. **Memory**: Higher memory usage due to Python overhead
3. **Precision**: Minor numerical differences due to library implementations
4. **Visualization**: Basic compared to RViz-based C++ version

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install --upgrade numpy scipy scikit-learn open3d opencv-python
```

**Memory Issues with Large Datasets**
- Reduce `useful_corner_num` parameter
- Increase `voxel_size` for downsampling
- Process in smaller batches

**Poor Loop Detection Performance**
- Adjust `similarity_threshold` (try 0.6-0.8)
- Modify `icp_threshold` (try 0.3-0.7)
- Check point cloud quality and density

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

Please maintain compatibility with the original C++ parameter interface.

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite the original BTC paper:

```bibtex
@article{btc_paper,
    title={BTC: A Binary and Triangle Combined Descriptor for 3D Place Recognition},
    author={[Authors]},
    journal={[Journal]},
    year={[Year]}
}
```

## Acknowledgments

This Python implementation is based on the original C++ BTC implementation. Thanks to the original authors for their excellent work.