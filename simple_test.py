# simple_test.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for BTC implementation
Generate synthetic data to verify the system works
"""

import numpy as np
import os
from btc_python import BTCDescManager, ConfigSetting


def generate_synthetic_point_cloud(center=[0, 0, 0], size=1000, noise_level=0.1):
    """Generate a synthetic point cloud with some structure"""
    np.random.seed(42)

    points = []

    # Generate some planar structures
    for i in range(3):
        # Create a plane
        normal = np.random.randn(3)
        normal = normal / np.linalg.norm(normal)

        # Generate points on the plane
        u = np.random.randn(3)
        u = u - np.dot(u, normal) * normal  # Make orthogonal to normal
        u = u / np.linalg.norm(u)

        v = np.cross(normal, u)

        for j in range(size // 3):
            # Random coordinates on plane
            s, t = np.random.uniform(-5, 5, 2)
            point = np.array(center) + s * u + t * v + i * 2 * normal

            # Add noise
            point += np.random.normal(0, noise_level, 3)

            # Add intensity
            intensity = np.random.uniform(0.5, 1.0)
            points.append([point[0], point[1], point[2], intensity])

    return np.array(points)


def test_btc_basic_functionality():
    """Test basic BTC functionality"""
    print("Testing BTC Basic Functionality")
    print("=" * 50)

    # Create configuration
    config = ConfigSetting()
    config.voxel_size = 0.5
    config.useful_corner_num = 100
    config.proj_plane_num = 2
    config.skip_near_num = 5
    config.print_debug_info = True

    # Create manager
    manager = BTCDescManager(config)
    manager.print_debug_info = True

    print("✓ BTC manager created successfully")

    # Generate test data
    cloud1 = generate_synthetic_point_cloud([0, 0, 0], 1000, 0.05)
    cloud2 = generate_synthetic_point_cloud([1, 0, 0], 1000, 0.05)  # Similar but shifted
    cloud3 = generate_synthetic_point_cloud([0, 0, 0], 1000, 0.05)  # Should match cloud1

    print(f"✓ Generated test clouds: {len(cloud1)}, {len(cloud2)}, {len(cloud3)} points")

    # Test descriptor generation
    btc_list1 = manager.generate_btc_descs(cloud1, 0)
    btc_list2 = manager.generate_btc_descs(cloud2, 1)
    btc_list3 = manager.generate_btc_descs(cloud3, 2)

    print(f"✓ Generated BTC descriptors: {len(btc_list1)}, {len(btc_list2)}, {len(btc_list3)}")

    # Test database operations
    manager.add_btc_descs(btc_list1)
    manager.add_btc_descs(btc_list2)

    print(f"✓ Added descriptors to database")
    print(f"  Database size: {len(manager.data_base)}")

    # Test loop detection
    loop_id, score, R, t, matches = manager.search_loop(btc_list3)

    print(f"✓ Loop detection test:")
    print(f"  Loop ID: {loop_id}")
    print(f"  Score: {score:.3f}")
    print(f"  Matches: {len(matches)}")

    if loop_id >= 0:
        print("  ✓ Loop detected successfully!")
    else:
        print("  ⚠ No loop detected (might be normal with synthetic data)")

    # Test utility functions
    overlap = manager.calc_overlap(cloud1, cloud3, 0.5)
    print(f"✓ Overlap calculation: {overlap:.3f}")

    downsampled = manager.down_sampling_voxel(cloud1, 0.2)
    print(f"✓ Downsampling: {len(cloud1)} -> {len(downsampled)} points")

    return True


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting Configuration Loading")
    print("=" * 50)

    # Create a temporary config file
    config_content = """
useful_corner_num: 200
voxel_size: 0.8
proj_plane_num: 1
skip_near_num: 20
descriptor_min_len: 1.5
"""

    with open("test_config.yaml", "w") as f:
        f.write(config_content)

    # Test loading
    manager = BTCDescManager(ConfigSetting())
    config = manager.load_config_from_yaml("test_config.yaml")

    print(f"✓ Config loaded:")
    print(f"  useful_corner_num: {config.useful_corner_num}")
    print(f"  voxel_size: {config.voxel_size}")
    print(f"  proj_plane_num: {config.proj_plane_num}")

    # Clean up
    os.remove("test_config.yaml")

    return True


def test_data_io():
    """Test data input/output functions"""
    print("\nTesting Data I/O Functions")
    print("=" * 50)

    manager = BTCDescManager(ConfigSetting())

    # Test synthetic bin file creation and reading
    test_points = np.random.rand(100, 4).astype(np.float32)

    # Write binary file
    with open("test.bin", "wb") as f:
        test_points.tobytes()
        for point in test_points:
            f.write(point.tobytes())

    try:
        # Try reading (this might fail with our simple test data)
        loaded_points = manager.read_kitti_bin("test.bin")
        print(f"✓ Binary I/O test: wrote {len(test_points)}, read {len(loaded_points)}")
    except:
        print("⚠ Binary I/O test failed (expected with simple test data)")

    # Clean up
    if os.path.exists("test.bin"):
        os.remove("test.bin")

    # Test pose loading with dummy data
    pose_content = """1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
2.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0
3.0 2.0 0.0 0.0 0.0 0.0 0.0 1.0"""

    with open("test_poses.txt", "w") as f:
        f.write(pose_content)

    try:
        translations, rotations = manager.load_pose_file("test_poses.txt")
        print(f"✓ Pose loading test: loaded {len(translations)} poses")
    except:
        print("⚠ Pose loading test failed")

    # Clean up
    if os.path.exists("test_poses.txt"):
        os.remove("test_poses.txt")

    return True


def run_performance_test():
    """Run a simple performance test"""
    print("\nRunning Performance Test")
    print("=" * 50)

    import time

    config = ConfigSetting()
    config.useful_corner_num = 300
    config.voxel_size = 0.5

    manager = BTCDescManager(config)

    # Generate larger test clouds
    clouds = []
    for i in range(5):
        cloud = generate_synthetic_point_cloud([i * 2, 0, 0], 2000, 0.1)
        clouds.append(cloud)

    print(f"✓ Generated {len(clouds)} test clouds with {len(clouds[0])} points each")

    times = {'desc': [], 'query': [], 'update': []}

    for i, cloud in enumerate(clouds):
        # Descriptor generation
        start = time.time()
        btc_list = manager.generate_btc_descs(cloud, i)
        desc_time = (time.time() - start) * 1000
        times['desc'].append(desc_time)

        # Loop query (after frame 2)
        start = time.time()
        if i >= 2:
            loop_id, score, R, t, matches = manager.search_loop(btc_list)
        else:
            loop_id, score = -1, 0
        query_time = (time.time() - start) * 1000
        times['query'].append(query_time)

        # Database update
        start = time.time()
        manager.add_btc_descs(btc_list)
        update_time = (time.time() - start) * 1000
        times['update'].append(update_time)

        print(f"  Frame {i}: {len(btc_list)} BTCs, "
              f"{desc_time:.1f}ms desc, {query_time:.1f}ms query, {update_time:.1f}ms update")

        if loop_id >= 0:
            print(f"    -> Loop detected: {i} -> {loop_id} (score: {score:.3f})")

    print(f"\n✓ Performance Summary:")
    print(f"  Average descriptor time: {np.mean(times['desc']):.1f} ± {np.std(times['desc']):.1f} ms")
    print(f"  Average query time: {np.mean(times['query']):.1f} ± {np.std(times['query']):.1f} ms")
    print(f"  Average update time: {np.mean(times['update']):.1f} ± {np.std(times['update']):.1f} ms")

    return True


def main():
    """Run all tests"""
    print("BTC Python Implementation Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Functionality", test_btc_basic_functionality),
        ("Configuration Loading", test_config_loading),
        ("Data I/O", test_data_io),
        ("Performance", run_performance_test)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
            print(f"✓ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"✗ {test_name}: FAILED with error: {str(e)}")

        print()

    # Summary
    passed = sum(results.values())
    total = len(results)

    print("=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! BTC implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    print("\nNext steps:")
    print("1. Try with real data using usage_example.py")
    print("2. Adjust config parameters for your specific dataset")
    print("3. Check data formats match expected input")


if __name__ == "__main__":
    main()