#!/usr/bin/env python3
"""
Test script for feature extraction
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.feature_extractor import SnoringFeatureExtractor


def test_feature_extraction():
    """Test feature extraction with synthetic data"""
    print("🧪 Testing Feature Extraction")
    print("=" * 40)
    
    # Create synthetic test data
    print("📊 Creating synthetic test data...")
    
    # 80 samples (8 seconds * 10 samples per second)
    n_samples = 80
    
    # Create synthetic sensor data
    test_data = pd.DataFrame({
        'b100': np.random.normal(100, 20, n_samples),      # Breath channel
        'b400': np.random.normal(200, 40, n_samples),      # Snore channel
        'b1000': np.random.normal(150, 30, n_samples),     # Speech channel
        'env': np.random.normal(80, 15, n_samples),         # High frequency
        'x': np.random.normal(0, 0.1, n_samples),          # Accelerometer X
        'y': np.random.normal(0, 0.1, n_samples),          # Accelerometer Y
        'z': np.random.normal(0, 0.1, n_samples)           # Accelerometer Z
    })
    
    print(f"✅ Created test data with shape: {test_data.shape}")
    print(f"   Columns: {list(test_data.columns)}")
    
    # Initialize feature extractor
    print("\n🌳 Initializing feature extractor...")
    extractor = SnoringFeatureExtractor()
    
    print(f"   Window size: {extractor.window_size}s")
    print(f"   Step size: {extractor.step_size}s")
    print(f"   Sampling rate: {extractor.sampling_rate} Hz")
    print(f"   Measurements per window: {extractor.measurements_per_window}")
    
    # Test feature extraction from window
    print("\n🔍 Testing feature extraction from window...")
    
    try:
        features = extractor.extract_features_from_window(test_data)
        print(f"✅ Successfully extracted features!")
        print(f"   Feature count: {len(features)}")
        print(f"   Expected: 39 features")
        
        if len(features) == 39:
            print("   ✅ Feature count matches expected!")
        else:
            print("   ❌ Feature count mismatch!")
            return False
        
        # Test individual feature extraction
        print("\n🔬 Testing individual feature extraction...")
        
        # Audio features
        audio_features = extractor.extract_audio_features(test_data['b100'].values)
        print(f"   Audio features per channel: {len(audio_features)}")
        
        # Accelerometer features
        accel_features = extractor.extract_accelerometer_features(test_data['x'].values)
        print(f"   Accelerometer features per axis: {len(accel_features)}")
        
        # Mixed features
        audio_feature_dicts = {}
        for channel in extractor.audio_channels:
            audio_feature_dicts[channel] = extractor.extract_audio_features(test_data[channel].values)
        
        mixed_features = extractor.extract_mixed_features(audio_feature_dicts)
        print(f"   Mixed features: {len(mixed_features)}")
        
        # Verify feature names
        print("\n📝 Testing feature names...")
        feature_names = extractor.get_feature_names()
        print(f"   Feature names count: {len(feature_names)}")
        
        if len(feature_names) == 39:
            print("   ✅ Feature names count matches!")
        else:
            print("   ❌ Feature names count mismatch!")
            return False
        
        # Display some feature names
        print("   Sample feature names:")
        for i, name in enumerate(feature_names[:10]):
            print(f"     {i}: {name}")
        if len(feature_names) > 10:
            print(f"     ... and {len(feature_names) - 10} more")
        
        # Test with missing columns
        print("\n⚠️ Testing with missing columns...")
        test_data_missing = test_data.drop(columns=['b1000', 'z'])
        features_missing = extractor.extract_features_from_window(test_data_missing)
        
        if len(features_missing) == 39:
            print("   ✅ Handles missing columns correctly!")
        else:
            print("   ❌ Error handling missing columns!")
            return False
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sliding_windows():
    """Test sliding window creation"""
    print("\n🪟 Testing Sliding Windows")
    print("=" * 40)
    
    try:
        # Create test data with 300 rows (30 seconds)
        n_rows = 300
        test_data = pd.DataFrame({
            'b100': np.random.normal(100, 20, n_rows),
            'b400': np.random.normal(200, 40, n_rows),
            'b1000': np.random.normal(150, 30, n_rows),
            'env': np.random.normal(80, 15, n_rows),
            'x': np.random.normal(0, 0.1, n_rows),
            'y': np.random.normal(0, 0.1, n_rows),
            'z': np.random.normal(0, 0.1, n_rows)
        })
        
        # Add time column for sliding windows
        test_data['time'] = np.arange(n_rows) * 0.1  # 0.1 second intervals
        
        # Initialize extractor
        extractor = SnoringFeatureExtractor()
        
        # Create sliding windows
        windows = extractor.create_sliding_windows(test_data, pd.Timestamp.now())
        
        print(f"✅ Created {len(windows)} sliding windows")
        
        # Calculate expected windows: (total_rows - window_size) / step_size + 1
        # But since step_size is in seconds and we have 10 measurements per second
        step_measurements = extractor.step_size * extractor.sampling_rate  # 1 * 10 = 10
        expected_windows = (n_rows - extractor.measurements_per_window) // step_measurements + 1
        
        print(f"   Expected windows: {expected_windows}")
        print(f"   Step measurements: {step_measurements}")
        
        if len(windows) == expected_windows:
            print("   ✅ Window count matches expected!")
        else:
            print("   ❌ Window count mismatch!")
            return False
        
        # Check window boundaries
        first_window = windows[0]
        last_window = windows[-1]
        
        print(f"   First window: rows {first_window[0]}-{first_window[1]}")
        print(f"   Last window: rows {last_window[0]}-{last_window[1]}")
        
        # Verify window size
        for i, (start, end, time) in enumerate(windows):
            if end - start != extractor.measurements_per_window:
                print(f"   ❌ Window {i} has wrong size: {end - start}")
                return False
        
        print("   ✅ All windows have correct size!")
        return True
        
    except Exception as e:
        print(f"❌ Error during sliding window test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 Starting Feature Extraction Tests")
    print("=" * 50)
    
    # Test 1: Feature extraction
    test1_passed = test_feature_extraction()
    
    # Test 2: Sliding windows
    test2_passed = test_sliding_windows()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Feature Extraction: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Sliding Windows: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Feature extraction is working correctly.")
        print("\n🎯 Next steps:")
        print("   1. Run training: python train_snoring_classifier.py")
        print("   2. Check your data format in snoring_data/")
        print("   3. Verify CSV columns match expected format")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 