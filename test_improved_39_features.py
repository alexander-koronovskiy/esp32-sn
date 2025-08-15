#!/usr/bin/env python3
"""
Test improved 39 features with better thresholds and logarithmic transforms
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.feature_extractor import SnoringFeatureExtractor


def test_improved_39_features():
    """Test the improved 39-feature extraction"""
    print("🧪 Testing Improved 39-Feature Extraction")
    print("=" * 50)
    
    # Create improved extractor
    extractor = SnoringFeatureExtractor()
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"📋 Total features: {len(feature_names)}")
    
    # Check feature categories
    audio_features = [i for i, name in enumerate(feature_names) if any(band in name for band in ['b100', 'b400', 'b1000', 'env'])]
    accel_features = [i for i, name in enumerate(feature_names) if 'accel' in name]
    mixed_features = [i for i, name in enumerate(feature_names) if 'ratio' in name]
    
    print(f"\n📊 Feature Categories:")
    print(f"   Audio features: {len(audio_features)} (indices: {audio_features[:5]}...)")
    print(f"   Accelerometer: {len(accel_features)} (indices: {accel_features})")
    print(f"   Mixed features: {len(mixed_features)} (indices: {mixed_features})")
    
    # Show some audio feature names
    print(f"\n🎵 Sample Audio Feature Names:")
    for i in range(min(20, len(audio_features))):
        idx = audio_features[i]
        print(f"   {idx:2d}: {feature_names[idx]}")
    
    # Test with sample data
    print(f"\n🧪 Testing with Sample Audio Data:")
    
    # Create sample audio data (similar to what we saw in CSV)
    sample_audio = np.array([2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 219, 224])
    
    print(f"   Sample audio data: {sample_audio}")
    print(f"   Data shape: {sample_audio.shape}")
    print(f"   Min: {np.min(sample_audio)}, Max: {np.max(sample_audio)}")
    print(f"   Mean: {np.mean(sample_audio):.2f}, Std: {np.std(sample_audio):.2f}")
    
    # Extract features
    features = extractor.extract_audio_features(sample_audio)
    
    print(f"\n🔍 Extracted Features:")
    for key, value in features.items():
        print(f"   {key}: {value:.6f}")
    
    # Test specific improvements
    print(f"\n✅ Feature Improvements:")
    
    # 1. Improved threshold (60th percentile instead of 75th)
    improved_threshold = np.percentile(sample_audio, 60)
    improved_ratio = np.mean(sample_audio > improved_threshold)
    print(f"   Improved threshold (60th percentile): {improved_threshold:.2f}")
    print(f"   Improved ratio: {improved_ratio:.4f}")
    
    # 2. Logarithmic transforms
    log_data = np.log1p(sample_audio)
    print(f"   Log data: min={np.min(log_data):.4f}, max={np.max(log_data):.4f}")
    print(f"   Log mean: {np.mean(log_data):.4f}, Log std: {np.std(log_data):.4f}")
    
    # Test with a window of data
    print(f"\n🪟 Testing Window Feature Extraction:")
    
    # Create sample window data
    sample_window = pd.DataFrame({
        'time': range(80),
        'x': np.random.randn(80) * 100,
        'y': np.random.randn(80) * 100,
        'z': np.random.randn(80) * 100,
        'b100': np.random.randint(1, 10, 80),
        'b400': np.random.randint(1, 8, 80),
        'b1000': np.random.randint(1, 6, 80),
        'env': np.random.randint(20, 50, 80)
    })
    
    print(f"   Window shape: {sample_window.shape}")
    print(f"   Columns: {list(sample_window.columns)}")
    
    # Extract features from window
    window_features = extractor.extract_features_from_window(sample_window)
    
    print(f"   Extracted features shape: {window_features.shape}")
    print(f"   Expected: 39 features")
    
    # Show some feature values
    print(f"\n🔍 Sample Feature Values:")
    for i in range(min(10, len(feature_names))):
        print(f"   {feature_names[i]}: {window_features[i]:.6f}")
    
    print(f"\n✅ Improved 39-feature extraction test completed!")
    print(f"   Total features: {len(feature_names)}")
    print(f"   Audio features: {len(audio_features)}")
    print(f"   Key improvements:")
    print(f"     - Audio threshold: 60th percentile (was 75th)")
    print(f"     - Accelerometer threshold: 40th percentile (was 50th)")
    print(f"     - Logarithmic transforms for better scaling")
    print(f"     - Improved mixed feature ratios")


if __name__ == "__main__":
    test_improved_39_features() 