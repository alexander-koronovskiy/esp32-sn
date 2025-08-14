#!/usr/bin/env python3
"""
Fix CSV loading with proper separator and test feature extraction
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.feature_extractor import SnoringFeatureExtractor


def fix_csv_loading():
    """Fix CSV loading and test feature extraction"""
    print("🔧 Fixing CSV Loading and Testing Feature Extraction")
    print("=" * 50)
    
    # Find CSV files
    snoring_data_dir = Path("snoring_data")
    csv_files = list(snoring_data_dir.rglob("*.csv"))
    print(f"✅ Found {len(csv_files)} CSV files")
    
    if len(csv_files) > 0:
        # Load first CSV file with proper separator
        first_csv = csv_files[0]
        print(f"\n📁 Loading CSV file: {first_csv}")
        
        try:
            # Load with semicolon separator
            raw_data = pd.read_csv(first_csv, sep=';')
            print(f"   📊 Fixed CSV shape: {raw_data.shape}")
            print(f"   📋 Fixed CSV columns: {list(raw_data.columns)}")
            print(f"   📊 First 3 rows:")
            print(raw_data.head(3))
            
            # Check data types and values
            print(f"\n🔍 Data Analysis:")
            print(f"   Data types: {raw_data.dtypes.to_dict()}")
            
            # Check for non-zero values in audio columns
            audio_columns = ['b100', 'b400', 'b1000', 'env']
            print(f"\n🎵 Audio Columns Analysis:")
            for col in audio_columns:
                if col in raw_data.columns:
                    non_zero_count = (raw_data[col] != 0).sum()
                    unique_values = raw_data[col].nunique()
                    min_val = raw_data[col].min()
                    max_val = raw_data[col].max()
                    mean_val = raw_data[col].mean()
                    print(f"   {col}: {non_zero_count} non-zero values, {unique_values} unique values")
                    print(f"      Range: [{min_val}, {max_val}], Mean: {mean_val:.2f}")
                else:
                    print(f"   {col}: Column not found!")
            
            # Check accelerometer columns
            accel_columns = ['x', 'y', 'z']
            print(f"\n📱 Accelerometer Columns Analysis:")
            for col in accel_columns:
                if col in raw_data.columns:
                    non_zero_count = (raw_data[col] != 0).sum()
                    unique_values = raw_data[col].nunique()
                    min_val = raw_data[col].min()
                    max_val = raw_data[col].max()
                    mean_val = raw_data[col].mean()
                    print(f"   {col}: {non_zero_count} non-zero values, {unique_values} unique values")
                    print(f"      Range: [{min_val}, {max_val}], Mean: {mean_val:.2f}")
                else:
                    print(f"   {col}: Column not found!")
            
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            import traceback
            traceback.print_exc()
    
    # Now test feature extraction with fixed data loading
    print(f"\n🧪 Testing Feature Extraction with Fixed CSV Loading...")
    
    try:
        # Create a simple test with the first CSV
        if len(csv_files) > 0:
            test_csv = csv_files[0]
            test_data = pd.read_csv(test_csv, sep=';')
            
            print(f"   📊 Test data shape: {test_data.shape}")
            
            # Check if we have the right columns
            required_columns = ['b100', 'b400', 'b1000', 'env', 'x', 'y', 'z']
            missing_columns = [col for col in required_columns if col not in test_data.columns]
            
            if missing_columns:
                print(f"   ⚠️ Missing columns: {missing_columns}")
            else:
                print(f"   ✅ All required columns present")
                
                # Test feature extraction on this data
                extractor = SnoringFeatureExtractor()
                
                # Create a simple window (first 80 rows)
                window_size = 80
                if len(test_data) >= window_size:
                    window_data = test_data.head(window_size)
                    print(f"   🪟 Testing with window of {len(window_data)} rows")
                    
                    # Extract features manually for this window
                    features = extractor.extract_features_from_window(window_data)
                    print(f"   🔍 Extracted features shape: {features.shape}")
                    print(f"   📊 Features: {features}")
                    
                    # Check if features are non-zero
                    non_zero_count = np.sum(features != 0)
                    print(f"   🔍 Non-zero features: {non_zero_count}/{len(features)}")
                    
                    if non_zero_count > 0:
                        print(f"   ✅ Feature extraction is working!")
                        print(f"   📈 Sample non-zero features:")
                        non_zero_indices = np.where(features != 0)[0]
                        for idx in non_zero_indices[:5]:
                            print(f"      Feature {idx}: {features[idx]}")
                    else:
                        print(f"   ❌ All features are still zero!")
        
    except Exception as e:
        print(f"   ❌ Error during feature extraction test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    fix_csv_loading() 