#!/usr/bin/env python3
"""
Debug feature extraction process to find why all features are 0.0
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from features.feature_extractor import SnoringFeatureExtractor


def debug_feature_extraction():
    """Debug the feature extraction process"""
    print("🐛 Debugging Feature Extraction")
    print("=" * 50)
    
    # Load raw data first
    print("📊 Loading raw data...")
    
    # Find CSV files manually
    snoring_data_dir = Path("snoring_data")
    csv_files = list(snoring_data_dir.rglob("*.csv"))
    print(f"✅ Found {len(csv_files)} CSV files")
    
    if len(csv_files) > 0:
        # Load first CSV file
        first_csv = csv_files[0]
        print(f"\n📁 Examining first CSV file: {first_csv}")
        
        try:
            # Load raw CSV
            raw_data = pd.read_csv(first_csv)
            print(f"   📊 Raw CSV shape: {raw_data.shape}")
            print(f"   📋 Raw CSV columns: {list(raw_data.columns)}")
            print(f"   📊 First 5 rows:")
            print(raw_data.head())
            
            # Check data types and values
            print(f"\n🔍 Data Analysis:")
            print(f"   Data types: {raw_data.dtypes.to_dict()}")
            print(f"   Non-null counts: {raw_data.count().to_dict()}")
            
            # Check for non-zero values
            numeric_columns = raw_data.select_dtypes(include=[np.number]).columns
            print(f"\n🔢 Numeric columns: {list(numeric_columns)}")
            
            for col in numeric_columns:
                non_zero_count = (raw_data[col] != 0).sum()
                unique_values = raw_data[col].nunique()
                print(f"   {col}: {non_zero_count} non-zero values, {unique_values} unique values")
                if non_zero_count > 0:
                    print(f"      Sample values: {raw_data[col].dropna().head(3).tolist()}")
            
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
    
    # Now test feature extraction on a small sample
    print(f"\n🧪 Testing Feature Extraction on Small Sample...")
    
    try:
        # Create feature extractor
        extractor = SnoringFeatureExtractor()
        
        # Load a small amount of data
        data_loader = SnoringDataLoader("snoring_data")
        features, labels, feature_names = data_loader.load_all_data()
        
        print(f"   📊 Extracted features shape: {features.shape}")
        print(f"   📋 Feature names: {feature_names[:5]}... (showing first 5)")
        
        # Check if features are all zero
        non_zero_features = np.sum(features != 0, axis=0)
        print(f"\n🔍 Feature Non-Zero Counts:")
        for i, count in enumerate(non_zero_features):
            if count > 0:
                print(f"   Feature {i} ({feature_names[i]}): {count} non-zero values")
        
        # Check feature statistics
        print(f"\n📈 Feature Statistics:")
        print(f"   Min values: {features.min(axis=0)[:5]}...")
        print(f"   Max values: {features.max(axis=0)[:5]}...")
        print(f"   Mean values: {features.mean(axis=0)[:5]}...")
        print(f"   Std values: {features.std(axis=0)[:5]}...")
        
        # Check if all features are exactly 0
        all_zero = np.all(features == 0)
        print(f"\n🚨 All features are zero: {all_zero}")
        
        if all_zero:
            print("   ❌ CRITICAL ISSUE: All extracted features are 0.0!")
            print("   🔍 This means feature extraction is broken.")
        
    except Exception as e:
        print(f"   ❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_feature_extraction() 