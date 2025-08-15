#!/usr/bin/env python3
"""
Check Window-Annotation Overlap
================================
Verify if sliding windows overlap with annotation periods
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from utils.data_loader import SnoringDataLoader
from features.feature_extractor import SnoringFeatureExtractor

def check_window_annotation_overlap():
    """Check if sliding windows overlap with annotation periods"""
    
    print("🔍 Checking Window-Annotation Overlap")
    print("=" * 50)
    
    # Initialize data loader and feature extractor
    data_dir = Path("snoring_data/19/4_2025_08_12_23.43")
    data_loader = SnoringDataLoader(data_dir)
    extractor = SnoringFeatureExtractor()
    
    # Load annotations
    print("📊 Loading annotations...")
    annotation_file = data_loader.find_annotation_file()
    if annotation_file:
        annotations = data_loader.parse_annotations(annotation_file)
    else:
        print("   ❌ No annotation file found")
        return
    print(f"   Loaded {len(annotations)} annotations")
    
    for i, (label, start_time, end_time) in enumerate(annotations):
        print(f"   {i+1}. {label}: {start_time} - {end_time}")
        duration = (end_time - start_time).total_seconds()
        print(f"      Duration: {duration:.1f}s")
    
    # Get CSV files
    print("\n📁 Loading CSV files...")
    csv_files = data_loader.get_csv_files_sorted()
    print(f"   Found {len(csv_files)} CSV files")
    
    # Get base date
    base_date = data_loader.get_base_date_from_folder()
    print(f"   Base date: {base_date}")
    
    # Check first few CSV files
    print("\n🧪 Testing window creation and annotation overlap...")
    
    total_windows = 0
    overlapping_windows = 0
    
    for csv_file in csv_files[:5]:  # Check first 5 files
        print(f"\n   📄 {csv_file.name}")
        
        try:
            # Extract features with correct base date
            features, window_times = extractor.extract_features_from_csv(csv_file, base_date)
            print(f"      Extracted {len(features)} windows")
            
            if len(window_times) > 0:
                first_window = window_times[0]
                last_window = window_times[-1]
                print(f"      First window: {first_window}")
                print(f"      Last window: {last_window}")
                
                # Check each window for overlap with annotations
                for j, window_time in enumerate(window_times):
                    total_windows += 1
                    
                    # Window spans 8 seconds
                    window_start = window_time
                    window_end = window_time + timedelta(seconds=8)
                    
                    # Check overlap with each annotation
                    for ann_label, ann_start, ann_end in annotations:
                        if (window_start <= ann_end and window_end >= ann_start):
                            overlap_start = max(window_start, ann_start)
                            overlap_end = min(window_end, ann_end)
                            overlap_duration = (overlap_end - overlap_start).total_seconds()
                            
                            if overlap_duration > 0:
                                overlapping_windows += 1
                                print(f"         ✅ Window {j+1} overlaps with {ann_label}: {overlap_duration:.1f}s")
                                break
                    else:
                        if j < 3:  # Show first 3 non-overlapping windows
                            print(f"         ❌ Window {j+1}: No overlap")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Total windows checked: {total_windows}")
    print(f"   Windows overlapping annotations: {overlapping_windows}")
    print(f"   Overlap ratio: {overlapping_windows/total_windows*100:.1f}%")
    
    if overlapping_windows == 0:
        print("\n⚠️  PROBLEM: No windows overlap with annotations!")
        print("   This means all samples will be labeled as 'No Snoring'")
        print("   Possible causes:")
        print("   1. Window times are incorrect")
        print("   2. Annotation times are incorrect")
        print("   3. Time scale mismatch")
    else:
        print(f"\n✅ SUCCESS: {overlapping_windows} windows overlap with annotations")

if __name__ == "__main__":
    check_window_annotation_overlap() 