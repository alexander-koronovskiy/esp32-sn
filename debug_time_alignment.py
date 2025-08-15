#!/usr/bin/env python3
"""
Debug time alignment between windows and annotations
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.feature_extractor import SnoringFeatureExtractor
from utils.data_loader import SnoringDataLoader


def debug_time_alignment():
    """Debug time alignment issues"""
    print("🔍 Debugging Time Alignment Issues")
    print("=" * 50)
    
    # Create extractor and data loader
    extractor = SnoringFeatureExtractor()
    data_loader = SnoringDataLoader("snoring_data")
    
    # Find annotation file
    annotation_file = data_loader.find_annotation_file()
    if not annotation_file:
        print("❌ No annotation file found")
        return
    
    print(f"📁 Annotation file: {annotation_file}")
    
    # Parse annotations
    annotations = data_loader.parse_annotations(annotation_file)
    print(f"📊 Loaded {len(annotations)} annotations:")
    
    for i, ann in enumerate(annotations):
        label, start_time, end_time = ann  # Unpack tuple
        print(f"   {i+1}. {label}: {start_time} - {end_time}")
        print(f"      Duration: {(end_time - start_time).total_seconds():.1f}s")
    
    # Get CSV files
    csv_files = data_loader.get_csv_files_sorted()
    print(f"\n📁 Found {len(csv_files)} CSV files")
    
    # Show first few CSV files with their start times
    print(f"\n🔍 First 10 CSV files and their start times:")
    for i, csv_file in enumerate(csv_files[:10]):
        start_time = data_loader.parse_time_from_filename(csv_file)
        print(f"   {i+1:2d}. {csv_file.name} -> {start_time}")
    
    # Create timeline
    timeline = data_loader.create_temporal_timeline(csv_files)
    print(f"\n⏰ Timeline created with {len(timeline)} entries")
    
    # Show first few timeline entries
    print(f"\n🔍 First 5 timeline entries:")
    for i, (csv_file, start_time, start_row) in enumerate(timeline[:5]):
        print(f"   {i+1}. {csv_file.name}")
        print(f"      Start time: {start_time}")
        print(f"      Start row: {start_row}")
        
        # Load CSV to check time column
        try:
            df = pd.read_csv(csv_file, sep=';')
            if 'time' in df.columns:
                time_col = df['time'].values
                print(f"      Time column: min={time_col.min()}, max={time_col.max()}")
                print(f"      First few time values: {time_col[:5]}")
            else:
                print(f"      ⚠️  No 'time' column found")
        except Exception as e:
            print(f"      ❌ Error loading CSV: {e}")
    
    # Test window creation for first CSV
    print(f"\n🧪 Testing window creation for first CSV: {csv_files[0]}")
    
    # Get base date from data loader
    base_date = data_loader.get_base_date_from_folder()
    
    # Extract features with correct base date
    try:
        features, window_times = extractor.extract_features_from_csv(csv_files[0], base_date)
        print(f"   Extracted {len(features)} windows")
        if window_times:
            print(f"   First window time: {window_times[0]}")
            print(f"   Last window time: {window_times[-1]}")
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
    
    # Check annotation time format
    print(f"\n🔍 Annotation time format analysis:")
    if annotations:
        first_ann = annotations[0]
        label, start_time, end_time = first_ann  # Unpack tuple
        print(f"   First annotation: {label}")
        print(f"   First annotation start: {start_time}")
        print(f"   First annotation end: {end_time}")
        print(f"   Start time type: {type(start_time)}")
        print(f"   End time type: {type(end_time)}")
        
        # Check if times are in the right range
        if timeline:
            first_csv_time = timeline[0][1]
            print(f"   First CSV start time: {first_csv_time}")
            
            # Convert both to datetime if needed
            if isinstance(start_time, datetime) and isinstance(first_csv_time, datetime):
                time_diff = (start_time - first_csv_time).total_seconds()
                print(f"   Time difference: {time_diff:.1f}s")
            else:
                print(f"   ⚠️  Cannot calculate time difference - different types")
    
    print(f"\n🔍 Summary of potential issues:")
    print(f"   1. CSV time column parsing")
    print(f"   2. Filename time parsing")
    print(f"   3. Window time calculation")
    print(f"   4. Annotation time parsing")
    print(f"   5. Time zone handling")


if __name__ == "__main__":
    debug_time_alignment() 