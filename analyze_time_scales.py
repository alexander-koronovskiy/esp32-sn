#!/usr/bin/env python3
"""
Analyze Time Scales
===================
Detailed analysis of CSV file times vs annotation times
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from utils.data_loader import SnoringDataLoader

def analyze_time_scales():
    """Analyze time scales between CSV files and annotations"""
    
    print("🔍 Analyzing Time Scales")
    print("=" * 50)
    
    # Initialize data loader
    data_dir = Path("snoring_data/19/4_2025_08_12_23.43")
    data_loader = SnoringDataLoader(data_dir)
    
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
    print("\n📁 Analyzing CSV files...")
    csv_files = data_loader.get_csv_files_sorted()
    print(f"   Found {len(csv_files)} CSV files")
    
    # Get base date
    base_date = data_loader.get_base_date_from_folder()
    print(f"   Base date: {base_date}")
    
    # Analyze first 10 CSV files in detail
    print("\n🧪 Detailed CSV Analysis (first 10 files):")
    
    csv_times = []
    
    for i, csv_file in enumerate(csv_files[:10]):
        print(f"\n   📄 {csv_file.name}")
        
        # Parse time from filename
        try:
            filename_time = data_loader.parse_time_from_filename(csv_file.name)
            print(f"      Filename time: {filename_time}")
            
            # Load CSV data to check actual time column
            try:
                data = pd.read_csv(csv_file, sep=';')
                if 'time' in data.columns:
                    time_col = data['time']
                    min_time = time_col.min()
                    max_time = time_col.max()
                    print(f"      Time column: min={min_time}, max={max_time}")
                    
                    # Convert time column values to actual time
                    # Assuming time column starts from 0 and increments by 1
                    # Each row = 0.1 seconds (10 Hz sampling rate)
                    first_row_time = filename_time + timedelta(seconds=min_time * 0.1)
                    last_row_time = filename_time + timedelta(seconds=max_time * 0.1)
                    print(f"      First row time: {first_row_time}")
                    print(f"      Last row time: {last_row_time}")
                    
                    csv_times.append({
                        'filename': csv_file.name,
                        'filename_time': filename_time,
                        'first_row_time': first_row_time,
                        'last_row_time': last_row_time,
                        'time_col_min': min_time,
                        'time_col_max': max_time
                    })
                    
                else:
                    print(f"      ❌ No 'time' column found")
                    
            except Exception as e:
                print(f"      ❌ Error reading CSV: {e}")
                
        except Exception as e:
            print(f"      ❌ Error parsing filename: {e}")
    
    # Summary analysis
    print(f"\n📊 Time Scale Summary:")
    print(f"   CSV files analyzed: {len(csv_times)}")
    
    if csv_times:
        # Find time range
        all_start_times = [t['first_row_time'] for t in csv_times]
        all_end_times = [t['last_row_time'] for t in csv_times]
        
        overall_start = min(all_start_times)
        overall_end = max(all_end_times)
        
        print(f"   Overall CSV time range: {overall_start} - {overall_end}")
        print(f"   Total CSV duration: {(overall_end - overall_start).total_seconds():.1f}s")
        
        # Check annotation overlap with overall CSV range
        print(f"\n🔍 Annotation vs CSV Range Overlap:")
        for i, (label, ann_start, ann_end) in enumerate(annotations):
            # Check if annotation overlaps with CSV range
            if (ann_start <= overall_end and ann_end >= overall_start):
                overlap_start = max(ann_start, overall_start)
                overlap_end = min(ann_end, overall_end)
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                print(f"   ✅ {label}: Overlaps with CSV range ({overlap_duration:.1f}s)")
            else:
                print(f"   ❌ {label}: NO OVERLAP with CSV range")
                print(f"      Annotation: {ann_start} - {ann_end}")
                print(f"      CSV range: {overall_start} - {overall_end}")
        
        # Check if we need to look at different CSV files
        print(f"\n🔍 CSV File Time Distribution:")
        morning_files = [t for t in csv_times if t['filename_time'].hour < 12]
        afternoon_files = [t for t in csv_times if 12 <= t['filename_time'].hour < 18]
        evening_files = [t for t in csv_times if t['filename_time'].hour >= 18]
        
        print(f"   Morning files (00:00-11:59): {len(morning_files)}")
        print(f"   Afternoon files (12:00-17:59): {len(afternoon_files)}")
        print(f"   Evening files (18:00-23:59): {len(evening_files)}")
        
        # Check if there are evening CSV files that might contain the snoring data
        if evening_files:
            print(f"\n🌙 Evening CSV files found:")
            for t in evening_files[:5]:  # Show first 5
                print(f"   {t['filename']}: {t['filename_time']} ({t['first_row_time']} - {t['last_row_time']})")
        else:
            print(f"\n⚠️  No evening CSV files found!")
            print(f"   This explains why windows don't overlap with evening annotations")

if __name__ == "__main__":
    analyze_time_scales() 