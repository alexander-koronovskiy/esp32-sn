#!/usr/bin/env python3
"""
Test CSV Sorting
================
Test the corrected CSV file sorting logic
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils.data_loader import SnoringDataLoader

def test_csv_sorting():
    """Test CSV file sorting with midnight crossing fix"""
    
    print("🧪 Testing CSV File Sorting")
    print("=" * 40)
    
    # Initialize data loader
    data_dir = Path("snoring_data/19/4_2025_08_12_23.43")
    data_loader = SnoringDataLoader(data_dir)
    
    # Get sorted CSV files
    print("📁 Getting sorted CSV files...")
    csv_files = data_loader.get_csv_files_sorted()
    print(f"   Found {len(csv_files)} CSV files")
    
    # Show first 20 files with their parsed times
    print("\n🔍 First 20 CSV files (sorted):")
    
    for i, csv_file in enumerate(csv_files[:20]):
        try:
            parsed_time = data_loader.parse_time_from_filename(csv_file.name)
            print(f"   {i+1:2d}. {csv_file.name} -> {parsed_time}")
        except Exception as e:
            print(f"   {i+1:2d}. {csv_file.name} -> ERROR: {e}")
    
    # Check for evening files (23:xx) and morning files (00:xx)
    print(f"\n🌙 Evening files (23:xx):")
    evening_files = [f for f in csv_files if '23' in f.name.split('_')[1][:2]]
    for f in evening_files[:5]:
        print(f"   {f.name}")
    
    print(f"\n🌅 Morning files (00:xx):")
    morning_files = [f for f in csv_files if '00' in f.name.split('_')[1][:2]]
    for f in morning_files[:5]:
        print(f"   {f.name}")
    
    # Check if evening files come before morning files
    evening_indices = [i for i, f in enumerate(csv_files) if '23' in f.name.split('_')[1][:2]]
    morning_indices = [i for i, f in enumerate(csv_files) if '00' in f.name.split('_')[1][:2]]
    
    if evening_indices and morning_indices:
        max_evening_index = max(evening_indices)
        min_morning_index = min(morning_indices)
        
        print(f"\n📊 Sorting Check:")
        print(f"   Max evening file index: {max_evening_index}")
        print(f"   Min morning file index: {min_morning_index}")
        
        if max_evening_index < min_morning_index:
            print(f"   ✅ SUCCESS: Evening files come before morning files")
        else:
            print(f"   ❌ FAILURE: Morning files come before evening files")
    else:
        print(f"\n⚠️  Cannot check sorting: missing evening or morning files")

if __name__ == "__main__":
    test_csv_sorting() 