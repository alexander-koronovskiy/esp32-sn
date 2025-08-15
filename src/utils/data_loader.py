"""
Data Loader for Snoring Classification
Handles CSV files, annotations, and temporal alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime, timedelta
import glob
import os


class SnoringDataLoader:
    """
    Loads and processes snoring data with annotations
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing snoring data
        """
        self.data_dir = Path(data_dir)
        self.annotation_file = None
        self.annotations = []
        
    def find_annotation_file(self) -> Optional[Path]:
        """
        Find annotation file in data directory
        
        Returns:
            Path to annotation file or None
        """
        # Look for files ending with _ann.txt
        ann_files = list(self.data_dir.rglob("*_ann.txt"))
        
        if ann_files:
            # Sort by modification time, take the most recent
            ann_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return ann_files[0]
        
        return None
    
    def parse_annotations(self, ann_file: Path) -> List[Tuple[str, datetime, datetime]]:
        """
        Parse annotation file
        
        Args:
            ann_file: Path to annotation file
            
        Returns:
            List of (label, start_time, end_time) tuples
        """
        # Set annotation_file first
        self.annotation_file = ann_file
        
        annotations = []
        
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        label = parts[0].strip()
                        start_time_str = parts[1].strip()
                        end_time_str = parts[2].strip()
                        
                        try:
                            start_time = self.parse_time_string(start_time_str, ann_file)
                            end_time = self.parse_time_string(end_time_str, ann_file, start_time)
                            
                            annotations.append((label, start_time, end_time))
                        except ValueError as e:
                            print(f"Warning: Could not parse time in line: {line}, error: {e}")
        
        # Save annotations to self.annotations for later use
        self.annotations = annotations
        print(f"   📊 Loaded {len(annotations)} annotations")
        
        return annotations
    
    def parse_time_string(self, time_str: str, annotation_file: Path = None, reference_time: datetime = None) -> datetime:
        """
        Parse time string in format HH:MM:SS.mmm
        
        Args:
            time_str: Time string
            annotation_file: Annotation file path for getting base date
            reference_time: Reference time for handling midnight crossing
            
        Returns:
            datetime object
        """
        # Handle milliseconds
        if '.' in time_str:
            time_part, ms_part = time_str.split('.')
            milliseconds = int(ms_part)
        else:
            time_part = time_str
            milliseconds = 0
        
        # Parse HH:MM:SS
        time_parts = time_part.split(':')
        if len(time_parts) != 3:
            raise ValueError(f"Invalid time format: {time_str}")
        
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2])
        
        # Get base date from annotation file folder name
        if annotation_file:
            parent_folder = annotation_file.parent.name
            # Try different date formats
            date_match = None
            
            # Format 1: '250812_234316887' -> '250812'
            date_match = re.search(r'(\d{6})_', parent_folder)
            if date_match:
                date_str = date_match.group(1)
                day = int(date_str[:2])
                month = int(date_str[2:4])
                year = 2000 + int(date_str[4:6])  # Assume 20xx
                base_date = datetime(year, month, day).date()
                print(f"   📅 Using base date from folder '{parent_folder}' (format 1): {base_date}")
                
                # Create datetime
                dt = datetime.combine(base_date, datetime.min.time().replace(
                    hour=hour, minute=minute, second=second, microsecond=milliseconds * 1000
                ))
                
                # Handle midnight crossing for end times
                if reference_time and hour < 12 and reference_time.hour >= 12:
                    # If this is an end time (early morning) and reference is late night, add one day
                    dt += timedelta(days=1)
                    print(f"   🌙 Midnight crossing detected, adjusted to next day: {dt}")
                
                return dt
            
            # Format 2: '4_2025_08_12_23.43' -> '2025_08_12'
            date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', parent_folder)
            if date_match:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                base_date = datetime(year, month, day).date()
                print(f"   📅 Using base date from folder '{parent_folder}' (format 2): {base_date}")
                
                # Create datetime
                dt = datetime.combine(base_date, datetime.min.time().replace(
                    hour=hour, minute=minute, second=second, microsecond=milliseconds * 1000
                ))
                
                # Handle midnight crossing for end times
                if reference_time and hour < 12 and reference_time.hour >= 12:
                    # If this is an end time (early morning) and reference is late night, add one day
                    dt += timedelta(days=1)
                    print(f"   🌙 Midnight crossing detected, adjusted to next day: {dt}")
                
                return dt
            
            # Format 3: '2025-08-12' -> '2025-08-12'
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', parent_folder)
            if date_match:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                base_date = datetime(year, month, day).date()
                print(f"   📅 Using base date from folder '{parent_folder}' (format 3): {base_date}")
                
                # Create datetime
                dt = datetime.combine(base_date, datetime.min.time().replace(
                    hour=hour, minute=minute, second=second, microsecond=milliseconds * 1000
                ))
                
                # Handle midnight crossing for end times
                if reference_time and hour < 12 and reference_time.hour >= 12:
                    # If this is an end time (early morning) and reference is late night, add one day
                    dt += timedelta(days=1)
                    print(f"   🌙 Midnight crossing detected, adjusted to next day: {dt}")
                
                return dt
            
            print(f"   ⚠️  Could not parse date from folder '{parent_folder}', using today")
        else:
            print(f"   ⚠️  No annotation file, using today")
        
        # Fallback to today
        today = datetime.now().date()
        print(f"   ⚠️  Using fallback date: {today}")
        return datetime.combine(today, datetime.min.time().replace(
            hour=hour, minute=minute, second=second, microsecond=milliseconds * 1000
        ))
    
    def parse_time_from_filename(self, filename) -> datetime:
        """
        Parse time from CSV filename
        
        Args:
            filename: CSV filename (e.g., '3_100917.csv') or Path object
            
        Returns:
            datetime object
        """
        # Convert Path to string if needed
        filename_str = str(filename)
        
        # Extract time part (e.g., '100917' from '3_100917.csv')
        match = re.search(r'_(\d{6})\.csv$', filename_str)
        if not match:
            raise ValueError(f"Cannot parse time from filename: {filename_str}")
        
        time_str = match.group(1)
        
        # Parse HH:MM:SS format
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        # Get base date from folder
        base_date = self.get_base_date_from_folder()
        
        # Create datetime
        dt = datetime(base_date.year, base_date.month, base_date.day, hour, minute, second)
        
        return dt
    
    def get_csv_files_sorted(self) -> List[Path]:
        """
        Get CSV files sorted by time
        
        Returns:
            List of CSV file paths sorted by time
        """
        csv_files = list(self.data_dir.rglob("*.csv"))
        
        # Filter out settings.csv and sort by time
        csv_files = [f for f in csv_files if 'settings.csv' not in f.name]
        
        # Sort by time parsed from filename
        # Handle midnight crossing: evening files (23:xx) should come before morning files (00:xx)
        def sort_key(filename):
            dt = self.parse_time_from_filename(filename.name)
            # If hour is 23 (evening), subtract 24 hours to make it come before 00 (morning)
            if dt.hour == 23:
                dt = dt - timedelta(hours=24)
            return dt
        
        csv_files.sort(key=sort_key)
        
        return csv_files
    
    def create_temporal_timeline(self, csv_files: List[Path]) -> List[Tuple[Path, datetime, int]]:
        """
        Create temporal timeline from CSV files
        
        Args:
            csv_files: List of CSV file paths
            
        Returns:
            List of (file_path, start_time, start_row) tuples
        """
        timeline = []
        current_row = 0
        
        for i, csv_file in enumerate(csv_files):
            start_time = self.parse_time_from_filename(csv_file.name)
            
            # Load CSV to get number of rows
            try:
                data = pd.read_csv(csv_file, sep=';')
                num_rows = len(data)
                
                timeline.append((csv_file, start_time, current_row))
                
                # Calculate next start row (continuous timeline)
                if i < len(csv_files) - 1:
                    # Check if next file is consecutive
                    next_file = csv_files[i + 1]
                    next_start_time = self.parse_time_from_filename(next_file.name)
                    
                    # If there's a gap, adjust the timeline
                    time_diff = (next_start_time - start_time).total_seconds()
                    expected_rows = int(time_diff * 10)  # 10 measurements per second
                    
                    if abs(expected_rows - num_rows) > 5:  # Allow small tolerance
                        print(f"   ⚠️  Time gap detected between {csv_file.name} and {next_file.name}")
                        print(f"      Expected: {expected_rows} rows, got: {num_rows} rows")
                
                current_row += num_rows
                
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
                continue
        
        return timeline
    
    def get_labels_for_windows(self, window_times: List[datetime]) -> List[int]:
        """
        Get labels for windows based on annotations
        
        Args:
            window_times: List of window start times
            
        Returns:
            List of labels (1 for snoring, 0 for no snoring)
        """
        # Load annotations if not already loaded
        if not hasattr(self, 'annotations') or self.annotations is None:
            if not hasattr(self, 'annotation_file') or self.annotation_file is None:
                self.annotation_file = self.find_annotation_file()
            if self.annotation_file:
                self.annotations = self.parse_annotations(self.annotation_file)
                print(f"   📊 Loaded {len(self.annotations)} annotations for labeling")
            else:
                print(f"   ⚠️  No annotation file found, all labels will be 0")
                return [0] * len(window_times)
        
        labels = []
        
        for window_time in window_times:
            window_end = window_time + timedelta(seconds=8)  # 8-second window
            
            # Check if window overlaps with any W period
            is_snoring = False
            for label, start_time, end_time in self.annotations:
                if label == 'W':  # Snoring period
                    # Check if window overlaps with snoring period
                    if (window_time < end_time and window_end > start_time):
                        is_snoring = True
                        break
            
            labels.append(1 if is_snoring else 0)  # 1 = snoring, 0 = no snoring
        
        return labels
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all data and create features with labels
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        # Find annotation file
        self.annotation_file = self.find_annotation_file()
        if not self.annotation_file:
            raise FileNotFoundError("No annotation file found")
        
        print(f"Found annotation file: {self.annotation_file}")
        
        # Parse annotations
        self.annotations = self.parse_annotations(self.annotation_file)
        print(f"Loaded {len(self.annotations)} annotations")
        
        # Get sorted CSV files
        csv_files = self.get_csv_files_sorted()
        print(f"Found {len(csv_files)} CSV files")
        
        # Create temporal timeline
        timeline = self.create_temporal_timeline(csv_files)
        
        # Load all data into one DataFrame
        all_data = []
        all_times = []
        
        for csv_file, start_time, start_row in timeline:
            try:
                data = pd.read_csv(csv_file, sep=';')
                data['file_time'] = start_time
                data['global_row'] = start_row + np.arange(len(data))
                
                all_data.append(data)
                
                # Create time column for each row
                row_times = [start_time + timedelta(seconds=i/10) for i in range(len(data))]
                all_times.extend(row_times)
                
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Create sliding windows
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from features.feature_extractor import SnoringFeatureExtractor
        
        extractor = SnoringFeatureExtractor()
        windows = extractor.create_sliding_windows(combined_data, all_times[0])
        
        print(f"Created {len(windows)} sliding windows")
        
        # Extract features for each window
        features_list = []
        window_times = []
        
        for start_row, end_row, window_time in windows:
            if end_row <= len(combined_data):
                window_data = combined_data.iloc[start_row:end_row]
                features = extractor.extract_features_from_window(window_data)
                
                features_list.append(features)
                window_times.append(window_time)
        
        features_array = np.array(features_list)
        print(f"Extracted features shape: {features_array.shape}")
        
        # Get labels for windows
        labels = self.get_labels_for_windows(window_times)
        labels_array = np.array(labels)
        
        print(f"Labels shape: {labels_array.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")
        
        # Get feature names
        feature_names = extractor.get_feature_names()
        
        return features_array, labels_array, feature_names 

    def get_base_date_from_folder(self) -> datetime.date:
        """
        Get base date from annotation file folder name
        
        Returns:
            Base date for all time calculations
        """
        # Set annotation_file if not already set
        if not hasattr(self, 'annotation_file') or self.annotation_file is None:
            self.annotation_file = self.find_annotation_file()
        
        if self.annotation_file:
            parent_folder = self.annotation_file.parent.name
            # Try different date formats
            date_match = None
            
            # Format 1: '250812_234316887' -> '250812'
            date_match = re.search(r'(\d{6})_', parent_folder)
            if date_match:
                date_str = date_match.group(1)
                day = int(date_str[:2])
                month = int(date_str[2:4])
                year = 2000 + int(date_str[4:6])  # Assume 20xx
                base_date = datetime(year, month, day).date()
                print(f"   📅 Using base date from folder '{parent_folder}' (format 1): {base_date}")
                return base_date
            
            # Format 2: '4_2025_08_12_23.43' -> '2025_08_12'
            date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', parent_folder)
            if date_match:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                base_date = datetime(year, month, day).date()
                print(f"   📅 Using base date from folder '{parent_folder}' (format 2): {base_date}")
                return base_date
            
            # Format 3: '2025-08-12' -> '2025-08-12'
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', parent_folder)
            if date_match:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                base_date = datetime(year, month, day).date()
                print(f"   📅 Using base date from folder '{parent_folder}' (format 3): {base_date}")
                return base_date
            
            print(f"   ⚠️  Could not parse date from folder '{parent_folder}', using today")
        else:
            print(f"   ⚠️  No annotation file, using today")
        
        # Fallback to today
        today = datetime.now().date()
        print(f"   ⚠️  Using fallback date: {today}")
        return today 