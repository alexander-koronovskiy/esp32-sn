"""
Improved Feature Extractor for Snoring Classification
Extracts 55 features from 8-second sliding windows with better audio features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
from datetime import datetime, timedelta


class ImprovedSnoringFeatureExtractor:
    """
    Extracts 55 features from sliding windows of sensor data
    - 44 audio features (11 per channel × 4 channels)
    - 9 accelerometer features (3 per axis × 3 axes)
    - 2 mixed features (ratios)
    """
    
    def __init__(self, window_size: int = 8, step_size: int = 1, sampling_rate: int = 10):
        """
        Initialize improved feature extractor
        
        Args:
            window_size: Window size in seconds (default: 8)
            step_size: Step size in seconds (default: 1)
            sampling_rate: Measurements per second (default: 10)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.measurements_per_window = window_size * sampling_rate  # 80
        
        # Audio channels
        self.audio_channels = ['b100', 'b400', 'b1000', 'env']
        
        # Accelerometer axes
        self.accel_axes = ['x', 'y', 'z']
        
    def parse_time_from_filename(self, filename: str) -> datetime:
        """
        Parse time from CSV filename
        
        Args:
            filename: CSV filename (e.g., '3_100917.csv')
            
        Returns:
            datetime object
        """
        # Extract time part (e.g., '100917' from '3_100917.csv')
        match = re.search(r'_(\d{6})\.csv$', filename)
        if not match:
            raise ValueError(f"Cannot parse time from filename: {filename}")
        
        time_str = match.group(1)
        
        # Parse HH:MM:SS format
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        # Assume date is today (or from parent folder name)
        today = datetime.now().date()
        return datetime.combine(today, datetime.min.time().replace(
            hour=hour, minute=minute, second=second
        ))
    
    def create_sliding_windows(self, data: pd.DataFrame, start_time: datetime) -> List[Tuple[int, int, datetime]]:
        """
        Create sliding windows with time information
        
        Args:
            data: DataFrame with sensor data
            start_time: Start time of the data
            
        Returns:
            List of (start_row, end_row, window_start_time) tuples
        """
        windows = []
        total_rows = len(data)
        
        # Calculate how many measurements we can step
        step_measurements = self.step_size * self.sampling_rate  # 10
        
        current_row = 0
        current_time = start_time
        
        while current_row + self.measurements_per_window <= total_rows:
            end_row = current_row + self.measurements_per_window
            
            windows.append((current_row, end_row, current_time))
            
            current_row += step_measurements
            current_time += timedelta(seconds=self.step_size)
        
        return windows
    
    def extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract 11 improved features for one audio channel
        
        Args:
            audio_data: Array of audio values for one channel
            
        Returns:
            Dictionary with 11 features
        """
        if len(audio_data) == 0:
            return {
                'mean': 0.0, 'max': 0.0, 'std': 0.0,
                'relative_std': 0.0, 'low_threshold_ratio': 0.0,
                'high_spike_ratio': 0.0, 'trend': 0.0, 'regularity': 0.0,
                'log_mean': 0.0, 'log_std': 0.0, 'log_max': 0.0
            }
        
        # Basic statistics
        mean_level = np.mean(audio_data)
        max_level = np.max(audio_data)
        std_dev = np.std(audio_data)
        
        # Avoid division by zero
        if mean_level < 1e-8:
            mean_level = 1e-8
        
        # 1. Средний уровень сигнала
        mean_level = float(mean_level)
        
        # 2. Максимальный уровень
        max_level = float(max_level)
        
        # 3. Изменчивость (стандартное отклонение)
        std_dev = float(std_dev)
        
        # 4. Относительная изменчивость
        relative_std = std_dev / mean_level
        
        # 5. Доля низких значений (ниже 25-го перцентиля) - НИЗКИЙ ПОРОГ
        low_threshold = np.percentile(audio_data, 25)
        low_threshold_ratio = np.mean(audio_data < low_threshold)
        
        # 6. Доля высоких всплесков (выше 90-го перцентиля) - ВЫСОКИЙ ПОРОГ
        high_threshold = np.percentile(audio_data, 90)
        high_spike_ratio = np.mean(audio_data > high_threshold)
        
        # 7. Тенденция изменения (линейный тренд)
        if len(audio_data) > 1:
            x = np.arange(len(audio_data))
            trend = np.polyfit(x, audio_data, 1)[0]
        else:
            trend = 0.0
        
        # 8. Регулярность (автокорреляция с лагом 1)
        if len(audio_data) > 1:
            regularity = np.corrcoef(audio_data[:-1], audio_data[1:])[0, 1]
            if np.isnan(regularity):
                regularity = 0.0
        else:
            regularity = 0.0
        
        # 9. Логарифмические признаки (для лучшей чувствительности)
        # Добавляем 1 для избежания log(0)
        log_data = np.log1p(audio_data)  # log(1 + x)
        log_mean = float(np.mean(log_data))
        log_std = float(np.std(log_data))
        log_max = float(np.max(log_data))
        
        return {
            'mean': mean_level,
            'max': max_level,
            'std': std_dev,
            'relative_std': float(relative_std),
            'low_threshold_ratio': float(low_threshold_ratio),
            'high_spike_ratio': float(high_spike_ratio),
            'trend': float(trend),
            'regularity': float(regularity),
            'log_mean': log_mean,
            'log_std': log_std,
            'log_max': log_max
        }
    
    def extract_accelerometer_features(self, accel_data: np.ndarray) -> Dict[str, float]:
        """
        Extract 3 features for one accelerometer axis
        
        Args:
            accel_data: Array of accelerometer values for one axis
            
        Returns:
            Dictionary with 3 features
        """
        if len(accel_data) < 2:
            return {
                'mean_activity': 0.0,
                'max_activity': 0.0,
                'movement_ratio': 0.0
            }
        
        # Вычисляем дельты (изменения между соседними измерениями)
        deltas = np.abs(np.diff(accel_data))
        
        # 1. Общая активность (средняя амплитуда движений)
        mean_activity = np.mean(deltas)
        
        # 2. Максимальная активность (самый сильный импульс)
        max_activity = np.max(deltas)
        
        # 3. Доля времени с движением (выше порога)
        movement_threshold = np.percentile(deltas, 50)  # медиана
        movement_ratio = np.mean(deltas > movement_threshold)
        
        return {
            'mean_activity': float(mean_activity),
            'max_activity': float(max_activity),
            'movement_ratio': float(movement_ratio)
        }
    
    def extract_mixed_features(self, audio_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Extract 2 mixed features
        
        Args:
            audio_features: Dictionary with features for each audio channel
            
        Returns:
            Dictionary with 2 mixed features
        """
        # 1. b400/b100 ratio
        b400_mean = audio_features['b400']['mean']
        b100_mean = audio_features['b100']['mean']
        b400_b100_ratio = b400_mean / (b100_mean + 1e-8)
        
        # 2. b400/b1000 ratio
        b1000_mean = audio_features['b1000']['mean']
        b400_b1000_ratio = b400_mean / (b1000_mean + 1e-8)
        
        return {
            'b400_b100_ratio': float(b400_b100_ratio),
            'b400_b1000_ratio': float(b400_b1000_ratio)
        }
    
    def extract_features_from_window(self, window_data: pd.DataFrame) -> np.ndarray:
        """
        Extract all 55 features from one window
        
        Args:
            window_data: DataFrame with 80 rows of sensor data
            
        Returns:
            Array of 55 features
        """
        features = []
        
        # Extract audio features (44 features)
        for channel in self.audio_channels:
            if channel in window_data.columns:
                channel_data = window_data[channel].values
                channel_features = self.extract_audio_features(channel_data)
                features.extend([channel_features[key] for key in [
                    'mean', 'max', 'std', 'relative_std', 'low_threshold_ratio',
                    'high_spike_ratio', 'trend', 'regularity', 'log_mean', 'log_std', 'log_max'
                ]])
            else:
                # If channel missing, fill with zeros
                features.extend([0.0] * 11)
        
        # Extract accelerometer features (9 features)
        for axis in self.accel_axes:
            if axis in window_data.columns:
                axis_data = window_data[axis].values
                axis_features = self.extract_accelerometer_features(axis_data)
                features.extend([axis_features[key] for key in [
                    'mean_activity', 'max_activity', 'movement_ratio'
                ]])
            else:
                # If axis missing, fill with zeros
                features.extend([0.0] * 3)
        
        # Extract mixed features (2 features)
        audio_features = {}
        for channel in self.audio_channels:
            if channel in window_data.columns:
                channel_data = window_data[channel].values
                audio_features[channel] = self.extract_audio_features(channel_data)
            else:
                audio_features[channel] = {'mean': 0.0}
        
        mixed_features = self.extract_mixed_features(audio_features)
        features.extend([mixed_features[key] for key in [
            'b400_b100_ratio', 'b400_b1000_ratio'
        ]])
        
        return np.array(features)
    
    def extract_features_from_csv(self, csv_path: Path) -> Tuple[np.ndarray, List[datetime]]:
        """
        Extract features from one CSV file
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (features_array, window_times)
        """
        # Load CSV data
        data = pd.read_csv(csv_path, sep=';')
        
        # Parse start time from filename
        start_time = self.parse_time_from_filename(csv_path.name)
        
        # Create sliding windows
        windows = self.create_sliding_windows(data, start_time)
        
        features_list = []
        window_times = []
        
        for start_row, end_row, window_time in windows:
            window_data = data.iloc[start_row:end_row]
            features = self.extract_features_from_window(window_data)
            
            features_list.append(features)
            window_times.append(window_time)
        
        return np.array(features_list), window_times
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names
        
        Returns:
            List of 55 feature names
        """
        feature_names = []
        
        # Audio features (44)
        for channel in self.audio_channels:
            for feature in ['mean', 'max', 'std', 'relative_std', 'low_threshold_ratio',
                          'high_spike_ratio', 'trend', 'regularity', 'log_mean', 'log_std', 'log_max']:
                feature_names.append(f"{channel}_{feature}")
        
        # Accelerometer features (9)
        for axis in self.accel_axes:
            for feature in ['mean_activity', 'max_activity', 'movement_ratio']:
                feature_names.append(f"accel_{axis}_{feature}")
        
        # Mixed features (2)
        feature_names.extend(['b400_b100_ratio', 'b400_b1000_ratio'])
        
        return feature_names 