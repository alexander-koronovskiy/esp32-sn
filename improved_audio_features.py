#!/usr/bin/env python3
"""
Improved Audio Feature Extraction with Low Thresholds and Logarithmic Features
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from features.feature_extractor import SnoringFeatureExtractor


def create_improved_extractor():
    """Create an improved feature extractor with better audio features"""
    
    class ImprovedSnoringFeatureExtractor(SnoringFeatureExtractor):
        """Improved feature extractor with better audio features"""
        
        def extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, float]:
            """
            Extract improved audio features with low thresholds and log transforms
            
            Args:
                audio_data: Array of audio values for one channel
                
            Returns:
                Dictionary with improved features
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
        
        def get_feature_names(self) -> List[str]:
            """Get improved feature names"""
            feature_names = []
            
            # Audio features (44 = 11 per channel × 4 channels)
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
        
        def extract_features_from_window(self, window_data: pd.DataFrame) -> np.ndarray:
            """Extract all features from one window"""
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
    
    return ImprovedSnoringFeatureExtractor()


def test_improved_features():
    """Test the improved feature extraction"""
    print("🧪 Testing Improved Audio Feature Extraction")
    print("=" * 50)
    
    # Create improved extractor
    extractor = create_improved_extractor()
    
    # Load data
    print("📊 Loading data...")
    data_loader = SnoringDataLoader("snoring_data")
    features, labels, feature_names = data_loader.load_all_data()
    
    print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Check feature names
    print(f"\n📋 Feature Categories:")
    audio_features = [i for i, name in enumerate(feature_names) if any(band in name for band in ['b100', 'b400', 'b1000', 'env'])]
    accel_features = [i for i, name in enumerate(feature_names) if 'accel' in name]
    mixed_features = [i for i, name in enumerate(feature_names) if 'ratio' in name]
    
    print(f"   Audio features: {len(audio_features)} (indices: {audio_features[:5]}...)")
    print(f"   Accelerometer: {len(accel_features)} (indices: {accel_features})")
    print(f"   Mixed features: {len(mixed_features)} (indices: {mixed_features})")
    
    # Analyze feature distributions by class
    print(f"\n🔍 Analyzing Improved Feature Distributions by Class...")
    
    # Class 0 (No Snoring)
    class_0_mask = labels == 0
    features_class_0 = features[class_0_mask]
    
    # Class 1 (Snoring) - very few samples
    class_1_mask = labels == 1
    features_class_1 = features[class_1_mask]
    
    print(f"   Class 0 samples: {np.sum(class_0_mask)}")
    print(f"   Class 1 samples: {np.sum(class_1_mask)}")
    
    # Analyze audio features with new thresholds
    print(f"\n📊 Improved Audio Features Analysis:")
    print(f"   {'=' * 50}")
    
    # Test different feature types
    test_features = [
        ('b100_low_threshold_ratio', 'Low threshold ratio (25th percentile)'),
        ('b100_high_spike_ratio', 'High spike ratio (90th percentile)'),
        ('b100_log_mean', 'Logarithmic mean'),
        ('b100_log_std', 'Logarithmic standard deviation'),
        ('b400_low_threshold_ratio', 'Low threshold ratio (25th percentile)'),
        ('b400_high_spike_ratio', 'High spike ratio (90th percentile)'),
        ('b400_log_mean', 'Logarithmic mean'),
        ('env_low_threshold_ratio', 'Low threshold ratio (25th percentile)'),
        ('env_high_spike_ratio', 'High spike ratio (90th percentile)')
    ]
    
    for feature_name, description in test_features:
        try:
            # Find feature index
            feature_idx = None
            for i, name in enumerate(feature_names):
                if name == feature_name:
                    feature_idx = i
                    break
            
            if feature_idx is not None:
                # Get values for both classes
                values_class_0 = features_class_0[:, feature_idx]
                values_class_1 = features_class_1[:, feature_idx]
                
                # Basic statistics
                mean_0 = np.mean(values_class_0)
                mean_1 = np.mean(values_class_1)
                std_0 = np.std(values_class_0)
                std_1 = np.std(values_class_1)
                
                # Check if there's a meaningful difference
                diff = abs(mean_1 - mean_0)
                relative_diff = diff / (std_0 + std_1 + 1e-8)
                
                # Determine if feature is discriminative
                is_discriminative = relative_diff > 0.3  # Lower threshold for improved features
                
                print(f"   {feature_name}:")
                print(f"     Description: {description}")
                print(f"     Class 0: mean={mean_0:.4f}, std={std_0:.4f}")
                print(f"     Class 1: mean={mean_1:.4f}, std={std_1:.4f}")
                print(f"     Difference: {diff:.4f} (relative: {relative_diff:.4f})")
                print(f"     Discriminative: {'✅ YES' if is_discriminative else '❌ NO'}")
                print()
            
        except Exception as e:
            print(f"   ❌ Error analyzing {feature_name}: {e}")
    
    # Create visualization
    print(f"\n🎨 Creating Improved Features Visualization...")
    create_improved_features_visualization(features, labels, feature_names, audio_features)
    
    print(f"\n✅ Improved feature analysis completed!")


def create_improved_features_visualization(features, labels, feature_names, audio_features):
    """Create visualization for improved features"""
    # Create output directory
    output_dir = Path("improved_features_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Select some key improved features
    key_features = [
        'b100_low_threshold_ratio', 'b100_high_spike_ratio', 'b100_log_mean',
        'b400_low_threshold_ratio', 'b400_high_spike_ratio', 'b400_log_mean',
        'env_low_threshold_ratio', 'env_high_spike_ratio', 'env_log_mean'
    ]
    
    # Find indices
    feature_indices = []
    feature_labels = []
    for name in key_features:
        for i, feature_name in enumerate(feature_names):
            if feature_name == name:
                feature_indices.append(i)
                feature_labels.append(name)
                break
    
    if not feature_indices:
        print("   ❌ No improved features found for visualization")
        return
    
    # Create subplots
    n_features = len(feature_indices)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each feature
    for i, (idx, name) in enumerate(zip(feature_indices, feature_labels)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get values for both classes
        class_0_mask = labels == 0
        class_1_mask = labels == 1
        
        values_0 = features[class_0_mask, idx]
        values_1 = features[class_1_mask, idx]
        
        # Create box plot
        data_to_plot = [values_0, values_1]
        labels_plot = ['No Snoring', 'Snoring']
        colors = ['lightblue', 'lightcoral']
        
        bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{name}', fontsize=10)
        ax.set_ylabel('Feature Value')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_0, mean_1 = np.mean(values_0), np.mean(values_1)
        diff = abs(mean_1 - mean_0)
        ax.text(0.02, 0.98, f'Diff: {diff:.3f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Improved Audio Features: Comparison Between Classes', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'improved_audio_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   🎨 Visualization saved to: {output_dir}")


if __name__ == "__main__":
    test_improved_features() 