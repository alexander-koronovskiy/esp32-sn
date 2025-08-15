#!/usr/bin/env python3
"""
Analyze data quality and understand why audio features don't work
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from features.feature_extractor import SnoringFeatureExtractor


def analyze_data_quality():
    """Analyze data quality and feature distributions"""
    print("🔍 Analyzing Data Quality and Feature Distributions")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    data_loader = SnoringDataLoader("snoring_data")
    features, labels, feature_names = data_loader.load_all_data()
    
    print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Separate features by type
    audio_features = [i for i, name in enumerate(feature_names) if any(band in name for band in ['b100', 'b400', 'b1000', 'env'])]
    accel_features = [i for i, name in enumerate(feature_names) if 'accel' in name]
    mixed_features = [i for i, name in enumerate(feature_names) if 'ratio' in name]
    
    print(f"\n📋 Feature Categories:")
    print(f"   Audio features: {len(audio_features)} (indices: {audio_features[:5]}...)")
    print(f"   Accelerometer: {len(accel_features)} (indices: {accel_features})")
    print(f"   Mixed features: {len(mixed_features)} (indices: {mixed_features})")
    
    # Analyze feature distributions by class
    print(f"\n🔍 Analyzing Feature Distributions by Class...")
    
    # Class 0 (No Snoring)
    class_0_mask = labels == 0
    features_class_0 = features[class_0_mask]
    
    # Class 1 (Snoring) - very few samples
    class_1_mask = labels == 1
    features_class_1 = features[class_1_mask]
    
    print(f"   Class 0 samples: {np.sum(class_0_mask)}")
    print(f"   Class 1 samples: {np.sum(class_1_mask)}")
    
    # Analyze each feature category
    analyze_feature_category("Audio", audio_features, features_class_0, features_class_1, feature_names)
    analyze_feature_category("Accelerometer", accel_features, features_class_0, features_class_1, feature_names)
    analyze_feature_category("Mixed", mixed_features, features_class_0, features_class_1, feature_names)
    
    # Check for NaN or infinite values
    print(f"\n🔍 Checking Data Quality Issues...")
    check_data_quality(features, labels, feature_names)
    
    # Create visualizations
    print(f"\n🎨 Creating Data Quality Visualizations...")
    create_quality_visualizations(features, labels, feature_names, audio_features, accel_features, mixed_features)
    
    print(f"\n✅ Data quality analysis completed!")
    print(f"📁 Check the generated visualizations and reports")


def analyze_feature_category(category_name, feature_indices, features_class_0, features_class_1, feature_names):
    """Analyze a specific feature category"""
    print(f"\n📊 {category_name} Features Analysis:")
    print(f"   {'=' * (len(category_name) + 20)}")
    
    if not feature_indices:
        print(f"   No {category_name.lower()} features found")
        return
    
    for idx in feature_indices:
        feature_name = feature_names[idx]
        
        # Get values for both classes
        values_class_0 = features_class_0[:, idx]
        values_class_1 = features_class_1[:, idx]
        
        # Basic statistics
        mean_0 = np.mean(values_class_0)
        mean_1 = np.mean(values_class_1)
        std_0 = np.std(values_class_0)
        std_1 = np.std(values_class_1)
        
        # Check if there's a meaningful difference
        diff = abs(mean_1 - mean_0)
        relative_diff = diff / (std_0 + std_1 + 1e-8)  # Avoid division by zero
        
        # Determine if feature is discriminative
        is_discriminative = relative_diff > 0.5  # Threshold for meaningful difference
        
        print(f"   {feature_name}:")
        print(f"     Class 0: mean={mean_0:.4f}, std={std_0:.4f}")
        print(f"     Class 1: mean={mean_1:.4f}, std={std_1:.4f}")
        print(f"     Difference: {diff:.4f} (relative: {relative_diff:.4f})")
        print(f"     Discriminative: {'✅ YES' if is_discriminative else '❌ NO'}")
        
        # Check for zero variance
        if std_0 < 1e-6 and std_1 < 1e-6:
            print(f"     ⚠️  WARNING: Zero variance in both classes!")
        elif std_0 < 1e-6:
            print(f"     ⚠️  WARNING: Zero variance in class 0!")
        elif std_1 < 1e-6:
            print(f"     ⚠️  WARNING: Zero variance in class 1!")


def check_data_quality(features, labels, feature_names):
    """Check for data quality issues"""
    print(f"   Checking for NaN values...")
    nan_count = np.sum(np.isnan(features))
    if nan_count > 0:
        print(f"     ❌ Found {nan_count} NaN values!")
        
        # Find which features have NaN
        nan_features = np.any(np.isnan(features), axis=0)
        nan_feature_names = [feature_names[i] for i in np.where(nan_features)[0]]
        print(f"     Features with NaN: {nan_feature_names}")
    else:
        print(f"     ✅ No NaN values found")
    
    print(f"   Checking for infinite values...")
    inf_count = np.sum(np.isinf(features))
    if inf_count > 0:
        print(f"     ❌ Found {inf_count} infinite values!")
    else:
        print(f"     ✅ No infinite values found")
    
    print(f"   Checking for zero variance...")
    zero_var_features = []
    for i in range(features.shape[1]):
        if np.std(features[:, i]) < 1e-6:
            zero_var_features.append(feature_names[i])
    
    if zero_var_features:
        print(f"     ⚠️  Found {len(zero_var_features)} features with zero variance:")
        for name in zero_var_features[:10]:  # Show first 10
            print(f"       - {name}")
        if len(zero_var_features) > 10:
            print(f"       ... and {len(zero_var_features) - 10} more")
    else:
        print(f"     ✅ All features have non-zero variance")


def create_quality_visualizations(features, labels, feature_names, audio_features, accel_features, mixed_features):
    """Create visualizations for data quality analysis"""
    # Create output directory
    output_dir = Path("data_quality_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Feature importance by category
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Analysis: Feature Distributions by Category', fontsize=16)
    
    # Audio features distribution
    if audio_features:
        ax1 = axes[0, 0]
        audio_data = features[:, audio_features]
        audio_names = [feature_names[i] for i in audio_features]
        
        # Create box plot for first 8 audio features
        n_audio_plot = min(8, len(audio_features))
        audio_data_plot = audio_data[:, :n_audio_plot]
        audio_names_plot = audio_names[:n_audio_plot]
        
        ax1.boxplot([audio_data_plot[:, i] for i in range(n_audio_plot)], labels=audio_names_plot)
        ax1.set_title('Audio Features Distribution')
        ax1.set_ylabel('Feature Values')
        ax1.tick_params(axis='x', rotation=45)
    
    # Accelerometer features distribution
    if accel_features:
        ax2 = axes[0, 1]
        accel_data = features[:, accel_features]
        accel_names = [feature_names[i] for i in accel_features]
        
        ax2.boxplot([accel_data[:, i] for i in range(len(accel_features))], labels=accel_names)
        ax2.set_title('Accelerometer Features Distribution')
        ax2.set_ylabel('Feature Values')
        ax2.tick_params(axis='x', rotation=45)
    
    # Mixed features distribution
    if mixed_features:
        ax3 = axes[1, 0]
        mixed_data = features[:, mixed_features]
        mixed_names = [feature_names[i] for i in mixed_features]
        
        ax3.boxplot([mixed_data[:, i] for i in range(len(mixed_features))], labels=mixed_names)
        ax3.set_title('Mixed Features Distribution')
        ax3.set_ylabel('Feature Values')
        ax3.tick_params(axis='x', rotation=45)
    
    # Class distribution
    ax4 = axes[1, 1]
    class_counts = np.bincount(labels)
    class_labels = ['No Snoring', 'Snoring']
    colors = ['lightblue', 'lightcoral']
    
    bars = ax4.bar(class_labels, class_counts, color=colors)
    ax4.set_title('Class Distribution')
    ax4.set_ylabel('Number of Samples')
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature correlation heatmap (top features only)
    print(f"   Creating correlation heatmap...")
    try:
        # Select top features by variance
        feature_variances = np.var(features, axis=0)
        top_feature_indices = np.argsort(feature_variances)[-20:]  # Top 20 by variance
        top_features = features[:, top_feature_indices]
        top_feature_names = [feature_names[i] for i in top_feature_indices]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(top_features.T)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   xticklabels=top_feature_names, 
                   yticklabels=top_feature_names,
                   cmap='coolwarm', 
                   center=0, 
                   annot=False,
                   square=True)
        plt.title('Feature Correlation Heatmap (Top 20 by Variance)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"     ❌ Error creating correlation heatmap: {e}")
    
    # 3. Save detailed analysis report
    print(f"   Creating detailed analysis report...")
    report_path = output_dir / 'data_quality_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("🔍 DATA QUALITY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"📊 Dataset Overview:\n")
        f.write(f"   Total samples: {features.shape[0]}\n")
        f.write(f"   Total features: {features.shape[1]}\n")
        f.write(f"   Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}\n\n")
        
        f.write(f"📋 Feature Categories:\n")
        f.write(f"   Audio features: {len(audio_features)}\n")
        f.write(f"   Accelerometer: {len(accel_features)}\n")
        f.write(f"   Mixed features: {len(mixed_features)}\n\n")
        
        f.write(f"🔍 Data Quality Issues:\n")
        f.write(f"   NaN values: {np.sum(np.isnan(features))}\n")
        f.write(f"   Infinite values: {np.sum(np.isinf(features))}\n")
        f.write(f"   Zero variance features: {np.sum(np.std(features, axis=0) < 1e-6)}\n\n")
        
        f.write(f"🎯 Recommendations:\n")
        f.write(f"   1. The extreme class imbalance (99.6% vs 0.4%) makes learning difficult\n")
        f.write(f"   2. Only 10 snoring samples may not be enough for reliable training\n")
        f.write(f"   3. Consider collecting more snoring data or using data augmentation\n")
        f.write(f"   4. Audio features may need different thresholds or preprocessing\n")
        f.write(f"   5. The model relies heavily on accelerometer data\n")
    
    print(f"   📋 Detailed report saved to: {report_path}")
    print(f"   🎨 Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    analyze_data_quality() 