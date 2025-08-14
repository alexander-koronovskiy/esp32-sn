#!/usr/bin/env python3
"""
Analyze raw features without normalization to understand why they're not discriminative
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


def analyze_raw_features():
    """Analyze raw features without normalization"""
    print("🔍 Analyzing Raw Features")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    data_loader = SnoringDataLoader("snoring_data")
    features, labels, feature_names = data_loader.load_all_data()
    
    print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(features, columns=feature_names)
    df['label'] = labels
    
    print(f"\n📊 Feature Statistics:")
    print(f"   Features shape: {df.shape}")
    print(f"   Label 0 samples: {np.sum(labels == 0)}")
    print(f"   Label 1 samples: {np.sum(labels == 1)}")
    
    # Analyze feature distributions for each class
    print(f"\n🔍 Analyzing Feature Distributions by Class:")
    
    # Get samples for each class
    class_0_samples = df[df['label'] == 0].iloc[:, :-1]  # Exclude label column
    class_1_samples = df[df['label'] == 1].iloc[:, :-1]
    
    print(f"   Class 0 samples: {class_0_samples.shape}")
    print(f"   Class 1 samples: {class_1_samples.shape}")
    
    # Calculate basic statistics for each class
    print(f"\n📈 Basic Statistics by Class:")
    
    # For each feature, compare statistics between classes
    feature_analysis = []
    
    for i, feature_name in enumerate(feature_names):
        class_0_values = class_0_samples.iloc[:, i]
        class_1_values = class_1_samples.iloc[:, i]
        
        # Basic statistics
        class_0_mean = class_0_values.mean()
        class_0_std = class_0_values.std()
        class_1_mean = class_1_values.mean()
        class_1_std = class_1_values.std()
        
        # Difference between means
        mean_diff = abs(class_1_mean - class_0_mean)
        
        # Coefficient of variation (std/mean) for each class
        cv_0 = class_0_std / (abs(class_0_mean) + 1e-8)
        cv_1 = class_1_std / (abs(class_1_mean) + 1e-8)
        
        # Store analysis
        feature_analysis.append({
            'feature': feature_name,
            'class_0_mean': class_0_mean,
            'class_0_std': class_0_std,
            'class_1_mean': class_1_mean,
            'class_1_std': class_1_std,
            'mean_difference': mean_diff,
            'cv_0': cv_0,
            'cv_1': cv_1,
            'discriminative_power': mean_diff / (class_0_std + class_1_std + 1e-8)
        })
    
    # Convert to DataFrame and sort by discriminative power
    analysis_df = pd.DataFrame(feature_analysis)
    analysis_df = analysis_df.sort_values('discriminative_power', ascending=False)
    
    print(f"\n🏆 Top 10 Most Discriminative Features:")
    print(analysis_df.head(10)[['feature', 'mean_difference', 'discriminative_power']].to_string(index=False))
    
    print(f"\n❌ Bottom 10 Least Discriminative Features:")
    print(analysis_df.tail(10)[['feature', 'mean_difference', 'discriminative_power']].to_string(index=False))
    
    # Check if any features have significant differences
    significant_features = analysis_df[analysis_df['discriminative_power'] > 0.1]
    print(f"\n🔍 Features with discriminative_power > 0.1: {len(significant_features)}")
    
    if len(significant_features) > 0:
        print("   These features might be useful for classification:")
        for _, row in significant_features.head(5).iterrows():
            print(f"   - {row['feature']}: power={row['discriminative_power']:.4f}")
    else:
        print("   ⚠️ No features have significant discriminative power!")
        print("   This explains why the tree has only 1 node.")
    
    # Create visualizations
    create_feature_analysis_plots(df, analysis_df)
    
    return analysis_df


def create_feature_analysis_plots(df, analysis_df):
    """Create visualization plots for feature analysis"""
    print(f"\n🎨 Creating Feature Analysis Plots...")
    
    # Create output directory
    output_dir = Path("feature_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Feature importance by discriminative power
    plt.figure(figsize=(12, 8))
    top_features = analysis_df.head(20)
    plt.barh(range(len(top_features)), top_features['discriminative_power'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Discriminative Power')
    plt.title('Top 20 Most Discriminative Features')
    plt.tight_layout()
    plt.savefig(output_dir / "discriminative_power.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean differences between classes
    plt.figure(figsize=(12, 8))
    top_features = analysis_df.head(20)
    plt.barh(range(len(top_features)), top_features['mean_difference'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Absolute Mean Difference Between Classes')
    plt.title('Top 20 Features by Mean Difference')
    plt.tight_layout()
    plt.savefig(output_dir / "mean_differences.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature value distributions for top 5 features
    top_5_features = analysis_df.head(5)['feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_5_features):
        if i < len(axes):
            # Plot distributions for each class
            class_0_data = df[df['label'] == 0][feature]
            class_1_data = df[df['label'] == 1][feature]
            
            axes[i].hist(class_0_data, alpha=0.7, label='No Snoring', bins=30, density=True)
            axes[i].hist(class_1_data, alpha=0.7, label='Snoring', bins=30, density=True)
            axes[i].set_title(f'{feature}')
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(top_5_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis data
    analysis_df.to_csv(output_dir / "feature_analysis.csv", index=False)
    
    print(f"   📊 Plots saved to: {output_dir}")
    print(f"   📋 Analysis data saved to: {output_dir}/feature_analysis.csv")


if __name__ == "__main__":
    analyze_raw_features() 