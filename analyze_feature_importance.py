#!/usr/bin/env python3
"""
Feature Importance Analysis and Visualization
===========================================

This script analyzes the feature importance of the trained snoring classifier
and creates comprehensive visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_model_data():
    """Load the model data from JSON and text files"""
    models_dir = Path("models")
    
    # Load feature names
    feature_names_path = models_dir / "improved_39_features_names.txt"
    with open(feature_names_path, 'r') as f:
        feature_names = []
        for line in f.readlines():
            if ':' in line:
                feature_names.append(line.split(':', 1)[1].strip())
    
    # Load training results
    results_path = models_dir / "improved_39_features_training_results.json"
    with open(results_path, 'r') as f:
        training_results = json.load(f)
    
    return feature_names, training_results

def analyze_feature_importance(feature_names, feature_importances):
    """Analyze feature importance from the training results"""
    
    # Create DataFrame for analysis
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Add feature categories
    def categorize_feature(feature_name):
        if feature_name.startswith('b100_'):
            return 'Audio (b100)'
        elif feature_name.startswith('b400_'):
            return 'Audio (b400)'
        elif feature_name.startswith('b1000_'):
            return 'Audio (b1000)'
        elif feature_name.startswith('env_'):
            return 'Audio (env)'
        elif feature_name.startswith('accel_'):
            return 'Accelerometer'
        elif feature_name.startswith('b400_b100') or feature_name.startswith('b400_b1000'):
            return 'Mixed Features'
        else:
            return 'Other'
    
    importance_df['category'] = importance_df['feature'].apply(categorize_feature)
    
    return importance_df

def create_feature_importance_plots(importance_df):
    """Create comprehensive feature importance visualizations"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Feature Importance Analysis - Snoring Classifier', fontsize=20, fontweight='bold')
    
    # 1. Top features bar plot
    top_features = importance_df.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'], fontsize=10)
    axes[0, 0].set_xlabel('Feature Importance', fontsize=12)
    axes[0, 0].set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()
    
    # 2. Feature importance by category
    category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
    axes[0, 1].pie(category_importance.values, labels=category_importance.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
    
    # 3. Feature importance distribution
    axes[1, 0].hist(importance_df['importance'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Feature Importance', fontsize=12)
    axes[1, 0].set_ylabel('Number of Features', fontsize=12)
    axes[1, 0].set_title('Distribution of Feature Importances', fontsize=14, fontweight='bold')
    axes[1, 0].axvline(importance_df['importance'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {importance_df["importance"].mean():.4f}')
    axes[1, 0].axvline(importance_df['importance'].median(), color='green', linestyle='--', 
                       label=f'Median: {importance_df["importance"].median():.4f}')
    axes[1, 0].legend()
    
    # 4. Feature importance by category (box plot)
    category_data = [importance_df[importance_df['category'] == cat]['importance'].values 
                    for cat in importance_df['category'].unique()]
    category_labels = importance_df['category'].unique()
    
    axes[1, 1].boxplot(category_data, labels=category_labels)
    axes[1, 1].set_ylabel('Feature Importance', fontsize=12)
    axes[1, 1].set_title('Feature Importance Distribution by Category', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_detailed_feature_analysis(importance_df):
    """Create detailed analysis of individual features"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Detailed Feature Analysis - Snoring Classifier', fontsize=20, fontweight='bold')
    
    # 1. Audio features comparison
    audio_features = importance_df[importance_df['category'].str.contains('Audio')]
    audio_channels = ['b100', 'b400', 'b1000', 'env']
    
    audio_data = []
    audio_labels = []
    for channel in audio_channels:
        channel_features = audio_features[audio_features['feature'].str.startswith(f'{channel}_')]
        audio_data.append(channel_features['importance'].values)
        audio_labels.append(channel)
    
    axes[0, 0].boxplot(audio_data, labels=audio_labels)
    axes[0, 0].set_ylabel('Feature Importance', fontsize=12)
    axes[0, 0].set_title('Audio Features by Channel', fontsize=14, fontweight='bold')
    
    # 2. Accelerometer features
    accel_features = importance_df[importance_df['category'] == 'Accelerometer']
    accel_axes = ['x', 'y', 'z']
    
    accel_data = []
    for axis in accel_axes:
        axis_features = accel_features[accel_features['feature'].str.contains(f'accel_{axis}')]
        accel_data.append(axis_features['importance'].values)
    
    axes[0, 1].boxplot(accel_data, labels=accel_axes)
    axes[0, 1].set_ylabel('Feature Importance', fontsize=12)
    axes[0, 1].set_title('Accelerometer Features by Axis', fontsize=14, fontweight='bold')
    
    # 3. Feature types within each category
    feature_types = ['mean', 'max', 'std', 'relative_std', 'high_threshold_ratio', 'trend', 'regularity']
    
    # Audio feature types
    audio_type_data = []
    for feature_type in feature_types:
        type_features = audio_features[audio_features['feature'].str.endswith(feature_type)]
        audio_type_data.append(type_features['importance'].values)
    
    axes[1, 0].boxplot(audio_type_data, labels=feature_types)
    axes[1, 0].set_ylabel('Feature Importance', fontsize=12)
    axes[1, 0].set_title('Audio Features by Type', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Mixed features
    mixed_features = importance_df[importance_df['category'] == 'Mixed Features']
    if not mixed_features.empty:
        axes[1, 1].bar(range(len(mixed_features)), mixed_features['importance'])
        axes[1, 1].set_xticks(range(len(mixed_features)))
        axes[1, 1].set_xticklabels(mixed_features['feature'], rotation=45)
        axes[1, 1].set_ylabel('Feature Importance', fontsize=12)
        axes[1, 1].set_title('Mixed Features', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No mixed features found', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Mixed Features', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_summary_report(importance_df, training_results):
    """Create a comprehensive summary report"""
    
    # Calculate statistics
    total_features = len(importance_df)
    significant_features = len(importance_df[importance_df['importance'] > 0])
    zero_importance_features = len(importance_df[importance_df['importance'] == 0])
    
    # Category breakdown
    category_stats = importance_df.groupby('category').agg({
        'importance': ['count', 'sum', 'mean', 'std']
    }).round(4)
    
    # Top features
    top_5_features = importance_df.head(5)
    
    # Create report
    report = f"""
Feature Importance Analysis Report
=================================

Model Performance:
- Test Accuracy: {training_results.get('test_metrics', {}).get('accuracy', 'N/A')}
- Test Precision: {training_results.get('test_metrics', {}).get('precision', 'N/A')}
- Test Recall: {training_results.get('test_metrics', {}).get('recall', 'N/A')}
- Test F1-Score: {training_results.get('test_metrics', {}).get('f1_score', 'N/A')}

Feature Statistics:
- Total Features: {total_features}
- Significant Features (importance > 0): {significant_features}
- Zero Importance Features: {zero_importance_features}
- Feature Utilization: {significant_features/total_features*100:.1f}%

Top 5 Most Important Features:
"""
    
    for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
        report += f"{i}. {row['feature']}: {row['importance']:.6f}\n"
    
    report += f"\nCategory Breakdown:\n"
    report += str(category_stats)
    
    report += f"\n\nFeature Importance Summary:\n"
    report += f"- Highest Importance: {importance_df['importance'].max():.6f}\n"
    report += f"- Lowest Importance: {importance_df['importance'].min():.6f}\n"
    report += f"- Mean Importance: {importance_df['importance'].mean():.6f}\n"
    report += f"- Median Importance: {importance_df['importance'].median():.6f}\n"
    report += f"- Standard Deviation: {importance_df['importance'].std():.6f}\n"
    
    return report

def main():
    """Main function to run the feature importance analysis"""
    
    print("🔍 Loading model data...")
    try:
        feature_names, training_results = load_model_data()
        print(f"✅ Data loaded successfully with {len(feature_names)} features")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("\n📊 Analyzing feature importance...")
    feature_importances = training_results['feature_importances']
    importance_df = analyze_feature_importance(feature_names, feature_importances)
    
    # Create output directory
    output_dir = Path("feature_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("\n🎨 Creating visualizations...")
    
    # 1. Main feature importance plots
    fig1 = create_feature_importance_plots(importance_df)
    fig1.savefig(output_dir / "feature_importance_overview.png", dpi=300, bbox_inches='tight')
    print("   ✅ Saved: feature_importance_overview.png")
    
    # 2. Detailed feature analysis
    fig2 = create_detailed_feature_analysis(importance_df)
    fig2.savefig(output_dir / "detailed_feature_analysis.png", dpi=300, bbox_inches='tight')
    print("   ✅ Saved: detailed_feature_analysis.png")
    
    # 3. Save feature importance data
    importance_df.to_csv(output_dir / "feature_importance_analysis.csv", index=False)
    print("   ✅ Saved: feature_importance_analysis.csv")
    
    # 4. Create and save summary report
    report = create_summary_report(importance_df, training_results)
    with open(output_dir / "feature_importance_report.txt", 'w') as f:
        f.write(report)
    print("   ✅ Saved: feature_importance_report.txt")
    
    # 5. Display top features
    print(f"\n🏆 Top 10 Most Important Features:")
    print("=" * 50)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.6f}")
    
    print(f"\n📁 All files saved to: {output_dir}")
    print("🎉 Feature importance analysis completed!")

if __name__ == "__main__":
    main() 