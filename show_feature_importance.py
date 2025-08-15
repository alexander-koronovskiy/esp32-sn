#!/usr/bin/env python3
"""
Quick Feature Importance Display
===============================

Simple script to display feature importance in terminal
"""

import json
from pathlib import Path

def load_feature_data():
    """Load feature names and importance data"""
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
    
    return feature_names, training_results['feature_importances']

def display_feature_importance():
    """Display feature importance in a nice format"""
    
    print("🔍 Loading feature importance data...")
    try:
        feature_names, importances = load_feature_data()
        print(f"✅ Loaded {len(feature_names)} features\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Create list of (name, importance) tuples and sort by importance
    feature_data = list(zip(feature_names, importances))
    feature_data.sort(key=lambda x: x[1], reverse=True)
    
    # Display header
    print("🏆 FEATURE IMPORTANCE RANKING")
    print("=" * 60)
    print(f"{'Rank':<4} {'Feature':<35} {'Importance':<12} {'Category':<15}")
    print("-" * 60)
    
    # Display features
    for i, (name, importance) in enumerate(feature_data, 1):
        # Determine category
        if name.startswith('b100_'):
            category = 'Audio (b100)'
        elif name.startswith('b400_'):
            category = 'Audio (b400)'
        elif name.startswith('b1000_'):
            category = 'Audio (b1000)'
        elif name.startswith('env_'):
            category = 'Audio (env)'
        elif name.startswith('accel_'):
            category = 'Accelerometer'
        elif name.startswith('b400_b100') or name.startswith('b400_b1000'):
            category = 'Mixed'
        else:
            category = 'Other'
        
        # Format importance
        if importance > 0:
            importance_str = f"{importance:.6f}"
        else:
            importance_str = "0.000000"
        
        print(f"{i:<4} {name:<35} {importance_str:<12} {category:<15}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)
    
    total_features = len(feature_data)
    significant_features = len([f for f in feature_data if f[1] > 0])
    zero_features = total_features - significant_features
    
    print(f"Total Features: {total_features}")
    print(f"Significant Features (importance > 0): {significant_features}")
    print(f"Zero Importance Features: {zero_features}")
    print(f"Feature Utilization: {significant_features/total_features*100:.1f}%")
    
    # Top features by category
    print(f"\n🏆 TOP FEATURES BY CATEGORY")
    print("=" * 60)
    
    categories = {}
    for name, importance in feature_data:
        if name.startswith('b100_'):
            cat = 'Audio (b100)'
        elif name.startswith('b400_'):
            cat = 'Audio (b400)'
        elif name.startswith('b1000_'):
            cat = 'Audio (b1000)'
        elif name.startswith('env_'):
            cat = 'Audio (env)'
        elif name.startswith('accel_'):
            cat = 'Accelerometer'
        elif name.startswith('b400_b100') or name.startswith('b400_b1000'):
            cat = 'Mixed'
        else:
            cat = 'Other'
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, importance))
    
    for category, features in categories.items():
        if features:
            top_feature = max(features, key=lambda x: x[1])
            print(f"{category:<20}: {top_feature[0]:<30} {top_feature[1]:.6f}")

if __name__ == "__main__":
    display_feature_importance() 