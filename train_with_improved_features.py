#!/usr/bin/env python3
"""
Train model with improved audio features (55 total features)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.improved_feature_extractor import ImprovedSnoringFeatureExtractor
from utils.data_loader import SnoringDataLoader
from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def train_with_improved_features():
    """Train model with improved features"""
    print("🚀 Training Model with Improved Audio Features")
    print("=" * 60)
    
    # Create improved extractor
    print("🔧 Creating improved feature extractor...")
    extractor = ImprovedSnoringFeatureExtractor()
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"📋 Total features: {len(feature_names)}")
    
    # Check feature categories
    audio_features = [i for i, name in enumerate(feature_names) if any(band in name for band in ['b100', 'b400', 'b1000', 'env'])]
    accel_features = [i for i, name in enumerate(feature_names) if 'accel' in name]
    mixed_features = [i for i, name in enumerate(feature_names) if 'ratio' in name]
    
    print(f"📊 Feature Categories:")
    print(f"   Audio features: {len(audio_features)} (indices: {audio_features[:5]}...)")
    print(f"   Accelerometer: {len(accel_features)} (indices: {accel_features})")
    print(f"   Mixed features: {len(mixed_features)} (indices: {mixed_features})")
    
    # Load data with improved extractor
    print(f"\n📊 Loading data with improved features...")
    data_loader = SnoringDataLoader("snoring_data")
    
    # Find annotation file
    annotation_file = data_loader.find_annotation_file()
    if not annotation_file:
        raise FileNotFoundError("No annotation file found")
    
    print(f"Found annotation file: {annotation_file}")
    
    # Parse annotations
    annotations = data_loader.parse_annotations(annotation_file)
    print(f"Loaded {len(annotations)} annotations")
    
    # Get sorted CSV files
    csv_files = data_loader.get_csv_files_sorted()
    print(f"Found {len(csv_files)} CSV files")
    
    # Create temporal timeline
    timeline = data_loader.create_temporal_timeline(csv_files)
    
    # Load and combine data with improved extractor
    print(f"   🔧 Using improved feature extractor...")
    
    all_features = []
    all_times = []
    
    for csv_file, start_time, start_row in timeline:
        try:
            # Extract features using improved extractor
            features, window_times = extractor.extract_features_from_csv(csv_file)
            
            # Skip if no features extracted
            if features.shape[0] == 0:
                print(f"   ⚠️  No features extracted from {csv_file.name}")
                continue
            
            # Adjust times based on start_time
            adjusted_times = []
            for wt in window_times:
                # Calculate time offset from start
                time_offset = (wt - start_time).total_seconds()
                adjusted_time = start_time + timedelta(seconds=time_offset)
                adjusted_times.append(adjusted_time)
            
            all_features.append(features)
            all_times.extend(adjusted_times)
            
        except Exception as e:
            print(f"   ⚠️  Error processing {csv_file.name}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features extracted from any CSV files")
    
    # Combine all features
    combined_features = np.vstack(all_features)
    print(f"   📊 Combined features shape: {combined_features.shape}")
    
    # Get labels for windows
    labels = data_loader.get_labels_for_windows(all_times)
    labels_array = np.array(labels)
    print(f"   🏷️  Labels shape: {labels_array.shape}")
    
    print(f"✅ Data loaded: {combined_features.shape[0]} samples, {combined_features.shape[1]} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")
    
    # Create and train classifier
    print(f"\n🎯 Creating and training classifier...")
    
    # Use better parameters for improved features
    classifier = SnoringDecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=1,
        criterion='entropy',
        random_state=42
    )
    
    # Disable hyperparameter tuning to use our improved parameters
    def forced_train(features, labels, feature_names):
        """Train without hyperparameter tuning"""
        print("   🚫 Skipping hyperparameter tuning (using improved parameters)")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(features, labels)
        
        # Fit scaler
        classifier.fit_scaler(X_train)
        
        # Train directly with improved parameters
        print("   🎯 Training with improved parameters...")
        classifier.classifier.fit(X_train, y_train)
        
        # Store feature names
        classifier.feature_names = feature_names
        
        # Get feature importances
        if hasattr(classifier.classifier, 'feature_importances_'):
            classifier.feature_importances = classifier.classifier.feature_importances_
        
        # Evaluate on test set
        X_test_scaled = classifier.transform_features(X_test)
        test_predictions = classifier.classifier.predict(X_test_scaled)
        test_probabilities = classifier.classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(y_test, test_predictions, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test, test_predictions, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test, test_predictions, pos_label=1, zero_division=0)
        
        # Store results
        classifier.training_history = {
            'improved_parameters': {
                'max_depth': 15,
                'min_samples_leaf': 1,
                'criterion': 'entropy'
            },
            'test_metrics': {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1
            },
            'data_shapes': {
                'train': X_train.shape,
                'validation': X_val.shape,
                'test': X_test.shape
            },
            'feature_importances': classifier.feature_importances.tolist() if classifier.feature_importances is not None else []
        }
        
        print(f"   Training completed!")
        print(f"   Test accuracy: {test_accuracy:.4f}")
        print(f"   Test precision: {test_precision:.4f}")
        print(f"   Test recall: {test_recall:.4f}")
        print(f"   Test F1-score: {test_f1:.4f}")
        
        return classifier.training_history
    
    # Replace the train method
    classifier.train = forced_train
    
    # Train the model
    print(f"   🚀 Starting training...")
    training_results = classifier.train(combined_features, labels, feature_names)
    
    # Check tree structure
    tree = classifier.classifier.tree_
    n_nodes = len(tree.children_left)
    max_depth = tree.max_depth
    
    print(f"\n🌳 Model Structure:")
    print(f"   Tree nodes: {n_nodes}")
    print(f"   Tree depth: {max_depth}")
    
    # Check feature importances
    if hasattr(classifier.classifier, 'feature_importances_'):
        importances = classifier.classifier.feature_importances_
        non_zero_importances = np.sum(importances > 0.001)
        print(f"   Significant feature importances: {non_zero_importances}/{len(feature_names)}")
        
        if non_zero_importances > 0:
            print(f"   🔍 Top 15 most important features:")
            top_indices = np.argsort(importances)[-15:][::-1]
            for i, idx in enumerate(top_indices):
                if importances[idx] > 0.001:
                    feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                    print(f"      {i+1:2d}. {feature_name}: {importances[idx]:.6f}")
    
    # Save the improved model
    print(f"\n💾 Saving improved model...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "improved_snoring_decision_tree.pkl"
    classifier.save_model(str(model_path))
    
    # Save feature names
    feature_names_path = models_dir / "improved_feature_names.txt"
    with open(feature_names_path, 'w') as f:
        for i, name in enumerate(feature_names):
            f.write(f"{i:2d}: {name}\n")
    
    # Save training results
    results_path = models_dir / "improved_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    print(f"   ✅ Model saved to: {model_path}")
    print(f"   ✅ Feature names saved to: {feature_names_path}")
    print(f"   ✅ Training results saved to: {results_path}")
    
    # Create summary report
    print(f"\n📋 Creating Summary Report...")
    summary_path = models_dir / "improved_model_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("🚀 IMPROVED SNORING CLASSIFIER - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"📊 Dataset Information:\n")
        f.write(f"   Total samples: {combined_features.shape[0]}\n")
        f.write(f"   Total features: {combined_features.shape[1]}\n")
        f.write(f"   Label distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}\n\n")
        
        f.write(f"🔧 Feature Categories:\n")
        f.write(f"   Audio features: {len(audio_features)} (11 per channel × 4 channels)\n")
        f.write(f"   Accelerometer: {len(accel_features)} (3 per axis × 3 axes)\n")
        f.write(f"   Mixed features: {len(mixed_features)} (ratios)\n\n")
        
        f.write(f"🎯 Model Parameters:\n")
        f.write(f"   max_depth: 15\n")
        f.write(f"   min_samples_leaf: 1\n")
        f.write(f"   criterion: entropy\n\n")
        
        f.write(f"🌳 Model Structure:\n")
        f.write(f"   Tree nodes: {n_nodes}\n")
        f.write(f"   Tree depth: {max_depth}\n")
        f.write(f"   Significant features: {non_zero_importances}/{len(feature_names)}\n\n")
        
        f.write(f"📈 Performance Metrics:\n")
        f.write(f"   Test accuracy: {training_results['test_metrics']['accuracy']:.4f}\n")
        f.write(f"   Test precision: {training_results['test_metrics']['precision']:.4f}\n")
        f.write(f"   Test recall: {training_results['test_metrics']['recall']:.4f}\n")
        f.write(f"   Test F1-score: {training_results['test_metrics']['f1_score']:.4f}\n\n")
        
        f.write(f"🔍 Key Improvements:\n")
        f.write(f"   1. Low threshold ratios (25th percentile) for better sensitivity\n")
        f.write(f"   2. High spike ratios (90th percentile) for detecting anomalies\n")
        f.write(f"   3. Logarithmic transforms (log_mean, log_std, log_max) for better scaling\n")
        f.write(f"   4. Increased total features from 39 to 55\n")
        f.write(f"   5. Better audio feature discrimination\n\n")
        
        f.write(f"📁 Files Created:\n")
        f.write(f"   - Model: {model_path}\n")
        f.write(f"   - Feature names: {feature_names_path}\n")
        f.write(f"   - Training results: {results_path}\n")
        f.write(f"   - Summary: {summary_path}\n")
    
    print(f"   📋 Summary report saved to: {summary_path}")
    
    print(f"\n🎉 Training with improved features completed successfully!")
    print(f"📊 Model performance:")
    print(f"   Accuracy: {training_results['test_metrics']['accuracy']:.4f}")
    print(f"   Precision: {training_results['test_metrics']['precision']:.4f}")
    print(f"   Recall: {training_results['test_metrics']['recall']:.4f}")
    print(f"   F1-Score: {training_results['test_metrics']['f1_score']:.4f}")
    
    return classifier


if __name__ == "__main__":
    train_with_improved_features() 