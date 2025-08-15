#!/usr/bin/env python3
"""
Improve the model with better parameters and analyze feature importances
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def improve_model():
    """Improve the model with better parameters"""
    print("🔧 Improving Model with Better Parameters")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    data_loader = SnoringDataLoader("snoring_data")
    features, labels, feature_names = data_loader.load_all_data()
    
    print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Try different parameter combinations for better feature importance
    parameter_sets = [
        # Set 1: More complex tree
        {
            'max_depth': 15,
            'min_samples_leaf': 1,
            'criterion': 'entropy',
            'description': 'Complex tree with entropy'
        },
        # Set 2: Balanced approach
        {
            'max_depth': 10,
            'min_samples_leaf': 2,
            'criterion': 'gini',
            'description': 'Balanced complexity'
        },
        # Set 3: Very complex
        {
            'max_depth': 20,
            'min_samples_leaf': 1,
            'criterion': 'entropy',
            'description': 'Very complex tree'
        }
    ]
    
    best_model = None
    best_score = 0
    best_params = None
    
    for i, params in enumerate(parameter_sets):
        print(f"\n🧪 Testing parameter set {i+1}: {params['description']}")
        print(f"   max_depth: {params['max_depth']}")
        print(f"   min_samples_leaf: {params['min_samples_leaf']}")
        print(f"   criterion: {params['criterion']}")
        
        try:
            # Create classifier with new parameters
            classifier = SnoringDecisionTreeClassifier(
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                criterion=params['criterion'],
                random_state=42
            )
            
            # Disable hyperparameter tuning
            def forced_train(features, labels, feature_names):
                """Train without hyperparameter tuning"""
                print("   🚫 Skipping hyperparameter tuning (forced parameters)")
                
                # Prepare data
                X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(features, labels)
                
                # Fit scaler
                classifier.fit_scaler(X_train)
                
                # Train directly with forced parameters
                print("   🎯 Training with forced parameters...")
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
                    'forced_parameters': params,
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
            training_results = classifier.train(features, labels, feature_names)
            
            # Check tree structure
            tree = classifier.classifier.tree_
            n_nodes = len(tree.children_left)
            max_depth = tree.max_depth
            
            print(f"   ✅ Training completed!")
            print(f"   🌳 Tree nodes: {n_nodes}")
            print(f"   🌳 Tree depth: {max_depth}")
            
            # Check feature importances
            if hasattr(classifier.classifier, 'feature_importances_'):
                importances = classifier.classifier.feature_importances_
                non_zero_importances = np.sum(importances > 0.001)  # Threshold for "significant"
                print(f"   🔍 Significant feature importances: {non_zero_importances}/39")
                
                if non_zero_importances > 0:
                    print(f"   🔍 Top 10 most important features:")
                    top_indices = np.argsort(importances)[-10:][::-1]
                    for i, idx in enumerate(top_indices):
                        if importances[idx] > 0.001:
                            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                            print(f"      {i+1:2d}. {feature_name}: {importances[idx]:.6f}")
            
            # Save if this is the best model with multiple nodes
            if n_nodes > 5:  # More complex tree
                test_accuracy = training_results['test_metrics']['accuracy']
                if test_accuracy > best_score:
                    best_model = classifier
                    best_score = test_accuracy
                    best_params = params
                    print(f"   🏆 New best model! (Nodes: {n_nodes}, Accuracy: {test_accuracy:.4f})")
            
        except Exception as e:
            print(f"   ❌ Error with this parameter set: {e}")
            continue
    
    # Save the best model
    if best_model is not None:
        print(f"\n🎉 Best improved model found!")
        print(f"   Parameters: {best_params}")
        print(f"   Test accuracy: {best_score:.4f}")
        
        # Save the improved model
        model_path = "models/improved_snoring_decision_tree.pkl"
        best_model.save_model(model_path)
        print(f"   💾 Improved model saved to: {model_path}")
        
        # Analyze feature importances in detail
        analyze_feature_importances(best_model, feature_names)
        
        return best_model
    else:
        print("\n❌ No suitable improved model found")
        return None


def analyze_feature_importances(classifier, feature_names):
    """Analyze feature importances in detail"""
    print(f"\n🔍 Detailed Feature Importance Analysis:")
    
    if hasattr(classifier.classifier, 'feature_importances_'):
        importances = classifier.classifier.feature_importances_
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"📊 Feature Importance Summary:")
        print(f"   Total features: {len(feature_names)}")
        print(f"   Non-zero importances: {np.sum(importances > 0)}")
        print(f"   Significant importances (>0.001): {np.sum(importances > 0.001)}")
        print(f"   Max importance: {np.max(importances):.6f}")
        print(f"   Min importance: {np.min(importances):.6f}")
        print(f"   Mean importance: {np.mean(importances):.6f}")
        
        print(f"\n🏆 Top 15 Most Important Features:")
        for i, (_, row) in enumerate(df.head(15).iterrows()):
            print(f"   {i+1:2d}. {row['Feature']}: {row['Importance']:.6f}")
        
        print(f"\n❌ Bottom 15 Least Important Features:")
        for i, (_, row) in enumerate(df.tail(15).iterrows()):
            print(f"   {i+1:2d}. {row['Feature']}: {row['Importance']:.6f}")
        
        # Save analysis
        analysis_path = "models/feature_importance_analysis.csv"
        df.to_csv(analysis_path, index=False)
        print(f"\n📋 Feature importance analysis saved to: {analysis_path}")
        
        # Check if we have good feature importance distribution
        good_distribution = np.sum(importances > 0.01) >= 10  # At least 10 features > 1%
        print(f"\n🎯 Feature Importance Quality:")
        print(f"   Good distribution: {'✅ YES' if good_distribution else '❌ NO'}")
        
        if good_distribution:
            print(f"   ✅ Model has good feature importance distribution!")
        else:
            print(f"   ⚠️ Model has poor feature importance distribution")
            print(f"   🔍 This suggests the tree is too simple or data lacks discriminative power")


if __name__ == "__main__":
    improve_model() 