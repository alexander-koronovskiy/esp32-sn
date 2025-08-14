#!/usr/bin/env python3
"""
Fix model parameters and retrain with better settings
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def fix_and_retrain():
    """Fix model parameters and retrain"""
    print("🔧 Fixing Model Parameters and Retraining")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    data_loader = SnoringDataLoader("snoring_data")
    features, labels, feature_names = data_loader.load_all_data()
    
    print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Try different parameter combinations
    parameter_sets = [
        # Set 1: More relaxed constraints
        {
            'max_depth': 10,
            'min_samples_leaf': 2,
            'criterion': 'gini',
            'description': 'Relaxed constraints'
        },
        # Set 2: Balanced approach
        {
            'max_depth': 8,
            'min_samples_leaf': 3,
            'criterion': 'entropy',
            'description': 'Balanced approach'
        },
        # Set 3: Very relaxed
        {
            'max_depth': 15,
            'min_samples_leaf': 1,
            'criterion': 'gini',
            'description': 'Very relaxed constraints'
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
            
            # Train the model
            training_results = classifier.train(features, labels, feature_names)
            
            # Check if tree has more than 1 node
            tree = classifier.classifier.tree_
            n_nodes = len(tree.children_left)
            max_depth = tree.max_depth
            
            print(f"   ✅ Training completed!")
            print(f"   🌳 Tree nodes: {n_nodes}")
            print(f"   🌳 Tree depth: {max_depth}")
            
            # Check feature importances
            if hasattr(classifier.classifier, 'feature_importances_'):
                non_zero_importances = np.sum(classifier.classifier.feature_importances_ > 0)
                print(f"   🔍 Non-zero feature importances: {non_zero_importances}/39")
            
            # Evaluate on test set
            if 'test_metrics' in training_results:
                test_accuracy = training_results['test_metrics']['accuracy']
                print(f"   📊 Test accuracy: {test_accuracy:.4f}")
                
                # Save if this is the best model with multiple nodes
                if n_nodes > 1 and test_accuracy > best_score:
                    best_model = classifier
                    best_score = test_accuracy
                    best_params = params
                    print(f"   🏆 New best model! (Nodes: {n_nodes}, Accuracy: {test_accuracy:.4f})")
            
        except Exception as e:
            print(f"   ❌ Error with this parameter set: {e}")
            continue
    
    # Save the best model
    if best_model is not None:
        print(f"\n🎉 Best model found!")
        print(f"   Parameters: {best_params}")
        print(f"   Test accuracy: {best_score:.4f}")
        
        # Save the improved model
        model_path = "models/improved_snoring_decision_tree.pkl"
        best_model.save_model(model_path)
        print(f"   💾 Improved model saved to: {model_path}")
        
        # Show feature importances
        if hasattr(best_model.classifier, 'feature_importances_'):
            importances = best_model.classifier.feature_importances_
            non_zero_features = np.where(importances > 0)[0]
            
            print(f"\n🔍 Feature Importances Analysis:")
            print(f"   Non-zero importances: {len(non_zero_features)}/39")
            
            if len(non_zero_features) > 0:
                print(f"   Top 10 most important features:")
                top_indices = np.argsort(importances)[-10:][::-1]
                for i, idx in enumerate(top_indices):
                    if importances[idx] > 0:
                        feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                        print(f"      {i+1:2d}. {feature_name}: {importances[idx]:.6f}")
        
        return best_model
    else:
        print("\n❌ No suitable model found with multiple nodes")
        print("   This suggests the data might be too imbalanced or simple")
        return None


if __name__ == "__main__":
    fix_and_retrain() 