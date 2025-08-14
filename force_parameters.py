#!/usr/bin/env python3
"""
Force training with specific parameters without GridSearchCV
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def force_train_with_parameters():
    """Force training with specific parameters"""
    print("🔧 Force Training with Specific Parameters")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    data_loader = SnoringDataLoader("snoring_data")
    features, labels, feature_names = data_loader.load_all_data()
    
    print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Force specific parameters
    forced_params = {
        'max_depth': 8,
        'min_samples_leaf': 2,
        'criterion': 'gini',
        'random_state': 42
    }
    
    print(f"\n🧪 Training with forced parameters:")
    print(f"   max_depth: {forced_params['max_depth']}")
    print(f"   min_samples_leaf: {forced_params['min_samples_leaf']}")
    print(f"   criterion: {forced_params['criterion']}")
    
    try:
        # Create classifier with forced parameters
        classifier = SnoringDecisionTreeClassifier(
            max_depth=forced_params['max_depth'],
            min_samples_leaf=forced_params['min_samples_leaf'],
            criterion=forced_params['criterion'],
            random_state=forced_params['random_state']
        )
        
        # Disable hyperparameter tuning by modifying the train method
        original_train = classifier.train
        
        def forced_train(features, labels, feature_names):
            """Train without hyperparameter tuning"""
            print("🚫 Skipping hyperparameter tuning (forced parameters)")
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(features, labels)
            
            # Fit scaler
            classifier.fit_scaler(X_train)
            
            # Train directly with forced parameters
            print("🎯 Training with forced parameters...")
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
                'forced_parameters': forced_params,
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
            
            print(f"\nTraining completed!")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Test precision: {test_precision:.4f}")
            print(f"Test recall: {test_recall:.4f}")
            print(f"Test F1-score: {test_f1:.4f}")
            
            return classifier.training_history
        
        # Replace the train method
        classifier.train = forced_train
        
        # Train the model
        training_results = classifier.train(features, labels, feature_names)
        
        # Check tree structure
        tree = classifier.classifier.tree_
        n_nodes = len(tree.children_left)
        max_depth = tree.max_depth
        
        print(f"\n✅ Training completed!")
        print(f"🌳 Tree nodes: {n_nodes}")
        print(f"🌳 Tree depth: {max_depth}")
        
        # Check feature importances
        if hasattr(classifier.classifier, 'feature_importances_'):
            importances = classifier.classifier.feature_importances_
            non_zero_importances = np.sum(importances > 0)
            print(f"🔍 Non-zero feature importances: {non_zero_importances}/39")
            
            if non_zero_importances > 0:
                print(f"🔍 Top 10 most important features:")
                top_indices = np.argsort(importances)[-10:][::-1]
                for i, idx in enumerate(top_indices):
                    if importances[idx] > 0:
                        feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                        print(f"   {i+1:2d}. {feature_name}: {importances[idx]:.6f}")
        
        # Save the improved model
        model_path = "models/forced_parameters_snoring_decision_tree.pkl"
        classifier.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        return classifier
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    force_train_with_parameters() 