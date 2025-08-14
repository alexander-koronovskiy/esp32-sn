#!/usr/bin/env python3
"""
Balance classes using SMOTE and retrain the model
"""

import sys
import os
from pathlib import Path
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def balance_and_retrain():
    """Balance classes and retrain the model"""
    print("⚖️ Balancing Classes and Retraining")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    data_loader = SnoringDataLoader("snoring_data")
    features, labels, feature_names = data_loader.load_all_data()
    
    print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"📊 Original label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Try different balancing strategies
    balancing_strategies = [
        {
            'name': 'SMOTE (Oversampling)',
            'method': SMOTE(random_state=42, k_neighbors=min(5, np.sum(labels == 1) - 1))
        },
        {
            'name': 'Random Under-sampling',
            'method': RandomUnderSampler(random_state=42)
        },
        {
            'name': 'SMOTE + ENN',
            'method': SMOTEENN(random_state=42)
        }
    ]
    
    best_model = None
    best_score = 0
    best_strategy = None
    
    for strategy in balancing_strategies:
        print(f"\n🧪 Testing strategy: {strategy['name']}")
        
        try:
            # Apply balancing
            if strategy['name'] == 'SMOTE (Oversampling)':
                # For SMOTE, we need at least 2 samples of minority class
                if np.sum(labels == 1) < 2:
                    print("   ⚠️ Skipping SMOTE - need at least 2 minority samples")
                    continue
                
                # Ensure k_neighbors is valid
                k_neighbors = min(5, np.sum(labels == 1) - 1)
                if k_neighbors < 1:
                    print("   ⚠️ Skipping SMOTE - k_neighbors < 1")
                    continue
                
                balancer = SMOTE(random_state=42, k_neighbors=k_neighbors)
            else:
                balancer = strategy['method']
            
            # Balance the data
            features_balanced, labels_balanced = balancer.fit_resample(features, labels)
            
            print(f"   📊 Balanced distribution: {dict(zip(*np.unique(labels_balanced, return_counts=True)))}")
            
            # Create classifier with forced parameters
            classifier = SnoringDecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=2,
                criterion='gini',
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
                    'balancing_strategy': strategy['name'],
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
            training_results = classifier.train(features_balanced, labels_balanced, feature_names)
            
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
                non_zero_importances = np.sum(importances > 0)
                print(f"   🔍 Non-zero feature importances: {non_zero_importances}/39")
                
                if non_zero_importances > 0:
                    print(f"   🔍 Top 5 most important features:")
                    top_indices = np.argsort(importances)[-5:][::-1]
                    for i, idx in enumerate(top_indices):
                        if importances[idx] > 0:
                            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                            print(f"      {i+1}. {feature_name}: {importances[idx]:.6f}")
            
            # Save if this is the best model with multiple nodes
            if n_nodes > 1:
                test_accuracy = training_results['test_metrics']['accuracy']
                if test_accuracy > best_score:
                    best_model = classifier
                    best_score = test_accuracy
                    best_strategy = strategy['name']
                    print(f"   🏆 New best model! (Nodes: {n_nodes}, Accuracy: {test_score:.4f})")
            
        except Exception as e:
            print(f"   ❌ Error with this strategy: {e}")
            continue
    
    # Save the best model
    if best_model is not None:
        print(f"\n🎉 Best balanced model found!")
        print(f"   Strategy: {best_strategy}")
        print(f"   Test accuracy: {best_score:.4f}")
        
        # Save the improved model
        model_path = "models/balanced_snoring_decision_tree.pkl"
        best_model.save_model(model_path)
        print(f"   💾 Balanced model saved to: {model_path}")
        
        return best_model
    else:
        print("\n❌ No suitable balanced model found")
        print("   This suggests the data might be too simple or features lack discriminative power")
        return None


if __name__ == "__main__":
    balance_and_retrain() 