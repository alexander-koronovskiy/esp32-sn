"""
Decision Tree Classifier for Snoring Classification
2 classes: No Snoring / Snoring
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import joblib
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class SnoringDecisionTreeClassifier:
    """
    Decision Tree classifier for snoring detection
    """
    
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 10, 
                 criterion: str = 'gini', random_state: int = 42):
        """
        Initialize classifier
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_leaf: Minimum samples in leaf
            criterion: Split criterion ('gini' or 'entropy')
            random_state: Random seed
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        
        # Initialize components
        self.scaler = RobustScaler()
        self.classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Training results
        self.feature_names = None
        self.feature_importances = None
        self.training_history = {}
        
    def prepare_data(self, features: np.ndarray, labels: np.ndarray, 
                    test_size: float = 0.15, val_size: float = 0.15) -> Tuple:
        """
        Prepare train/validation/test split
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Test set size
            val_size: Validation set size
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, 
            random_state=self.random_state, stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Data split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_scaler(self, X_train: np.ndarray):
        """
        Fit scaler on training data
        
        Args:
            X_train: Training features
        """
        self.scaler.fit(X_train)
        print(f"Scaler fitted on {X_train.shape[0]} training samples")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        return self.scaler.transform(X)
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using validation set
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Best parameters
        """
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_leaf': [5, 10, 15, 20],
            'criterion': ['gini', 'entropy']
        }
        
        # Use GridSearchCV with validation set
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=self.random_state, class_weight='balanced'),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation score: {best_score:.4f}")
        
        # Update classifier with best parameters
        self.classifier = DecisionTreeClassifier(
            **best_params,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # Update instance variables
        self.max_depth = best_params['max_depth']
        self.min_samples_leaf = best_params['min_samples_leaf']
        self.criterion = best_params['criterion']
        
        return best_params
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              feature_names: List[str] = None) -> Dict:
        """
        Train the classifier
        
        Args:
            features: Feature array
            labels: Label array
            feature_names: List of feature names
            
        Returns:
            Training results dictionary
        """
        print("Starting training process...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Prepare data split
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            features, labels
        )
        
        # Fit scaler on training data
        self.fit_scaler(X_train)
        
        # Transform features
        X_train_scaled = self.transform_features(X_train)
        X_val_scaled = self.transform_features(X_val)
        X_test_scaled = self.transform_features(X_test)
        
        # Hyperparameter tuning
        best_params = self.hyperparameter_tuning(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Train final model on train+validation
        X_train_val = np.vstack([X_train_scaled, X_val_scaled])
        y_train_val = np.concatenate([y_train, y_val])
        
        print("Training final model on train+validation data...")
        self.classifier.fit(X_train_val, y_train_val)
        
        # Get feature importances
        self.feature_importances = self.classifier.feature_importances_
        
        # Evaluate on test set
        test_predictions = self.classifier.predict(X_test_scaled)
        test_probabilities = self.classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(y_test, test_predictions, pos_label=1, zero_division=0)
        test_recall = recall_score(y_test, test_predictions, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test, test_predictions, pos_label=1, zero_division=0)
        
        # Store results
        self.training_history = {
            'best_parameters': best_params,
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
            'feature_importances': self.feature_importances.tolist()
        }
        
        print(f"\nTraining completed!")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test precision: {test_precision:.4f}")
        print(f"Test recall: {test_recall:.4f}")
        print(f"Test F1-score: {test_f1:.4f}")
        
        return self.training_history
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            features: Features to predict on
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Transform features
        features_scaled = self.transform_features(features)
        
        # Make predictions
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def get_feature_importance_plot(self, save_path: Optional[str] = None):
        """
        Create feature importance plot
        
        Args:
            save_path: Path to save plot
        """
        if self.feature_importances is None or self.feature_names is None:
            print("No feature importances available. Train the model first.")
            return
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Decision Tree Feature Importances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """
        Save trained model
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        # Save classifier
        joblib.dump(self.classifier, model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler
        if scaler_path is None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Save training history
        history_path = model_path.replace('.pkl', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Load trained model
        
        Args:
            model_path: Path to model
            scaler_path: Path to scaler
        """
        # Load classifier
        self.classifier = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load scaler
        if scaler_path is None:
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        
        # Update parameters
        self.max_depth = self.classifier.max_depth
        self.min_samples_leaf = self.classifier.min_samples_leaf
        self.criterion = self.classifier.criterion
        
        # Load training history if available
        history_path = model_path.replace('.pkl', '_history.json')
        if Path(history_path).exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            print(f"Training history loaded from {history_path}")
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model info
        """
        info = {
            'model_type': 'Decision Tree',
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'feature_count': len(self.feature_names) if self.feature_names else 39,
            'class_names': self.classifier.classes_.tolist() if hasattr(self.classifier, 'classes_') else None,
            'training_history': self.training_history
        }
        
        return info 