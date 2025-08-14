#!/usr/bin/env python3
"""
Main script for training Snoring Decision Tree Classifier
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.data_loader import SnoringDataLoader
from models.decision_tree_classifier import SnoringDecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    """Main training function"""
    print("🚀 Starting Snoring Classification Training")
    print("=" * 50)
    
    # Configuration
    data_dir = "snoring_data"
    model_save_dir = "models"
    
    # Create model directory
    Path(model_save_dir).mkdir(exist_ok=True)
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"❌ Data directory '{data_dir}' not found!")
        print("Please ensure your data is in the 'snoring_data' folder")
        return
    
    try:
        # Step 1: Load and prepare data
        print("\n📊 Step 1: Loading and preparing data...")
        data_loader = SnoringDataLoader(data_dir)
        
        features, labels, feature_names = data_loader.load_all_data()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Number of features: {len(feature_names)}")
        
        # Step 2: Initialize classifier
        print("\n🌳 Step 2: Initializing Decision Tree classifier...")
        classifier = SnoringDecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            criterion='gini',
            random_state=42
        )
        
        print("✅ Classifier initialized!")
        
        # Step 3: Train the model
        print("\n🎯 Step 3: Training the model...")
        training_results = classifier.train(features, labels, feature_names)
        
        print("✅ Training completed!")
        
        # Step 4: Create visualizations
        print("\n📈 Step 4: Creating visualizations...")
        
        # Feature importance plot
        feature_plot_path = Path(model_save_dir) / "feature_importances.png"
        classifier.get_feature_importance_plot(str(feature_plot_path))
        
        # Confusion matrix (if we have test predictions)
        if 'test_metrics' in training_results:
            print(f"   Test accuracy: {training_results['test_metrics']['accuracy']:.4f}")
            print(f"   Test precision: {training_results['test_metrics']['precision']:.4f}")
            print(f"   Test recall: {training_results['test_metrics']['recall']:.4f}")
            print(f"   Test F1-score: {training_results['test_metrics']['f1_score']:.4f}")
        
        # Step 5: Save the model
        print("\n💾 Step 5: Saving the model...")
        model_path = Path(model_save_dir) / "snoring_decision_tree.pkl"
        classifier.save_model(str(model_path))
        
        print("✅ Model saved successfully!")
        
        # Step 6: Display model information
        print("\n📋 Step 6: Model information:")
        model_info = classifier.get_model_info()
        
        for key, value in model_info.items():
            if key != 'training_history':
                print(f"   {key}: {value}")
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Feature importance plot: {feature_plot_path}")
        
        # Step 7: Export to different formats
        print("\n🔄 Step 7: Exporting to different formats...")
        
        # Export feature names
        feature_names_path = Path(model_save_dir) / "feature_names.txt"
        with open(feature_names_path, 'w') as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i}: {name}\n")
        print(f"   Feature names saved to: {feature_names_path}")
        
        # Export training summary
        summary_path = Path(model_save_dir) / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("SNORING CLASSIFICATION TRAINING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model Type: {model_info['model_type']}\n")
            f.write(f"Max Depth: {model_info['max_depth']}\n")
            f.write(f"Min Samples Leaf: {model_info['min_samples_leaf']}\n")
            f.write(f"Criterion: {model_info['criterion']}\n")
            f.write(f"Feature Count: {model_info['feature_count']}\n")
            f.write(f"Class Names: {model_info['class_names']}\n\n")
            
            if 'test_metrics' in training_results:
                f.write("TEST METRICS:\n")
                for metric, value in training_results['test_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        print(f"   Training summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎯 Next steps:")
    print("   1. Check the saved model in 'models/' directory")
    print("   2. Use the model for predictions")
    print("   3. Export to TFLite/Keras if needed")
    print("   4. Deploy to ESP32")


if __name__ == "__main__":
    main() 