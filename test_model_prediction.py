#!/usr/bin/env python3
"""
Test script for the trained snoring classifier
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def test_model_prediction():
    """Test the trained model with synthetic data"""
    print("🧪 Testing Trained Model")
    print("=" * 40)
    
    # Load the trained model
    model_path = "models/snoring_decision_tree.pkl"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Load model
        classifier = SnoringDecisionTreeClassifier()
        classifier.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Create synthetic test data (39 features)
        print("\n📊 Creating synthetic test data...")
        test_features = np.random.rand(5, 39)  # 5 test samples, 39 features
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions, probabilities = classifier.predict(test_features)
        
        print(f"✅ Predictions completed!")
        print(f"   Test samples: {len(test_features)}")
        print(f"   Features per sample: {test_features.shape[1]}")
        print(f"   Predictions: {predictions}")
        print(f"   Probabilities shape: {probabilities.shape}")
        
        # Show prediction details
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            class_name = "Snoring" if pred == 1 else "No Snoring"
            confidence = prob[pred] * 100
            print(f"   Sample {i+1}: {class_name} (confidence: {confidence:.1f}%)")
        
        print("\n🎉 Model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_prediction() 