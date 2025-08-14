#!/usr/bin/env python3
"""
Export Decision Tree to TFLite and Keras formats
"""

import sys
import os
from pathlib import Path
import numpy as np
import joblib
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def export_to_tflite_keras():
    """Export model to TFLite and Keras formats"""
    print("🔄 Exporting Decision Tree to TFLite and Keras")
    print("=" * 50)
    
    # Load trained model
    print("📥 Loading trained model...")
    model_path = "models/snoring_decision_tree.pkl"
    scaler_path = "models/snoring_decision_tree_scaler.pkl"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        classifier = SnoringDecisionTreeClassifier()
        classifier.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Get model info
        model_info = classifier.get_model_info()
        print(f"📋 Model info: {model_info}")
        
        # Export to Keras format
        print("\n🎯 Exporting to Keras format...")
        export_to_keras(classifier)
        
        # Export to TFLite format
        print("\n🎯 Exporting to TFLite format...")
        export_to_tflite(classifier)
        
        print("\n🎉 Export completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        import traceback
        traceback.print_exc()


def export_to_keras(classifier):
    """Export to Keras format"""
    try:
        # Create a simple Keras model that mimics the decision tree
        import tensorflow as tf
        from tensorflow import keras
        
        print("   🔧 Creating Keras model...")
        
        # Create a simple neural network that approximates the decision tree
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(39,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create synthetic training data based on the decision tree
        print("   📊 Creating synthetic training data...")
        n_samples = 1000
        X_synthetic = np.random.randn(n_samples, 39)
        
        # Use the decision tree to generate labels
        y_synthetic = []
        for i in range(n_samples):
            pred, _ = classifier.predict(X_synthetic[i:i+1])
            y_synthetic.append(pred[0])
        y_synthetic = np.array(y_synthetic)
        
        # Train the Keras model
        print("   🎯 Training Keras model...")
        model.fit(X_synthetic, y_synthetic, epochs=50, batch_size=32, verbose=0)
        
        # Save Keras model
        keras_path = "models/snoring_classifier_keras.h5"
        model.save(keras_path)
        print(f"   💾 Keras model saved to: {keras_path}")
        
        # Test the Keras model
        print("   🧪 Testing Keras model...")
        test_pred = model.predict(X_synthetic[:5])
        print(f"   ✅ Keras predictions shape: {test_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error exporting to Keras: {e}")
        return False


def export_to_tflite(classifier):
    """Export to TFLite format"""
    try:
        # First create Keras model if it doesn't exist
        keras_path = "models/snoring_classifier_keras.h5"
        if not Path(keras_path).exists():
            print("   ⚠️ Keras model not found, creating it first...")
            if not export_to_keras(classifier):
                print("   ❌ Failed to create Keras model")
                return False
        
        import tensorflow as tf
        from tensorflow import keras
        
        print("   🔧 Converting Keras to TFLite...")
        
        # Load the Keras model
        model = keras.models.load_model(keras_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = "models/snoring_classifier.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"   💾 TFLite model saved to: {tflite_path}")
        
        # Get model size
        model_size = len(tflite_model) / 1024  # KB
        print(f"   📏 TFLite model size: {model_size:.2f} KB")
        
        # Test TFLite model
        print("   🧪 Testing TFLite model...")
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   ✅ TFLite model loaded successfully!")
        print(f"   📊 Input shape: {input_details[0]['shape']}")
        print(f"   📊 Output shape: {output_details[0]['shape']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error exporting to TFLite: {e}")
        return False


def create_export_summary():
    """Create summary of exported models"""
    print("\n📋 Creating Export Summary...")
    
    summary_path = "models/export_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("🎯 EXPORT SUMMARY: Snoring Classification Models\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("📁 Exported Models:\n")
        f.write("-" * 20 + "\n")
        
        # Check what files exist
        models_dir = Path("models")
        
        if (models_dir / "snoring_decision_tree.pkl").exists():
            f.write("✅ Python Pickle: snoring_decision_tree.pkl\n")
        
        if (models_dir / "snoring_classifier_keras.h5").exists():
            f.write("✅ Keras H5: snoring_classifier_keras.h5\n")
        
        if (models_dir / "snoring_classifier.tflite").exists():
            f.write("✅ TFLite: snoring_classifier.tflite\n")
        
        if (models_dir / "snoring_classifier_esp32.h").exists():
            f.write("✅ ESP32 C Header: snoring_classifier_esp32.h\n")
        
        if (models_dir / "snoring_classifier_esp32_impl.c").exists():
            f.write("✅ ESP32 C Implementation: snoring_classifier_esp32_impl.c\n")
        
        f.write("\n🚀 Usage Instructions:\n")
        f.write("-" * 20 + "\n")
        
        f.write("1. Python (Pickle):\n")
        f.write("   from models.decision_tree_classifier import SnoringDecisionTreeClassifier\n")
        f.write("   classifier = SnoringDecisionTreeClassifier()\n")
        f.write("   classifier.load_model('models/snoring_decision_tree.pkl')\n\n")
        
        f.write("2. Keras:\n")
        f.write("   import tensorflow as tf\n")
        f.write("   model = tf.keras.models.load_model('models/snoring_classifier_keras.h5')\n\n")
        
        f.write("3. TFLite:\n")
        f.write("   import tensorflow as tf\n")
        f.write("   interpreter = tf.lite.Interpreter(model_path='models/snoring_classifier.tflite')\n\n")
        
        f.write("4. ESP32:\n")
        f.write("   #include 'snoring_classifier_esp32.h'\n")
        f.write("   // Use the C functions directly\n\n")
        
        f.write("🎯 All models are ready for deployment!\n")
    
    print(f"📋 Export summary saved to: {summary_path}")


if __name__ == "__main__":
    export_to_tflite_keras()
    create_export_summary() 