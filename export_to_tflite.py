#!/usr/bin/env python3
"""
Export trained Decision Tree to TFLite format
"""

import sys
import os
from pathlib import Path
import numpy as np
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.decision_tree_classifier import SnoringDecisionTreeClassifier


def export_to_tflite(model_path: str, output_dir: str = "models"):
    """
    Export trained model to TFLite format
    
    Args:
        model_path: Path to trained model (.pkl)
        output_dir: Directory to save exported models
    """
    print("🔄 Exporting Decision Tree to TFLite...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load trained model
    print("📥 Loading trained model...")
    classifier = SnoringDecisionTreeClassifier()
    classifier.load_model(model_path)
    
    print("✅ Model loaded successfully!")
    
    # Get model info
    model_info = classifier.get_model_info()
    print(f"📋 Model info: {model_info}")
    
    # Export feature names
    feature_names_path = Path(output_dir) / "feature_names.txt"
    if classifier.feature_names:
        with open(feature_names_path, 'w') as f:
            for i, name in enumerate(classifier.feature_names):
                f.write(f"{i}: {name}\n")
        print(f"📝 Feature names exported to: {feature_names_path}")
    
    # Export model parameters for ESP32
    esp32_params_path = Path(output_dir) / "esp32_model_params.json"
    esp32_params = {
        'feature_count': int(model_info['feature_count']),
        'max_depth': int(model_info['max_depth']),
        'min_samples_leaf': int(model_info['min_samples_leaf']),
        'criterion': str(model_info['criterion']),
        'class_names': [int(x) for x in model_info['class_names']] if model_info['class_names'] else [],
        'feature_importances': [float(x) for x in classifier.feature_importances] if classifier.feature_importances is not None else [],
        'tree_structure': {
            'n_leaves': int(classifier.classifier.tree_.n_leaves),
            'n_nodes': int(len(classifier.classifier.tree_.children_left)),
            'max_depth': int(classifier.classifier.tree_.max_depth)
        }
    }
    
    import json
    with open(esp32_params_path, 'w') as f:
        json.dump(esp32_params, f, indent=2)
    print(f"📊 ESP32 parameters exported to: {esp32_params_path}")
    
    # Export decision tree structure for C implementation
    tree = classifier.classifier.tree_
    
    # Extract tree structure
    tree_structure = {
        'children_left': [int(x) for x in tree.children_left.tolist()],
        'children_right': [int(x) for x in tree.children_right.tolist()],
        'feature': [int(x) for x in tree.feature.tolist()],
        'threshold': [float(x) for x in tree.threshold.tolist()],
        'value': [[float(y) for y in x[0]] for x in tree.value.tolist()],
        'n_leaves': int(tree.n_leaves),
        'n_nodes': int(len(tree.children_left)),
        'max_depth': int(tree.max_depth)
    }
    
    tree_structure_path = Path(output_dir) / "tree_structure.json"
    with open(tree_structure_path, 'w') as f:
        json.dump(tree_structure, f, indent=2)
    print(f"🌳 Tree structure exported to: {tree_structure_path}")
    
    # Create C header file for ESP32
    c_header_path = Path(output_dir) / "snoring_classifier_esp32.h"
    create_c_header(c_header_path, tree_structure, model_info)
    print(f"📋 C header file created: {c_header_path}")
    
    # Create simplified C implementation
    c_impl_path = Path(output_dir) / "snoring_classifier_esp32_impl.c"
    create_c_implementation(c_impl_path, tree_structure, model_info)
    print(f"⚙️ C implementation created: {c_impl_path}")
    
    print("\n🎉 Export completed successfully!")
    print(f"📁 All files saved to: {output_dir}")
    print("\n📋 Exported files:")
    print(f"   - Feature names: {feature_names_path}")
    print(f"   - ESP32 parameters: {esp32_params_path}")
    print(f"   - Tree structure: {tree_structure_path}")
    print(f"   - C header: {c_header_path}")
    print(f"   - C implementation: {c_impl_path}")


def create_c_header(header_path: Path, tree_structure: dict, model_info: dict):
    """Create C header file for ESP32"""
    
    header_content = f"""/*
 * ESP32 Snoring Classifier - Auto-generated Header
 * Generated from trained Decision Tree model
 */

#ifndef SNORING_CLASSIFIER_ESP32_H
#define SNORING_CLASSIFIER_ESP32_H

// Model configuration
#define FEATURE_COUNT {model_info['feature_count']}
#define MAX_DEPTH {model_info['max_depth']}
#define N_NODES {tree_structure['n_nodes']}
#define N_LEAVES {tree_structure['n_leaves']}

// Tree structure arrays
extern const int children_left[{tree_structure['n_nodes']}];
extern const int children_right[{tree_structure['n_nodes']}];
extern const int feature[{tree_structure['n_nodes']}];
extern const float threshold[{tree_structure['n_nodes']}];
extern const float value[{tree_structure['n_nodes']}][2];  // 2 classes

// Function declarations
int predict_snoring(float* features);
void extract_features(float* window_data, float* features);

#endif // SNORING_CLASSIFIER_ESP32_H
"""
    
    with open(header_path, 'w') as f:
        f.write(header_content)


def create_c_implementation(impl_path: Path, tree_structure: dict, model_info: dict):
    """Create C implementation file for ESP32"""
    
    # Create arrays for tree structure
    children_left_str = ", ".join(map(str, tree_structure['children_left']))
    children_right_str = ", ".join(map(str, tree_structure['children_right']))
    feature_str = ", ".join(map(str, tree_structure['feature']))
    threshold_str = ", ".join(map(str, tree_structure['threshold']))
    
    # Create value array
    value_array = []
    for i, val in enumerate(tree_structure['value']):
        # val is list with 2 values for 2 classes
        if len(val) >= 2:
            value_array.append(f"{{{val[0]:.6f}, {val[1]:.6f}}}")
        else:
            value_array.append("{0.0, 0.0}")
    value_str = ", ".join(value_array)
    
    impl_content = f"""/*
 * ESP32 Snoring Classifier - Auto-generated Implementation
 * Generated from trained Decision Tree model
 */

#include "snoring_classifier_esp32.h"
#include <math.h>

// Tree structure arrays
const int children_left[{tree_structure['n_nodes']}] = {{{children_left_str}}};
const int children_right[{tree_structure['n_nodes']}] = {{{children_right_str}}};
const int feature[{tree_structure['n_nodes']}] = {{{feature_str}}};
const float threshold[{tree_structure['n_nodes']}] = {{{threshold_str}}};
const float value[{tree_structure['n_nodes']}][2] = {{{value_str}}};

// Decision tree prediction
int predict_snoring(float* features) {{
    int node = 0;
    
    while (children_left[node] != -1) {{
        float feature_val = features[feature[node]];
        if (feature_val <= threshold[node]) {{
            node = children_left[node];
        }} else {{
            node = children_right[node];
        }}
    }}
    
    // Return class with highest probability
    if (value[node][1] > value[node][0]) {{
        return 1;  // Snoring
    }} else {{
        return 0;  // No snoring
    }}
}}

// Feature extraction (simplified version)
void extract_features(float* window_data, float* features) {{
    // This is a placeholder - implement actual feature extraction
    // based on your specific sensor data format
    
    // For now, just copy data (you need to implement the actual feature extraction)
    for (int i = 0; i < FEATURE_COUNT; i++) {{
        features[i] = window_data[i];
    }}
}}
"""
    
    with open(impl_path, 'w') as f:
        f.write(impl_content)


def main():
    """Main function"""
    # Check if model exists
    model_path = "models/snoring_decision_tree.pkl"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using: python train_snoring_classifier.py")
        return
    
    # Export to TFLite
    export_to_tflite(model_path)
    
    print("\n🎯 Next steps:")
    print("   1. Copy the generated C files to your ESP32 project")
    print("   2. Implement the feature extraction function for your sensor data")
    print("   3. Compile and flash to ESP32")
    print("   4. Test the classifier with real sensor data")


if __name__ == "__main__":
    main() 