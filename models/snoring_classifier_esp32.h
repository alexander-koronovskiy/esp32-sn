/*
 * ESP32 Snoring Classifier - Auto-generated Header
 * Generated from trained Decision Tree model
 */

#ifndef SNORING_CLASSIFIER_ESP32_H
#define SNORING_CLASSIFIER_ESP32_H

// Model configuration
#define FEATURE_COUNT 39
#define MAX_DEPTH 3
#define N_NODES 1
#define N_LEAVES 1

// Tree structure arrays
extern const int children_left[1];
extern const int children_right[1];
extern const int feature[1];
extern const float threshold[1];
extern const float value[1][2];  // 2 classes

// Function declarations
int predict_snoring(float* features);
void extract_features(float* window_data, float* features);

#endif // SNORING_CLASSIFIER_ESP32_H
