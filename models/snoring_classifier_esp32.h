/*
 * ESP32 Snoring Classifier - Auto-generated Header
 * Generated from trained Decision Tree model
 */

#ifndef SNORING_CLASSIFIER_ESP32_H
#define SNORING_CLASSIFIER_ESP32_H

// Model configuration
#define FEATURE_COUNT 39
#define MAX_DEPTH 3
#define N_NODES 11
#define N_LEAVES 6

// Tree structure arrays
extern const int children_left[11];
extern const int children_right[11];
extern const int feature[11];
extern const float threshold[11];
extern const float value[11][2];  // 2 classes

// Function declarations
int predict_snoring(float* features);
void extract_features(float* window_data, float* features);

#endif // SNORING_CLASSIFIER_ESP32_H
