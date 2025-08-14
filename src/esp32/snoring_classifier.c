/*
 * ESP32 Snoring Classifier - C Implementation
 * Optimized for MicroPython and bare metal ESP32
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Configuration
#define WINDOW_SIZE 80
#define FEATURE_COUNT 39
#define AUDIO_CHANNELS 4
#define ACCEL_AXES 3
#define AUDIO_FEATURES_PER_CHANNEL 7
#define ACCEL_FEATURES_PER_AXIS 3
#define MIXED_FEATURES 2

// Feature extraction functions
float calculate_mean(float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

float calculate_max(float* data, int size) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

float calculate_std(float* data, int size, float mean) {
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    return sqrtf(variance / size);
}

float calculate_relative_std(float std, float mean) {
    return std / (mean + 1e-8f);
}

float calculate_high_threshold_ratio(float* data, int size) {
    // Sort data to find 75th percentile
    float sorted_data[WINDOW_SIZE];
    memcpy(sorted_data, data, size * sizeof(float));
    
    // Simple bubble sort (can be optimized)
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (sorted_data[j] > sorted_data[j + 1]) {
                float temp = sorted_data[j];
                sorted_data[j] = sorted_data[j + 1];
                sorted_data[j + 1] = temp;
            }
        }
    }
    
    int threshold_idx = (int)(0.75f * size);
    float threshold = sorted_data[threshold_idx];
    
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (data[i] > threshold) {
            count++;
        }
    }
    
    return (float)count / size;
}

float calculate_trend(float* data, int size) {
    if (size < 2) return 0.0f;
    return (data[size - 1] - data[0]) / size;
}

float calculate_regularity(float* data, int size) {
    if (size < 2) return 0.0f;
    
    float sum_diffs = 0.0f;
    for (int i = 1; i < size; i++) {
        sum_diffs += fabsf(data[i] - data[i - 1]);
    }
    
    return 1.0f / (1.0f + sum_diffs / (size - 1));
}

// Audio feature extraction
void extract_audio_features(float* audio_data, float* features) {
    float mean = calculate_mean(audio_data, WINDOW_SIZE);
    float max_val = calculate_max(audio_data, WINDOW_SIZE);
    float std = calculate_std(audio_data, WINDOW_SIZE, mean);
    
    features[0] = mean;                                    // mean
    features[1] = max_val;                                 // max
    features[2] = std;                                     // std
    features[3] = calculate_relative_std(std, mean);       // relative_std
    features[4] = calculate_high_threshold_ratio(audio_data, WINDOW_SIZE); // high_threshold_ratio
    features[5] = calculate_trend(audio_data, WINDOW_SIZE); // trend
    features[6] = calculate_regularity(audio_data, WINDOW_SIZE); // regularity
}

// Accelerometer feature extraction
void extract_accelerometer_features(float* accel_data, float* features) {
    // Calculate deltas
    float deltas[WINDOW_SIZE - 1];
    for (int i = 1; i < WINDOW_SIZE; i++) {
        deltas[i - 1] = fabsf(accel_data[i] - accel_data[i - 1]);
    }
    
    float mean_activity = calculate_mean(deltas, WINDOW_SIZE - 1);
    float max_activity = calculate_max(deltas, WINDOW_SIZE - 1);
    
    // Calculate movement ratio
    float movement_threshold = mean_activity; // Simplified threshold
    int moving_count = 0;
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        if (deltas[i] > movement_threshold) {
            moving_count++;
        }
    }
    float movement_ratio = (float)moving_count / (WINDOW_SIZE - 1);
    
    features[0] = mean_activity;    // mean_activity
    features[1] = max_activity;     // max_activity
    features[2] = movement_ratio;   // movement_ratio
}

// Mixed features
void extract_mixed_features(float* audio_features, float* mixed_features) {
    // b400/b100 ratio
    float b400_mean = audio_features[1 * AUDIO_FEATURES_PER_CHANNEL]; // b400 mean
    float b100_mean = audio_features[0 * AUDIO_FEATURES_PER_CHANNEL]; // b100 mean
    mixed_features[0] = b400_mean / (b100_mean + 1e-8f);
    
    // b400/b1000 ratio
    float b1000_mean = audio_features[2 * AUDIO_FEATURES_PER_CHANNEL]; // b1000 mean
    mixed_features[1] = b400_mean / (b1000_mean + 1e-8f);
}

// Main feature extraction function
void extract_all_features(float* window_data, float* features) {
    int feature_idx = 0;
    
    // Audio features (28 features)
    for (int ch = 0; ch < AUDIO_CHANNELS; ch++) {
        float* channel_data = &window_data[ch * WINDOW_SIZE];
        float* channel_features = &features[feature_idx];
        extract_audio_features(channel_data, channel_features);
        feature_idx += AUDIO_FEATURES_PER_CHANNEL;
    }
    
    // Accelerometer features (9 features)
    for (int axis = 0; axis < ACCEL_AXES; axis++) {
        float* axis_data = &window_data[(AUDIO_CHANNELS + axis) * WINDOW_SIZE];
        float* axis_features = &features[feature_idx];
        extract_accelerometer_features(axis_data, axis_features);
        feature_idx += ACCEL_FEATURES_PER_AXIS;
    }
    
    // Mixed features (2 features)
    float* mixed_features = &features[feature_idx];
    extract_mixed_features(features, mixed_features);
}

// Simple decision tree prediction (placeholder)
// In real implementation, this would use the actual trained tree
int predict_snoring(float* features) {
    // Simple heuristic based on audio features
    float b100_mean = features[0];   // b100 mean
    float b400_mean = features[7];   // b400 mean
    float b1000_mean = features[14]; // b1000 mean
    
    // If b400 (snore range) is significantly higher than others, likely snoring
    if (b400_mean > (b100_mean + b1000_mean) * 0.6f) {
        return 1; // Snoring
    } else {
        return 0; // No snoring
    }
}

// Example usage
int main() {
    printf("ESP32 Snoring Classifier\n");
    
    // Example window data (7 channels * 80 samples)
    float window_data[7 * WINDOW_SIZE];
    
    // Fill with example data
    for (int ch = 0; ch < 7; ch++) {
        for (int i = 0; i < WINDOW_SIZE; i++) {
            window_data[ch * WINDOW_SIZE + i] = 100.0f + ch * 50.0f + i * 0.1f;
        }
    }
    
    // Extract features
    float features[FEATURE_COUNT];
    extract_all_features(window_data, features);
    
    printf("Extracted %d features\n", FEATURE_COUNT);
    
    // Make prediction
    int prediction = predict_snoring(features);
    printf("Prediction: %s\n", prediction ? "Snoring" : "No Snoring");
    
    return 0;
} 