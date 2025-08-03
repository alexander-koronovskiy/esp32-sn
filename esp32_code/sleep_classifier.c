// Автоматически сгенерированный код для ESP32
// Классификатор стадий сна

#include <stdint.h>

#define NUM_FEATURES 8
#define NUM_CLASSES 5
#define NUM_NODES 9
#define QUANTIZATION_SCALE 100

// Массивы дерева решений
const int8_t tree_feature[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0
};

const int8_t tree_threshold[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0
};

const int8_t tree_children_left[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1
};

const int8_t tree_children_right[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1
};

int predict_sleep_stage(float* features) {
    int node = 0;
    
    while (tree_children_left[node] != -1) {
        int feature_idx = tree_feature[node];
        float threshold = (float)tree_threshold[node] / QUANTIZATION_SCALE;
        
        if (features[feature_idx] <= threshold) {
            node = tree_children_left[node];
        } else {
            node = tree_children_right[node];
        }
    }
    
    // Возвращаем класс (упрощенно)
    return node % NUM_CLASSES;
}
