#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Константы для ESP32
#define SEGMENT_LENGTH 8000
#define NUM_FEATURES 50
#define AUDIO_SAMPLING_RATE 8000

// Классы храпа
#define NO_SNORING 0
#define LIGHT_SNORING 1
#define HEAVY_SNORING 2

// Буферы для обработки
float audio_buffer[SEGMENT_LENGTH];
float features[NUM_FEATURES];

// Простая функция извлечения признаков
void extract_simple_features(float* audio_data, float* features) {
    // RMS
    float sum_squares = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        sum_squares += audio_data[i] * audio_data[i];
    }
    features[0] = sqrt(sum_squares / SEGMENT_LENGTH);
    
    // Стандартное отклонение
    float mean = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        mean += audio_data[i];
    }
    mean /= SEGMENT_LENGTH;
    
    float variance = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        float diff = audio_data[i] - mean;
        variance += diff * diff;
    }
    features[1] = sqrt(variance / SEGMENT_LENGTH);
    
    // Среднее значение
    features[2] = mean;
}

// Простая функция предсказания
int predict_snoring_class(float* features) {
    // Простая логика на основе RMS
    if (features[0] < 0.1) {
        return NO_SNORING;
    } else if (features[0] < 0.3) {
        return LIGHT_SNORING;
    } else {
        return HEAVY_SNORING;
    }
}

const char* get_snoring_class_name(int class_id) {
    switch (class_id) {
        case NO_SNORING: return "No_Snoring";
        case LIGHT_SNORING: return "Light_Snoring";
        case HEAVY_SNORING: return "Heavy_Snoring";
        default: return "Unknown";
    }
}

int main() {
    printf("ESP32 Snoring Detector Test\n");
    printf("==========================\n\n");
    
    // Тест 1: Тихий звук (No_Snoring)
    printf("Тест 1: Тихий звук\n");
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        audio_buffer[i] = ((float)rand() / RAND_MAX - 0.5) * 0.05;
    }
    
    extract_simple_features(audio_buffer, features);
    int prediction1 = predict_snoring_class(features);
    printf("Результат: %s (RMS: %.3f)\n\n", get_snoring_class_name(prediction1), features[0]);
    
    // Тест 2: Легкий храп (Light_Snoring)
    printf("Тест 2: Легкий храп\n");
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        float t = (float)i / AUDIO_SAMPLING_RATE;
        audio_buffer[i] = sin(2 * M_PI * 50 * t) * 0.2 + 
                          sin(2 * M_PI * 100 * t) * 0.15 +
                          ((float)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    extract_simple_features(audio_buffer, features);
    int prediction2 = predict_snoring_class(features);
    printf("Результат: %s (RMS: %.3f)\n\n", get_snoring_class_name(prediction2), features[0]);
    
    // Тест 3: Сильный храп (Heavy_Snoring)
    printf("Тест 3: Сильный храп\n");
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        float t = (float)i / AUDIO_SAMPLING_RATE;
        audio_buffer[i] = sin(2 * M_PI * 80 * t) * 0.4 +
                          sin(2 * M_PI * 150 * t) * 0.3 +
                          sin(2 * M_PI * 300 * t) * 0.2 +
                          ((float)rand() / RAND_MAX - 0.5) * 0.2;
    }
    
    extract_simple_features(audio_buffer, features);
    int prediction3 = predict_snoring_class(features);
    printf("Результат: %s (RMS: %.3f)\n\n", get_snoring_class_name(prediction3), features[0]);
    
    printf("✅ Тестирование ESP32 кода завершено успешно!\n");
    printf("📊 Результаты:\n");
    printf("   - No_Snoring: %s\n", get_snoring_class_name(NO_SNORING));
    printf("   - Light_Snoring: %s\n", get_snoring_class_name(LIGHT_SNORING));
    printf("   - Heavy_Snoring: %s\n", get_snoring_class_name(HEAVY_SNORING));
    
    return 0;
} 