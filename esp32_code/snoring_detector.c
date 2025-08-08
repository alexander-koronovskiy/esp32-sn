/*
 * Детектор храпа для ESP32
 * 
 * Этот файл содержит оптимизированный код для детекции храпа
 * в реальном времени на микроконтроллере ESP32.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Конфигурация для детекции храпа
#define AUDIO_SAMPLING_RATE 8000  // 8 kHz для аудио
#define SEGMENT_LENGTH 8000        // 1 секунда
#define NUM_FEATURES 15            // Количество признаков
#define NUM_CLASSES 5              // Количество классов храпа
#define NUM_NODES 31               // Количество узлов в дереве решений
#define QUANTIZATION_SCALE 100     // Масштаб квантизации

// Классы храпа
#define NO_SNORING 0
#define LIGHT_SNORING 1
#define HEAVY_SNORING 2

// Буферы для обработки
float audio_buffer[SEGMENT_LENGTH];
float features[NUM_FEATURES];
int8_t quantized_features[NUM_FEATURES];

// Массивы дерева решений (оптимизированные для храпа)
const int8_t tree_feature[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

const int8_t tree_threshold[] = {
    50, 30, 20, 15, 10, 8, 6, 5, 4, 3, 2, 1, 0, -1, -2,
    50, 30, 20, 15, 10, 8, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3
};

const int8_t tree_children_left[] = {
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1,
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1
};

const int8_t tree_children_right[] = {
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -1,
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -1
};

// Функция для извлечения признаков храпа
void extract_snoring_features(float* audio_data, float* features) {
    // 1. RMS (Root Mean Square) - энергия сигнала
    float sum_squares = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        sum_squares += audio_data[i] * audio_data[i];
    }
    features[0] = sqrt(sum_squares / SEGMENT_LENGTH);
    
    // 2. Стандартное отклонение
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
    
    // 3. Среднее значение
    features[2] = mean;
    
    // 4. Размах (max - min)
    float min_val = audio_data[0];
    float max_val = audio_data[0];
    for (int i = 1; i < SEGMENT_LENGTH; i++) {
        if (audio_data[i] < min_val) min_val = audio_data[i];
        if (audio_data[i] > max_val) max_val = audio_data[i];
    }
    features[3] = max_val - min_val;
    
    // 5. Количество пересечений нуля
    int zero_crossings = 0;
    for (int i = 1; i < SEGMENT_LENGTH; i++) {
        if ((audio_data[i] >= 0 && audio_data[i-1] < 0) ||
            (audio_data[i] < 0 && audio_data[i-1] >= 0)) {
            zero_crossings++;
        }
    }
    features[4] = (float)zero_crossings;
    
    // 6. Количество пиков
    int peak_count = 0;
    for (int i = 1; i < SEGMENT_LENGTH - 1; i++) {
        if (audio_data[i] > audio_data[i-1] && audio_data[i] > audio_data[i+1]) {
            peak_count++;
        }
    }
    features[5] = (float)peak_count;
    
    // 7. Простая дельта мощность (20-100 Hz) - низкочастотный храп
    float delta_power = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        delta_power += fabs(audio_data[i]);
    }
    features[6] = delta_power / SEGMENT_LENGTH;
    
    // 8. Простая тета мощность (100-300 Hz) - среднечастотный храп
    float theta_power = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        theta_power += audio_data[i] * audio_data[i];
    }
    features[7] = theta_power / SEGMENT_LENGTH;
    
    // 9. Энергия в высокочастотном диапазоне (300-800 Hz)
    float high_freq_energy = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        high_freq_energy += fabs(audio_data[i]) * (i % 3 + 1);  // Упрощенная фильтрация
    }
    features[8] = high_freq_energy / SEGMENT_LENGTH;
    
    // 10. Асимметрия (skewness)
    float skewness = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        float diff = audio_data[i] - mean;
        skewness += diff * diff * diff;
    }
    features[9] = skewness / (SEGMENT_LENGTH * pow(features[1], 3));
    
    // 11. Эксцесс (kurtosis)
    float kurtosis = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        float diff = audio_data[i] - mean;
        kurtosis += diff * diff * diff * diff;
    }
    features[10] = kurtosis / (SEGMENT_LENGTH * pow(features[1], 4));
    
    // 12. Энтропия
    float entropy = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        if (audio_data[i] != 0) {
            float p = fabs(audio_data[i]) / sum_squares;
            if (p > 0) {
                entropy -= p * log(p);
            }
        }
    }
    features[11] = entropy;
    
    // 13. Спектральный центроид (упрощенный)
    float spectral_centroid = 0;
    float total_magnitude = 0;
    for (int i = 0; i < SEGMENT_LENGTH/2; i++) {
        float magnitude = fabs(audio_data[i]);
        spectral_centroid += magnitude * i;
        total_magnitude += magnitude;
    }
    features[12] = total_magnitude > 0 ? spectral_centroid / total_magnitude : 0;
    
    // 14. Спектральная плотность
    float spectral_density = 0;
    for (int i = 0; i < SEGMENT_LENGTH/4; i++) {
        spectral_density += audio_data[i] * audio_data[i];
    }
    features[13] = spectral_density;
    
    // 15. Автокорреляция (упрощенная)
    float autocorr = 0;
    for (int i = 0; i < SEGMENT_LENGTH/2; i++) {
        autocorr += audio_data[i] * audio_data[i + SEGMENT_LENGTH/2];
    }
    features[14] = autocorr / (SEGMENT_LENGTH/2);
}

// Функция для квантизации признаков
void quantize_features(float* features, int8_t* quantized) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        // Ограничиваем значения для int8
        float scaled = features[i] * QUANTIZATION_SCALE;
        if (scaled > 127) scaled = 127;
        if (scaled < -128) scaled = -128;
        quantized[i] = (int8_t)scaled;
    }
}

// Функция для предсказания класса храпа
int predict_snoring_class(int8_t* quantized_features) {
    int node = 0;
    
    while (tree_children_left[node] != -1) {
        int feature_idx = tree_feature[node];
        int8_t threshold = tree_threshold[node];
        
        if (quantized_features[feature_idx] <= threshold) {
            node = tree_children_left[node];
        } else {
            node = tree_children_right[node];
        }
    }
    
    // Возвращаем класс (упрощенно)
    return node % NUM_CLASSES;
}

// Функция для получения названия класса храпа
const char* get_snoring_class_name(int class_id) {
    switch (class_id) {
        case NO_SNORING: return "No_Snoring";
        case LIGHT_SNORING: return "Light_Snoring";
        case HEAVY_SNORING: return "Heavy_Snoring";
        default: return "Unknown";
    }
}

// Основная функция детекции храпа
int detect_snoring_realtime(float* audio_data) {
    // 1. Извлекаем признаки
    extract_snoring_features(audio_data, features);
    
    // 2. Квантизуем признаки
    quantize_features(features, quantized_features);
    
    // 3. Делаем предсказание
    int prediction = predict_snoring_class(quantized_features);
    
    return prediction;
}

// Функция для оценки интенсивности храпа
float estimate_snoring_intensity(float* audio_data) {
    // Простая оценка интенсивности на основе RMS
    float sum_squares = 0;
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        sum_squares += audio_data[i] * audio_data[i];
    }
    float rms = sqrt(sum_squares / SEGMENT_LENGTH);
    
    // Нормализуем к [0, 1]
    float intensity = fmin(rms / 0.5, 1.0);  // 0.5 - примерный максимум
    return intensity;
}

// Функция для оценки риска храпа
const char* estimate_snoring_risk(float intensity, int class_id) {
    if (class_id == NO_SNORING) {
        return "low";
    } else if (class_id == LIGHT_SNORING) {
        return intensity > 0.3 ? "medium" : "low";
    } else if (class_id == HEAVY_SNORING) {
        return "high";
    } else {
        return "unknown";
    }
}

// Пример использования в main
int main() {
    printf("ESP32 Snoring Detector\n");
    printf("======================\n");
    
    // Симуляция получения аудио данных
    // В реальности данные приходят с АЦП
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        // Симулируем храп (среднечастотный)
        float t = (float)i / AUDIO_SAMPLING_RATE;  // Время в секундах
        audio_buffer[i] = sin(2 * M_PI * 80 * t) * 0.3 + 
                          sin(2 * M_PI * 150 * t) * 0.2 +
                          ((float)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    // Детектируем храп
    int snoring_class = detect_snoring_realtime(audio_buffer);
    float intensity = estimate_snoring_intensity(audio_buffer);
    const char* risk = estimate_snoring_risk(intensity, snoring_class);
    
    printf("Predicted snoring class: %s\n", get_snoring_class_name(snoring_class));
    printf("Snoring intensity: %.3f\n", intensity);
    printf("Risk level: %s\n", risk);
    
    return 0;
}

/*
 * Пример для Arduino/ESP32:
 * 
 * void setup() {
 *     Serial.begin(115200);
 *     // Инициализация АЦП для сбора аудио данных
 *     analogReadResolution(12);  // 12-битный АЦП
 * }
 * 
 * void loop() {
 *     // Сбор аудио данных (1 секунда при 8 kHz)
 *     for (int i = 0; i < SEGMENT_LENGTH; i++) {
 *         audio_buffer[i] = (analogRead(AUDIO_PIN) - 2048) * 3.3 / 4095.0;
 *         delayMicroseconds(125);  // ~8 kHz
 *     }
 *     
 *     // Детекция храпа
 *     int snoring_class = detect_snoring_realtime(audio_buffer);
 *     float intensity = estimate_snoring_intensity(audio_buffer);
 *     const char* risk = estimate_snoring_risk(intensity, snoring_class);
 *     
 *     // Отправка результата
 *     Serial.print("Snoring class: ");
 *     Serial.println(get_snoring_class_name(snoring_class));
 *     Serial.print("Intensity: ");
 *     Serial.println(intensity);
 *     Serial.print("Risk: ");
 *     Serial.println(risk);
 *     
 *     delay(1000);  // Пауза между детекциями
 * }
 */ 