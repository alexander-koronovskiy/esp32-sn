/*
 * Пример использования классификатора стадий сна на ESP32
 * 
 * Этот файл показывает, как использовать сгенерированный код
 * для классификации стадий сна в реальном времени.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sleep_classifier.h"

// Буфер для данных ЭЭГ (10 секунд при 64 Hz)
#define EEG_BUFFER_SIZE 640
float eeg_buffer[EEG_BUFFER_SIZE];

// Буфер для признаков
float features[NUM_FEATURES];

// Функция для извлечения признаков (упрощенная версия)
void extract_features(float* eeg_data, float* features) {
    // 1. RMS (Root Mean Square)
    float sum_squares = 0;
    for (int i = 0; i < EEG_BUFFER_SIZE; i++) {
        sum_squares += eeg_data[i] * eeg_data[i];
    }
    features[0] = sqrt(sum_squares / EEG_BUFFER_SIZE);
    
    // 2. Стандартное отклонение
    float mean = 0;
    for (int i = 0; i < EEG_BUFFER_SIZE; i++) {
        mean += eeg_data[i];
    }
    mean /= EEG_BUFFER_SIZE;
    
    float variance = 0;
    for (int i = 0; i < EEG_BUFFER_SIZE; i++) {
        float diff = eeg_data[i] - mean;
        variance += diff * diff;
    }
    features[1] = sqrt(variance / EEG_BUFFER_SIZE);
    
    // 3. Среднее значение
    features[2] = mean;
    
    // 4. Размах (max - min)
    float min_val = eeg_data[0];
    float max_val = eeg_data[0];
    for (int i = 1; i < EEG_BUFFER_SIZE; i++) {
        if (eeg_data[i] < min_val) min_val = eeg_data[i];
        if (eeg_data[i] > max_val) max_val = eeg_data[i];
    }
    features[3] = max_val - min_val;
    
    // 5. Количество пересечений нуля
    int zero_crossings = 0;
    for (int i = 1; i < EEG_BUFFER_SIZE; i++) {
        if ((eeg_data[i] >= 0 && eeg_data[i-1] < 0) ||
            (eeg_data[i] < 0 && eeg_data[i-1] >= 0)) {
            zero_crossings++;
        }
    }
    features[4] = (float)zero_crossings;
    
    // 6. Количество пиков
    int peak_count = 0;
    for (int i = 1; i < EEG_BUFFER_SIZE - 1; i++) {
        if (eeg_data[i] > eeg_data[i-1] && eeg_data[i] > eeg_data[i+1]) {
            peak_count++;
        }
    }
    features[5] = (float)peak_count;
    
    // 7. Простая дельта мощность (0.5-4 Hz)
    // Упрощенная версия без FFT
    float delta_power = 0;
    for (int i = 0; i < EEG_BUFFER_SIZE; i++) {
        delta_power += fabs(eeg_data[i]);
    }
    features[6] = delta_power / EEG_BUFFER_SIZE;
    
    // 8. Простая тета мощность (4-8 Hz)
    // Упрощенная версия
    float theta_power = 0;
    for (int i = 0; i < EEG_BUFFER_SIZE; i++) {
        theta_power += eeg_data[i] * eeg_data[i];
    }
    features[7] = theta_power / EEG_BUFFER_SIZE;
}

// Функция для нормализации признаков
void normalize_features(float* features) {
    // Простая нормализация (в реальности нужен обученный scaler)
    for (int i = 0; i < NUM_FEATURES; i++) {
        // Ограничиваем значения
        if (features[i] > 100.0f) features[i] = 100.0f;
        if (features[i] < -100.0f) features[i] = -100.0f;
    }
}

// Функция для получения названия стадии сна
const char* get_sleep_stage_name(int stage) {
    switch (stage) {
        case 0: return "Wake";
        case 1: return "N1";
        case 2: return "N2";
        case 3: return "N3";
        case 4: return "REM";
        default: return "Unknown";
    }
}

// Основная функция классификации
int classify_sleep_stage_realtime(float* eeg_data) {
    // 1. Извлекаем признаки
    extract_features(eeg_data, features);
    
    // 2. Нормализуем признаки
    normalize_features(features);
    
    // 3. Делаем предсказание
    int prediction = predict_sleep_stage(features);
    
    return prediction;
}

// Пример использования в main
int main() {
    printf("ESP32 Sleep Stage Classifier\n");
    printf("============================\n");
    
    // Симуляция получения данных ЭЭГ
    // В реальности данные приходят с АЦП
    for (int i = 0; i < EEG_BUFFER_SIZE; i++) {
        // Симулируем сигнал ЭЭГ (например, стадия Wake)
        float t = (float)i / 64.0f;  // Время в секундах
        eeg_buffer[i] = sin(2 * M_PI * 10 * t) * 0.5 + 
                        sin(2 * M_PI * 20 * t) * 0.3 +
                        ((float)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    // Классифицируем стадию сна
    int sleep_stage = classify_sleep_stage_realtime(eeg_buffer);
    
    printf("Predicted sleep stage: %s\n", get_sleep_stage_name(sleep_stage));
    
    return 0;
}

/*
 * Пример для Arduino/ESP32:
 * 
 * void setup() {
 *     Serial.begin(115200);
 *     // Инициализация АЦП для сбора данных ЭЭГ
 * }
 * 
 * void loop() {
 *     // Сбор данных ЭЭГ (10 секунд при 64 Hz)
 *     for (int i = 0; i < EEG_BUFFER_SIZE; i++) {
 *         eeg_buffer[i] = analogRead(EEG_PIN) * 3.3 / 4095.0;
 *         delay(15);  // ~64 Hz
 *     }
 *     
 *     // Классификация
 *     int stage = classify_sleep_stage_realtime(eeg_buffer);
 *     
 *     // Отправка результата
 *     Serial.print("Sleep stage: ");
 *     Serial.println(get_sleep_stage_name(stage));
 *     
 *     delay(1000);  // Пауза между классификациями
 * }
 */ 