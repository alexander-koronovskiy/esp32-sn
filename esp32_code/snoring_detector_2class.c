/*
 * Snoring Detector для ESP32 - 2-классовая классификация
 * Адаптирован для работы с реальными данными
 * 
 * Классы:
 * - 0: No_Snoring (W - бодрствование)
 * - 1: Snoring (храп)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/adc.h"
#include "driver/i2c.h"
#include "esp_log.h"
#include "esp_timer.h"

// Константы
#define SNORING_NO 0      // No Snoring (W)
#define SNORING_YES 1     // Snoring

#define SAMPLE_RATE 8000  // Частота дискретизации
#define BUFFER_SIZE 8000  // 1 секунда данных
#define FEATURE_COUNT 25  // Количество признаков

// Структуры данных
typedef struct {
    float x, y, z;        // Акселерометр
    float bpm;            // Пульс
    float breath_nn;      // Нейросеть дыхания
    float snore_nn;       // Нейросеть храпа
    float signal_nn;      // Нейросеть сигнала
    float splash_nn;      // Нейросеть всплесков
    float b100, b400, b1000; // Полосовые фильтры
    float env;            // Огибающая
    float mf, af, arf, sf, r_th; // Дополнительные признаки
    float status_1, status_2, status_3; // Статусы
    float ml, eog;        // Дополнительные метки
} snoring_features_t;

typedef struct {
    int class_id;         // ID класса (0 или 1)
    float confidence;     // Уверенность
    float p_snore;        // Вероятность храпа
    bool is_snoring;      // Бинарный индикатор
} snoring_prediction_t;

// Глобальные переменные
static const char *TAG = "SNORING_DETECTOR_2CLASS";
static float audio_buffer[BUFFER_SIZE];
static snoring_features_t current_features;
static snoring_prediction_t current_prediction;

// Прототипы функций
void init_snoring_detector(void);
void extract_features_from_csv(float* csv_data);
snoring_prediction_t classify_snoring(snoring_features_t features);
const char* get_snoring_class_name(int class_id);
float estimate_snoring_risk(snoring_prediction_t prediction);
void process_audio_sample(float sample);
void process_accelerometer(float x, float y, float z);
void update_snoring_status(void);

// Инициализация детектора храпа
void init_snoring_detector(void) {
    ESP_LOGI(TAG, "Инициализация 2-классового детектора храпа");
    
    // Инициализация буферов
    memset(audio_buffer, 0, sizeof(audio_buffer));
    memset(&current_features, 0, sizeof(current_features));
    memset(&current_prediction, 0, sizeof(current_prediction));
    
    // Установка значений по умолчанию
    current_features.breath_nn = 95.0f;
    current_features.snore_nn = 142.0f;
    current_features.signal_nn = 0.0f;
    current_features.splash_nn = 0.0f;
    current_features.b100 = 5.0f;
    current_features.b400 = 3.0f;
    current_features.b1000 = 10.0f;
    current_features.env = 50.0f;
    
    ESP_LOGI(TAG, "Детектор храпа инициализирован");
}

// Извлечение признаков из CSV данных
void extract_features_from_csv(float* csv_data) {
    if (csv_data == NULL) {
        ESP_LOGE(TAG, "Ошибка: csv_data == NULL");
        return;
    }
    
    // Акселерометр
    current_features.x = csv_data[1];
    current_features.y = csv_data[2];
    current_features.z = csv_data[3];
    
    // Пульс
    current_features.bpm = csv_data[4];
    current_features.bpm = csv_data[5]; // bpm_by_nn
    current_features.bpm = csv_data[6]; // bpm_by_acf
    
    // Дополнительные признаки
    current_features.mf = csv_data[7];
    current_features.af = csv_data[8];
    current_features.arf = csv_data[9];
    current_features.sf = csv_data[10];
    current_features.r_th = csv_data[11];
    
    // Статусы
    current_features.status_1 = csv_data[12];
    current_features.status_2 = csv_data[13];
    current_features.status_3 = csv_data[14];
    
    // Дополнительные метки
    current_features.ml = csv_data[15];
    
    // Нейронные сети
    current_features.breath_nn = csv_data[16];
    current_features.snore_nn = csv_data[17];
    current_features.signal_nn = csv_data[18];
    current_features.splash_nn = csv_data[19];
    
    // EOG
    current_features.eog = csv_data[20];
    
    // Полосовые фильтры
    current_features.b100 = csv_data[21];
    current_features.b400 = csv_data[22];
    current_features.b1000 = csv_data[23];
    
    // Огибающая
    current_features.env = csv_data[24];
    
    ESP_LOGI(TAG, "Признаки извлечены из CSV");
}

// Классификация храпа (упрощенная логика)
snoring_prediction_t classify_snoring(snoring_features_t features) {
    snoring_prediction_t prediction;
    
    // Простая эвристическая классификация
    float snore_score = 0.0f;
    
    // Анализ нейронных сетей
    if (features.snore_nn > 150.0f) {
        snore_score += 0.3f;
    }
    if (features.signal_nn > 50.0f) {
        snore_score += 0.2f;
    }
    if (features.splash_nn > 50.0f) {
        snore_score += 0.2f;
    }
    
    // Анализ полосовых фильтров
    if (features.b1000 > 100.0f) {
        snore_score += 0.2f;
    }
    if (features.env > 100.0f) {
        snore_score += 0.1f;
    }
    
    // Анализ соотношений
    float snore_breath_ratio = features.snore_nn / (features.breath_nn + 1e-6f);
    if (snore_breath_ratio > 1.5f) {
        snore_score += 0.2f;
    }
    
    // Определение класса
    if (snore_score > 0.5f) {
        prediction.class_id = SNORING_YES;
        prediction.is_snoring = true;
        prediction.p_snore = fminf(snore_score, 1.0f);
        prediction.confidence = fminf(snore_score, 1.0f);
    } else {
        prediction.class_id = SNORING_NO;
        prediction.is_snoring = false;
        prediction.p_snore = snore_score;
        prediction.confidence = 1.0f - snore_score;
    }
    
    return prediction;
}

// Получение названия класса
const char* get_snoring_class_name(int class_id) {
    switch (class_id) {
        case SNORING_NO:
            return "No_Snoring";
        case SNORING_YES:
            return "Snoring";
        default:
            return "Unknown";
    }
}

// Оценка риска храпа
float estimate_snoring_risk(snoring_prediction_t prediction) {
    if (prediction.is_snoring) {
        if (prediction.p_snore > 0.8f) {
            return 0.9f; // Высокий риск
        } else if (prediction.p_snore > 0.5f) {
            return 0.6f; // Средний риск
        } else {
            return 0.3f; // Низкий риск
        }
    } else {
        return 0.1f; // Минимальный риск
    }
}

// Обработка аудио сэмпла
void process_audio_sample(float sample) {
    // Простая обработка аудио (в реальности здесь будет FFT)
    static int buffer_index = 0;
    
    audio_buffer[buffer_index] = sample;
    buffer_index = (buffer_index + 1) % BUFFER_SIZE;
    
    // Обновление признаков каждые 100 сэмплов
    if (buffer_index % 100 == 0) {
        // Вычисление RMS
        float rms = 0.0f;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            rms += audio_buffer[i] * audio_buffer[i];
        }
        rms = sqrtf(rms / BUFFER_SIZE);
        
        // Обновление огибающей
        current_features.env = rms;
        
        // Классификация
        current_prediction = classify_snoring(current_features);
        
        ESP_LOGI(TAG, "Аудио обработано: RMS=%.2f, класс=%s, p_snore=%.3f", 
                 rms, get_snoring_class_name(current_prediction.class_id), 
                 current_prediction.p_snore);
    }
}

// Обработка данных акселерометра
void process_accelerometer(float x, float y, float z) {
    current_features.x = x;
    current_features.y = y;
    current_features.z = z;
    
    // Вычисление величины ускорения
    float magnitude = sqrtf(x*x + y*y + z*z);
    
    // Обновление статуса каждые 10 измерений
    static int accel_counter = 0;
    if (++accel_counter >= 10) {
        accel_counter = 0;
        
        // Простая логика определения активности
        if (magnitude > 2000.0f) {
            ESP_LOGI(TAG, "Высокая активность: magnitude=%.2f", magnitude);
        }
    }
}

// Обновление статуса храпа
void update_snoring_status(void) {
    // Обновление каждую секунду
    static uint32_t last_update = 0;
    uint32_t current_time = esp_timer_get_time() / 1000000; // в секундах
    
    if (current_time - last_update >= 1) {
        last_update = current_time;
        
        // Логирование текущего статуса
        ESP_LOGI(TAG, "Статус храпа: %s (уверенность=%.3f, риск=%.3f)", 
                 get_snoring_class_name(current_prediction.class_id),
                 current_prediction.confidence,
                 estimate_snoring_risk(current_prediction));
        
        // Отправка данных (в реальности здесь будет отправка по WiFi/BLE)
        // send_snoring_data(&current_prediction);
    }
}

// Основная задача детектора храпа
void snoring_detector_task(void *pvParameters) {
    ESP_LOGI(TAG, "Задача детектора храпа запущена");
    
    // Инициализация
    init_snoring_detector();
    
    // Основной цикл
    while (1) {
        // Обновление статуса
        update_snoring_status();
        
        // Задержка
        vTaskDelay(pdMS_TO_TICKS(100)); // 100 мс
    }
}

// Функция для тестирования
void test_snoring_detector(void) {
    ESP_LOGI(TAG, "Тестирование детектора храпа");
    
    // Тест 1: Нормальное дыхание
    snoring_features_t test_features = {
        .breath_nn = 95.0f,
        .snore_nn = 142.0f,
        .signal_nn = 0.0f,
        .splash_nn = 0.0f,
        .b100 = 5.0f,
        .b400 = 3.0f,
        .b1000 = 10.0f,
        .env = 50.0f
    };
    
    snoring_prediction_t prediction = classify_snoring(test_features);
    ESP_LOGI(TAG, "Тест 1 - Нормальное дыхание: класс=%s, p_snore=%.3f", 
             get_snoring_class_name(prediction.class_id), prediction.p_snore);
    
    // Тест 2: Храп
    test_features.snore_nn = 200.0f;
    test_features.signal_nn = 80.0f;
    test_features.b1000 = 150.0f;
    test_features.env = 120.0f;
    
    prediction = classify_snoring(test_features);
    ESP_LOGI(TAG, "Тест 2 - Храп: класс=%s, p_snore=%.3f", 
             get_snoring_class_name(prediction.class_id), prediction.p_snore);
    
    ESP_LOGI(TAG, "Тестирование завершено");
}

// API функции для внешнего использования
snoring_prediction_t get_current_prediction(void) {
    return current_prediction;
}

snoring_features_t get_current_features(void) {
    return current_features;
}

void set_csv_features(float* csv_data) {
    extract_features_from_csv(csv_data);
    current_prediction = classify_snoring(current_features);
}

// Инициализация системы
void snoring_system_init(void) {
    ESP_LOGI(TAG, "Инициализация системы детекции храпа");
    
    // Создание задачи детектора
    xTaskCreate(snoring_detector_task, "snoring_detector", 4096, NULL, 5, NULL);
    
    // Тестирование
    test_snoring_detector();
    
    ESP_LOGI(TAG, "Система детекции храпа инициализирована");
} 