# Оптимизация проекта классификации стадий сна под микроконтроллеры ESP32

## Ограничения ESP32
- **RAM**: до 512 КБ SRAM
- **Flash**: до 4 МБ
- **CPU**: 240 МГц, без FPU
- **Поддержка**: TensorFlow Lite for Microcontrollers

## Шаг 1: Анализ текущей архитектуры

### Текущие проблемы:
1. **Слишком много признаков** (~200+ признаков на сегмент)
2. **Сложные модели** (Random Forest, SVM, Neural Networks)
3. **Float32 вычисления** (медленно на ESP32)
4. **Большие размеры моделей** (сотни КБ)
5. **Сложная предобработка** (спектральный анализ, PCA)

## Шаг 2: Оптимизация извлечения признаков

### 2.1 Упрощение признаков
```python
# Только самые важные признаки:
- RMS (Root Mean Square)
- Стандартное отклонение
- Количество пересечений нуля
- Простые статистики (min, max, mean)
- Базовая спектральная мощность (без сложных вычислений)
```

### 2.2 Уменьшение количества признаков
- **Цель**: максимум 20-30 признаков
- **Метод**: Feature selection + упрощение
- **Результат**: экономия памяти и времени

## Шаг 3: Оптимизация моделей

### 3.1 Выбор простой модели
```python
# Рекомендуемые модели для ESP32:
1. Decision Tree (самая простая)
2. Logistic Regression
3. Простая нейронная сеть (1-2 слоя)
4. K-Nearest Neighbors (с ограниченным K)
```

### 3.2 Квантизация моделей
```python
# Преобразование float32 → int8
- Экономия памяти: 75%
- Ускорение вычислений: 3-4x
- Потеря точности: 2-5%
```

## Шаг 4: Оптимизация данных

### 4.1 Уменьшение размера входных данных
```python
# Оптимизации:
- Downsampling: 256 Hz → 64 Hz
- Уменьшение длины сегмента: 30 сек → 10 сек
- Уменьшение количества каналов: 2 → 1
```

### 4.2 Сжатие данных
```python
# Методы сжатия:
- Delta encoding
- Run-length encoding
- Простые алгоритмы сжатия
```

## Шаг 5: Создание оптимизированного экстрактора признаков

### 5.1 Минимальный набор признаков
```python
def extract_minimal_features(segment):
    """Извлекает только самые важные признаки для ESP32."""
    features = {}
    
    # Базовые статистики
    features['rms'] = np.sqrt(np.mean(segment**2))
    features['std'] = np.std(segment)
    features['mean'] = np.mean(segment)
    features['range'] = np.max(segment) - np.min(segment)
    
    # Простые частотные признаки
    features['zero_crossings'] = np.sum(np.diff(np.sign(segment)) != 0)
    features['peak_count'] = len(find_peaks_simple(segment))
    
    # Простая спектральная мощность
    features['low_freq_power'] = extract_simple_power(segment, 0.5, 4.0)
    features['high_freq_power'] = extract_simple_power(segment, 8.0, 30.0)
    
    return features
```

## Шаг 6: Создание простой модели

### 6.1 Decision Tree для ESP32
```python
class ESP32DecisionTree:
    """Простая реализация дерева решений для ESP32."""
    
    def __init__(self, max_depth=5, max_features=10):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None
    
    def fit(self, X, y):
        """Обучение дерева."""
        # Упрощенная реализация CART
        pass
    
    def predict(self, X):
        """Предсказание."""
        # Простой обход дерева
        pass
```

## Шаг 7: Квантизация и оптимизация

### 7.1 Квантизация признаков
```python
def quantize_features(features, scale=100):
    """Квантизация признаков в int8."""
    return np.round(features * scale).astype(np.int8)
```

### 7.2 Квантизация модели
```python
def quantize_model(model):
    """Квантизация весов модели."""
    # TensorFlow Lite квантизация
    # или ручная квантизация
    pass
```

## Шаг 8: Создание C-кода для ESP32

### 8.1 Генерация C-кода
```python
def generate_esp32_code(model, features_config):
    """Генерирует C-код для ESP32."""
    
    # Генерация кода извлечения признаков
    generate_feature_extraction_code(features_config)
    
    # Генерация кода модели
    generate_model_code(model)
    
    # Генерация основного кода
    generate_main_code()
```

### 8.2 Структура C-кода
```c
// feature_extraction.h
typedef struct {
    float rms;
    float std;
    float mean;
    int zero_crossings;
    int peak_count;
    float low_freq_power;
    float high_freq_power;
} features_t;

// model.h
typedef struct {
    int8_t weights[100];  // Квантизованные веса
    int8_t biases[5];     // Квантизованные смещения
} model_t;

// main.c
void classify_sleep_stage(float* eeg_data, int* prediction) {
    features_t features = extract_features(eeg_data);
    *prediction = predict_model(features);
}
```

## Шаг 9: Тестирование и валидация

### 9.1 Тестирование на реальном железе
```python
def test_on_esp32():
    """Тестирование на реальном ESP32."""
    
    # Загрузка тестовых данных
    test_data = load_test_data()
    
    # Симуляция ESP32
    predictions = simulate_esp32_inference(test_data)
    
    # Сравнение с оригинальной моделью
    compare_with_original(predictions)
```

### 9.2 Оптимизация производительности
```python
def optimize_performance():
    """Оптимизация производительности."""
    
    # Профилирование памяти
    memory_usage = profile_memory_usage()
    
    # Профилирование времени
    inference_time = profile_inference_time()
    
    # Оптимизация на основе профиля
    optimize_based_on_profile(memory_usage, inference_time)
```

## Шаг 10: Создание финального решения

### 10.1 Структура проекта для ESP32
```
esp32_sleep_classifier/
├── src/
│   ├── feature_extraction.c
│   ├── feature_extraction.h
│   ├── model.c
│   ├── model.h
│   ├── main.c
│   └── config.h
├── data/
│   ├── model_weights.h
│   └── model_config.h
├── examples/
│   └── basic_example.ino
└── README.md
```

### 10.2 Конфигурация для ESP32
```c
// config.h
#define SAMPLING_RATE 64
#define SEGMENT_LENGTH 640  // 10 секунд
#define NUM_FEATURES 8
#define NUM_CLASSES 5
#define MODEL_SIZE 2048  // 2KB
#define MAX_MEMORY_USAGE 50000  // 50KB
```

## Результаты оптимизации

### Ожидаемые улучшения:
- **Размер модели**: 500KB → 2KB (99.6% уменьшение)
- **Время инференса**: 100ms → 10ms (90% ускорение)
- **Использование памяти**: 200KB → 50KB (75% экономия)
- **Точность**: 85% → 80% (5% потеря точности)

### Компромиссы:
- ✅ Работает на ESP32
- ✅ Быстрая инференция
- ✅ Низкое энергопотребление
- ❌ Потеря точности
- ❌ Упрощенная модель

## Следующие шаги

1. **Реализация упрощенного экстрактора признаков**
2. **Создание простой модели (Decision Tree)**
3. **Квантизация и оптимизация**
4. **Генерация C-кода**
5. **Тестирование на ESP32**
6. **Финальная оптимизация**

Этот план обеспечит работоспособность системы классификации стадий сна на ESP32 с приемлемой точностью и производительностью. 