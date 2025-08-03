# Исправления TypeError: '<' not supported between instances of 'str' and 'int'

## 🐛 Проблема
Ошибка возникала при сравнении строковых названий признаков с числовыми значениями в различных частях кода.

## 🔧 Примененные исправления

### 1. **ESP32FeatureSelector** (`src/features/esp32_extractor.py`)

**Проблема**: Сравнение строк с числами в `feature_names`

**Исправления**:
```python
# Добавлена проверка типов
if not all(isinstance(name, str) for name in feature_names):
    raise ValueError("Все названия признаков должны быть строками")

# Преобразование scores в float
self.feature_importance[feature_names[i]] = float(score)
```

### 2. **ESP32Classifier** (`src/models/esp32_classifier.py`)

**Проблема**: Сравнение строк с числами в `feature_names`

**Исправления**:
```python
# Проверка типов в train()
if not all(isinstance(name, str) for name in feature_names):
    raise ValueError("Все названия признаков должны быть строками")

# Преобразование в int в генерации C-кода
file.write(f"#define NUM_CLASSES {int(quantized_model['n_classes'])}\n")
file.write(f"#define QUANTIZATION_SCALE {int(quantized_model['quantization_scale'])}\n")
```

### 3. **ESP32FeatureExtractor** (`src/features/esp32_extractor.py`)

**Проблема**: NaN и бесконечные значения в признаках

**Исправления**:
```python
# Преобразование всех значений в float
features[f"{ch_name}_rms"] = float(np.sqrt(np.mean(signal_data**2)))

# Проверка на NaN и бесконечность
for key, value in features.items():
    if np.isnan(value) or np.isinf(value):
        features[key] = 0.0

# Безопасная нормализация
if ch1_std < 1e-8:
    ch1_norm = np.zeros_like(ch1)
else:
    ch1_norm = (ch1 - np.mean(ch1)) / ch1_std
```

### 4. **esp32_demo.py**

**Проблема**: Смешанные типы данных в DataFrame

**Исправления**:
```python
# Преобразование в числовой формат
for col in features_clean.columns:
    features_clean[col] = pd.to_numeric(features_clean[col], errors='coerce')

# Убеждение, что названия признаков - строки
feature_names = [str(col) for col in features_clean.columns]
selected_features_str = [str(feature) for feature in selected_features]
```

### 5. **Квантизация признаков**

**Проблема**: Сравнение строк с числами в квантизации

**Исправления**:
```python
# Проверка типов в квантизации
if not isinstance(value, (int, float)):
    value = float(value)
```

## 🧪 Тестирование

Создан тестовый скрипт `test_esp32_fixes.py` для проверки всех исправлений:

```bash
python test_esp32_fixes.py
```

Тесты проверяют:
- ✅ Извлечение признаков без NaN/бесконечности
- ✅ Селекцию признаков с правильными типами
- ✅ Обучение классификатора
- ✅ Квантизацию модели
- ✅ Операции с DataFrame

## 📊 Результаты исправлений

### До исправлений:
- ❌ TypeError при сравнении строк с числами
- ❌ NaN значения в признаках
- ❌ Бесконечные значения
- ❌ Проблемы с типами данных

### После исправлений:
- ✅ Все сравнения типов безопасны
- ✅ NaN значения заменяются на 0.0
- ✅ Бесконечные значения обрабатываются
- ✅ Все типы данных корректны
- ✅ Добавлены проверки типов

## 🚀 Использование

Теперь можно безопасно запускать ESP32 демонстрацию:

```bash
# Активировать виртуальное окружение
source venv/bin/activate

# Запустить демонстрацию
python esp32_demo.py

# Или запустить тесты
python test_esp32_fixes.py
```

## 🔍 Дополнительные улучшения

1. **Логирование ошибок**: Добавлено подробное логирование для отладки
2. **Валидация данных**: Проверка типов на всех этапах
3. **Безопасная обработка**: Graceful handling ошибок
4. **Документация**: Подробные комментарии в коде

## 📝 Заключение

Все проблемы с TypeError исправлены. Код теперь безопасно обрабатывает:
- Смешанные типы данных
- NaN и бесконечные значения
- Сравнения строк с числами
- Преобразования типов

Система готова к использованию на ESP32! 🎉 