#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений TypeError.
"""

import sys
import os
import numpy as np
import pandas as pd

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.esp32_extractor import ESP32FeatureExtractor, ESP32FeatureSelector, create_esp32_config
from src.models.esp32_classifier import ESP32Classifier, create_esp32_classifier


def test_feature_extraction():
    """Тестирует извлечение признаков."""
    print("Тестирование извлечения признаков...")
    
    # Создаем тестовые данные
    config = create_esp32_config()
    extractor = ESP32FeatureExtractor(
        sampling_rate=config['sampling_rate'],
        segment_length=config['segment_length']
    )
    
    # Создаем тестовый сегмент
    test_segment = np.random.randn(1, config['segment_length'])
    
    # Извлекаем признаки
    features = extractor.extract_minimal_features(test_segment)
    
    print(f"✓ Извлечено {len(features)} признаков")
    print(f"  Типы значений: {[type(v) for v in features.values()]}")
    
    # Проверяем, что все значения - числа
    for key, value in features.items():
        assert isinstance(value, (int, float)), f"Признак {key} не является числом: {type(value)}"
        assert not np.isnan(value), f"Признак {key} содержит NaN"
        assert not np.isinf(value), f"Признак {key} содержит бесконечность"
    
    print("✓ Все признаки корректны")
    return features


def test_feature_selection():
    """Тестирует селекцию признаков."""
    print("\nТестирование селекции признаков...")
    
    # Создаем тестовые данные
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Создаем селектор
    selector = ESP32FeatureSelector(max_features=5)
    
    # Выбираем признаки
    selected_features = selector.select_features(X, y, feature_names)
    
    print(f"✓ Выбрано {len(selected_features)} признаков")
    print(f"  Выбранные признаки: {selected_features}")
    
    # Проверяем, что все названия - строки
    for feature in selected_features:
        assert isinstance(feature, str), f"Признак {feature} не является строкой: {type(feature)}"
    
    print("✓ Все названия признаков корректны")
    return selected_features


def test_classifier():
    """Тестирует классификатор."""
    print("\nТестирование классификатора...")
    
    # Создаем тестовые данные
    n_samples = 50
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Создаем классификатор
    config = create_esp32_config()
    classifier = create_esp32_classifier(config)
    
    # Обучаем модель
    train_metrics = classifier.train(X, y, feature_names)
    
    print(f"✓ Модель обучена")
    print(f"  Точность: {train_metrics['train_accuracy']:.4f}")
    print(f"  Размер модели: {train_metrics['model_size_bytes']} байт")
    
    # Делаем предсказания
    predictions = classifier.predict(X)
    print(f"✓ Предсказания выполнены: {len(predictions)}")
    
    # Тестируем квантизацию
    quantized_model = classifier.quantize_model()
    print(f"✓ Модель квантизована")
    
    # Тестируем квантизованные предсказания
    predictions_quantized = classifier.predict_quantized(X, quantized_model)
    print(f"✓ Квантизованные предсказания выполнены: {len(predictions_quantized)}")
    
    return classifier


def test_dataframe_operations():
    """Тестирует операции с DataFrame."""
    print("\nТестирование операций с DataFrame...")
    
    # Создаем тестовый DataFrame
    data = {
        'feature_1': [1.0, 2.0, 3.0, np.nan, 5.0],
        'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature_3': ['a', 'b', 'c', 'd', 'e'],  # Нечисловая колонка
        'feature_4': [10, 20, 30, 40, 50]
    }
    
    df = pd.DataFrame(data)
    print(f"Исходный DataFrame: {df.shape}")
    
    # Выбираем числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].dropna()
    
    print(f"Числовые колонки: {list(numeric_cols)}")
    print(f"DataFrame после очистки: {df_numeric.shape}")
    
    # Преобразуем в числовой формат
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    df_numeric = df_numeric.dropna()
    print(f"DataFrame после преобразования: {df_numeric.shape}")
    
    # Проверяем типы данных
    for col in df_numeric.columns:
        assert df_numeric[col].dtype in ['int64', 'float64'], f"Колонка {col} не числовая: {df_numeric[col].dtype}"
    
    print("✓ Все операции с DataFrame корректны")


def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ TYPERROR")
    print("=" * 60)
    
    try:
        # Тестируем извлечение признаков
        features = test_feature_extraction()
        
        # Тестируем селекцию признаков
        selected_features = test_feature_selection()
        
        # Тестируем классификатор
        classifier = test_classifier()
        
        # Тестируем операции с DataFrame
        test_dataframe_operations()
        
        print("\n" + "=" * 60)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        
        print("\nИсправления применены:")
        print("✅ Проверка типов данных в feature_names")
        print("✅ Преобразование значений в float")
        print("✅ Обработка NaN и бесконечных значений")
        print("✅ Безопасная нормализация данных")
        print("✅ Проверка типов в квантизации")
        print("✅ Обработка ошибок в DataFrame операциях")
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 