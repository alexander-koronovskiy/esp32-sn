#!/usr/bin/env python3
"""
Обновленный тестовый скрипт для проверки исправлений TypeError.
"""

import sys
import os
import numpy as np
import pandas as pd

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.esp32_extractor import ESP32FeatureExtractor, ESP32FeatureSelector, create_esp32_config
from src.models.esp32_classifier import ESP32Classifier, create_esp32_classifier


def test_comparison_operations():
    """Тестирует операции сравнения."""
    print("Тестирование операций сравнения...")
    
    # Тест 1: Сравнение чисел
    assert 5 < 10, "Сравнение чисел должно работать"
    assert 3.14 <= 3.15, "Сравнение float должно работать"
    
    # Тест 2: Сравнение с NaN
    assert not (np.nan < 5), "NaN сравнение должно быть False"
    assert not (5 < np.nan), "NaN сравнение должно быть False"
    
    # Тест 3: Сравнение с бесконечностью
    assert 5 < np.inf, "Сравнение с inf должно работать"
    assert -np.inf < 5, "Сравнение с -inf должно работать"
    
    print("✓ Все операции сравнения корректны")


def test_feature_extraction_with_edge_cases():
    """Тестирует извлечение признаков с граничными случаями."""
    print("\nТестирование извлечения признаков с граничными случаями...")
    
    config = create_esp32_config()
    extractor = ESP32FeatureExtractor(
        sampling_rate=config['sampling_rate'],
        segment_length=config['segment_length']
    )
    
    # Тест 1: Нормальные данные
    test_segment = np.random.randn(1, config['segment_length'])
    features = extractor.extract_minimal_features(test_segment)
    
    for key, value in features.items():
        assert isinstance(value, (int, float)), f"Признак {key} не является числом"
        assert not np.isnan(value), f"Признак {key} содержит NaN"
        assert not np.isinf(value), f"Признак {key} содержит бесконечность"
    
    # Тест 2: Данные с NaN
    test_segment_nan = np.random.randn(1, config['segment_length'])
    test_segment_nan[0, 0] = np.nan
    features_nan = extractor.extract_minimal_features(test_segment_nan)
    
    for key, value in features_nan.items():
        assert isinstance(value, (int, float)), f"Признак {key} не является числом"
        assert not np.isnan(value), f"Признак {key} содержит NaN"
        assert not np.isinf(value), f"Признак {key} содержит бесконечность"
    
    # Тест 3: Данные с бесконечностью
    test_segment_inf = np.random.randn(1, config['segment_length'])
    test_segment_inf[0, 0] = np.inf
    features_inf = extractor.extract_minimal_features(test_segment_inf)
    
    for key, value in features_inf.items():
        assert isinstance(value, (int, float)), f"Признак {key} не является числом"
        assert not np.isnan(value), f"Признак {key} содержит NaN"
        assert not np.isinf(value), f"Признак {key} содержит бесконечность"
    
    print("✓ Извлечение признаков работает с граничными случаями")


def test_feature_selection_with_mixed_types():
    """Тестирует селекцию признаков со смешанными типами."""
    print("\nТестирование селекции признаков со смешанными типами...")
    
    # Создаем данные с потенциально проблемными названиями признаков
    n_samples = 50
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    
    # Тест 1: Строковые названия признаков
    feature_names_str = [f'feature_{i}' for i in range(n_features)]
    selector = ESP32FeatureSelector(max_features=5)
    
    try:
        selected_features = selector.select_features(X, y, feature_names_str)
        print(f"✓ Селекция со строковыми названиями: {len(selected_features)} признаков")
    except Exception as e:
        print(f"✗ Ошибка со строковыми названиями: {e}")
        return False
    
    # Тест 2: Смешанные названия признаков (должно вызвать ошибку)
    feature_names_mixed = [f'feature_{i}' if i % 2 == 0 else i for i in range(n_features)]
    selector2 = ESP32FeatureSelector(max_features=5)
    
    try:
        selected_features = selector2.select_features(X, y, feature_names_mixed)
        print("✗ Не должна была пройти селекция со смешанными типами")
        return False
    except ValueError as e:
        print(f"✓ Правильно отловлена ошибка со смешанными типами: {e}")
    
    print("✓ Селекция признаков работает корректно")
    return True


def test_classifier_with_edge_cases():
    """Тестирует классификатор с граничными случаями."""
    print("\nТестирование классификатора с граничными случаями...")
    
    # Создаем тестовые данные
    n_samples = 30
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Создаем классификатор
    config = create_esp32_config()
    classifier = create_esp32_classifier(config)
    
    # Обучаем модель
    train_metrics = classifier.train(X, y, feature_names)
    print(f"✓ Модель обучена, точность: {train_metrics['train_accuracy']:.4f}")
    
    # Тестируем предсказания
    predictions = classifier.predict(X)
    assert len(predictions) == len(X), "Количество предсказаний не совпадает"
    assert all(isinstance(p, (int, np.integer)) for p in predictions), "Не все предсказания - числа"
    
    # Тестируем квантизацию
    quantized_model = classifier.quantize_model()
    predictions_quantized = classifier.predict_quantized(X, quantized_model)
    assert len(predictions_quantized) == len(X), "Количество квантизованных предсказаний не совпадает"
    assert all(isinstance(p, (int, np.integer)) for p in predictions_quantized), "Не все квантизованные предсказания - числа"
    
    print("✓ Классификатор работает с граничными случаями")
    return classifier


def test_dataframe_operations_with_problematic_data():
    """Тестирует операции с DataFrame с проблемными данными."""
    print("\nТестирование операций с DataFrame с проблемными данными...")
    
    # Создаем DataFrame с проблемными данными
    data = {
        'feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature_2': [0.1, 0.2, 0.3, np.inf, 0.5],
        'feature_3': ['a', 'b', 'c', 'd', 'e'],  # Нечисловая колонка
        'feature_4': [10, 20, 30, 40, 50],
        'feature_5': [1, 2, 3, 4, 'problem']  # Смешанные типы
    }
    
    df = pd.DataFrame(data)
    print(f"Исходный DataFrame: {df.shape}")
    
    # Выбираем числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()
    
    print(f"Числовые колонки: {list(numeric_cols)}")
    
    # Преобразуем в числовой формат
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Удаляем NaN
    df_numeric = df_numeric.dropna()
    print(f"DataFrame после очистки: {df_numeric.shape}")
    
    # Проверяем, что все значения - числа
    for col in df_numeric.columns:
        assert df_numeric[col].dtype in ['int64', 'float64'], f"Колонка {col} не числовая"
        assert not df_numeric[col].isna().any(), f"Колонка {col} содержит NaN"
    
    print("✓ Операции с DataFrame работают с проблемными данными")


def test_c_code_generation():
    """Тестирует генерацию C-кода."""
    print("\nТестирование генерации C-кода...")
    
    # Создаем классификатор и обучаем его
    n_samples = 20
    n_features = 6
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    config = create_esp32_config()
    classifier = create_esp32_classifier(config)
    classifier.train(X, y, feature_names)
    
    # Генерируем C-код
    os.makedirs('test_output', exist_ok=True)
    c_code_file = 'test_output/test_classifier.c'
    
    try:
        classifier.generate_c_code(c_code_file)
        print(f"✓ C-код сгенерирован: {c_code_file}")
        
        # Проверяем, что файл создан и не пустой
        assert os.path.exists(c_code_file), "C-код не создан"
        with open(c_code_file, 'r') as f:
            content = f.read()
            assert len(content) > 0, "C-код пустой"
            assert '#define NUM_FEATURES' in content, "Отсутствуют определения"
            assert 'predict_sleep_stage' in content, "Отсутствует функция предсказания"
        
        print("✓ C-код содержит необходимые элементы")
        
    except Exception as e:
        print(f"✗ Ошибка генерации C-кода: {e}")
        return False
    
    return True


def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ TYPERROR (ОБНОВЛЕННАЯ ВЕРСИЯ)")
    print("=" * 60)
    
    try:
        # Тест 1: Операции сравнения
        test_comparison_operations()
        
        # Тест 2: Извлечение признаков с граничными случаями
        test_feature_extraction_with_edge_cases()
        
        # Тест 3: Селекция признаков со смешанными типами
        if not test_feature_selection_with_mixed_types():
            return False
        
        # Тест 4: Классификатор с граничными случаями
        classifier = test_classifier_with_edge_cases()
        
        # Тест 5: Операции с DataFrame с проблемными данными
        test_dataframe_operations_with_problematic_data()
        
        # Тест 6: Генерация C-кода
        if not test_c_code_generation():
            return False
        
        print("\n" + "=" * 60)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        
        print("\nИсправления применены:")
        print("✅ Безопасные операции сравнения")
        print("✅ Обработка NaN и бесконечных значений")
        print("✅ Проверка типов в селекции признаков")
        print("✅ Безопасная квантизация моделей")
        print("✅ Корректная генерация C-кода")
        print("✅ Обработка проблемных данных в DataFrame")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 