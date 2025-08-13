#!/usr/bin/env python3
"""
Демонстрация Decision Tree классификатора храпа с 30 признаками.
Архитектура согласно техническому заданию.
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Tuple

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent / 'src'))

from models.snoring_classifier_decision_tree import SnoringDecisionTreeClassifier
from features.decision_tree_extractor import DecisionTreeFeatureExtractor, create_feature_extractor_config


def create_synthetic_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает синтетические данные для тестирования.
    
    Args:
        n_samples: Количество сэмплов
        
    Returns:
        X: Признаки (n_samples, 30)
        y: Метки классов (n_samples,)
    """
    print(f"🔧 Создание синтетических данных: {n_samples} сэмплов...")
    
    # Генерируем 30 признаков
    X = np.random.randn(n_samples, 30)
    
    # Создаем метки классов (3 класса)
    # 0: No Snoring, 1: Light Snoring, 2: Heavy Snoring
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
    
    print(f"✅ Создано {n_samples} сэмплов с 30 признаками")
    print(f"📊 Распределение классов: {np.bincount(y)}")
    
    return X, y


def load_real_data(data_path: str = "snoring_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает реальные данные из CSV файлов.
    
    Args:
        data_path: Путь к папке с данными
        
    Returns:
        X: Признаки
        y: Метки классов
    """
    print(f"📁 Загрузка реальных данных из {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"❌ Папка {data_path} не найдена")
        return None, None
    
    features_list = []
    labels_list = []
    
    # Проходим по всем подпапкам
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # Загружаем CSV
                    csv_data = pd.read_csv(file_path).values
                    
                    if csv_data.shape[1] >= 25:  # Проверяем количество колонок
                        # Создаем экстрактор признаков
                        extractor = DecisionTreeFeatureExtractor()
                        
                        # Извлекаем 30 признаков
                        features = extractor.extract_features_from_csv(csv_data)
                        features_list.append(features)
                        
                        # Определяем метку класса
                        # W = No Snoring (0), отсутствие метки = Snoring (1)
                        label = 1  # По умолчанию Snoring
                        labels_list.append(label)
                        
                except Exception as e:
                    print(f"⚠️ Ошибка при обработке {file}: {e}")
                    continue
    
    if not features_list:
        print("❌ Не удалось загрузить данные")
        return None, None
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"✅ Загружено {len(X)} реальных сэмплов")
    print(f"🔢 Количество признаков: {X.shape[1]}")
    print(f"📊 Распределение меток: {np.bincount(y)}")
    
    return X, y


def test_decision_tree_classifier():
    """Тестирует Decision Tree классификатор."""
    print("=" * 60)
    print("🌳 ТЕСТИРОВАНИЕ DECISION TREE КЛАССИФИКАТОРА")
    print("=" * 60)
    
    # 1. Создаем классификатор
    print("\n🔧 Создание Decision Tree классификатора...")
    classifier = SnoringDecisionTreeClassifier(
        max_depth=5,           # Ограничение сложности согласно ТЗ
        min_samples_leaf=10,   # Минимальное количество сэмплов в листе
        criterion='gini'       # Критерий разделения
    )
    
    print(f"✅ Классификатор создан:")
    print(f"   - Максимальная глубина: {classifier.max_depth}")
    print(f"   - Минимальные сэмплы в листе: {classifier.min_samples_leaf}")
    print(f"   - Критерий: {classifier.criterion}")
    print(f"   - Количество классов: {classifier.class_count}")
    
    # 2. Тест с синтетическими данными
    print("\n🧪 Тест с синтетическими данными...")
    X_synthetic, y_synthetic = create_synthetic_data(1000)
    
    try:
        # Обучаем модель
        metrics = classifier.train(X_synthetic, y_synthetic)
        print(f"✅ Модель обучена на синтетических данных!")
        print(f"📊 Точность: {metrics['accuracy']:.3f}")
        print(f"🔢 Количество признаков: {metrics['feature_count']}")
        
        # Тестируем предсказания
        y_pred = classifier.predict(X_synthetic[:10])
        y_proba = classifier.predict_proba(X_synthetic[:10])
        
        print(f"🔮 Тест предсказаний:")
        print(f"   - Предсказания: {y_pred}")
        print(f"   - Вероятности: {y_proba.shape}")
        
        # Применяем постобработку
        processed_predictions = classifier.apply_postprocessing(y_pred, y_proba)
        print(f"   - После постобработки: {processed_predictions}")
        
    except Exception as e:
        print(f"❌ Ошибка при работе с синтетическими данными: {e}")
    
    # 3. Тест с реальными данными
    print("\n📊 Тест с реальными данными...")
    X_real, y_real = load_real_data()
    
    if X_real is not None and y_real is not None:
        try:
            # Обучаем модель на реальных данных
            print("📚 Обучение на реальных данных...")
            metrics = classifier.train(X_real, y_real)
            print(f"✅ Модель обучена на реальных данных!")
            print(f"📊 Точность: {metrics['accuracy']:.3f}")
            print(f"🔢 Количество признаков: {metrics['feature_count']}")
            
            # Тестируем предсказания
            print("🔮 Тестирование предсказания на реальных данных...")
            y_pred = classifier.predict(X_real[:10])
            y_proba = classifier.predict_proba(X_real[:10])
            
            print(f"   - Предсказания: {y_pred}")
            print(f"   - Вероятности: {y_proba.shape}")
            
            # Применяем постобработку
            processed_predictions = classifier.apply_postprocessing(y_pred, y_proba)
            print(f"   - После постобработки: {processed_predictions}")
            
        except Exception as e:
            print(f"❌ Ошибка при работе с реальными данными: {e}")
    
    # 4. Информация о модели
    print("\n📋 Информация о модели:")
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        print(f"   - {key}: {value}")
    
    # 5. Сохранение модели
    print("\n💾 Сохранение модели...")
    try:
        classifier.save_model("models/decision_tree_snoring_classifier.pkl")
        print("✅ Модель сохранена!")
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


def test_feature_extractor():
    """Тестирует экстрактор признаков."""
    print("\n" + "=" * 60)
    print("🔍 ТЕСТИРОВАНИЕ ЭКСТРАКТОРА ПРИЗНАКОВ")
    print("=" * 60)
    
    # Создаем экстрактор
    extractor = DecisionTreeFeatureExtractor()
    
    # Тест с синтетическими аудио и акселерометром
    print("\n🧪 Тест экстрактора с синтетическими данными...")
    
    # Создаем синтетические данные
    audio_data = np.random.randn(8000)  # 1 секунда аудио
    accel_data = np.random.randn(8000, 3)  # 1 секунда акселерометра
    
    try:
        # Извлекаем 30 признаков
        features = extractor.extract_30_features(audio_data, accel_data)
        print(f"✅ Извлечено {len(features)} признаков")
        print(f"🔢 Размер вектора признаков: {features.shape}")
        
        # Проверяем, что получилось ровно 30 признаков
        if len(features) == 30:
            print("✅ Количество признаков соответствует архитектуре (30)")
        else:
            print(f"❌ Неверное количество признаков: {len(features)}")
            
    except Exception as e:
        print(f"❌ Ошибка при извлечении признаков: {e}")
    
    # Тест с CSV данными
    print("\n📊 Тест экстрактора с CSV данными...")
    
    # Создаем синтетические CSV данные
    csv_data = np.random.randn(1, 25)  # 1 сэмпл, 25 колонок
    
    try:
        features = extractor.extract_features_from_csv(csv_data)
        print(f"✅ Извлечено {len(features)} признаков из CSV")
        print(f"🔢 Размер вектора признаков: {features.shape}")
        
        if len(features) == 30:
            print("✅ Количество признаков соответствует архитектуре (30)")
        else:
            print(f"❌ Неверное количество признаков: {len(features)}")
            
    except Exception as e:
        print(f"❌ Ошибка при извлечении признаков из CSV: {e}")
    
    # Конфигурация экстрактора
    print("\n⚙️ Конфигурация экстрактора:")
    config = create_feature_extractor_config()
    for key, value in config.items():
        print(f"   - {key}: {value}")


if __name__ == "__main__":
    print("🚀 Запуск демонстрации Decision Tree классификатора храпа")
    print("📋 Архитектура: 30 признаков, 3 класса, ограничения сложности")
    
    # Тестируем экстрактор признаков
    test_feature_extractor()
    
    # Тестируем классификатор
    test_decision_tree_classifier()
    
    print("\n🎉 Демонстрация завершена!") 