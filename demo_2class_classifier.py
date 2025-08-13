#!/usr/bin/env python3
"""
Демонстрация 2-классового SnoringClassifier
Адаптирован для работы с реальными данными
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append('src')

def create_synthetic_real_data(n_samples: int = 1000) -> tuple:
    """
    Создание синтетических данных, имитирующих реальные CSV данные
    
    Args:
        n_samples: Количество сэмплов
        
    Returns:
        X: Матрица признаков
        y: Вектор меток
    """
    print(f"🎭 Создание синтетических данных ({n_samples} сэмплов)")
    
    # Параметры для генерации
    np.random.seed(42)
    
    # Генерация признаков (25 колонок как в реальных данных)
    X = np.zeros((n_samples, 25))
    
    # Акселерометр (x, y, z)
    X[:, 1] = np.random.normal(0, 2000, n_samples)  # x
    X[:, 2] = np.random.normal(0, 2000, n_samples)  # y  
    X[:, 3] = np.random.normal(0, 2000, n_samples)  # z
    
    # Пульс и физиологические данные
    X[:, 4] = np.random.normal(20, 10, n_samples)   # bpm
    X[:, 5] = np.random.normal(20, 10, n_samples)   # bpm_by_nn
    X[:, 6] = np.random.normal(20, 10, n_samples)   # bpm_by_acf
    
    # Дополнительные признаки
    X[:, 7:16] = np.random.normal(0, 100, (n_samples, 9))  # mf, af, arf, sf, r_th, status_1,2,3, ml
    
    # Нейронные сети
    X[:, 16] = np.random.normal(95, 5, n_samples)   # breath_nn
    X[:, 17] = np.random.normal(142, 10, n_samples) # snore_nn
    X[:, 18] = np.random.normal(0, 50, n_samples)   # signal_nn
    X[:, 19] = np.random.normal(0, 50, n_samples)   # splash_nn
    
    # EOG
    X[:, 20] = np.random.normal(0, 10, n_samples)   # eog
    
    # Полосовые фильтры
    X[:, 21] = np.random.normal(5, 3, n_samples)    # b100
    X[:, 22] = np.random.normal(3, 2, n_samples)    # b400
    X[:, 23] = np.random.normal(10, 5, n_samples)   # b1000
    
    # Огибающая
    X[:, 24] = np.random.normal(50, 20, n_samples)  # env
    
    # Создание меток: 70% No Snoring (W), 30% Snoring
    y = np.random.choice(['W', ''], size=n_samples, p=[0.7, 0.3])
    
    # Добавление паттернов храпа
    snoring_indices = np.where(y == '')[0]
    
    for idx in snoring_indices:
        # Увеличиваем признаки храпа
        X[idx, 17] += np.random.normal(50, 20)  # snore_nn
        X[idx, 23] += np.random.normal(100, 50) # b1000
        X[idx, 24] += np.random.normal(100, 50) # env
        
        # Уменьшаем признаки дыхания
        X[idx, 16] -= np.random.normal(20, 10)  # breath_nn
        X[idx, 21] -= np.random.normal(2, 1)    # b100
    
    print(f"✅ Создано {n_samples} сэмплов")
    print(f"📊 Распределение классов: {np.bincount([1 if label == 'W' else 0 for label in y])}")
    
    return X, y

def test_2class_classifier():
    """Тестирование 2-классового классификатора"""
    try:
        from src.models.snoring_classifier_2class import SnoringClassifier2Class
        
        print("🧪 Тестирование SnoringClassifier2Class")
        
        # Создание синтетических данных
        X, y = create_synthetic_real_data(1000)
        
        # Создание классификатора
        classifier = SnoringClassifier2Class(model_type='random_forest')
        
        # Обучение модели
        print("\n📚 Обучение модели...")
        training_result = classifier.train(X, y, data_type='csv')
        
        print(f"✅ Модель обучена!")
        print(f"📊 Точность: {training_result['accuracy']:.3f}")
        print(f"🔢 Количество признаков: {training_result['feature_count']}")
        print(f"📈 Распределение классов: {training_result['class_distribution']}")
        
        # Оценка модели
        print("\n📊 Оценка модели...")
        evaluation_result = classifier.evaluate(X, y, data_type='csv')
        
        print(f"📋 Результаты оценки:")
        print(f"   Точность: {evaluation_result['accuracy']:.3f}")
        print(f"   Матрица ошибок: {evaluation_result['confusion_matrix']}")
        
        # Тестирование предсказания
        print("\n🔮 Тестирование предсказания...")
        
        # Тест 1: No Snoring (W)
        test_no_snoring = X[y == 'W'][:1]
        prediction_no = classifier.predict(test_no_snoring, data_type='csv')
        
        print(f"📋 Тест No Snoring:")
        print(f"   Класс: {prediction_no['class']}")
        print(f"   Вероятность храпа: {prediction_no['p_snore']:.3f}")
        print(f"   Есть храп: {prediction_no['is_snoring']}")
        
        # Тест 2: Snoring
        test_snoring = X[y == ''][:1]
        prediction_snoring = classifier.predict(test_snoring, data_type='csv')
        
        print(f"📋 Тест Snoring:")
        print(f"   Класс: {prediction_snoring['class']}")
        print(f"   Вероятность храпа: {prediction_snoring['p_snore']:.3f}")
        print(f"   Есть храп: {prediction_snoring['is_snoring']}")
        
        # Тестирование на новых данных
        print("\n🆕 Тестирование на новых данных...")
        new_data = create_synthetic_real_data(100)[0]  # Только признаки
        
        predictions = []
        for i in range(min(5, len(new_data))):  # Тестируем первые 5 сэмплов
            pred = classifier.predict(new_data[i:i+1], data_type='csv')
            predictions.append(pred)
            print(f"   Сэмпл {i+1}: {pred['class']} (p_snore={pred['p_snore']:.3f})")
        
        # Сохранение модели
        output_dir = "models"
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "snoring_classifier_2class_demo.pkl")
        classifier.save_model(model_path)
        print(f"\n💾 Модель сохранена в {model_path}")
        
        # Информация о модели
        model_info = classifier.get_model_info()
        print(f"\n📋 Информация о модели:")
        print(f"   Тип: {model_info['model_type']}")
        print(f"   Классы: {model_info['class_names']}")
        print(f"   Количество классов: {model_info['class_count']}")
        print(f"   Описание: {model_info['description']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("📝 Убедитесь, что файл src/models/snoring_classifier_2class.py создан")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_with_real_data():
    """Тестирование с реальными данными (если доступны)"""
    print("\n🔍 Проверка реальных данных...")
    
    data_dir = "snoring_data"
    if not os.path.exists(data_dir):
        print(f"❌ Папка {data_dir} не найдена")
        return False
    
    try:
        from load_real_data import load_snoring_dataset, prepare_training_data
        
        # Загрузка реальных данных
        all_data, all_labels, file_names = load_snoring_dataset(data_dir)
        
        if not all_data:
            print("❌ Не удалось загрузить реальные данные")
            return False
        
        # Подготовка данных
        X, y = prepare_training_data(all_data, all_labels)
        
        if len(X) == 0:
            print("❌ Недостаточно реальных данных")
            return False
        
        print(f"✅ Загружено {len(X)} реальных сэмплов")
        
        # Тестирование классификатора
        from src.models.snoring_classifier_2class import SnoringClassifier2Class
        
        classifier = SnoringClassifier2Class(model_type='random_forest')
        
        # Обучение на реальных данных
        print("📚 Обучение на реальных данных...")
        training_result = classifier.train(X, y, data_type='csv')
        
        print(f"✅ Модель обучена на реальных данных!")
        print(f"📊 Точность: {training_result['accuracy']:.3f}")
        print(f"🔢 Количество признаков: {training_result['feature_count']}")
        
        # Тестирование предсказания на реальных данных
        print("\n🔮 Тестирование предсказания на реальных данных...")
        test_sample = X[0:1]  # Первый сэмпл
        prediction = classifier.predict(test_sample, data_type='csv')
        
        print(f"📋 Результат предсказания:")
        print(f"   Класс: {prediction['class']}")
        print(f"   ID класса: {prediction['class_id']}")
        print(f"   Вероятности: {prediction['probabilities']}")
        print(f"   Вероятность храпа: {prediction['p_snore']:.3f}")
        print(f"   Есть храп: {prediction['is_snoring']}")
        print(f"   Уверенность: {prediction['confidence']:.3f}")
        
        # Сохранение модели
        output_dir = "models"
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "snoring_classifier_2class_real.pkl")
        classifier.save_model(model_path)
        print(f"💾 Модель на реальных данных сохранена в {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при работе с реальными данными: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 Демонстрация 2-классового SnoringClassifier")
    print("=" * 60)
    
    # Тест 1: Синтетические данные
    print("\n🎭 ТЕСТ 1: Синтетические данные")
    print("-" * 40)
    
    success1 = test_2class_classifier()
    
    # Тест 2: Реальные данные
    print("\n🔍 ТЕСТ 2: Реальные данные")
    print("-" * 40)
    
    success2 = test_with_real_data()
    
    # Итоги
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    if success1:
        print("✅ Тест с синтетическими данными: УСПЕШНО")
    else:
        print("❌ Тест с синтетическими данными: НЕУДАЧНО")
    
    if success2:
        print("✅ Тест с реальными данными: УСПЕШНО")
    else:
        print("❌ Тест с реальными данными: НЕУДАЧНО")
    
    if success1 or success2:
        print("\n🎉 Адаптация SnoringClassifier для 2 классов завершена!")
        print("📝 Модель готова к использованию с реальными данными")
    else:
        print("\n⚠️  Требуется дополнительная настройка")

if __name__ == "__main__":
    main() 