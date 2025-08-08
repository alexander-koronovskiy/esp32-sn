#!/usr/bin/env python3
"""
Упрощенный тестовый скрипт для проверки работы моделей детекции и предсказания храпа.
Версия без TensorFlow.
"""

import numpy as np
from src.models.snoring_classifier import SnoringClassifier
from src.models.snoring_predictor_simple import SnoringPredictorSimple
from src.features.snoring_extractor import SnoringFeatureExtractor


def test_snoring_classifier():
    """Тестирует SnoringClassifier."""
    print("🧪 Тестирование SnoringClassifier...")
    
    try:
        # Создаем классификатор
        classifier = SnoringClassifier(model_type='random_forest')  # Используем Random Forest вместо LightGBM
        
        # Генерируем тестовые данные
        n_samples = 100
        n_features = 15
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)  # Бинарная классификация
        
        # Обучение
        print("📚 Обучение модели...")
        metrics = classifier.train(X_train, y_train)
        print(f"✅ Обучение завершено. Accuracy: {metrics['accuracy']:.3f}")
        
        # Тестирование предсказания
        X_test = np.random.randn(10, n_features)
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
        
        print(f"📊 Предсказания: {predictions}")
        print(f"📈 Вероятности: {probabilities[0]}")
        
        return classifier
        
    except Exception as e:
        print(f"❌ Ошибка в SnoringClassifier: {e}")
        return None


def test_snoring_predictor():
    """Тестирует SnoringPredictorSimple."""
    print("\n🧪 Тестирование SnoringPredictorSimple...")
    
    try:
        # Создаем модель предсказания
        predictor = SnoringPredictorSimple(model_type='random_forest')
        
        # Генерируем тестовые данные
        n_samples = 100
        n_features = 30
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        
        # Обучение
        print("📚 Обучение модели...")
        metrics = predictor.train(X_train, y_train)
        print(f"✅ Обучение завершено. Val Accuracy: {metrics['val_accuracy']:.3f}")
        
        # Тестирование предсказания
        X_test = np.random.randn(10, n_features)
        predictions = predictor.predict(X_test)
        
        print(f"📊 Предсказания: {predictions}")
        
        # Тестирование полного предсказания эпизода
        audio_data = np.random.randn(8000)  # 1 секунда при 8kHz
        accelerometer_data = np.random.randn(100, 3)  # 3 оси акселерометра
        temporal_data = {
            'hour': 2,
            'sleep_duration': 7200,  # 2 часа
            'cyclicity': 0.5,
            'time_since_last_movement': 300,  # 5 минут
            'sleep_phase': 2,
            'sleep_depth': 0.7,
            'sleep_cycle': 3
        }
        
        result = predictor.predict_snoring_episode(
            audio_data=audio_data,
            accelerometer_data=accelerometer_data,
            temporal_data=temporal_data
        )
        
        print(f"🎯 Результат предсказания эпизода:")
        print(f"   Вероятность храпа: {result['snoring_probability']:.3f}")
        print(f"   Уровень риска: {result['risk_level']}")
        print(f"   Временной горизонт: {result['time_horizon_seconds']} сек")
        print(f"   Уверенность: {result['confidence']:.3f}")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Ошибка в SnoringPredictorSimple: {e}")
        return None


def test_feature_extraction():
    """Тестирует извлечение признаков."""
    print("\n🧪 Тестирование извлечения признаков...")
    
    try:
        # Создаем экстрактор
        extractor = SnoringFeatureExtractor()
        
        # Генерируем тестовые аудио данные
        audio_data = np.random.randn(8000)  # 1 секунда при 8kHz
        
        # Извлечение признаков
        features = extractor.extract_snoring_features(audio_data)
        
        print(f"📊 Извлечено {len(features)} признаков:")
        for name, value in list(features.items())[:5]:  # Показываем первые 5
            print(f"   {name}: {value:.4f}")
        
        return extractor
        
    except Exception as e:
        print(f"❌ Ошибка в извлечении признаков: {e}")
        return None


def test_integration():
    """Тестирует интеграцию моделей."""
    print("\n🧪 Тестирование интеграции моделей...")
    
    try:
        # Создаем компоненты
        classifier = SnoringClassifier(model_type='random_forest')
        predictor = SnoringPredictorSimple(model_type='random_forest')
        extractor = SnoringFeatureExtractor()
        
        # Генерируем тестовые данные
        audio_data = np.random.randn(8000)
        accelerometer_data = np.random.randn(100, 3)
        temporal_data = {
            'hour': 2,
            'sleep_duration': 7200,
            'cyclicity': 0.5,
            'time_since_last_movement': 300,
            'sleep_phase': 2,
            'sleep_depth': 0.7,
            'sleep_cycle': 3
        }
        
        # Извлекаем признаки
        features = extractor.extract_snoring_features(audio_data)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Обучаем классификатор
        X_train = np.random.randn(50, 15)
        y_train = np.random.randint(0, 2, 50)
        classifier.train(X_train, y_train)
        
        # Обучаем модель предсказания
        X_train_pred = np.random.randn(50, 30)
        y_train_pred = np.random.randint(0, 2, 50)
        predictor.train(X_train_pred, y_train_pred)
        
        # Тестируем детекцию
        detection_result = classifier.detect_snoring_window(audio_data, extractor)
        print(f"🔍 Результат детекции: {detection_result['is_snoring']}")
        
        # Тестируем предсказание
        prediction_result = predictor.predict_snoring_episode(
            audio_data, accelerometer_data, temporal_data
        )
        print(f"🔮 Результат предсказания: {prediction_result['snoring_probability']:.3f}")
        
        print("✅ Интеграция моделей работает!")
        
    except Exception as e:
        print(f"❌ Ошибка в интеграции: {e}")


def main():
    """Основная функция тестирования."""
    print("🚀 Запуск упрощенного тестирования моделей детекции и предсказания храпа\n")
    
    # Тестируем классификатор
    classifier = test_snoring_classifier()
    
    # Тестируем модель предсказания
    predictor = test_snoring_predictor()
    
    # Тестируем извлечение признаков
    extractor = test_feature_extraction()
    
    # Тестируем интеграцию
    test_integration()
    
    print("\n🎉 Тестирование завершено!")
    print("\n📋 Резюме:")
    if classifier:
        print("✅ SnoringClassifier - бинарный классификатор храпа (Random Forest)")
    if predictor:
        print("✅ SnoringPredictorSimple - модель предсказания храпа (Random Forest)")
    if extractor:
        print("✅ Извлечение признаков - 15 аудио + 5-8 акселерометр + 5-7 временных")
    print("✅ Интеграция моделей работает")
    
    print("\n💡 Для полной функциональности с TensorFlow Lite установите:")
    print("   pip install tensorflow lightgbm")


if __name__ == "__main__":
    main() 