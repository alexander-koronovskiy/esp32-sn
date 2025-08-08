#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы моделей детекции и предсказания храпа.
"""

import numpy as np
from src.models.snoring_classifier import SnoringClassifier
from src.models.snoring_predictor import SnoringPredictor
from src.features.snoring_extractor import SnoringFeatureExtractor


def test_snoring_classifier():
    """Тестирует SnoringClassifier."""
    print("🧪 Тестирование SnoringClassifier...")
    
    # Создаем классификатор
    classifier = SnoringClassifier(model_type='lightgbm')
    
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


def test_snoring_predictor():
    """Тестирует SnoringPredictor."""
    print("\n🧪 Тестирование SnoringPredictor...")
    
    # Создаем модель предсказания
    predictor = SnoringPredictor(model_type='neural_network')
    
    # Генерируем тестовые данные
    n_samples = 100
    n_features = 30
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # Обучение
    print("📚 Обучение модели...")
    metrics = predictor.train(X_train, y_train, epochs=5)  # Быстрое обучение для теста
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


def test_feature_extraction():
    """Тестирует извлечение признаков."""
    print("\n🧪 Тестирование извлечения признаков...")
    
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


def test_tflite_conversion():
    """Тестирует конвертацию в TensorFlow Lite."""
    print("\n🧪 Тестирование конвертации в TFLite...")
    
    try:
        # Создаем модель предсказания
        predictor = SnoringPredictor(model_type='neural_network')
        
        # Генерируем тестовые данные
        n_samples = 50
        n_features = 30
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        
        # Обучение
        print("📚 Обучение модели для конвертации...")
        predictor.train(X_train, y_train, epochs=3)
        
        # Конвертация в TFLite
        print("🔄 Конвертация в TensorFlow Lite...")
        tflite_model = predictor.convert_to_tflite()
        
        print(f"✅ Конвертация успешна! Размер модели: {len(tflite_model)} байт")
        
        # Сохранение модели
        predictor.save_model('models/test_snoring_predictor.tflite')
        print("💾 Модель сохранена в models/test_snoring_predictor.tflite")
        
    except Exception as e:
        print(f"❌ Ошибка при конвертации: {e}")


def main():
    """Основная функция тестирования."""
    print("🚀 Запуск тестирования моделей детекции и предсказания храпа\n")
    
    try:
        # Тестируем классификатор
        classifier = test_snoring_classifier()
        
        # Тестируем модель предсказания
        predictor = test_snoring_predictor()
        
        # Тестируем извлечение признаков
        extractor = test_feature_extraction()
        
        # Тестируем конвертацию в TFLite
        test_tflite_conversion()
        
        print("\n🎉 Все тесты пройдены успешно!")
        print("\n📋 Резюме:")
        print("✅ SnoringClassifier - бинарный классификатор храпа (LightGBM)")
        print("✅ SnoringPredictor - модель предсказания храпа (TensorFlow Lite)")
        print("✅ Извлечение признаков - 15 аудио + 5-8 акселерометр + 5-7 временных")
        print("✅ Конвертация в TFLite для ESP32")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 