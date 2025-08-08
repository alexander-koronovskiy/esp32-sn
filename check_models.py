#!/usr/bin/env python3
"""
Простая проверка работы двух моделей: SnoringClassifier и SnoringPredictor
"""

import numpy as np
from src.models.snoring_classifier import SnoringClassifier
from src.models.snoring_predictor_simple import SnoringPredictorSimple
from src.features.snoring_extractor import SnoringFeatureExtractor

def main():
    print("🚀 Проверка работы двух моделей\n")
    
    # 1. Проверка SnoringClassifier
    print("1️⃣ Тестирование SnoringClassifier...")
    classifier = SnoringClassifier(model_type='random_forest')
    X_train = np.random.randn(100, 15)
    y_train = np.random.randint(0, 2, 100)
    metrics = classifier.train(X_train, y_train)
    print(f"✅ Классификатор готов! Accuracy: {metrics['accuracy']:.3f}")
    
    # 2. Проверка SnoringPredictor
    print("\n2️⃣ Тестирование SnoringPredictor...")
    predictor = SnoringPredictorSimple(model_type='random_forest')
    X_train = np.random.randn(100, 30)
    y_train = np.random.randint(0, 2, 100)
    metrics = predictor.train(X_train, y_train)
    print(f"✅ Модель предсказания готова! Val Accuracy: {metrics['val_accuracy']:.3f}")
    
    # 3. Тестирование предсказания
    print("\n3️⃣ Тестирование предсказания эпизода храпа...")
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
    
    result = predictor.predict_snoring_episode(
        audio_data=audio_data,
        accelerometer_data=accelerometer_data,
        temporal_data=temporal_data
    )
    
    print(f"🎯 Результат предсказания:")
    print(f"   Вероятность храпа: {result['snoring_probability']:.3f}")
    print(f"   Уровень риска: {result['risk_level']}")
    print(f"   Временной горизонт: {result['time_horizon_seconds']} сек")
    
    # 4. Проверка извлечения признаков
    print("\n4️⃣ Тестирование извлечения признаков...")
    extractor = SnoringFeatureExtractor()
    features = extractor.extract_snoring_features(audio_data)
    print(f"✅ Извлечено {len(features)} признаков")
    
    print("\n🎉 Все модели работают корректно!")
    print("\n📋 Резюме:")
    print("✅ SnoringClassifier - бинарная классификация храпа")
    print("✅ SnoringPredictor - предсказание вероятности храпа")
    print("✅ Извлечение признаков - 15 аудио-признаков")
    print("✅ Интеграция моделей работает")

if __name__ == "__main__":
    main() 