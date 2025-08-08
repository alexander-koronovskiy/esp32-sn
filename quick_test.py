#!/usr/bin/env python3
"""
Быстрый тест моделей без сложных зависимостей
"""

import numpy as np
import sys
import os

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Тестирует импорты моделей"""
    print("🔍 Тестирование импортов...")
    
    try:
        from models.snoring_classifier import SnoringClassifier
        print("✅ SnoringClassifier импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта SnoringClassifier: {e}")
        return False
    
    try:
        from models.snoring_predictor_simple import SnoringPredictorSimple
        print("✅ SnoringPredictorSimple импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта SnoringPredictorSimple: {e}")
        return False
    
    try:
        from features.snoring_extractor import SnoringFeatureExtractor
        print("✅ SnoringFeatureExtractor импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта SnoringFeatureExtractor: {e}")
        return False
    
    return True

def test_snoring_classifier():
    """Тестирует SnoringClassifier"""
    print("\n🧪 Тестирование SnoringClassifier...")
    
    try:
        from models.snoring_classifier import SnoringClassifier
        
        # Создаем классификатор с Random Forest (без LightGBM)
        classifier = SnoringClassifier(model_type='random_forest')
        
        # Генерируем тестовые данные
        X_train = np.random.randn(50, 15)
        y_train = np.random.randint(0, 2, 50)
        
        # Обучение
        metrics = classifier.train(X_train, y_train)
        print(f"✅ Классификатор обучен! Accuracy: {metrics['accuracy']:.3f}")
        
        # Тестирование
        X_test = np.random.randn(5, 15)
        predictions = classifier.predict(X_test)
        print(f"📊 Предсказания: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в SnoringClassifier: {e}")
        return False

def test_snoring_predictor():
    """Тестирует SnoringPredictorSimple"""
    print("\n🧪 Тестирование SnoringPredictorSimple...")
    
    try:
        from models.snoring_predictor_simple import SnoringPredictorSimple
        
        # Создаем модель предсказания
        predictor = SnoringPredictorSimple(model_type='random_forest')
        
        # Генерируем тестовые данные
        X_train = np.random.randn(50, 30)
        y_train = np.random.randint(0, 2, 50)
        
        # Обучение
        metrics = predictor.train(X_train, y_train)
        print(f"✅ Модель предсказания обучена! Val Accuracy: {metrics['val_accuracy']:.3f}")
        
        # Тестирование предсказания
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
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в SnoringPredictorSimple: {e}")
        return False

def test_feature_extraction():
    """Тестирует извлечение признаков"""
    print("\n🧪 Тестирование извлечения признаков...")
    
    try:
        from features.snoring_extractor import SnoringFeatureExtractor
        
        # Создаем экстрактор
        extractor = SnoringFeatureExtractor()
        
        # Генерируем тестовые аудио данные
        audio_data = np.random.randn(8000)
        
        # Извлечение признаков
        features = extractor.extract_snoring_features(audio_data)
        
        print(f"✅ Извлечено {len(features)} признаков")
        print(f"📊 Примеры признаков:")
        for i, (name, value) in enumerate(list(features.items())[:3]):
            print(f"   {name}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в извлечении признаков: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Быстрый тест моделей детекции и предсказания храпа\n")
    
    # Проверяем импорты
    if not test_imports():
        print("\n❌ Проблемы с импортами. Проверьте зависимости.")
        return
    
    # Тестируем компоненты
    classifier_ok = test_snoring_classifier()
    predictor_ok = test_snoring_predictor()
    features_ok = test_feature_extraction()
    
    print("\n" + "="*50)
    print("📋 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("="*50)
    
    if classifier_ok:
        print("✅ SnoringClassifier - работает")
    else:
        print("❌ SnoringClassifier - ошибки")
    
    if predictor_ok:
        print("✅ SnoringPredictor - работает")
    else:
        print("❌ SnoringPredictor - ошибки")
    
    if features_ok:
        print("✅ Извлечение признаков - работает")
    else:
        print("❌ Извлечение признаков - ошибки")
    
    if classifier_ok and predictor_ok and features_ok:
        print("\n🎉 ВСЕ МОДЕЛИ РАБОТАЮТ КОРРЕКТНО!")
        print("\n💡 Для полной функциональности установите:")
        print("   pip install tensorflow lightgbm")
    else:
        print("\n⚠️  Есть проблемы с некоторыми компонентами")
        print("Проверьте установку зависимостей")

if __name__ == "__main__":
    main() 