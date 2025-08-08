#!/usr/bin/env python3
"""
Максимально простой тест моделей
"""

import numpy as np
import sys
import os

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_classifier():
    """Тестирует только классификатор"""
    print("🧪 Тестирование SnoringClassifier...")
    
    try:
        from models.snoring_classifier import SnoringClassifier
        
        # Создаем классификатор
        classifier = SnoringClassifier(model_type='random_forest')
        
        # Простые тестовые данные
        X_train = np.random.randn(20, 15)
        y_train = np.random.randint(0, 2, 20)
        
        # Обучение
        metrics = classifier.train(X_train, y_train)
        print(f"✅ Классификатор работает! Accuracy: {metrics['accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в классификаторе: {e}")
        return False

def test_predictor():
    """Тестирует только модель предсказания"""
    print("\n🧪 Тестирование SnoringPredictorSimple...")
    
    try:
        from models.snoring_predictor_simple import SnoringPredictorSimple
        
        # Создаем модель
        predictor = SnoringPredictorSimple(model_type='random_forest')
        
        # Простые тестовые данные
        X_train = np.random.randn(20, 30)
        y_train = np.random.randint(0, 2, 20)
        
        # Обучение
        metrics = predictor.train(X_train, y_train)
        print(f"✅ Модель предсказания работает! Val Accuracy: {metrics['val_accuracy']:.3f}")
        
        # Простое предсказание
        X_test = np.random.randn(1, 30)
        prob = predictor.predict(X_test)[0]
        print(f"📊 Тестовое предсказание: {prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в модели предсказания: {e}")
        return False

def test_features():
    """Тестирует извлечение признаков"""
    print("\n🧪 Тестирование извлечения признаков...")
    
    try:
        from features.snoring_extractor import SnoringFeatureExtractor
        
        # Создаем экстрактор
        extractor = SnoringFeatureExtractor()
        
        # Простые аудио данные
        audio_data = np.random.randn(8000)
        
        # Извлечение признаков
        features = extractor.extract_snoring_features(audio_data)
        
        print(f"✅ Извлечение признаков работает! Получено {len(features)} признаков")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в извлечении признаков: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 Простой тест моделей\n")
    
    # Тестируем компоненты по отдельности
    classifier_ok = test_classifier()
    predictor_ok = test_predictor()
    features_ok = test_features()
    
    print("\n" + "="*40)
    print("📋 РЕЗУЛЬТАТЫ:")
    print("="*40)
    
    if classifier_ok:
        print("✅ SnoringClassifier - РАБОТАЕТ")
    else:
        print("❌ SnoringClassifier - ОШИБКА")
    
    if predictor_ok:
        print("✅ SnoringPredictor - РАБОТАЕТ")
    else:
        print("❌ SnoringPredictor - ОШИБКА")
    
    if features_ok:
        print("✅ Извлечение признаков - РАБОТАЕТ")
    else:
        print("❌ Извлечение признаков - ОШИБКА")
    
    if classifier_ok and predictor_ok and features_ok:
        print("\n🎉 ВСЕ МОДЕЛИ РАБОТАЮТ!")
    else:
        print("\n⚠️  Есть проблемы с некоторыми компонентами")

if __name__ == "__main__":
    main() 