#!/usr/bin/env python3
"""
Диагностика проблемы с SnoringPredictor
"""

import sys
import os
import traceback

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_imports():
    """Диагностика импортов"""
    print("🔍 Диагностика импортов...")
    
    try:
        print("Попытка импорта SnoringPredictorSimple...")
        from models.snoring_predictor_simple import SnoringPredictorSimple
        print("✅ SnoringPredictorSimple импортирован успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта SnoringPredictorSimple:")
        print(f"   {e}")
        traceback.print_exc()
        return False

def debug_creation():
    """Диагностика создания объекта"""
    print("\n🔍 Диагностика создания объекта...")
    
    try:
        from models.snoring_predictor_simple import SnoringPredictorSimple
        predictor = SnoringPredictorSimple(model_type='random_forest')
        print("✅ Объект SnoringPredictorSimple создан успешно")
        return predictor
    except Exception as e:
        print(f"❌ Ошибка создания объекта:")
        print(f"   {e}")
        traceback.print_exc()
        return None

def debug_training(predictor):
    """Диагностика обучения"""
    print("\n🔍 Диагностика обучения...")
    
    try:
        import numpy as np
        
        # Генерируем тестовые данные
        X_train = np.random.randn(50, 30)
        y_train = np.random.randint(0, 2, 50)
        
        print(f"   Размер X_train: {X_train.shape}")
        print(f"   Размер y_train: {y_train.shape}")
        
        # Обучение
        metrics = predictor.train(X_train, y_train)
        print(f"✅ Обучение завершено успешно!")
        print(f"   Val Accuracy: {metrics['val_accuracy']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обучения:")
        print(f"   {e}")
        traceback.print_exc()
        return False

def debug_prediction(predictor):
    """Диагностика предсказания"""
    print("\n🔍 Диагностика предсказания...")
    
    try:
        import numpy as np
        
        # Тестовые данные
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
        
        print(f"   Размер audio_data: {audio_data.shape}")
        print(f"   Размер accelerometer_data: {accelerometer_data.shape}")
        
        # Предсказание
        result = predictor.predict_snoring_episode(
            audio_data=audio_data,
            accelerometer_data=accelerometer_data,
            temporal_data=temporal_data
        )
        
        print(f"✅ Предсказание завершено успешно!")
        print(f"   Результат: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка предсказания:")
        print(f"   {e}")
        traceback.print_exc()
        return False

def main():
    """Основная функция диагностики"""
    print("🚀 Диагностика SnoringPredictor\n")
    
    # Проверяем импорты
    if not debug_imports():
        print("\n❌ Проблема с импортами")
        return
    
    # Проверяем создание объекта
    predictor = debug_creation()
    if predictor is None:
        print("\n❌ Проблема с созданием объекта")
        return
    
    # Проверяем обучение
    if not debug_training(predictor):
        print("\n❌ Проблема с обучением")
        return
    
    # Проверяем предсказание
    if not debug_prediction(predictor):
        print("\n❌ Проблема с предсказанием")
        return
    
    print("\n🎉 ВСЕ ПРОБЛЕМЫ РЕШЕНЫ!")

if __name__ == "__main__":
    main() 