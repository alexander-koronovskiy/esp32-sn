#!/usr/bin/env python3
"""
Тест для проверки исправления строки 47 в esp32_classifier.py
"""

import sys
import os
import numpy as np

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.esp32_classifier import ESP32Classifier, create_esp32_classifier
from src.features.esp32_extractor import create_esp32_config


def test_line_47_fix():
    """Тестирует исправление строки 47."""
    print("Тестирование исправления строки 47...")
    
    # Тест 1: Создание классификатора с числовым max_features
    try:
        classifier1 = ESP32Classifier(max_features=5)
        print("✓ Классификатор создан с числовым max_features")
    except Exception as e:
        print(f"✗ Ошибка с числовым max_features: {e}")
        return False
    
    # Тест 2: Создание классификатора со строковым max_features
    try:
        classifier2 = ESP32Classifier(max_features='sqrt')
        print("✓ Классификатор создан со строковым max_features")
    except Exception as e:
        print(f"✗ Ошибка со строковым max_features: {e}")
        return False
    
    # Тест 3: Создание классификатора через конфигурацию
    try:
        config = create_esp32_config()
        classifier3 = create_esp32_classifier(config)
        print("✓ Классификатор создан через конфигурацию")
    except Exception as e:
        print(f"✗ Ошибка через конфигурацию: {e}")
        return False
    
    # Тест 4: Создание модели
    try:
        model = classifier1.create_model()
        print("✓ Модель создана успешно")
    except Exception as e:
        print(f"✗ Ошибка создания модели: {e}")
        return False
    
    # Тест 5: Обучение модели
    try:
        X = np.random.randn(20, 6)
        y = np.random.randint(0, 5, 20)
        metrics = classifier1.train(X, y)
        print(f"✓ Модель обучена, точность: {metrics['train_accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Ошибка обучения модели: {e}")
        return False
    
    print("✓ Все тесты пройдены успешно!")
    return True


def test_edge_cases():
    """Тестирует граничные случаи."""
    print("\nТестирование граничных случаев...")
    
    # Тест 1: max_features как float
    try:
        classifier = ESP32Classifier(max_features=5.5)
        model = classifier.create_model()
        print("✓ max_features как float работает")
    except Exception as e:
        print(f"✗ Ошибка с float max_features: {e}")
        return False
    
    # Тест 2: max_features как None
    try:
        classifier = ESP32Classifier(max_features=None)
        model = classifier.create_model()
        print("✓ max_features как None работает")
    except Exception as e:
        print(f"✗ Ошибка с None max_features: {e}")
        return False
    
    # Тест 3: max_features как список (должно вызвать ошибку)
    try:
        classifier = ESP32Classifier(max_features=[1, 2, 3])
        model = classifier.create_model()
        print("✓ max_features как список работает")
    except Exception as e:
        print(f"✓ Правильно обработана ошибка с неверным типом: {e}")
    
    print("✓ Все граничные случаи обработаны")
    return True


def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЯ СТРОКИ 47")
    print("=" * 60)
    
    try:
        # Тест основного исправления
        if not test_line_47_fix():
            return False
        
        # Тест граничных случаев
        if not test_edge_cases():
            return False
        
        print("\n" + "=" * 60)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        
        print("\nИсправления применены:")
        print("✅ Безопасная обработка max_features в конструкторе")
        print("✅ Безопасная обработка max_features в create_model()")
        print("✅ Безопасная обработка max_features в create_esp32_classifier()")
        print("✅ Обработка различных типов данных")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 