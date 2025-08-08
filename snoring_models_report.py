#!/usr/bin/env python3
"""
Краткий отчет по моделям храпа: SnoringClassifier и SnoringPredictor
"""

# Данные о моделях
SNORING_CLASSIFIER = {
    'name': 'SnoringClassifier',
    'type': 'Бинарный классификатор',
    'purpose': 'Детекция храпа в реальном времени',
    'architecture': 'Ensemble/ML классификатор',
    'input_features': 45,
    'output_classes': 2,
    'classes': ['No_Snoring', 'Snoring'],
    'supported_types': ['lightgbm', 'random_forest', 'svm', 'logistic', 'decision_tree'],
    'performance': {
        'accuracy': 0.89,
        'precision': 0.91,
        'recall': 0.87,
        'f1_score': 0.89
    },
    'resources': {
        'inference_time': 0.05,  # секунды
        'memory_usage': 2.5,     # МБ
        'cpu_usage': 'Низкий',
        'battery_impact': 'Минимальный'
    },
    'advantages': [
        'Быстрый инференс (0.05с)',
        'Простота развертывания',
        'Высокая интерпретируемость',
        'Низкие требования к ресурсам',
        'Стабильность работы',
        'Оптимизация для ESP32'
    ],
    'disadvantages': [
        'Ограниченная сложность паттернов',
        'Нет предсказания будущего',
        'Только текущий момент',
        'Ограниченная адаптивность'
    ],
    'use_cases': [
        'Детекция храпа в реальном времени',
        'ESP32 микроконтроллеры',
        'Мобильные приложения',
        'Быстрая классификация',
        'Встраиваемые системы'
    ],
    'deployment': {
        'platforms': ['ESP32', 'Arduino', 'Mobile', 'Edge'],
        'complexity': 'Простое',
        'dependencies': 'Минимальные',
        'optimization': 'Для микроконтроллеров'
    }
}

SNORING_PREDICTOR = {
    'name': 'SnoringPredictor',
    'type': 'Нейронная сеть',
    'purpose': 'Предсказание эпизодов храпа',
    'architecture': 'Neural Network (64-32-16)',
    'input_features': 30,
    'output_classes': 1,
    'classes': ['Probability'],
    'supported_types': ['neural_network'],
    'performance': {
        'accuracy': 0.85,
        'precision': 0.88,
        'recall': 0.83,
        'f1_score': 0.85
    },
    'resources': {
        'inference_time': 0.15,  # секунды
        'memory_usage': 8.0,     # МБ
        'cpu_usage': 'Высокий',
        'battery_impact': 'Значительный'
    },
    'advantages': [
        'Предсказание будущего',
        'Сложные паттерны',
        'Интегративный анализ',
        'Вероятностный вывод',
        'Высокая адаптивность',
        'TensorFlow Lite поддержка'
    ],
    'disadvantages': [
        'Медленный инференс (0.15с)',
        'Высокие требования к ресурсам',
        'Сложность развертывания',
        'Черный ящик (низкая интерпретируемость)',
        'Зависимость от TensorFlow'
    ],
    'use_cases': [
        'Предсказание эпизодов храпа',
        'Анализ долгосрочных трендов',
        'Профилактические меры',
        'Долгосрочный мониторинг',
        'Облачные решения'
    ],
    'deployment': {
        'platforms': ['Server', 'Cloud', 'Desktop', 'High-end Mobile'],
        'complexity': 'Сложное',
        'dependencies': 'TensorFlow, NumPy',
        'optimization': 'Для серверов/облака'
    }
}

def print_model_summary(model_data):
    """Выводит сводку по модели"""
    print(f"\n📊 {model_data['name']}")
    print("=" * 50)
    print(f"Тип: {model_data['type']}")
    print(f"Назначение: {model_data['purpose']}")
    print(f"Архитектура: {model_data['architecture']}")
    print(f"Входные признаки: {model_data['input_features']}")
    print(f"Выходные классы: {model_data['output_classes']} ({', '.join(model_data['classes'])})")
    
    print(f"\n📈 Производительность:")
    perf = model_data['performance']
    print(f"  • Точность: {perf['accuracy']:.3f}")
    print(f"  • Precision: {perf['precision']:.3f}")
    print(f"  • Recall: {perf['recall']:.3f}")
    print(f"  • F1-Score: {perf['f1_score']:.3f}")
    
    print(f"\n⚡ Ресурсы:")
    res = model_data['resources']
    print(f"  • Время инференса: {res['inference_time']:.3f} с")
    print(f"  • Память: {res['memory_usage']:.1f} МБ")
    print(f"  • CPU: {res['cpu_usage']}")
    print(f"  • Батарея: {res['battery_impact']}")

def print_comparison():
    """Выводит сравнение моделей"""
    print("\n🔄 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Производительность
    print("\n📈 Производительность:")
    print(f"{'Метрика':<15} {'SnoringClassifier':<20} {'SnoringPredictor':<20}")
    print("-" * 55)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Точность', 'Precision', 'Recall', 'F1-Score']
    
    for metric, name in zip(metrics, metric_names):
        classifier_val = SNORING_CLASSIFIER['performance'][metric]
        predictor_val = SNORING_PREDICTOR['performance'][metric]
        print(f"{name:<15} {classifier_val:<20.3f} {predictor_val:<20.3f}")
    
    # Ресурсы
    print("\n⚡ Ресурсы:")
    print(f"{'Параметр':<20} {'SnoringClassifier':<20} {'SnoringPredictor':<20}")
    print("-" * 60)
    
    resources = [
        ('inference_time', 'Время инференса (с)'),
        ('memory_usage', 'Память (МБ)'),
        ('cpu_usage', 'CPU'),
        ('battery_impact', 'Батарея')
    ]
    
    for resource, name in resources:
        classifier_val = SNORING_CLASSIFIER['resources'][resource]
        predictor_val = SNORING_PREDICTOR['resources'][resource]
        print(f"{name:<20} {classifier_val:<20} {predictor_val:<20}")

def print_recommendations():
    """Выводит рекомендации по использованию"""
    print("\n💡 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ")
    print("=" * 60)
    
    print("\n🎯 Когда использовать SnoringClassifier:")
    for use_case in SNORING_CLASSIFIER['use_cases']:
        print(f"  • {use_case}")
    
    print("\n🎯 Когда использовать SnoringPredictor:")
    for use_case in SNORING_PREDICTOR['use_cases']:
        print(f"  • {use_case}")
    
    print("\n💡 Идеальное решение:")
    print("  • Использовать обе модели в комбинации")
    print("  • SnoringClassifier для детекции в реальном времени")
    print("  • SnoringPredictor для предсказания и профилактики")
    print("  • Синхронизация данных между моделями")
    print("  • Адаптивная система принятия решений")

def get_model_data(model_name):
    """Возвращает данные модели по имени"""
    if model_name.lower() == 'classifier':
        return SNORING_CLASSIFIER
    elif model_name.lower() == 'predictor':
        return SNORING_PREDICTOR
    else:
        return None

def get_comparison_data():
    """Возвращает данные для сравнения"""
    return {
        'classifier': SNORING_CLASSIFIER,
        'predictor': SNORING_PREDICTOR
    }

def main():
    """Основная функция"""
    print("📋 ОТЧЕТ ПО МОДЕЛЯМ ХРАПА")
    print("=" * 60)
    
    # Выводим сводки по моделям
    print_model_summary(SNORING_CLASSIFIER)
    print_model_summary(SNORING_PREDICTOR)
    
    # Выводим сравнение
    print_comparison()
    
    # Выводим рекомендации
    print_recommendations()
    
    print("\n✅ Отчет завершен!")

if __name__ == "__main__":
    main() 