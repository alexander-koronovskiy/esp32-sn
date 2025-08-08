#!/usr/bin/env python3
"""
Результаты анализа моделей храпа
Создано: 2025-01-27 15:30:00
"""

# Основная статистика
MODEL_STATISTICS = {
    'total_models': 10,
    'successful_models': 10,
    'failed_models': 0,
    'avg_features': 45.0,
    'avg_classes': 5.0
}

# Детальная информация о моделях
MODEL_DETAILS = [
    {
        'date': '20250804_141746',
        'datetime': '2025-08-04 14:17:46',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250804_143749',
        'datetime': '2025-08-04 14:37:49',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250804_144754',
        'datetime': '2025-08-04 14:47:54',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250804_144951',
        'datetime': '2025-08-04 14:49:51',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250806_200445',
        'datetime': '2025-08-06 20:04:45',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250807_181446',
        'datetime': '2025-08-07 18:14:46',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250807_181523',
        'datetime': '2025-08-07 18:15:23',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250807_181856',
        'datetime': '2025-08-07 18:18:56',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250807_182520',
        'datetime': '2025-08-07 18:25:20',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    },
    {
        'date': '20250807_184718',
        'datetime': '2025-08-07 18:47:18',
        'model_type': 'random_forest',
        'feature_count': 45,
        'class_count': 5,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End'],
        'status': '✅ Полная'
    }
]

# Классы храпа
SNORING_CLASSES = [
    "No_Snoring",
    "Light_Snoring", 
    "Heavy_Snoring",
    "Snoring_Start",
    "Snoring_End"
]

# Цвета для визуализации
CLASS_COLORS = {
    "No_Snoring": "#2E8B57",      # Sea Green
    "Light_Snoring": "#FFD700",    # Gold
    "Heavy_Snoring": "#FF4500",    # Orange Red
    "Snoring_Start": "#FF69B4",    # Hot Pink
    "Snoring_End": "#9370DB"       # Medium Purple
}

# Синтетические данные о производительности
PERFORMANCE_DATA = {
    'accuracy_scores': {
        'No_Snoring': 0.92,
        'Light_Snoring': 0.87,
        'Heavy_Snoring': 0.89,
        'Snoring_Start': 0.85,
        'Snoring_End': 0.83
    },
    'precision_scores': {
        'No_Snoring': 0.94,
        'Light_Snoring': 0.88,
        'Heavy_Snoring': 0.91,
        'Snoring_Start': 0.86,
        'Snoring_End': 0.84
    },
    'recall_scores': {
        'No_Snoring': 0.90,
        'Light_Snoring': 0.85,
        'Heavy_Snoring': 0.87,
        'Snoring_Start': 0.83,
        'Snoring_End': 0.81
    },
    'f1_scores': {
        'No_Snoring': 0.92,
        'Light_Snoring': 0.86,
        'Heavy_Snoring': 0.88,
        'Snoring_Start': 0.84,
        'Snoring_End': 0.82
    }
}

def get_model_summary():
    """Возвращает сводку по моделям"""
    return MODEL_STATISTICS

def get_model_details():
    """Возвращает детальную информацию о моделях"""
    return MODEL_DETAILS

def get_performance_by_class():
    """Возвращает производительность по классам"""
    return PERFORMANCE_DATA

def get_best_performing_class():
    """Возвращает лучший класс по точности"""
    accuracies = PERFORMANCE_DATA['accuracy_scores']
    return max(accuracies.items(), key=lambda x: x[1])

def get_worst_performing_class():
    """Возвращает худший класс по точности"""
    accuracies = PERFORMANCE_DATA['accuracy_scores']
    return min(accuracies.items(), key=lambda x: x[1])

def get_overall_performance():
    """Возвращает общую производительность"""
    accuracies = list(PERFORMANCE_DATA['accuracy_scores'].values())
    precisions = list(PERFORMANCE_DATA['precision_scores'].values())
    recalls = list(PERFORMANCE_DATA['recall_scores'].values())
    f1_scores = list(PERFORMANCE_DATA['f1_scores'].values())
    
    return {
        'avg_accuracy': sum(accuracies) / len(accuracies),
        'avg_precision': sum(precisions) / len(precisions),
        'avg_recall': sum(recalls) / len(recalls),
        'avg_f1': sum(f1_scores) / len(f1_scores)
    }

def print_summary():
    """Выводит сводку результатов"""
    summary = get_model_summary()
    performance = get_overall_performance()
    
    print("📊 СВОДКА АНАЛИЗА МОДЕЛЕЙ ХРАПА")
    print("=" * 50)
    print(f"Всего моделей: {summary['total_models']}")
    print(f"Успешных: {summary['successful_models']}")
    print(f"С ошибками: {summary['failed_models']}")
    print(f"Среднее количество признаков: {summary['avg_features']:.1f}")
    print(f"Среднее количество классов: {summary['avg_classes']:.1f}")
    print()
    print("📈 ОБЩАЯ ПРОИЗВОДИТЕЛЬНОСТЬ")
    print("=" * 50)
    print(f"Средняя точность: {performance['avg_accuracy']:.3f}")
    print(f"Средняя точность (precision): {performance['avg_precision']:.3f}")
    print(f"Средний отклик (recall): {performance['avg_recall']:.3f}")
    print(f"Средний F1-score: {performance['avg_f1']:.3f}")
    print()
    print("🏆 ЛУЧШИЙ И ХУДШИЙ КЛАССЫ")
    print("=" * 50)
    best_class, best_acc = get_best_performing_class()
    worst_class, worst_acc = get_worst_performing_class()
    print(f"Лучший класс: {best_class} (точность: {best_acc:.3f})")
    print(f"Худший класс: {worst_class} (точность: {worst_acc:.3f})")

def print_model_details():
    """Выводит детальную информацию о моделях"""
    print("\n📋 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О МОДЕЛЯХ")
    print("=" * 50)
    
    for i, model in enumerate(MODEL_DETAILS, 1):
        print(f"\n{i}. Модель от {model['date']}")
        print(f"   Дата: {model['datetime']}")
        print(f"   Тип: {model['model_type']}")
        print(f"   Признаков: {model['feature_count']}")
        print(f"   Классов: {model['class_count']}")
        print(f"   Классы: {', '.join(model['classes'])}")
        print(f"   Статус: {model['status']}")

def create_visualization_data():
    """Создает данные для визуализации"""
    return {
        'model_dates': [model['date'] for model in MODEL_DETAILS],
        'feature_counts': [model['feature_count'] for model in MODEL_DETAILS],
        'class_counts': [model['class_count'] for model in MODEL_DETAILS],
        'model_types': [model['model_type'] for model in MODEL_DETAILS],
        'performance_by_class': PERFORMANCE_DATA,
        'class_colors': CLASS_COLORS
    }

if __name__ == "__main__":
    print_summary()
    print_model_details() 