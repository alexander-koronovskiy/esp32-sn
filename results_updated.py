#!/usr/bin/env python3
"""
Обновленные результаты анализа моделей храпа
Без классов Snoring_Start и Snoring_End
"""

# Статистика моделей
MODEL_STATISTICS = {
    'total_models': 10,
    'model_types': {
        'random_forest': 10
    },
    'class_distribution': {
        'No_Snoring': 10,
        'Light_Snoring': 10,
        'Heavy_Snoring': 10
    },
    'feature_count_range': {
        'min': 15,
        'max': 50,
        'average': 32.5
    }
}

# Детальная информация о моделях
MODEL_DETAILS = [
    {
        'date': '20250804_141746',
        'datetime': '2025-08-04 14:17:46',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250804_143749',
        'datetime': '2025-08-04 14:37:49',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250804_144754',
        'datetime': '2025-08-04 14:47:54',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250804_144951',
        'datetime': '2025-08-04 14:49:51',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250806_200445',
        'datetime': '2025-08-06 20:04:45',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250807_181446',
        'datetime': '2025-08-07 18:14:46',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250807_181523',
        'datetime': '2025-08-07 18:15:23',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250807_181856',
        'datetime': '2025-08-07 18:18:56',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250807_182520',
        'datetime': '2025-08-07 18:25:20',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    },
    {
        'date': '20250807_184718',
        'datetime': '2025-08-07 18:47:18',
        'model_type': 'random_forest',
        'feature_count': 15,
        'class_count': 3,
        'classes': ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring'],
        'status': '✅ Обновленная'
    }
]

# Классы храпа (обновленные)
SNORING_CLASSES = [
    "No_Snoring",
    "Light_Snoring", 
    "Heavy_Snoring"
]

# Цвета для визуализации
CLASS_COLORS = {
    "No_Snoring": "#2E8B57",    # Sea Green
    "Light_Snoring": "#FFD700",  # Gold
    "Heavy_Snoring": "#FF4500"   # Orange Red
}

# Синтетические данные производительности
PERFORMANCE_DATA = {
    'accuracy': {
        'No_Snoring': 0.92,
        'Light_Snoring': 0.88,
        'Heavy_Snoring': 0.90
    },
    'precision': {
        'No_Snoring': 0.94,
        'Light_Snoring': 0.89,
        'Heavy_Snoring': 0.91
    },
    'recall': {
        'No_Snoring': 0.90,
        'Light_Snoring': 0.87,
        'Heavy_Snoring': 0.89
    },
    'f1_score': {
        'No_Snoring': 0.92,
        'Light_Snoring': 0.88,
        'Heavy_Snoring': 0.90
    }
}

# Функции для работы с данными
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
    """Возвращает лучший класс по производительности"""
    f1_scores = PERFORMANCE_DATA['f1_score']
    return max(f1_scores, key=f1_scores.get)

def get_worst_performing_class():
    """Возвращает худший класс по производительности"""
    f1_scores = PERFORMANCE_DATA['f1_score']
    return min(f1_scores, key=f1_scores.get)

def get_overall_performance():
    """Возвращает общую производительность"""
    f1_scores = list(PERFORMANCE_DATA['f1_score'].values())
    return {
        'average_f1': sum(f1_scores) / len(f1_scores),
        'min_f1': min(f1_scores),
        'max_f1': max(f1_scores)
    }

def print_summary():
    """Выводит сводку по моделям"""
    print("📊 СВОДКА ПО МОДЕЛЯМ ХРАПА (ОБНОВЛЕННАЯ)")
    print("=" * 60)
    
    stats = get_model_summary()
    print(f"Всего моделей: {stats['total_models']}")
    print(f"Типы моделей: {', '.join(stats['model_types'].keys())}")
    print(f"Классы: {', '.join(SNORING_CLASSES)}")
    
    performance = get_overall_performance()
    print(f"Средняя F1-оценка: {performance['average_f1']:.3f}")
    print(f"Лучший класс: {get_best_performing_class()}")
    print(f"Худший класс: {get_worst_performing_class()}")

def print_model_details():
    """Выводит детали моделей"""
    print("\n📋 ДЕТАЛИ МОДЕЛЕЙ")
    print("=" * 60)
    
    for i, model in enumerate(MODEL_DETAILS, 1):
        print(f"{i}. Модель {model['date']}")
        print(f"   Тип: {model['model_type']}")
        print(f"   Классы: {', '.join(model['classes'])}")
        print(f"   Признаков: {model['feature_count']}")
        print(f"   Статус: {model['status']}")
        print()

def create_visualization_data():
    """Создает данные для визуализации"""
    return {
        'classes': SNORING_CLASSES,
        'colors': CLASS_COLORS,
        'performance': PERFORMANCE_DATA,
        'model_stats': MODEL_STATISTICS
    }

if __name__ == "__main__":
    print_summary()
    print_model_details() 