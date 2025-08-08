#!/usr/bin/env python3
"""
Анализ использования классов Snoring_Start и Snoring_End в моделях
"""

import os
import json
import glob
from typing import Dict, List, Set

def analyze_snoring_classes():
    """Анализирует использование классов храпа в моделях"""
    print("🔍 АНАЛИЗ КЛАССОВ ХРАПА В МОДЕЛЯХ")
    print("=" * 60)
    
    # Находим все модели
    model_files = glob.glob("models/snoring_model_*.json")
    
    if not model_files:
        print("❌ Модели не найдены в папке models/")
        return
    
    print(f"📊 Найдено моделей: {len(model_files)}")
    
    # Анализируем каждую модель
    models_with_start_end = []
    models_without_start_end = []
    all_class_names = set()
    
    for model_file in model_files:
        try:
            with open(model_file, 'r') as f:
                metadata = json.load(f)
            
            model_name = os.path.basename(model_file)
            class_names = metadata.get('class_names', [])
            model_type = metadata.get('model_type', 'unknown')
            
            all_class_names.update(class_names)
            
            has_start = 'Snoring_Start' in class_names
            has_end = 'Snoring_End' in class_names
            
            model_info = {
                'file': model_name,
                'model_type': model_type,
                'classes': class_names,
                'has_start': has_start,
                'has_end': has_end,
                'class_count': len(class_names)
            }
            
            if has_start and has_end:
                models_with_start_end.append(model_info)
            else:
                models_without_start_end.append(model_info)
                
        except Exception as e:
            print(f"❌ Ошибка чтения {model_file}: {e}")
    
    # Выводим результаты
    print(f"\n📈 СТАТИСТИКА:")
    print(f"   Всего моделей: {len(model_files)}")
    print(f"   Моделей с Snoring_Start и Snoring_End: {len(models_with_start_end)}")
    print(f"   Моделей без этих классов: {len(models_without_start_end)}")
    
    print(f"\n🎯 МОДЕЛИ С КЛАССАМИ Snoring_Start И Snoring_End:")
    print("-" * 60)
    
    if models_with_start_end:
        for model in models_with_start_end:
            print(f"📁 {model['file']}")
            print(f"   Тип модели: {model['model_type']}")
            print(f"   Классы: {', '.join(model['classes'])}")
            print(f"   Количество классов: {model['class_count']}")
            print()
    else:
        print("❌ Нет моделей с классами Snoring_Start и Snoring_End")
    
    print(f"\n❌ МОДЕЛИ БЕЗ КЛАССОВ Snoring_Start И Snoring_End:")
    print("-" * 60)
    
    if models_without_start_end:
        for model in models_without_start_end:
            print(f"📁 {model['file']}")
            print(f"   Тип модели: {model['model_type']}")
            print(f"   Классы: {', '.join(model['classes'])}")
            print(f"   Количество классов: {model['class_count']}")
            print()
    else:
        print("✅ Все модели содержат классы Snoring_Start и Snoring_End")
    
    # Анализируем типы моделей
    model_types = {}
    for model in models_with_start_end:
        model_type = model['model_type']
        if model_type not in model_types:
            model_types[model_type] = 0
        model_types[model_type] += 1
    
    print(f"\n🤖 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ МОДЕЛЕЙ:")
    print("-" * 60)
    for model_type, count in model_types.items():
        print(f"   {model_type}: {count} моделей")
    
    # Анализируем все уникальные классы
    print(f"\n📋 ВСЕ УНИКАЛЬНЫЕ КЛАССЫ В МОДЕЛЯХ:")
    print("-" * 60)
    for class_name in sorted(all_class_names):
        count = sum(1 for model in models_with_start_end if class_name in model['classes'])
        print(f"   {class_name}: {count} моделей")
    
    return {
        'models_with_start_end': models_with_start_end,
        'models_without_start_end': models_without_start_end,
        'all_class_names': all_class_names,
        'model_types': model_types
    }

def analyze_code_usage():
    """Анализирует использование классов в коде"""
    print(f"\n💻 АНАЛИЗ ИСПОЛЬЗОВАНИЯ В КОДЕ:")
    print("=" * 60)
    
    # Файлы для анализа
    code_files = [
        'src/train_snoring.py',
        'snoring_demo.py',
        'snoring_demo_enhanced.py',
        'create_snoring_visualizations.py',
        'esp32_code/snoring_detector.c'
    ]
    
    for file_path in code_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                has_start = 'Snoring_Start' in content
                has_end = 'Snoring_End' in content
                
                if has_start or has_end:
                    print(f"📄 {file_path}")
                    print(f"   Snoring_Start: {'✅' if has_start else '❌'}")
                    print(f"   Snoring_End: {'✅' if has_end else '❌'}")
                    print()
                    
            except Exception as e:
                print(f"❌ Ошибка чтения {file_path}: {e}")

def create_detailed_report():
    """Создает детальный отчет"""
    print(f"\n📊 ДЕТАЛЬНЫЙ ОТЧЕТ:")
    print("=" * 60)
    
    # Анализируем модели
    analysis = analyze_snoring_classes()
    
    # Анализируем код
    analyze_code_usage()
    
    # Создаем сводку
    print(f"\n📋 СВОДКА:")
    print("-" * 60)
    
    if analysis['models_with_start_end']:
        print(f"✅ Классы Snoring_Start и Snoring_End используются в {len(analysis['models_with_start_end'])} моделях")
        print(f"   Типы моделей: {', '.join(analysis['model_types'].keys())}")
        
        # Показываем примеры классов
        example_model = analysis['models_with_start_end'][0]
        print(f"   Пример классов: {', '.join(example_model['classes'])}")
    else:
        print("❌ Классы Snoring_Start и Snoring_End не найдены в моделях")
    
    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    print("1. Классы Snoring_Start и Snoring_End используются для:")
    print("   - Детекции начала эпизода храпа")
    print("   - Детекции окончания эпизода храпа")
    print("   - Анализа временных паттернов храпа")
    
    print("\n2. Эти классы важны для:")
    print("   - Определения длительности эпизодов храпа")
    print("   - Анализа частоты эпизодов")
    print("   - Предсказания будущих эпизодов")
    
    print("\n3. Модели с этими классами:")
    print("   - Более детальная классификация")
    print("   - Лучший анализ временных паттернов")
    print("   - Поддержка 5-классной классификации")

def main():
    """Основная функция"""
    print("🚀 Анализ использования классов Snoring_Start и Snoring_End")
    print("=" * 80)
    
    create_detailed_report()
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main() 