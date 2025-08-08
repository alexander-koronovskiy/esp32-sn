#!/usr/bin/env python3
"""
Быстрый просмотр информации о моделях
"""

import json
import os
import glob
from datetime import datetime
import pandas as pd

def load_model_info():
    """Загружает информацию о всех моделях"""
    print("📊 Загрузка информации о моделях...")
    
    models_info = []
    model_files = glob.glob("models/snoring_model_*_metadata.json")
    
    for file_path in model_files:
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            # Извлекаем дату из имени файла
            filename = os.path.basename(file_path)
            date_str = filename.replace('snoring_model_', '').replace('_metadata.json', '')
            
            # Проверяем наличие файлов модели
            model_file = file_path.replace('_metadata.json', '.pkl')
            scaler_file = file_path.replace('_metadata.json', '_scaler.pkl')
            
            model_exists = os.path.exists(model_file)
            scaler_exists = os.path.exists(scaler_file)
            
            models_info.append({
                'date': date_str,
                'datetime': datetime.strptime(date_str, '%Y%m%d_%H%M%S'),
                'model_type': metadata.get('model_type', 'Unknown'),
                'feature_count': len(metadata['feature_names']),
                'class_count': len(metadata['class_names']),
                'classes': metadata['class_names'],
                'max_depth': metadata.get('max_depth', 'N/A'),
                'max_features': metadata.get('max_features', 'N/A'),
                'model_exists': model_exists,
                'scaler_exists': scaler_exists,
                'status': '✅ Полная' if model_exists and scaler_exists else '⚠️ Частичная' if model_exists else '❌ Отсутствует'
            })
        except Exception as e:
            print(f"❌ Ошибка загрузки {file_path}: {e}")
    
    # Сортируем по дате
    models_info.sort(key=lambda x: x['datetime'])
    return models_info

def print_model_summary(models_info):
    """Выводит сводку по моделям"""
    print("\n" + "="*80)
    print("📋 СВОДКА ПО МОДЕЛЯМ ХРАПА")
    print("="*80)
    
    if not models_info:
        print("❌ Модели не найдены")
        return
    
    # Создаем DataFrame для красивого вывода
    df = pd.DataFrame(models_info)
    df['date_formatted'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Выводим основную информацию
    print(f"\nВсего моделей: {len(models_info)}")
    
    # Статистика по типам моделей
    model_types = df['model_type'].value_counts()
    print(f"\nТипы моделей:")
    for model_type, count in model_types.items():
        print(f"  • {model_type}: {count}")
    
    # Статистика по статусу
    status_counts = df['status'].value_counts()
    print(f"\nСтатус моделей:")
    for status, count in status_counts.items():
        print(f"  • {status}: {count}")
    
    # Средние значения
    avg_features = df['feature_count'].mean()
    avg_classes = df['class_count'].mean()
    print(f"\nСредние значения:")
    print(f"  • Признаков: {avg_features:.1f}")
    print(f"  • Классов: {avg_classes:.1f}")
    
    print("\n" + "="*80)

def print_detailed_model_info(models_info):
    """Выводит детальную информацию по каждой модели"""
    print("\n📊 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО МОДЕЛЯМ")
    print("="*80)
    
    for i, model in enumerate(models_info, 1):
        print(f"\n{i}. Модель от {model['date_formatted']}")
        print(f"   Дата создания: {model['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Тип модели: {model['model_type']}")
        print(f"   Количество признаков: {model['feature_count']}")
        print(f"   Количество классов: {model['class_count']}")
        print(f"   Классы: {', '.join(model['classes'])}")
        print(f"   Максимальная глубина: {model['max_depth']}")
        print(f"   Максимальные признаки: {model['max_features']}")
        print(f"   Статус: {model['status']}")
        
        if model['model_exists']:
            print(f"   ✅ Файл модели: {os.path.basename(model['date'])}.pkl")
        else:
            print(f"   ❌ Файл модели: отсутствует")
            
        if model['scaler_exists']:
            print(f"   ✅ Файл scaler: {os.path.basename(model['date'])}_scaler.pkl")
        else:
            print(f"   ❌ Файл scaler: отсутствует")

def analyze_feature_evolution(models_info):
    """Анализирует эволюцию признаков"""
    print("\n📈 АНАЛИЗ ЭВОЛЮЦИИ ПРИЗНАКОВ")
    print("="*80)
    
    # Создаем DataFrame для анализа
    df = pd.DataFrame(models_info)
    
    if len(df) < 2:
        print("Для анализа эволюции нужно минимум 2 модели")
        return
    
    # Анализируем изменение количества признаков
    feature_counts = df['feature_count'].values
    dates = df['datetime'].values
    
    print(f"Эволюция количества признаков:")
    for i in range(len(feature_counts)):
        date_str = dates[i].strftime('%Y-%m-%d')
        features = feature_counts[i]
        if i > 0:
            change = features - feature_counts[i-1]
            change_str = f" ({change:+d})" if change != 0 else " (без изменений)"
        else:
            change_str = " (базовая)"
        print(f"  • {date_str}: {features} признаков{change_str}")
    
    # Анализируем типы моделей
    print(f"\nЭволюция типов моделей:")
    for i, row in df.iterrows():
        date_str = row['datetime'].strftime('%Y-%m-%d')
        model_type = row['model_type']
        print(f"  • {date_str}: {model_type}")

def main():
    """Основная функция"""
    print("🚀 Быстрый просмотр моделей...")
    
    # Загружаем информацию о моделях
    models_info = load_model_info()
    
    if not models_info:
        print("❌ Модели не найдены")
        return
    
    # Выводим сводку
    print_model_summary(models_info)
    
    # Выводим детальную информацию
    print_detailed_model_info(models_info)
    
    # Анализируем эволюцию
    analyze_feature_evolution(models_info)
    
    print("\n✅ Анализ завершен!")

if __name__ == "__main__":
    main() 