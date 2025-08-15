#!/usr/bin/env python3
"""
Скрипт для отладки проблемы с метками
Почему все метки = 0 (только "не храп")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.utils.data_loader import SnoringDataLoader
from src.features.feature_extractor import SnoringFeatureExtractor


def debug_labels_for_folder(folder_path):
    """Отлаживает метки для одной папки"""
    print(f"\n🔍 Отладка меток для папки: {folder_path.name}")
    
    # Ищем файл аннотаций
    ann_files = list(folder_path.glob("*_ann.txt"))
    if not ann_files:
        print("⚠️ Файл аннотаций не найден")
        return
    
    ann_file = ann_files[0]
    print(f"📄 Файл аннотаций: {ann_file.name}")
    
    # Ищем CSV файлы
    csv_files = list(folder_path.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith('settings')]
    csv_files.sort()
    
    print(f"📊 Найдено CSV файлов: {len(csv_files)}")
    
    # Загружаем данные
    data_loader = SnoringDataLoader(str(folder_path))
    data_loader.parse_annotations(ann_file)
    
    print(f"📊 Загружено аннотаций: {len(data_loader.annotations)}")
    
    # Выводим аннотации
    print("\n📝 Аннотации:")
    for i, ann in enumerate(data_loader.annotations):
        label, start_time, end_time = ann
        duration = (end_time - start_time).total_seconds()
        print(f"  {i+1}. {label}: {start_time.strftime('%H:%M:%S.%f')[:-3]} - {end_time.strftime('%H:%M:%S.%f')[:-3]} ({duration:.1f}с)")
    
    # Извлекаем признаки и времена окон
    extractor = SnoringFeatureExtractor()
    
    all_features = []
    all_window_times = []
    
    for csv_file in csv_files[:3]:  # Только первые 3 файла для отладки
        try:
            features, window_times = extractor.extract_features_from_csv(csv_file)
            if features.shape[0] > 0:
                all_features.append(features)
                all_window_times.extend(window_times)
        except Exception as e:
            print(f"⚠️ Ошибка при обработке {csv_file.name}: {e}")
            continue
    
    if not all_features:
        print("⚠️ Не удалось извлечь признаки")
        return
    
    print(f"\n📊 Извлечено окон: {len(all_window_times)}")
    
    # Выводим первые несколько времен окон
    print("\n⏰ Первые 10 времен окон:")
    for i, window_time in enumerate(all_window_times[:10]):
        print(f"  {i+1}. {window_time.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Выводим последние несколько времен окон
    print("\n⏰ Последние 10 времен окон:")
    for i, window_time in enumerate(all_window_times[-10:]):
        print(f"  {len(all_window_times)-10+i+1}. {window_time.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Получаем метки
    print(f"\n🏷️ Получаем метки для {len(all_window_times)} окон...")
    labels = data_loader.get_labels_for_windows(all_window_times)
    
    if labels is None:
        print("❌ get_labels_for_windows вернул None")
        return
    
    labels = np.array(labels)
    print(f"✅ Получено меток: {len(labels)}")
    print(f"📊 Распределение меток: {np.bincount(labels)}")
    
    # Анализируем пересечения
    print(f"\n🔍 Анализ пересечений окон с аннотациями:")
    
    for i, (window_time, label) in enumerate(zip(all_window_times[:20], labels[:20])):  # Первые 20 окон
        window_end = window_time + timedelta(seconds=8)  # Окно 8 секунд
        
        # Ищем пересечения с аннотациями
        intersections = []
        for ann in data_loader.annotations:
            ann_label, ann_start, ann_end = ann
            if ann_start < window_end and window_time < ann_end:
                # Есть пересечение
                overlap_start = max(window_time, ann_start)
                overlap_end = min(window_end, ann_end)
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                intersections.append((ann_label, overlap_duration))
        
        if intersections:
            print(f"  Окно {i+1} ({window_time.strftime('%H:%M:%S.%f')[:-3]}): {intersections}")
        else:
            print(f"  Окно {i+1} ({window_time.strftime('%H:%M:%S.%f')[:-3]}): НЕТ пересечений")
    
    # Проверяем временные границы
    print(f"\n⏰ Временные границы:")
    print(f"  Первое окно: {all_window_times[0]}")
    print(f"  Последнее окно: {all_window_times[-1]}")
    print(f"  Первая аннотация: {data_loader.annotations[0][1] if data_loader.annotations else 'Нет'}")
    print(f"  Последняя аннотация: {data_loader.annotations[-1][2] if data_loader.annotations else 'Нет'}")


def main():
    """Основная функция"""
    print("🚀 Отладка проблемы с метками...")
    
    # Находим папки для тестирования
    snoring_data_dir = Path("snoring_data")
    if not snoring_data_dir.exists():
        print("❌ Папка snoring_data не найдена")
        return
    
    folders = []
    for item in snoring_data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            for subitem in item.iterdir():
                if subitem.is_dir() and not subitem.name.startswith('.'):
                    folders.append(subitem)
    
    print(f"📁 Найдено папок: {len(folders)}")
    
    # Отлаживаем каждую папку
    for folder in folders:
        debug_labels_for_folder(folder)
        break  # Только первую папку для отладки


if __name__ == "__main__":
    main() 