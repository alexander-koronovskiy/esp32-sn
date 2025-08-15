#!/usr/bin/env python3
"""
Детальная отладка временного выравнивания
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import re

# Добавляем путь к src
sys.path.append('src')

from utils.data_loader import SnoringDataLoader
from features.feature_extractor import SnoringFeatureExtractor

def debug_time_alignment_detailed():
    """Детальная отладка временного выравнивания"""
    
    print("🔍 ДЕТАЛЬНАЯ ОТЛАДКА ВРЕМЕННОГО ВЫРАВНИВАНИЯ")
    print("=" * 60)
    
    # Путь к данным
    data_path = Path("snoring_data/19/4_2025_08_12_23.43")
    
    # Инициализируем загрузчик и экстрактор
    data_loader = SnoringDataLoader(data_path)
    extractor = SnoringFeatureExtractor()
    
    # Загружаем аннотации
    print("\n📋 АННОТАЦИИ:")
    annotation_file = data_loader.find_annotation_file()
    annotations = data_loader.parse_annotations(annotation_file)
    for i, (label, start_time, end_time) in enumerate(annotations):
        duration = (end_time - start_time).total_seconds()
        print(f"  {i+1}. {label}: {start_time} - {end_time}")
        print(f"     Длительность: {duration:.1f}с")
    
    # Получаем отсортированные CSV файлы
    csv_files = data_loader.get_csv_files_sorted()
    print(f"\n📁 Всего CSV файлов: {len(csv_files)}")
    
    # Проверяем конкретные файлы вокруг периода аннотации
    print("\n🔍 АНАЛИЗ ФАЙЛОВ ВОКРУГ ПЕРИОДА АННОТАЦИИ:")
    
    # Основной период аннотации: 23:57:46 - 00:08:40
    target_start = datetime(2025, 8, 12, 23, 57, 46)
    target_end = datetime(2025, 8, 13, 0, 8, 40)
    
    print(f"  Целевой период: {target_start} - {target_end}")
    
    # Проверяем файлы, которые должны покрывать этот период
    relevant_files = []
    for csv_file in csv_files:
        try:
            # Парсим время из имени файла с правильной базовой датой
            base_date = data_loader.get_base_date_from_folder()
            start_time = extractor.parse_time_from_filename(csv_file.name, base_date)
            end_time = start_time + timedelta(seconds=30)  # 30 секунд на файл
            
            # Проверяем, пересекается ли с целевым периодом
            if (start_time < target_end and end_time > target_start):
                relevant_files.append((csv_file, start_time, end_time))
                print(f"  ✅ {csv_file.name}: {start_time} - {end_time} (ПЕРЕСЕКАЕТСЯ)")
            elif start_time < target_start + timedelta(minutes=5) or end_time > target_end - timedelta(minutes=5):
                print(f"  ⚠️  {csv_file.name}: {start_time} - {end_time} (БЛИЗКО)")
            else:
                print(f"  ❌ {csv_file.name}: {start_time} - {end_time}")
                
        except Exception as e:
            print(f"  ❌ Ошибка парсинга {csv_file.name}: {e}")
    
    print(f"\n📊 РЕЗУЛЬТАТ:")
    print(f"  Файлов, пересекающихся с аннотацией: {len(relevant_files)}")
    
    if relevant_files:
        print("\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРЕСЕКАЮЩИХСЯ ФАЙЛОВ:")
        for csv_file, start_time, end_time in relevant_files:
            print(f"\n  📄 {csv_file.name}:")
            
            # Загружаем данные
            try:
                data = pd.read_csv(csv_file, sep=';')
                print(f"     Строк в файле: {len(data)}")
                print(f"     Колонка 'time': {data['time'].min()} - {data['time'].max()}")
                
                # Проверяем, как это время соотносится с реальным временем
                time_diff = data['time'].max() - data['time'].min()
                real_time_diff = (end_time - start_time).total_seconds()
                print(f"     Разница в 'time': {time_diff}")
                print(f"     Реальное время: {real_time_diff:.1f}с")
                
                # Проверяем, есть ли данные в целевом периоде
                if start_time <= target_start <= end_time:
                    print(f"     ✅ Содержит начало аннотации")
                if start_time <= target_end <= end_time:
                    print(f"     ✅ Содержит конец аннотации")
                    
            except Exception as e:
                print(f"     ❌ Ошибка загрузки: {e}")
    
    else:
        print("\n⚠️  ПРОБЛЕМА: Нет файлов, пересекающихся с аннотацией!")
        print("   Это объясняет, почему все окна помечены как 'No Snoring'")
        
        # Проверяем, есть ли разрыв в данных
        print("\n🔍 ПРОВЕРКА РАЗРЫВА В ДАННЫХ:")
        
        # Ищем последний вечерний файл
        evening_files = [f for f in csv_files if '23' in f.name]
        if evening_files:
            last_evening = max(evening_files, key=lambda x: extractor.parse_time_from_filename(x.name))
            last_evening_time = extractor.parse_time_from_filename(last_evening.name)
            print(f"  Последний вечерний файл: {last_evening.name} ({last_evening_time})")
            
            # Ищем первый утренний файл
            morning_files = [f for f in csv_files if '00' in f.name]
            if morning_files:
                first_morning = min(morning_files, key=lambda x: extractor.parse_time_from_filename(x.name))
                first_morning_time = extractor.parse_time_from_filename(first_morning.name)
                print(f"  Первый утренний файл: {first_morning.name} ({first_morning_time})")
                
                gap = (first_morning_time - last_evening_time).total_seconds()
                print(f"  Разрыв в данных: {gap:.1f}с")
                
                if gap > 30:
                    print(f"  ⚠️  Есть разрыв в данных около полуночи!")

if __name__ == "__main__":
    debug_time_alignment_detailed() 