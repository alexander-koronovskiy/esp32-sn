#!/usr/bin/env python3
"""
Скрипт для анализа оригинальных аннотаций _ann.txt
"""

from pathlib import Path
from datetime import datetime, timedelta

def analyze_annotation_file(file_path):
    """Анализирует файл аннотаций и выводит детальную информацию"""
    print(f"\n📄 Анализ файла: {file_path.name}")
    print("=" * 60)
    
    periods = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and line.startswith('W,'):
                parts = line.split(',')
                if len(parts) == 3:
                    start_time_str = parts[1]
                    end_time_str = parts[2]
                    
                    try:
                        # Парсим время
                        start_time = datetime.strptime(start_time_str, '%H:%M:%S.%f')
                        end_time = datetime.strptime(end_time_str, '%H:%M:%S.%f')
                        
                        # Если время переходит через полночь, добавляем день
                        if end_time < start_time:
                            end_time += timedelta(days=1)
                        
                        duration = (end_time - start_time).total_seconds()
                        periods.append((start_time, end_time, duration))
                        
                        print(f"Период {line_num}:")
                        print(f"  Начало: {start_time_str}")
                        print(f"  Конец:  {end_time_str}")
                        print(f"  Длительность: {duration:.3f} сек ({duration/60:.2f} мин)")
                        
                        # Анализируем границы
                        start_sec = start_time.second + start_time.microsecond / 1000000
                        end_sec = end_time.second + end_time.microsecond / 1000000
                        
                        print(f"  Границы: {start_sec:.3f}с - {end_sec:.3f}с")
                        print()
                        
                    except ValueError as e:
                        print(f"⚠️ Ошибка парсинга времени в строке {line_num}: {line} - {e}")
                        continue
    
    if periods:
        # Общая статистика
        total_duration = sum(duration for _, _, duration in periods)
        avg_duration = total_duration / len(periods)
        min_duration = min(duration for _, _, duration in periods)
        max_duration = max(duration for _, _, duration in periods)
        
        print("📊 ОБЩАЯ СТАТИСТИКА:")
        print("-" * 40)
        print(f"Количество периодов: {len(periods)}")
        print(f"Общая длительность храпа: {total_duration:.3f} сек ({total_duration/60:.2f} мин)")
        print(f"Средняя длительность: {avg_duration:.3f} сек ({avg_duration/60:.2f} мин)")
        print(f"Минимальная длительность: {min_duration:.3f} сек ({min_duration/60:.2f} мин)")
        print(f"Максимальная длительность: {max_duration:.3f} сек ({max_duration/60:.2f} мин)")
        
        # Анализ распределения длительностей
        print(f"\n📈 РАСПРЕДЕЛЕНИЕ ДЛИТЕЛЬНОСТЕЙ:")
        print("-" * 40)
        
        short_periods = [d for d in [p[2] for p in periods] if d < 60]  # < 1 минуты
        medium_periods = [d for d in [p[2] for p in periods] if 60 <= d < 300]  # 1-5 минут
        long_periods = [d for d in [p[2] for p in periods] if d >= 300]  # >= 5 минут
        
        print(f"Короткие периоды (< 1 мин): {len(short_periods)}")
        if short_periods:
            print(f"  Примеры: {[f'{d:.1f}с' for d in short_periods]}")
        
        print(f"Средние периоды (1-5 мин): {len(medium_periods)}")
        if medium_periods:
            print(f"  Примеры: {[f'{d/60:.1f}мин' for d in medium_periods]}")
        
        print(f"Длинные периоды (>= 5 мин): {len(long_periods)}")
        if long_periods:
            print(f"  Примеры: {[f'{d/60:.1f}мин' for d in long_periods]}")
        
        # Анализ временных интервалов
        print(f"\n⏰ ВРЕМЕННЫЕ ИНТЕРВАЛЫ:")
        print("-" * 40)
        
        if len(periods) > 1:
            intervals = []
            for i in range(len(periods) - 1):
                current_end = periods[i][1]
                next_start = periods[i + 1][0]
                interval = (next_start - current_end).total_seconds()
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                min_interval = min(intervals)
                max_interval = max(intervals)
                
                print(f"Средний интервал между периодами: {avg_interval:.1f} сек ({avg_interval/60:.2f} мин)")
                print(f"Минимальный интервал: {min_interval:.1f} сек ({min_interval/60:.2f} мин)")
                print(f"Максимальный интервал: {max_interval:.1f} сек ({max_interval/60:.2f} мин)")
                
                # Анализ непрерывности
                continuous_threshold = 60  # 1 минута
                continuous_periods = [i for i in intervals if i <= continuous_threshold]
                print(f"Короткие интервалы (<= 1 мин): {len(continuous_periods)} из {len(intervals)}")
        
        # Анализ точности времени
        print(f"\n🎯 ТОЧНОСТЬ ВРЕМЕНИ:")
        print("-" * 40)
        
        millisecond_precision = []
        for start_time, end_time, _ in periods:
            start_ms = start_time.microsecond / 1000
            end_ms = end_time.microsecond / 1000
            millisecond_precision.extend([start_ms, end_ms])
        
        if millisecond_precision:
            avg_ms = sum(millisecond_precision) / len(millisecond_precision)
            print(f"Средняя точность (мс): {avg_ms:.1f} мс")
            print(f"Максимальная точность (мс): {max(millisecond_precision):.1f} мс")
            print(f"Минимальная точность (мс): {min(millisecond_precision):.1f} мс")
    
    return periods

def main():
    """Основная функция"""
    print("🔍 АНАЛИЗ ОРИГИНАЛЬНЫХ АННОТАЦИЙ _ann.txt")
    print("=" * 80)
    
    # Находим все файлы _ann.txt
    snoring_data_dir = Path("snoring_data")
    if not snoring_data_dir.exists():
        print("❌ Папка snoring_data не найдена")
        return
    
    all_annotations = []
    
    for item in snoring_data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            for subitem in item.iterdir():
                if subitem.is_dir() and not subitem.name.startswith('.'):
                    # Ищем оригинальные файлы аннотаций (без префикса model_)
                    ann_files = [f for f in subitem.glob("*_ann.txt") if not f.name.startswith('model_')]
                    
                    for ann_file in ann_files:
                        periods = analyze_annotation_file(ann_file)
                        if periods:
                            all_annotations.extend(periods)
    
    # Общая статистика по всем файлам
    if all_annotations:
        print(f"\n🎯 ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ФАЙЛАМ:")
        print("=" * 80)
        
        total_files = len(set(f.parent for f in snoring_data_dir.rglob("*_ann.txt") if not f.name.startswith('model_')))
        total_periods = len(all_annotations)
        total_duration = sum(duration for _, _, duration in all_annotations)
        avg_duration = total_duration / total_periods if total_periods > 0 else 0
        
        print(f"Всего файлов аннотаций: {total_files}")
        print(f"Всего периодов храпа: {total_periods}")
        print(f"Общая длительность храпа: {total_duration:.3f} сек ({total_duration/60:.2f} мин)")
        print(f"Средняя длительность периода: {avg_duration:.3f} сек ({avg_duration/60:.2f} мин)")
        
        # Анализ распределения по длительностям
        durations = [duration for _, _, duration in all_annotations]
        durations.sort()
        
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ДЛИТЕЛЬНОСТЕЙ:")
        print("-" * 50)
        
        # Квартили
        if len(durations) >= 4:
            q1_idx = len(durations) // 4
            q3_idx = 3 * len(durations) // 4
            median_idx = len(durations) // 2
            
            print(f"Q1 (25%): {durations[q1_idx]:.1f} сек")
            print(f"Медиана (50%): {durations[median_idx]:.1f} сек")
            print(f"Q3 (75%): {durations[q3_idx]:.1f} сек")
        
        print(f"Минимум: {min(durations):.1f} сек")
        print(f"Максимум: {max(durations):.1f} сек")
        
        # Группировка по длительностям
        very_short = [d for d in durations if d < 10]  # < 10 сек
        short = [d for d in durations if 10 <= d < 60]  # 10 сек - 1 мин
        medium = [d for d in durations if 60 <= d < 300]  # 1-5 мин
        long = [d for d in durations if 300 <= d < 600]  # 5-10 мин
        very_long = [d for d in durations if d >= 600]  # >= 10 мин
        
        print(f"\n📈 ГРУППИРОВКА ПО ДЛИТЕЛЬНОСТЯМ:")
        print("-" * 50)
        print(f"Очень короткие (< 10 сек): {len(very_short)} периодов")
        print(f"Короткие (10 сек - 1 мин): {len(short)} периодов")
        print(f"Средние (1-5 мин): {len(medium)} периодов")
        print(f"Длинные (5-10 мин): {len(long)} периодов")
        print(f"Очень длинные (>= 10 мин): {len(very_long)} периодов")
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main() 