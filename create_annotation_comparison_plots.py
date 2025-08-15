#!/usr/bin/env python3
"""
Скрипт для создания детальных графиков сравнения аннотаций
Оригинальные (_ann.txt) vs Модель (model_*_ann.txt)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

class AnnotationComparisonPlotter:
    """Создает детальные графики сравнения аннотаций"""
    
    def __init__(self):
        self.results = {}
        
    def parse_annotations(self, ann_file_path):
        """Парсит файл аннотаций и возвращает список периодов"""
        periods = []
        
        with open(ann_file_path, 'r') as f:
            for line in f:
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
                            
                            periods.append((start_time, end_time))
                        except ValueError as e:
                            print(f"⚠️ Ошибка парсинга времени: {line} - {e}")
                            continue
        
        return periods
    
    def create_comparison_plot(self, folder_path, original_periods, model_periods, folder_name):
        """Создает детальный график сравнения аннотаций"""
        
        # Создаем временную шкалу
        all_start_times = [p[0] for p in original_periods + model_periods]
        all_end_times = [p[1] for p in original_periods + model_periods]
        
        if not all_start_times or not all_end_times:
            return
        
        timeline_start = min(all_start_times)
        timeline_end = max(all_end_times)
        total_seconds = int((timeline_end - timeline_start).total_seconds())
        
        # Создаем основной график
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        
        # График 1: Оригинальные аннотации
        ax1.set_title(f'Оригинальные аннотации экспертов - {folder_name}', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Эксперты', fontsize=12)
        
        for i, (start_time, end_time) in enumerate(original_periods):
            start_sec = (start_time - timeline_start).total_seconds()
            end_sec = (end_time - timeline_start).total_seconds()
            duration = end_sec - start_sec
            
            # Создаем прямоугольник для периода
            rect = Rectangle((start_sec, -0.4), duration, 0.8, 
                           facecolor='#2E86AB', edgecolor='#1B4F72', linewidth=1, alpha=0.8)
            ax1.add_patch(rect)
            
            # Добавляем подпись с длительностью
            if duration > 60:  # Если больше минуты, показываем в минутах
                label = f"{duration/60:.1f} мин"
            else:
                label = f"{duration:.1f}с"
            
            ax1.text(start_sec + duration/2, 0, label, ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        
        ax1.set_xlim(0, total_seconds)
        ax1.set_ylim(-0.6, 0.6)
        ax1.set_xticks([])
        ax1.grid(True, alpha=0.3)
        
        # График 2: Аннотации модели
        ax2.set_title(f'Аннотации модели (8-секундные периоды) - {folder_name}', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Модель', fontsize=12)
        
        for i, (start_time, end_time) in enumerate(model_periods):
            start_sec = (start_time - timeline_start).total_seconds()
            end_sec = (end_time - timeline_start).total_seconds()
            duration = end_sec - start_sec
            
            # Создаем прямоугольник для периода
            rect = Rectangle((start_sec, -0.4), duration, 0.8, 
                           facecolor='#A23B72', edgecolor='#6B2B3D', linewidth=1, alpha=0.8)
            ax2.add_patch(rect)
            
            # Добавляем подпись с длительностью (всегда 8 секунд)
            label = "8с"
            ax2.text(start_sec + duration/2, 0, label, ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
        
        ax2.set_xlim(0, total_seconds)
        ax2.set_ylim(-0.6, 0.6)
        ax2.set_xticks([])
        ax2.grid(True, alpha=0.3)
        
        # График 3: Сравнение и пересечения
        ax3.set_title(f'Сравнение и пересечения - {folder_name}', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Пересечения', fontsize=12)
        ax3.set_xlabel('Время (секунды)', fontsize=12)
        
        # Создаем временную шкалу с шагом 1 секунда
        timeline = np.arange(0, total_seconds + 1)
        original_vector = np.zeros(total_seconds + 1, dtype=int)
        model_vector = np.zeros(total_seconds + 1, dtype=int)
        
        # Заполняем оригинальные аннотации
        for start_time, end_time in original_periods:
            start_idx = int((start_time - timeline_start).total_seconds())
            end_idx = int((end_time - timeline_start).total_seconds())
            start_idx = max(0, start_idx)
            end_idx = min(total_seconds + 1, end_idx)
            if start_idx < end_idx:
                original_vector[start_idx:end_idx] = 1
        
        # Заполняем аннотации модели
        for start_time, end_time in model_periods:
            start_idx = int((start_time - timeline_start).total_seconds())
            end_idx = int((end_time - timeline_start).total_seconds())
            start_idx = max(0, start_idx)
            end_idx = min(total_seconds + 1, end_idx)
            if start_idx < end_idx:
                model_vector[start_idx:end_idx] = 1
        
        # Вычисляем пересечения
        intersection = original_vector & model_vector
        only_original = original_vector & ~model_vector
        only_model = ~original_vector & model_vector
        
        # Рисуем пересечения
        ax3.fill_between(timeline, 0, intersection, color='#28A745', alpha=0.8, label='Пересечение (TP)')
        ax3.fill_between(timeline, 0, only_original, color='#007BFF', alpha=0.6, label='Только эксперты (FN)')
        ax3.fill_between(timeline, 0, only_model, color='#DC3545', alpha=0.6, label='Только модель (FP)')
        
        ax3.set_xlim(0, total_seconds)
        ax3.set_ylim(0, 1.1)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Добавляем временные метки на нижний график
        if total_seconds > 0:
            step = max(1, total_seconds // 15)  # Больше меток
            time_ticks = list(range(0, total_seconds + 1, step))
            ax3.set_xticks(time_ticks)
            ax3.set_xticklabels([f"{timeline_start + timedelta(seconds=s):%H:%M:%S}" 
                                for s in time_ticks], rotation=45, fontsize=9)
        
        # Добавляем статистику
        intersection_seconds = np.sum(intersection)
        only_original_seconds = np.sum(only_original)
        only_model_seconds = np.sum(only_model)
        total_original = np.sum(original_vector)
        total_model = np.sum(model_vector)
        
        # Вычисляем метрики
        if total_original > 0:
            recall = intersection_seconds / total_original
        else:
            recall = 0.0
            
        if total_model > 0:
            precision = intersection_seconds / total_model
        else:
            precision = 0.0
            
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Добавляем статистику на график
        stats_text = f"""Статистика:
Пересечение: {intersection_seconds}с
Только эксперты: {only_original_seconds}с  
Только модель: {only_model_seconds}с
Recall: {recall:.3f}
Precision: {precision:.3f}
F1-score: {f1_score:.3f}"""
        
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Сохраняем график
        output_dir = Path("results_visualization")
        output_dir.mkdir(exist_ok=True)
        
        plot_path = output_dir / f"annotation_comparison_detailed_{folder_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Детальный график сравнения сохранен: {plot_path}")
        
        plt.close()
        
        return {
            'intersection_seconds': int(intersection_seconds),
            'only_original_seconds': int(only_original_seconds),
            'only_model_seconds': int(only_model_seconds),
            'total_original_seconds': int(total_original),
            'total_model_seconds': int(total_model),
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score
        }
    
    def run_analysis(self):
        """Запускает анализ всех папок"""
        print("🚀 Начинаю создание графиков сравнения аннотаций...")
        
        # Находим папки для анализа
        snoring_data_dir = Path("snoring_data")
        if not snoring_data_dir.exists():
            raise FileNotFoundError("Папка snoring_data не найдена")
        
        folders = []
        for item in snoring_data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                for subitem in item.iterdir():
                    if subitem.is_dir() and not subitem.name.startswith('.'):
                        folders.append(subitem)
        
        print(f"📁 Найдено папок для анализа: {len(folders)}")
        
        # Анализируем каждую папку
        for folder in folders:
            try:
                print(f"\n🔍 Обрабатываю папку: {folder.name}")
                
                # Ищем файлы аннотаций
                ann_files = [f for f in folder.glob("*_ann.txt") if not f.name.startswith('model_')]
                model_ann_files = [f for f in folder.glob("model_*_ann.txt")]
                
                if not ann_files:
                    print(f"⚠️ Оригинальный файл аннотаций не найден")
                    continue
                
                if not model_ann_files:
                    print(f"⚠️ Файл аннотаций модели не найден")
                    continue
                
                original_ann_file = ann_files[0]
                model_ann_file = model_ann_files[0]
                
                print(f"📄 Оригинальные аннотации: {original_ann_file.name}")
                print(f"🤖 Аннотации модели: {model_ann_file.name}")
                
                # Парсим аннотации
                original_periods = self.parse_annotations(original_ann_file)
                model_periods = self.parse_annotations(model_ann_file)
                
                print(f"📊 Оригинальных периодов: {len(original_periods)}")
                print(f"🤖 Периодов модели: {len(model_periods)}")
                
                # Создаем график сравнения
                result = self.create_comparison_plot(folder, original_periods, model_periods, folder.name)
                if result:
                    self.results[folder.name] = result
                
            except Exception as e:
                print(f"❌ Ошибка при обработке {folder.name}: {e}")
                continue
        
        print(f"\n🎉 Создание графиков завершено! Обработано папок: {len(self.results)}")


def main():
    """Основная функция"""
    try:
        plotter = AnnotationComparisonPlotter()
        plotter.run_analysis()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 