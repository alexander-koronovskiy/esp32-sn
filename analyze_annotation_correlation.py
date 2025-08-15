#!/usr/bin/env python3
"""
Скрипт для анализа соотношения между оригинальными аннотациями (_ann.txt) 
и аннотациями модели (model_*_ann.txt)

ВАЖНО: Аннотации модели основаны на 8-секундном скользящем окне с шагом 1 секунда.
Каждое окно дает предсказание на 1 секунду (время начала окна).
Шаг между окнами = 1 секунда обеспечивает полное покрытие времени без пропусков.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class AnnotationCorrelationAnalyzer:
    """Анализатор корреляции между аннотациями"""
    
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
    
    def analyze_folder(self, folder_path):
        """Анализирует одну папку с данными"""
        folder_name = folder_path.name
        print(f"\n🔍 Анализирую папку: {folder_name}")
        
        # Ищем файлы аннотаций
        ann_files = [f for f in folder_path.glob("*_ann.txt") if not f.name.startswith('model_')]
        model_ann_files = [f for f in folder_path.glob("model_*_ann.txt")]
        
        if not ann_files:
            print(f"⚠️ Оригинальный файл аннотаций не найден")
            return None
        
        if not model_ann_files:
            print(f"⚠️ Файл аннотаций модели не найден")
            return None
        
        original_ann_file = ann_files[0]
        model_ann_file = model_ann_files[0]
        
        print(f"📄 Оригинальные аннотации: {original_ann_file.name}")
        print(f"🤖 Аннотации модели: {model_ann_file.name}")
        
        # Парсим аннотации
        original_periods = self.parse_annotations(original_ann_file)
        model_periods = self.parse_annotations(model_ann_file)
        
        print(f"📊 Оригинальных периодов: {len(original_periods)}")
        print(f"🤖 Периодов модели: {len(model_periods)}")
        
        # Анализируем корреляцию
        correlation_metrics = self.calculate_correlation_metrics(original_periods, model_periods)
        
        # Визуализируем
        self.visualize_correlation(original_periods, model_periods, folder_name)
        
        # Сохраняем результаты
        self.results[folder_name] = {
            'original_periods': len(original_periods),
            'model_periods': len(model_periods),
            'correlation_metrics': correlation_metrics
        }
        
        return correlation_metrics
    
    def calculate_correlation_metrics(self, original_periods, model_periods):
        """Вычисляет метрики корреляции между аннотациями
        
        ВАЖНО: Аннотации модели основаны на 8-секундном скользящем окне.
        Каждое окно дает предсказание на 1 секунду (время начала окна).
        Шаг между окнами = 1 секунда, что обеспечивает полное покрытие времени.
        """
        
        # Создаем временную шкалу с шагом 1 секунда
        all_start_times = [p[0] for p in original_periods + model_periods]
        all_end_times = [p[1] for p in original_periods + model_periods]
        
        if not all_start_times or not all_end_times:
            return {}
        
        timeline_start = min(all_start_times)
        timeline_end = max(all_end_times)
        
        # Создаем временные метки каждую секунду
        timeline = []
        current_time = timeline_start
        while current_time <= timeline_end:
            timeline.append(current_time)
            current_time += timedelta(seconds=1)
        
        # Создаем бинарные векторы для каждого типа аннотаций
        timeline_length = len(timeline)
        original_vector = np.zeros(timeline_length, dtype=int)
        model_vector = np.zeros(timeline_length, dtype=int)
        
        # Заполняем оригинальные аннотации
        for start_time, end_time in original_periods:
            start_idx = int((start_time - timeline_start).total_seconds())
            end_idx = int((end_time - timeline_start).total_seconds())
            
            start_idx = max(0, start_idx)
            end_idx = min(timeline_length, end_idx)
            
            if start_idx < end_idx:
                original_vector[start_idx:end_idx] = 1
        
        # Заполняем аннотации модели (каждая отметка = 1 секунда, полное покрытие)
        for start_time, end_time in model_periods:
            start_idx = int((start_time - timeline_start).total_seconds())
            end_idx = int((end_time - timeline_start).total_seconds())
            
            start_idx = max(0, start_idx)
            end_idx = min(timeline_length, end_idx)
            
            if start_idx < end_idx:
                model_vector[start_idx:end_idx] = 1
        
        # Вычисляем метрики
        total_seconds = len(timeline)
        original_snoring_seconds = np.sum(original_vector)
        model_snoring_seconds = np.sum(model_vector)
        
        # Пересечение (True Positive)
        intersection = np.sum(original_vector & model_vector)
        
        # Объединение (Total)
        union = np.sum(original_vector | model_vector)
        
        # False Positive (модель предсказала храп, но его нет)
        false_positive = np.sum((model_vector == 1) & (original_vector == 0))
        
        # False Negative (модель не предсказала храп, но он есть)
        false_negative = np.sum((model_vector == 0) & (original_vector == 1))
        
        # True Negative (модель правильно предсказала отсутствие храпа)
        true_negative = np.sum((model_vector == 0) & (original_vector == 0))
        
        # Вычисляем метрики
        if union > 0:
            jaccard_similarity = intersection / union
        else:
            jaccard_similarity = 0.0
        
        if original_snoring_seconds > 0:
            recall = intersection / original_snoring_seconds
        else:
            recall = 0.0
        
        if model_snoring_seconds > 0:
            precision = intersection / model_snoring_seconds
        else:
            precision = 0.0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Общая точность
        accuracy = (intersection + true_negative) / total_seconds
        
        metrics = {
            'total_seconds': total_seconds,
            'original_snoring_seconds': int(original_snoring_seconds),
            'model_snoring_seconds': int(model_snoring_seconds),
            'intersection_seconds': int(intersection),
            'union_seconds': int(union),
            'false_positive_seconds': int(false_positive),
            'false_negative_seconds': int(false_negative),
            'true_negative_seconds': int(true_negative),
            'jaccard_similarity': jaccard_similarity,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
        
        print(f"📊 Метрики корреляции (модель: 8с скользящее окно):")
        print(f"  Общее время: {total_seconds} сек")
        print(f"  Оригинальный храп: {original_snoring_seconds} сек")
        print(f"  Храп модели: {model_snoring_seconds} сек (8с периоды)")
        print(f"  Пересечение: {intersection} сек")
        print(f"  Jaccard similarity: {jaccard_similarity:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-score: {f1_score:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def visualize_correlation(self, original_periods, model_periods, folder_name):
        """Визуализирует корреляцию между аннотациями
        
        ВАЖНО: Аннотации модели основаны на 8-секундном скользящем окне.
        Каждое окно дает предсказание на 1 секунду (время начала окна).
        """
        
        # Создаем временную шкалу
        all_start_times = [p[0] for p in original_periods + model_periods]
        all_end_times = [p[1] for p in original_periods + model_periods]
        
        if not all_start_times or not all_end_times:
            return
        
        timeline_start = min(all_start_times)
        timeline_end = max(all_end_times)
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # График 1: Оригинальные аннотации
        ax1.set_title(f'Оригинальные аннотации - {folder_name}', fontsize=14)
        ax1.set_ylabel('Оригинальные')
        
        for i, (start_time, end_time) in enumerate(original_periods):
            start_sec = (start_time - timeline_start).total_seconds()
            end_sec = (end_time - timeline_start).total_seconds()
            ax1.barh(0, end_sec - start_sec, left=start_sec, height=0.8, 
                     color='blue', alpha=0.7, label='Храп' if i == 0 else "")
        
        ax1.set_xlim(0, (timeline_end - timeline_start).total_seconds())
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xticks([])
        ax1.legend()
        
        # График 2: Аннотации модели
        ax2.set_title(f'Аннотации модели (8с периоды) - {folder_name}', fontsize=14)
        ax2.set_ylabel('Модель (8с периоды)')
        ax2.set_xlabel('Время (секунды)')
        
        for i, (start_time, end_time) in enumerate(model_periods):
            start_sec = (start_time - timeline_start).total_seconds()
            end_sec = (end_time - timeline_start).total_seconds()
            ax2.barh(0, end_sec - start_sec, left=start_sec, height=0.8, 
                     color='red', alpha=0.7, label='Храп' if i == 0 else "")
        
        ax2.set_xlim(0, (timeline_end - timeline_start).total_seconds())
        ax2.set_ylim(-0.5, 0.5)
        ax2.legend()
        
        # Добавляем временные метки
        total_seconds = int((timeline_end - timeline_start).total_seconds())
        if total_seconds > 0:
            step = max(1, total_seconds // 10)
            time_ticks = list(range(0, total_seconds + 1, step))
            ax2.set_xticks(time_ticks)
            ax2.set_xticklabels([f"{timeline_start + timedelta(seconds=s):%H:%M:%S}" 
                                for s in time_ticks], rotation=45)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_dir = Path("results_visualization")
        output_dir.mkdir(exist_ok=True)
        
        plot_path = output_dir / f"annotation_correlation_{folder_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 График сохранен: {plot_path}")
        
        plt.close()
    
    def run_analysis(self):
        """Запускает анализ всех папок"""
        print("🚀 Начинаю анализ корреляции аннотаций...")
        
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
                self.analyze_folder(folder)
            except Exception as e:
                print(f"❌ Ошибка при анализе {folder.name}: {e}")
                continue
        
        # Создаем общий отчет
        self.create_summary_report()
        
        print(f"\n🎉 Анализ завершен! Обработано папок: {len(self.results)}")
    
    def create_summary_report(self):
        """Создает общий отчет по анализу"""
        if not self.results:
            print("⚠️ Нет результатов для отчета")
            return
        
        report_path = Path("results") / "annotation_correlation_analysis.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        print(f"\n📊 Создаю общий отчет: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("АНАЛИЗ КОРРЕЛЯЦИИ МЕЖДУ АННОТАЦИЯМИ\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего проанализировано папок: {len(self.results)}\n\n")
            
            # Общая статистика
            total_original_seconds = sum(r['correlation_metrics']['original_snoring_seconds'] 
                                       for r in self.results.values())
            total_model_seconds = sum(r['correlation_metrics']['model_snoring_seconds'] 
                                    for r in self.results.values())
            total_intersection = sum(r['correlation_metrics']['intersection_seconds'] 
                                   for r in self.results.values())
            
            avg_jaccard = np.mean([r['correlation_metrics']['jaccard_similarity'] 
                                  for r in self.results.values()])
            avg_recall = np.mean([r['correlation_metrics']['recall'] 
                                for r in self.results.values()])
            avg_precision = np.mean([r['correlation_metrics']['precision'] 
                                   for r in self.results.values()])
            avg_f1 = np.mean([r['correlation_metrics']['f1_score'] 
                             for r in self.results.values()])
            avg_accuracy = np.mean([r['correlation_metrics']['accuracy'] 
                                  for r in self.results.values()])
            
            f.write("ОБЩАЯ СТАТИСТИКА:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Общее время оригинального храпа: {total_original_seconds} сек\n")
            f.write(f"Общее время храпа модели: {total_model_seconds} сек (8с периоды)\n")
            f.write(f"Общее время пересечения: {total_intersection} сек\n")
            f.write(f"Средний Jaccard similarity: {avg_jaccard:.4f}\n")
            f.write(f"Средний Recall: {avg_recall:.4f}\n")
            f.write(f"Средний Precision: {avg_precision:.4f}\n")
            f.write(f"Средний F1-score: {avg_f1:.4f}\n")
            f.write(f"Средняя Accuracy: {avg_accuracy:.4f}\n\n")
            
            # Детальные результаты по папкам
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ПАПКАМ:\n")
            f.write("-" * 50 + "\n")
            
            for folder_name, result in self.results.items():
                metrics = result['correlation_metrics']
                f.write(f"\nПапка: {folder_name}\n")
                f.write(f"  Оригинальных периодов: {result['original_periods']}\n")
                f.write(f"  Периодов модели: {result['model_periods']} (8с периоды)\n")
                f.write(f"  Jaccard similarity: {metrics['jaccard_similarity']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  F1-score: {metrics['f1_score']:.4f}\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Пересечение: {metrics['intersection_seconds']} сек\n")
                f.write(f"  False Positive: {metrics['false_positive_seconds']} сек\n")
                f.write(f"  False Negative: {metrics['false_negative_seconds']} сек\n")
        
        print(f"✅ Общий отчет создан: {report_path}")


def main():
    """Основная функция"""
    try:
        analyzer = AnnotationCorrelationAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 