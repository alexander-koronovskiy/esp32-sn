#!/usr/bin/env python3
"""
Скрипт для создания сводного графика сравнения аннотаций по всем папкам
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

def create_summary_comparison_plot():
    """Создает сводный график сравнения по всем папкам"""
    
    # Данные из анализа корреляции
    data = {
        '4_2025_08_12_23.43': {
            'original_periods': 2,
            'model_periods': 189,
            'original_duration': 656,
            'model_duration': 437,
            'intersection': 135,
            'jaccard': 0.1409,
            'recall': 0.2058,
            'precision': 0.3089,
            'f1_score': 0.2470,
            'accuracy': 0.9905
        },
        '666_2025_08_11_10.08': {
            'original_periods': 4,
            'model_periods': 669,
            'original_duration': 826,
            'model_duration': 923,
            'intersection': 815,
            'jaccard': 0.8726,
            'recall': 0.9867,
            'precision': 0.8830,
            'f1_score': 0.9320,
            'accuracy': 0.8799
        }
    }
    
    # Создаем график
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    folders = list(data.keys())
    x = np.arange(len(folders))
    width = 0.35
    
    # График 1: Количество периодов
    ax1.bar(x - width/2, [data[f]['original_periods'] for f in folders], 
            width, label='Оригинальные', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, [data[f]['model_periods'] for f in folders], 
            width, label='Модель', color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Папки', fontsize=12)
    ax1.set_ylabel('Количество периодов', fontsize=12)
    ax1.set_title('Количество периодов по папкам', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace('_', '\n') for f in folders], fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, folder in enumerate(folders):
        ax1.text(i - width/2, data[folder]['original_periods'] + 1, 
                str(data[folder]['original_periods']), ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, data[folder]['model_periods'] + 1, 
                str(data[folder]['model_periods']), ha='center', va='bottom', fontweight='bold')
    
    # График 2: Длительность периодов
    ax2.bar(x - width/2, [data[f]['original_duration'] for f in folders], 
            width, label='Оригинальные', color='#2E86AB', alpha=0.8)
    ax2.bar(x + width/2, [data[f]['model_duration'] for f in folders], 
            width, label='Модель', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Папки', fontsize=12)
    ax2.set_ylabel('Длительность (секунды)', fontsize=12)
    ax2.set_title('Общая длительность периодов по папкам', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace('_', '\n') for f in folders], fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, folder in enumerate(folders):
        ax2.text(i - width/2, data[folder]['original_duration'] + 20, 
                f"{data[folder]['original_duration']}с", ha='center', va='bottom', fontweight='bold')
        ax2.text(i + width/2, data[folder]['model_duration'] + 20, 
                f"{data[folder]['model_duration']}с", ha='center', va='bottom', fontweight='bold')
    
    # График 3: Метрики качества
    metrics = ['jaccard', 'recall', 'precision', 'f1_score']
    metric_names = ['Jaccard', 'Recall', 'Precision', 'F1-score']
    
    x_metrics = np.arange(len(metrics))
    width_metrics = 0.35
    
    for i, folder in enumerate(folders):
        values = [data[folder][metric] for metric in metrics]
        ax3.bar(x_metrics + i * width_metrics, values, width_metrics, 
                label=folder.replace('_', '\n'), alpha=0.8)
    
    ax3.set_xlabel('Метрики', fontsize=12)
    ax3.set_ylabel('Значение', fontsize=12)
    ax3.set_title('Метрики качества по папкам', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_metrics + width_metrics/2)
    ax3.set_xticklabels(metric_names, fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # График 4: Пересечения и различия
    categories = ['Пересечение\n(TP)', 'Только\nэксперты (FN)', 'Только\nмодель (FP)']
    
    for i, folder in enumerate(folders):
        intersection = data[folder]['intersection']
        only_original = data[folder]['original_duration'] - intersection
        only_model = data[folder]['model_duration'] - intersection
        
        values = [intersection, only_original, only_model]
        x_pos = np.arange(len(categories)) + i * 0.3
        
        ax4.bar(x_pos, values, 0.25, label=folder.replace('_', '\n'), alpha=0.8)
    
    ax4.set_xlabel('Категории', fontsize=12)
    ax4.set_ylabel('Время (секунды)', fontsize=12)
    ax4.set_title('Временные характеристики по категориям', fontsize=14, fontweight='bold')
    ax4.set_xticks(np.arange(len(categories)) + 0.15)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_dir = Path("results_visualization")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / "annotation_comparison_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Сводный график сравнения сохранен: {plot_path}")
    
    plt.close()
    
    # Создаем таблицу сравнения
    create_comparison_table(data)

def create_comparison_table(data):
    """Создает таблицу сравнения в текстовом виде"""
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    table_path = output_dir / "annotation_comparison_table.txt"
    
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("ТАБЛИЦА СРАВНЕНИЯ АННОТАЦИЙ\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Заголовки таблицы
        headers = [
            'Папка',
            'Оригинальные\nпериоды',
            'Модель\nпериоды',
            'Оригинальная\nдлительность',
            'Модель\nдлительность',
            'Пересечение',
            'Jaccard',
            'Recall',
            'Precision',
            'F1-score',
            'Accuracy'
        ]
        
        # Выравнивание заголовков
        header_line = "| " + " | ".join(f"{h:^15}" for h in headers) + " |"
        separator = "|" + "|".join("-" * 17 for _ in headers) + "|"
        
        f.write(header_line + "\n")
        f.write(separator + "\n")
        
        # Данные
        for folder, metrics in data.items():
            row = [
                folder[:15],
                str(metrics['original_periods']),
                str(metrics['model_periods']),
                f"{metrics['original_duration']}с",
                f"{metrics['model_duration']}с",
                f"{metrics['intersection']}с",
                f"{metrics['jaccard']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['accuracy']:.4f}"
            ]
            
            row_line = "| " + " | ".join(f"{str(r):^15}" for r in row) + " |"
            f.write(row_line + "\n")
        
        f.write(separator + "\n\n")
        
        # Общая статистика
        total_original_periods = sum(data[f]['original_periods'] for f in data.keys())
        total_model_periods = sum(data[f]['model_periods'] for f in data.keys())
        total_original_duration = sum(data[f]['original_duration'] for f in data.keys())
        total_model_duration = sum(data[f]['model_duration'] for f in data.keys())
        total_intersection = sum(data[f]['intersection'] for f in data.keys())
        
        avg_jaccard = np.mean([data[f]['jaccard'] for f in data.keys()])
        avg_recall = np.mean([data[f]['recall'] for f in data.keys()])
        avg_precision = np.mean([data[f]['precision'] for f in data.keys()])
        avg_f1 = np.mean([data[f]['f1_score'] for f in data.keys()])
        avg_accuracy = np.mean([data[f]['accuracy'] for f in data.keys()])
        
        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Всего оригинальных периодов: {total_original_periods}\n")
        f.write(f"Всего периодов модели: {total_model_periods}\n")
        f.write(f"Общая оригинальная длительность: {total_original_duration} сек ({total_original_duration/60:.2f} мин)\n")
        f.write(f"Общая длительность модели: {total_model_duration} сек ({total_model_duration/60:.2f} мин)\n")
        f.write(f"Общее пересечение: {total_intersection} сек ({total_intersection/60:.2f} мин)\n\n")
        
        f.write("СРЕДНИЕ МЕТРИКИ:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Средний Jaccard: {avg_jaccard:.4f}\n")
        f.write(f"Средний Recall: {avg_recall:.4f}\n")
        f.write(f"Средний Precision: {avg_precision:.4f}\n")
        f.write(f"Средний F1-score: {avg_f1:.4f}\n")
        f.write(f"Средняя Accuracy: {avg_accuracy:.4f}\n")
    
    print(f"📋 Таблица сравнения сохранена: {table_path}")

def main():
    """Основная функция"""
    try:
        create_summary_comparison_plot()
        print("✅ Создание сводного графика завершено!")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 