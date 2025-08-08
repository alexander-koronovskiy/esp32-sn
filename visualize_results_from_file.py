#!/usr/bin/env python3
"""
Визуализация результатов из results.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Импортируем данные из results.py
try:
    from results import (
        MODEL_STATISTICS, 
        MODEL_DETAILS, 
        PERFORMANCE_DATA, 
        CLASS_COLORS,
        SNORING_CLASSES,
        get_overall_performance,
        get_best_performing_class,
        get_worst_performing_class
    )
except ImportError:
    print("❌ Файл results.py не найден. Сначала создайте его.")
    exit(1)

def create_comprehensive_visualization():
    """Создает комплексную визуализацию результатов"""
    print("🎨 Создание визуализации результатов...")
    
    # Создаем фигуру с подграфиками
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Статистика моделей
    ax1 = plt.subplot(3, 3, 1)
    model_stats = ['Всего', 'Успешных', 'С ошибками']
    model_values = [
        MODEL_STATISTICS['total_models'],
        MODEL_STATISTICS['successful_models'],
        MODEL_STATISTICS['failed_models']
    ]
    colors = ['#2E8B57', '#FFD700', '#FF4500']
    bars1 = ax1.bar(model_stats, model_values, color=colors, alpha=0.7)
    ax1.set_title('Статистика моделей', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Количество')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars1, model_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom')
    
    # 2. Производительность по классам
    ax2 = plt.subplot(3, 3, 2)
    classes = list(PERFORMANCE_DATA['accuracy_scores'].keys())
    accuracies = list(PERFORMANCE_DATA['accuracy_scores'].values())
    colors_list = [CLASS_COLORS[cls] for cls in classes]
    
    bars2 = ax2.bar(classes, accuracies, color=colors_list, alpha=0.7)
    ax2.set_title('Точность по классам', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Точность')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Добавляем значения на столбцы
    for bar, accuracy in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    # 3. Сравнение метрик
    ax3 = plt.subplot(3, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    avg_metrics = [
        np.mean(list(PERFORMANCE_DATA['accuracy_scores'].values())),
        np.mean(list(PERFORMANCE_DATA['precision_scores'].values())),
        np.mean(list(PERFORMANCE_DATA['recall_scores'].values())),
        np.mean(list(PERFORMANCE_DATA['f1_scores'].values()))
    ]
    
    bars3 = ax3.bar(metrics, avg_metrics, color='skyblue', alpha=0.7)
    ax3.set_title('Средние метрики', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Значение')
    ax3.set_ylim(0, 1)
    
    # Добавляем значения на столбцы
    for bar, metric in zip(bars3, avg_metrics):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{metric:.3f}', ha='center', va='bottom')
    
    # 4. Эволюция моделей во времени
    ax4 = plt.subplot(3, 3, 4)
    dates = [model['date'] for model in MODEL_DETAILS]
    feature_counts = [model['feature_count'] for model in MODEL_DETAILS]
    
    ax4.plot(range(len(dates)), feature_counts, 'o-', color='#9370DB', linewidth=2, markersize=8)
    ax4.set_title('Эволюция признаков', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Модель')
    ax4.set_ylabel('Количество признаков')
    ax4.grid(True, alpha=0.3)
    
    # 5. Распределение типов моделей
    ax5 = plt.subplot(3, 3, 5)
    model_types = [model['model_type'] for model in MODEL_DETAILS]
    type_counts = pd.Series(model_types).value_counts()
    
    ax5.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
            colors=['#FFD700', '#FF4500', '#2E8B57'])
    ax5.set_title('Распределение типов моделей', fontsize=14, fontweight='bold')
    
    # 6. Тепловая карта производительности
    ax6 = plt.subplot(3, 3, 6)
    performance_matrix = np.array([
        list(PERFORMANCE_DATA['accuracy_scores'].values()),
        list(PERFORMANCE_DATA['precision_scores'].values()),
        list(PERFORMANCE_DATA['recall_scores'].values()),
        list(PERFORMANCE_DATA['f1_scores'].values())
    ])
    
    sns.heatmap(performance_matrix, 
                xticklabels=classes,
                yticklabels=['Accuracy', 'Precision', 'Recall', 'F1'],
                annot=True, fmt='.3f', cmap='Blues', ax=ax6)
    ax6.set_title('Матрица производительности', fontsize=14, fontweight='bold')
    
    # 7. Сравнение лучшего и худшего классов
    ax7 = plt.subplot(3, 3, 7)
    best_class, best_acc = get_best_performing_class()
    worst_class, worst_acc = get_worst_performing_class()
    
    comparison_classes = [best_class, worst_class]
    comparison_accuracies = [best_acc, worst_acc]
    comparison_colors = [CLASS_COLORS[best_class], CLASS_COLORS[worst_class]]
    
    bars7 = ax7.bar(comparison_classes, comparison_accuracies, color=comparison_colors, alpha=0.7)
    ax7.set_title('Лучший vs Худший класс', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Точность')
    ax7.set_ylim(0, 1)
    
    # Добавляем значения на столбцы
    for bar, accuracy in zip(bars7, comparison_accuracies):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    # 8. Статистика по признакам
    ax8 = plt.subplot(3, 3, 8)
    feature_stats = {
        'Среднее количество': MODEL_STATISTICS['avg_features'],
        'Минимальное': min([m['feature_count'] for m in MODEL_DETAILS]),
        'Максимальное': max([m['feature_count'] for m in MODEL_DETAILS])
    }
    
    bars8 = ax8.bar(feature_stats.keys(), feature_stats.values(), color='lightcoral', alpha=0.7)
    ax8.set_title('Статистика признаков', fontsize=14, fontweight='bold')
    ax8.set_ylabel('Количество')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars8, feature_stats.values()):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom')
    
    # 9. Общая сводка
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    overall_perf = get_overall_performance()
    summary_text = f"""
ОБЩАЯ СВОДКА

📊 Модели:
• Всего: {MODEL_STATISTICS['total_models']}
• Успешных: {MODEL_STATISTICS['successful_models']}
• С ошибками: {MODEL_STATISTICS['failed_models']}

📈 Производительность:
• Средняя точность: {overall_perf['avg_accuracy']:.3f}
• Средняя precision: {overall_perf['avg_precision']:.3f}
• Средний recall: {overall_perf['avg_recall']:.3f}
• Средний F1: {overall_perf['avg_f1']:.3f}

🏆 Классы:
• Лучший: {best_class} ({best_acc:.3f})
• Худший: {worst_class} ({worst_acc:.3f})
"""
    
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results_comprehensive_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_charts():
    """Создает детальные графики производительности"""
    print("📊 Создание графиков производительности...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Сравнение всех метрик по классам
    classes = list(PERFORMANCE_DATA['accuracy_scores'].keys())
    x = np.arange(len(classes))
    width = 0.2
    
    metrics_data = {
        'Accuracy': list(PERFORMANCE_DATA['accuracy_scores'].values()),
        'Precision': list(PERFORMANCE_DATA['precision_scores'].values()),
        'Recall': list(PERFORMANCE_DATA['recall_scores'].values()),
        'F1-Score': list(PERFORMANCE_DATA['f1_scores'].values())
    }
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax1.bar(x + i*width, values, width, label=metric, alpha=0.7)
    
    ax1.set_title('Сравнение метрик по классам', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Классы')
    ax1.set_ylabel('Значение')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(classes, rotation=45)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 2. Радарная диаграмма производительности
    ax2 = plt.subplot(2, 2, 2, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    # Берем accuracy для радарной диаграммы
    values = list(PERFORMANCE_DATA['accuracy_scores'].values()) + [PERFORMANCE_DATA['accuracy_scores'][classes[0]]]
    
    ax2.plot(angles, values, 'o-', linewidth=2, color='#FF4500')
    ax2.fill(angles, values, alpha=0.25, color='#FF4500')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(classes)
    ax2.set_ylim(0, 1)
    ax2.set_title('Радарная диаграмма точности', fontsize=14, fontweight='bold')
    
    # 3. Тепловая карта корреляций
    ax3 = plt.subplot(2, 2, 3)
    
    # Создаем матрицу корреляций между метриками
    corr_matrix = np.corrcoef([
        list(PERFORMANCE_DATA['accuracy_scores'].values()),
        list(PERFORMANCE_DATA['precision_scores'].values()),
        list(PERFORMANCE_DATA['recall_scores'].values()),
        list(PERFORMANCE_DATA['f1_scores'].values())
    ])
    
    sns.heatmap(corr_matrix, 
                xticklabels=['Accuracy', 'Precision', 'Recall', 'F1'],
                yticklabels=['Accuracy', 'Precision', 'Recall', 'F1'],
                annot=True, fmt='.3f', cmap='Reds', ax=ax3)
    ax3.set_title('Корреляция метрик', fontsize=14, fontweight='bold')
    
    # 4. Распределение производительности
    ax4 = plt.subplot(2, 2, 4)
    
    all_scores = []
    for metric in ['accuracy_scores', 'precision_scores', 'recall_scores', 'f1_scores']:
        all_scores.extend(list(PERFORMANCE_DATA[metric].values()))
    
    ax4.hist(all_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_title('Распределение всех метрик', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Значение метрики')
    ax4.set_ylabel('Частота')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_performance_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("🚀 Визуализация результатов из results.py...")
    
    # Создаем комплексную визуализацию
    create_comprehensive_visualization()
    
    # Создаем графики производительности
    create_performance_charts()
    
    print("✅ Визуализация завершена!")
    print("📁 Результаты сохранены в:")
    print("   • results_comprehensive_visualization.png")
    print("   • results_performance_charts.png")

if __name__ == "__main__":
    main() 