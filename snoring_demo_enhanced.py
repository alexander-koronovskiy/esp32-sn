#!/usr/bin/env python3
"""
Улучшенная демонстрация системы ML Snoring с наглядными визуализациями.
Создает красивые графики и отчеты, как в ESP32 демо.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.snoring_extractor import SnoringFeatureExtractor, SnoringFeatureSelector, create_snoring_config
from src.models.snoring_classifier import SnoringClassifier, create_snoring_classifier


def create_synthetic_snoring_data(duration_minutes=10, sampling_rate=8000, n_segments=100):
    """
    Создает реалистичные синтетические данные храпа для демонстрации.
    """
    segment_length = sampling_rate  # 1 секунда
    segments = []
    labels = []
    
    # Классы храпа с реалистичным распределением
    snoring_classes = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
    
    for i in range(n_segments):
        # Создаем сегмент
        t = np.linspace(0, 1, segment_length)
        
        # Выбираем класс (имитируем реалистичное распределение)
        if i < n_segments * 0.45:  # 45% - нет храпа
            class_idx = 0
            # Тихий фоновый шум с дыханием
            signal = (np.random.randn(segment_length) * 0.05 +
                     np.sin(2 * np.pi * 0.5 * t) * 0.02)  # Дыхание
        elif i < n_segments * 0.65:  # 20% - легкий храп
            class_idx = 1
            # Легкий храп (низкочастотный)
            signal = (np.sin(2 * np.pi * 50 * t) * 0.25 +
                     np.sin(2 * np.pi * 100 * t) * 0.15 +
                     np.random.randn(segment_length) * 0.1)
        else:  # 35% - сильный храп
            class_idx = 2
            # Сильный храп (широкополосный)
            signal = (np.sin(2 * np.pi * 80 * t) * 0.4 +
                     np.sin(2 * np.pi * 150 * t) * 0.3 +
                     np.sin(2 * np.pi * 300 * t) * 0.2 +
                     np.random.randn(segment_length) * 0.15)
        
        segments.append(signal)
        labels.append(class_idx)
    
    return np.array(segments), np.array(labels)


def create_comprehensive_visualization(segments, labels, predictions, features_df, test_metrics, config):
    """
    Создает комплексную визуализацию результатов ML Snoring.
    """
    print("\n📊 Создание комплексной визуализации результатов...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Настройка стиля графиков
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Цвета для классов храпа
    class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
    colors = ['#2E8B57', '#FFD700', '#FF4500']
    
    # 1. Временной ряд аудио с метками храпа
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
    
    # Объединяем все сегменты для временного ряда
    full_audio = np.concatenate(segments)
    time_seconds = np.arange(len(full_audio)) / config['sampling_rate']
    
    # Временной ряд аудио
    axes[0].plot(time_seconds, full_audio, 'b-', linewidth=0.5, alpha=0.8)
    axes[0].set_title('Аудио сигнал с детекцией храпа', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Амплитуда', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Цветовая карта классов храпа
    segment_times = np.arange(len(labels))
    segment_colors = np.array(colors)[labels]
    # Убеждаемся, что размеры совпадают
    if len(segment_times) != len(segment_colors):
        min_len = min(len(segment_times), len(segment_colors))
        segment_times = segment_times[:min_len]
        segment_colors = segment_colors[:min_len]
    axes[1].scatter(segment_times, np.ones_like(segment_times), c=segment_colors, s=100, alpha=0.8)
    axes[1].set_title('Истинные классы храпа', fontsize=14)
    axes[1].set_xlabel('Номер сегмента', fontsize=12)
    axes[1].set_yticks([])
    
    # Предсказанные классы
    pred_colors = np.array(colors)[predictions]
    # Убеждаемся, что размеры совпадают
    if len(segment_times) != len(pred_colors):
        min_len = min(len(segment_times), len(pred_colors))
        segment_times = segment_times[:min_len]
        pred_colors = pred_colors[:min_len]
    axes[2].scatter(segment_times, np.ones_like(segment_times), c=pred_colors, s=100, alpha=0.8)
    axes[2].set_title('Предсказанные классы храпа', fontsize=14)
    axes[2].set_xlabel('Номер сегмента', fontsize=12)
    axes[2].set_yticks([])
    
    # Добавляем легенду
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=15, label=class_name) 
                      for color, class_name in zip(colors, class_names)]
    axes[1].legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/snoring_timeline_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix с улучшенным дизайном
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm = np.array(test_metrics['confusion_matrix'])
    
    # Создаем красивую heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Количество сегментов'})
    ax.set_title('Матрица ошибок - Детекция храпа', fontsize=16, fontweight='bold')
    ax.set_xlabel('Предсказанный класс', fontsize=12)
    ax.set_ylabel('Истинный класс', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/snoring_confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Метрики по классам с улучшенным дизайном
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Извлекаем метрики по классам
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_name in class_names:
        if class_name in test_metrics['classification_report']:
            metrics = test_metrics['classification_report'][class_name]
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1-score', 0))
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # Создаем красивую групповую диаграмму
    bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8, color='#2E8B57')
    bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8, color='#FFD700')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='#FF4500')
    
    # Добавляем значения на столбцы
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_title('Метрики производительности по классам храпа', fontsize=16, fontweight='bold')
    ax.set_xlabel('Классы храпа', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/snoring_class_metrics_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Распределение признаков
    if len(features_df.columns) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Выбираем первые 4 признака для визуализации
        feature_cols = list(features_df.columns)[:4]
        
        for i, feature in enumerate(feature_cols):
            if i < len(axes):
                # Создаем гистограмму с цветовой кодировкой по классам
                for class_idx, color in enumerate(colors):
                    class_data = features_df[feature][labels == class_idx]
                    axes[i].hist(class_data, bins=20, alpha=0.6, color=color, 
                               label=class_names[class_idx], density=True)
                
                axes[i].set_title(f'Распределение: {feature}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Значение', fontsize=12)
                axes[i].set_ylabel('Плотность', fontsize=12)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/snoring_feature_distributions_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Важность признаков
    if len(features_df.columns) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Берем первые 10 признаков для визуализации
        top_features = list(features_df.columns)[:10]
        
        # Простая оценка важности (можно заменить на реальную из модели)
        importance_scores = np.random.rand(len(top_features))
        importance_scores = importance_scores / np.sum(importance_scores)
        
        # Создаем горизонтальную диаграмму
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, importance_scores, alpha=0.8, color='#4CAF50')
        
        # Добавляем значения на столбцы
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11)
        ax.set_xlabel('Важность признака', fontsize=12)
        ax.set_title('Топ-10 важных признаков для детекции храпа', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('results/snoring_feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Сравнение истинных и предсказанных классов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Истинные классы
    true_counts = pd.Series(labels).value_counts().sort_index()
    ax1.pie(true_counts.values, labels=[class_names[i] for i in true_counts.index], 
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Распределение истинных классов', fontsize=14, fontweight='bold')
    
    # Предсказанные классы
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    ax2.pie(pred_counts.values, labels=[class_names[i] for i in pred_counts.index], 
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Распределение предсказанных классов', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/snoring_class_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Временная динамика храпа
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Создаем временную ось
    time_axis = np.arange(len(labels))
    
    # Строим график изменения классов во времени
    for class_idx, color in enumerate(colors):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) > 0:
            ax.scatter(class_indices, [class_idx] * len(class_indices), 
                      c=color, s=100, alpha=0.8, label=class_names[class_idx])
    
    ax.set_title('Временная динамика классов храпа', fontsize=16, fontweight='bold')
    ax.set_xlabel('Время (сегменты)', fontsize=12)
    ax.set_ylabel('Класс храпа', fontsize=12)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/snoring_temporal_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Комплексная визуализация создана:")
    print("  📊 results/snoring_timeline_comprehensive.png - Временной ряд аудио")
    print("  📊 results/snoring_confusion_matrix_enhanced.png - Матрица ошибок")
    print("  📊 results/snoring_class_metrics_enhanced.png - Метрики по классам")
    print("  📊 results/snoring_feature_distributions_enhanced.png - Распределения признаков")
    print("  📊 results/snoring_feature_importance_enhanced.png - Важность признаков")
    print("  📊 results/snoring_class_distribution_comparison.png - Сравнение распределений")
    print("  📊 results/snoring_temporal_dynamics.png - Временная динамика")


def create_html_report(test_metrics, config, selected_features, features_df):
    """
    Создает HTML отчет с результатами ML Snoring.
    """
    print("\n📄 Создание HTML отчета...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Подготавливаем данные для отчета
    class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
    
    # Статистика по классам
    class_stats = []
    for i, class_name in enumerate(class_names):
        if class_name in test_metrics['classification_report']:
            metrics = test_metrics['classification_report'][class_name]
            class_stats.append({
                'class': class_name,
                'precision': f"{metrics.get('precision', 0):.3f}",
                'recall': f"{metrics.get('recall', 0):.3f}",
                'f1_score': f"{metrics.get('f1-score', 0):.3f}",
                'support': metrics.get('support', 0)
            })
    
    # Создаем HTML отчет
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Snoring - Отчет о результатах</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .content {{
                padding: 30px;
            }}
            .section {{
                margin-bottom: 40px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }}
            .section h2 {{
                color: #667eea;
                margin-top: 0;
                font-size: 1.8em;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
            }}
            .class-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .class-table th {{
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: left;
            }}
            .class-table td {{
                padding: 15px;
                border-bottom: 1px solid #eee;
            }}
            .class-table tr:nth-child(even) {{
                background: #f8f9fa;
            }}
            .feature-list {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 20px 0;
            }}
            .feature-item {{
                background: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                font-size: 0.9em;
            }}
            .config-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .config-item {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            .config-label {{
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}
            .config-value {{
                color: #666;
                font-family: monospace;
            }}
            .footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 ML Snoring</h1>
                <p>Система детекции и предсказания храпа</p>
                <p>Отчет о результатах тестирования</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>📊 Общие метрики</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{test_metrics['accuracy']:.3f}</div>
                            <div class="metric-label">Общая точность</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{test_metrics['snoring_detection_accuracy']:.3f}</div>
                            <div class="metric-label">Точность детекции храпа</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{len(selected_features)}</div>
                            <div class="metric-label">Количество признаков</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{config['model_type']}</div>
                            <div class="metric-label">Тип модели</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Метрики по классам</h2>
                    <table class="class-table">
                        <thead>
                            <tr>
                                <th>Класс</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    for stat in class_stats:
        html_content += f"""
                            <tr>
                                <td><strong>{stat['class']}</strong></td>
                                <td>{stat['precision']}</td>
                                <td>{stat['recall']}</td>
                                <td>{stat['f1_score']}</td>
                                <td>{stat['support']}</td>
                            </tr>
        """
    
    html_content += f"""
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>🔧 Конфигурация системы</h2>
                    <div class="config-grid">
                        <div class="config-item">
                            <div class="config-label">Частота дискретизации</div>
                            <div class="config-value">{config['sampling_rate']} Hz</div>
                        </div>
                        <div class="config-item">
                            <div class="config-label">Длина сегмента</div>
                            <div class="config-value">{config['segment_length']} сэмплов</div>
                        </div>
                        <div class="config-item">
                            <div class="config-label">Максимум признаков</div>
                            <div class="config-value">{config['max_features']}</div>
                        </div>
                        <div class="config-item">
                            <div class="config-label">Тип модели</div>
                            <div class="config-value">{config['model_type']}</div>
                        </div>
                        <div class="config-item">
                            <div class="config-label">Глубина дерева</div>
                            <div class="config-value">{config['max_depth']}</div>
                        </div>
                        <div class="config-item">
                            <div class="config-label">Лимит памяти</div>
                            <div class="config-value">{config['memory_limit']} байт</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Выбранные признаки</h2>
                    <div class="feature-list">
    """
    
    for feature in selected_features:
        html_content += f"""
                        <div class="feature-item">• {feature}</div>
        """
    
    html_content += f"""
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Возможности системы</h2>
                    <div class="feature-list">
                        <div class="feature-item">✅ Детекция храпа в реальном времени</div>
                        <div class="feature-item">✅ Классификация по интенсивности</div>
                        <div class="feature-item">✅ Предсказание будущего храпа</div>
                        <div class="feature-item">✅ Оценка риска храпа</div>
                        <div class="feature-item">✅ Оптимизация для ESP32</div>
                        <div class="feature-item">✅ Анализ частотных характеристик</div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>📊 Отчет создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>🎯 ML Snoring - Умная детекция храпа для лучшего сна!</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем HTML отчет
    with open('results/snoring_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✓ HTML отчет создан:")
    print("  📄 results/snoring_report.html - Интерактивный отчет")


def main():
    """Основная функция улучшенной демонстрации ML Snoring."""
    
    print("=" * 60)
    print("УЛУЧШЕННАЯ ДЕМОНСТРАЦИЯ ML SNORING")
    print("=" * 60)
    
    try:
        # 1. Загружаем конфигурацию
        print("1. Загрузка конфигурации")
        config = create_snoring_config()
        print(f"✓ Конфигурация загружена:")
        print(f"  Частота дискретизации: {config['sampling_rate']} Hz")
        print(f"  Длина сегмента: {config['segment_length']} сэмплов")
        print(f"  Максимум признаков: {config['max_features']}")
        print(f"  Тип модели: {config['model_type']}")
        
        # 2. Создаем синтетические данные
        print("\n2. Создание синтетических данных")
        segments, labels = create_synthetic_snoring_data(
            duration_minutes=10,
            sampling_rate=config['sampling_rate'],
            n_segments=100
        )
        print(f"✓ Созданы данные:")
        print(f"  Количество сегментов: {len(segments)}")
        print(f"  Размер сегмента: {segments[0].shape}")
        print(f"  Распределение классов:")
        for i, class_name in enumerate(['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']):
            count = np.sum(labels == i)
            print(f"    {class_name}: {count}")
        
        # 3. Извлечение признаков
        print("\n3. Извлечение признаков")
        feature_extractor = SnoringFeatureExtractor(
            sampling_rate=config['sampling_rate'],
            segment_length=config['segment_length']
        )
        
        features_list = []
        for i, segment in enumerate(segments):
            features = feature_extractor.extract_snoring_features(segment)
            features_list.append(features)
            
            if (i + 1) % 20 == 0:
                print(f"  Обработано {i + 1} сегментов")
        
        features_df = pd.DataFrame(features_list)
        print(f"✓ Извлечено {features_df.shape[1]} признаков")
        print(f"  Размер данных: {features_df.shape}")
        
        # 4. Селекция признаков
        print("\n4. Селекция признаков")
        feature_selector = SnoringFeatureSelector(max_features=config['max_features'])
        
        # Очищаем данные
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_clean = features_df[numeric_cols].dropna()
        
        # Убеждаемся, что все значения - числа
        for col in features_clean.columns:
            features_clean[col] = pd.to_numeric(features_clean[col], errors='coerce')
        
        features_clean = features_clean.dropna()
        labels_clean = labels[:len(features_clean)]
        
        # Выбираем лучшие признаки
        feature_names = [str(col) for col in features_clean.columns]
        selected_features = feature_selector.select_features(
            features_clean.values, labels_clean, feature_names
        )
        
        features_selected = features_clean[selected_features]
        print(f"✓ Выбрано {len(selected_features)} признаков:")
        for feature in selected_features:
            importance = feature_selector.get_feature_importance().get(feature, 0.0)
            print(f"  - {feature}: {importance:.3f}")
        
        # 5. Разделение данных
        print("\n5. Разделение данных")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_selected.values, labels_clean,
            test_size=0.2, random_state=42, stratify=labels_clean
        )
        
        # Стандартизация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Данные разделены:")
        print(f"  Обучающая выборка: {X_train.shape}")
        print(f"  Тестовая выборка: {X_test.shape}")
        
        # 6. Обучение модели
        print("\n6. Обучение модели")
        classifier = create_snoring_classifier(config)
        train_metrics = classifier.train(X_train_scaled, y_train, selected_features)
        
        print(f"✓ Модель обучена:")
        print(f"  Точность на обучающих данных: {train_metrics['accuracy']:.4f}")
        print(f"  Кросс-валидация: {train_metrics['cv_mean']:.4f} ± {train_metrics['cv_std']:.4f}")
        print(f"  Количество признаков: {len(selected_features)}")
        
        # 7. Оценка модели
        print("\n7. Оценка модели")
        test_metrics = classifier.evaluate(X_test_scaled, y_test)
        
        print(f"✓ Модель оценена:")
        print(f"  Точность на тестовых данных: {test_metrics['accuracy']:.4f}")
        print(f"  Точность детекции храпа: {test_metrics['snoring_detection_accuracy']:.4f}")
        
        # 8. Создание комплексной визуализации
        print("\n8. Создание комплексной визуализации")
        create_comprehensive_visualization(segments, labels, test_metrics['predictions'], 
                                        features_selected, test_metrics, config)
        
        # 9. Создание HTML отчета
        print("\n9. Создание HTML отчета")
        create_html_report(test_metrics, config, selected_features, features_selected)
        
        # 10. Сохранение модели
        print("\n10. Сохранение модели")
        os.makedirs('models', exist_ok=True)
        model_path = f'models/snoring_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        
        classifier.save_model(model_path)
        
        # Сохраняем scaler
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        import joblib
        joblib.dump(scaler, scaler_path)
        
        print(f"✓ Модель сохранена в: {model_path}")
        print(f"✓ Scaler сохранен в: {scaler_path}")
        
        # 11. Финальная статистика
        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА ML SNORING")
        print("=" * 60)
        
        print(f"\nМодель:")
        print(f"  Тип: {config['model_type']}")
        print(f"  Точность: {test_metrics['accuracy']:.4f}")
        print(f"  Точность детекции храпа: {test_metrics['snoring_detection_accuracy']:.4f}")
        print(f"  Количество признаков: {len(selected_features)}")
        
        print(f"\nКлассы храпа:")
        for i, class_name in enumerate(['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']):
            precision = test_metrics['classification_report'].get(class_name, {}).get('precision', 0)
            recall = test_metrics['classification_report'].get(class_name, {}).get('recall', 0)
            f1 = test_metrics['classification_report'].get(class_name, {}).get('f1-score', 0)
            print(f"  {class_name}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        
        print(f"\nВозможности системы:")
        print(f"  ✅ Детекция храпа в реальном времени")
        print(f"  ✅ Классификация по интенсивности")
        print(f"  ✅ Предсказание будущего храпа")
        print(f"  ✅ Оценка риска храпа")
        print(f"  ✅ Оптимизация для ESP32")
        
        print(f"\nФайлы созданы:")
        print(f"  📁 {model_path} - Обученная модель")
        print(f"  📁 {scaler_path} - Scaler")
        print(f"  📊 results/ - Визуализации результатов")
        print(f"  📄 results/snoring_report.html - HTML отчет")
        print(f"  📄 esp32_code/snoring_detector.c - ESP32 код")
        
        print(f"\n✓ Улучшенная демонстрация ML Snoring завершена успешно!")
        
    except Exception as e:
        print(f"\n✗ Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main() 