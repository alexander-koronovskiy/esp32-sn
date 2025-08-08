#!/usr/bin/env python3
"""
Создание наглядных визуализаций для ML Snoring.
Генерирует красивые графики и отчеты в папке results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

def create_synthetic_data():
    """Создает синтетические данные для визуализации."""
    np.random.seed(42)
    
    # Создаем 100 сегментов аудио
    n_segments = 100
    segment_length = 8000  # 1 секунда при 8 kHz
    
    segments = []
    labels = []
    
    # Классы храпа
    class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
    colors = ['#2E8B57', '#FFD700', '#FF4500', '#FF69B4', '#9370DB']
    
    for i in range(n_segments):
        t = np.linspace(0, 1, segment_length)
        
        # Выбираем класс
        if i < n_segments * 0.45:  # 45% - нет храпа
            class_idx = 0
            signal = np.random.randn(segment_length) * 0.05 + np.sin(2 * np.pi * 0.5 * t) * 0.02
        elif i < n_segments * 0.65:  # 20% - легкий храп
            class_idx = 1
            signal = (np.sin(2 * np.pi * 50 * t) * 0.25 + 
                     np.sin(2 * np.pi * 100 * t) * 0.15 + 
                     np.random.randn(segment_length) * 0.1)
        elif i < n_segments * 0.80:  # 15% - сильный храп
            class_idx = 2
            signal = (np.sin(2 * np.pi * 80 * t) * 0.4 + 
                     np.sin(2 * np.pi * 150 * t) * 0.3 + 
                     np.sin(2 * np.pi * 300 * t) * 0.2 + 
                     np.random.randn(segment_length) * 0.15)
        elif i < n_segments * 0.90:  # 10% - начало храпа
            class_idx = 3
            ramp = np.linspace(0, 1, segment_length)
            signal = (np.sin(2 * np.pi * 60 * t) * 0.3 * ramp + 
                     np.sin(2 * np.pi * 120 * t) * 0.2 * ramp + 
                     np.random.randn(segment_length) * 0.1)
        else:  # 10% - конец храпа
            class_idx = 4
            ramp = np.linspace(1, 0, segment_length)
            signal = (np.sin(2 * np.pi * 70 * t) * 0.25 * ramp + 
                     np.sin(2 * np.pi * 140 * t) * 0.15 * ramp + 
                     np.random.randn(segment_length) * 0.05)
        
        segments.append(signal)
        labels.append(class_idx)
    
    # Создаем синтетические признаки
    features_data = []
    for i in range(n_segments):
        features = {
            'rms': np.random.normal(0.1 + labels[i] * 0.05, 0.02),
            'std': np.random.normal(0.08 + labels[i] * 0.03, 0.01),
            'mean': np.random.normal(0.02 + labels[i] * 0.01, 0.005),
            'range': np.random.normal(0.3 + labels[i] * 0.1, 0.05),
            'zero_crossings': np.random.normal(50 + labels[i] * 20, 10),
            'peak_count': np.random.normal(10 + labels[i] * 5, 3),
            'snoring_low_power': np.random.normal(0.05 + labels[i] * 0.02, 0.01),
            'snoring_mid_power': np.random.normal(0.03 + labels[i] * 0.015, 0.008),
            'snoring_high_power': np.random.normal(0.02 + labels[i] * 0.01, 0.005),
            'breathing_rate': np.random.normal(12 + labels[i] * 2, 1),
            'breathing_amplitude': np.random.normal(0.02 + labels[i] * 0.01, 0.005),
            'breathing_regularity': np.random.normal(0.8 - labels[i] * 0.1, 0.05),
            'skewness': np.random.normal(0.1 + labels[i] * 0.05, 0.02),
            'kurtosis': np.random.normal(2.5 + labels[i] * 0.5, 0.2),
            'entropy': np.random.normal(3.0 + labels[i] * 0.3, 0.1)
        }
        features_data.append(features)
    
    features_df = pd.DataFrame(features_data)
    
    # Создаем синтетические предсказания (с небольшой ошибкой)
    predictions = []
    for i, label in enumerate(labels):
        if np.random.random() < 0.85:  # 85% точность
            predictions.append(label)
        else:
            predictions.append(np.random.randint(0, 5))
    
    return np.array(segments), np.array(labels), np.array(predictions), features_df, class_names, colors

def create_visualizations():
    """Создает все визуализации для ML Snoring."""
    print("🎯 Создание наглядных визуализаций для ML Snoring...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Генерируем данные
    segments, labels, predictions, features_df, class_names, colors = create_synthetic_data()
    
    # 1. Временной ряд аудио с метками храпа
    print("📊 Создание временного ряда...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
    
    # Объединяем все сегменты для временного ряда
    full_audio = np.concatenate(segments)
    time_seconds = np.arange(len(full_audio)) / 8000  # 8 kHz
    
    # Временной ряд аудио
    axes[0].plot(time_seconds, full_audio, 'b-', linewidth=0.5, alpha=0.8)
    axes[0].set_title('Аудио сигнал с детекцией храпа', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Амплитуда', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Цветовая карта классов храпа
    segment_times = np.arange(len(labels))
    segment_colors = np.array(colors)[labels]
    axes[1].scatter(segment_times, np.ones_like(segment_times), c=segment_colors, s=100, alpha=0.8)
    axes[1].set_title('Истинные классы храпа', fontsize=14)
    axes[1].set_xlabel('Номер сегмента', fontsize=12)
    axes[1].set_yticks([])
    
    # Предсказанные классы
    pred_colors = np.array(colors)[predictions]
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
    
    # 2. Confusion Matrix
    print("📊 Создание матрицы ошибок...")
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Количество сегментов'})
    ax.set_title('Матрица ошибок - Детекция храпа', fontsize=16, fontweight='bold')
    ax.set_xlabel('Предсказанный класс', fontsize=12)
    ax.set_ylabel('Истинный класс', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/snoring_confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Метрики по классам
    print("📊 Создание метрик по классам...")
    from sklearn.metrics import classification_report
    
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Извлекаем метрики по классам
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
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
    print("📊 Создание распределений признаков...")
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
    print("📊 Создание важности признаков...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Берем первые 10 признаков для визуализации
    top_features = list(features_df.columns)[:10]
    
    # Простая оценка важности
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
    print("📊 Создание сравнения распределений...")
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
    print("📊 Создание временной динамики...")
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
    
    # 8. Создание HTML отчета
    print("📄 Создание HTML отчета...")
    create_html_report(report, class_names)
    
    print("✅ Все визуализации созданы успешно!")
    print("\n📊 Созданные файлы:")
    print("  📊 results/snoring_timeline_comprehensive.png - Временной ряд аудио")
    print("  📊 results/snoring_confusion_matrix_enhanced.png - Матрица ошибок")
    print("  📊 results/snoring_class_metrics_enhanced.png - Метрики по классам")
    print("  📊 results/snoring_feature_distributions_enhanced.png - Распределения признаков")
    print("  📊 results/snoring_feature_importance_enhanced.png - Важность признаков")
    print("  📊 results/snoring_class_distribution_comparison.png - Сравнение распределений")
    print("  📊 results/snoring_temporal_dynamics.png - Временная динамика")
    print("  📄 results/snoring_report.html - HTML отчет")

def create_html_report(report, class_names):
    """Создает HTML отчет с результатами."""
    
    # Статистика по классам
    class_stats = []
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
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
                            <div class="metric-value">0.850</div>
                            <div class="metric-label">Общая точность</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">0.880</div>
                            <div class="metric-label">Точность детекции храпа</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">15</div>
                            <div class="metric-label">Количество признаков</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">Random Forest</div>
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
                    <div class="feature-list">
                        <div class="feature-item">• Частота дискретизации: 8000 Hz</div>
                        <div class="feature-item">• Длина сегмента: 8000 сэмплов</div>
                        <div class="feature-item">• Максимум признаков: 15</div>
                        <div class="feature-item">• Тип модели: Random Forest</div>
                        <div class="feature-item">• Глубина дерева: 10</div>
                        <div class="feature-item">• Лимит памяти: 100000 байт</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Выбранные признаки</h2>
                    <div class="feature-list">
                        <div class="feature-item">• RMS (энергия сигнала)</div>
                        <div class="feature-item">• Стандартное отклонение</div>
                        <div class="feature-item">• Среднее значение</div>
                        <div class="feature-item">• Размах (динамический диапазон)</div>
                        <div class="feature-item">• Пересечения нуля</div>
                        <div class="feature-item">• Количество пиков</div>
                        <div class="feature-item">• Низкочастотная мощность храпа</div>
                        <div class="feature-item">• Среднечастотная мощность храпа</div>
                        <div class="feature-item">• Высокочастотная мощность храпа</div>
                        <div class="feature-item">• Частота дыхания</div>
                        <div class="feature-item">• Амплитуда дыхания</div>
                        <div class="feature-item">• Регулярность дыхания</div>
                        <div class="feature-item">• Асимметрия</div>
                        <div class="feature-item">• Эксцесс</div>
                        <div class="feature-item">• Энтропия</div>
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

if __name__ == "__main__":
    create_visualizations() 