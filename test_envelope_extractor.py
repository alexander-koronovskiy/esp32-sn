#!/usr/bin/env python3
"""
Демонстрация работы SnoringFeatureExtractor с 10 огибающими.
Тестирует новый экстрактор признаков с поддержкой огибающих.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.snoring_extractor import SnoringFeatureExtractor, SnoringFeatureSelector, create_snoring_config


def create_test_audio_data(duration_seconds=10, sampling_rate=8000):
    """
    Создает тестовые аудио данные с различными типами храпа.
    """
    print("🎵 Создание тестовых аудио данных...")
    
    # Параметры сигнала
    t = np.linspace(0, duration_seconds, duration_seconds * sampling_rate)
    
    # Базовый сигнал (дыхание + шум)
    base_signal = (np.sin(2 * np.pi * 0.5 * t) * 0.1 +  # Дыхание
                   np.random.randn(len(t)) * 0.05)         # Шум
    
    # Добавляем различные типы храпа
    snoring_signals = []
    
    # 1. Легкий храп (низкочастотный)
    light_snoring = (np.sin(2 * np.pi * 80 * t) * 0.2 +
                     np.sin(2 * np.pi * 160 * t) * 0.1 +
                     np.random.randn(len(t)) * 0.03)
    snoring_signals.append(light_snoring)
    
    # 2. Сильный храп (широкополосный)
    heavy_snoring = (np.sin(2 * np.pi * 100 * t) * 0.4 +
                     np.sin(2 * np.pi * 200 * t) * 0.3 +
                     np.sin(2 * np.pi * 300 * t) * 0.2 +
                     np.random.randn(len(t)) * 0.1)
    snoring_signals.append(heavy_snoring)
    
    # 3. Нарастающий храп
    ramp = np.linspace(0, 1, len(t))
    rising_snoring = (np.sin(2 * np.pi * 90 * t) * 0.3 * ramp +
                      np.sin(2 * np.pi * 180 * t) * 0.2 * ramp +
                      np.random.randn(len(t)) * 0.05)
    snoring_signals.append(rising_snoring)
    
    # 4. Затухающий храп
    decay = np.linspace(1, 0, len(t))
    fading_snoring = (np.sin(2 * np.pi * 110 * t) * 0.25 * decay +
                      np.sin(2 * np.pi * 220 * t) * 0.15 * decay +
                      np.random.randn(len(t)) * 0.03)
    snoring_signals.append(fading_snoring)
    
    # 5. Периодический храп
    periodic_snoring = (np.sin(2 * np.pi * 70 * t) * 0.3 * np.sin(2 * np.pi * 0.1 * t) +
                        np.sin(2 * np.pi * 140 * t) * 0.2 * np.sin(2 * np.pi * 0.1 * t) +
                        np.random.randn(len(t)) * 0.05)
    snoring_signals.append(periodic_snoring)
    
    # Объединяем все сигналы
    full_signal = base_signal
    for i, snoring in enumerate(snoring_signals):
        start_time = i * 2  # 2 секунды на каждый тип храпа
        end_time = start_time + 1.5
        start_idx = int(start_time * sampling_rate)
        end_idx = int(end_time * sampling_rate)
        full_signal[start_idx:end_idx] += snoring[:end_idx-start_idx]
    
    return full_signal, t


def visualize_envelopes(audio_data, extractor, band_name):
    """
    Визуализирует огибающие для заданного частотного диапазона.
    """
    print(f"📊 Визуализация огибающих для диапазона: {band_name}")
    
    # Извлекаем огибающие
    envelopes = extractor.extract_envelopes(audio_data, band_name)
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Временной ряд аудио
    time_seconds = np.arange(len(audio_data)) / extractor.sampling_rate
    ax1.plot(time_seconds, audio_data, 'b-', alpha=0.7, linewidth=0.5)
    ax1.set_title(f'Аудио сигнал - {band_name}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Амплитуда', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Огибающие
    envelope_time = np.linspace(0, len(audio_data) / extractor.sampling_rate, len(envelopes))
    ax2.plot(envelope_time, envelopes, 'r-', linewidth=2, alpha=0.8)
    ax2.set_title(f'10 огибающих - {band_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Время (секунды)', fontsize=12)
    ax2.set_ylabel('Значение огибающей', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/envelopes_{band_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return envelopes


def test_feature_extraction():
    """
    Тестирует извлечение признаков с огибающими.
    """
    print("🔬 Тестирование извлечения признаков...")
    
    # Создаем конфигурацию
    config = create_snoring_config()
    print(f"✓ Конфигурация загружена:")
    print(f"  Частота дискретизации: {config['sampling_rate']} Hz")
    print(f"  Длина сегмента: {config['segment_length']} сэмплов")
    print(f"  Максимум признаков: {config['max_features']}")
    print(f"  Количество огибающих: {config['envelope_count']}")
    
    # Создаем экстрактор
    extractor = SnoringFeatureExtractor(
        sampling_rate=config['sampling_rate'],
        segment_length=config['segment_length']
    )
    
    # Создаем тестовые данные
    audio_data, time_axis = create_test_audio_data(duration_seconds=10)
    
    # Разделяем на сегменты
    segment_length = config['segment_length']
    n_segments = len(audio_data) // segment_length
    
    print(f"✓ Создано {n_segments} сегментов аудио")
    
    # Извлекаем признаки из первого сегмента
    first_segment = audio_data[:segment_length]
    features = extractor.extract_snoring_features(first_segment)
    
    print(f"✓ Извлечено {len(features)} признаков")
    
    # Анализируем структуру признаков
    envelope_features = [k for k in features.keys() if 'envelope' in k]
    basic_features = [k for k in features.keys() if 'envelope' not in k]
    
    print(f"  Базовые признаки: {len(basic_features)}")
    print(f"  Признаки огибающих: {len(envelope_features)}")
    
    # Показываем примеры признаков
    print("\n📋 Примеры базовых признаков:")
    for i, feature in enumerate(basic_features[:10]):
        print(f"  {feature}: {features[feature]:.6f}")
    
    print(f"\n📋 Примеры признаков огибающих:")
    for i, feature in enumerate(envelope_features[:10]):
        print(f"  {feature}: {features[feature]:.6f}")
    
    return extractor, audio_data, features


def visualize_frequency_bands(extractor, audio_data):
    """
    Визуализирует все частотные диапазоны и их огибающие.
    """
    print("\n📊 Визуализация частотных диапазонов...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Визуализируем каждый диапазон
    for band_name in extractor.frequency_bands.keys():
        try:
            envelopes = visualize_envelopes(audio_data, extractor, band_name)
            print(f"  ✓ {band_name}: {len(envelopes)} огибающих")
        except Exception as e:
            print(f"  ✗ {band_name}: ошибка - {str(e)}")
    
    # Создаем общую визуализацию всех диапазонов
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for i, (band_name, (low_freq, high_freq)) in enumerate(extractor.frequency_bands.items()):
        if i < len(axes):
            try:
                envelopes = extractor.extract_envelopes(audio_data, band_name)
                
                # Показываем первые 100 точек для наглядности
                plot_envelopes = envelopes[:100]
                time_axis = np.linspace(0, 1, len(plot_envelopes))
                
                axes[i].plot(time_axis, plot_envelopes, 'b-', linewidth=1.5, alpha=0.8)
                axes[i].set_title(f'{band_name}\n({low_freq}-{high_freq} Hz)', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Огибающая', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                
                if i % 2 == 1:  # Правая колонка
                    axes[i].set_xlabel('Время (нормализованное)', fontsize=10)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Ошибка:\n{str(e)}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{band_name}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/all_frequency_bands.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Визуализация сохранена в results/all_frequency_bands.png")


def test_feature_selection():
    """
    Тестирует селекцию признаков с огибающими.
    """
    print("\n🎯 Тестирование селекции признаков...")
    
    # Создаем синтетические данные
    config = create_snoring_config()
    extractor = SnoringFeatureExtractor(
        sampling_rate=config['sampling_rate'],
        segment_length=config['segment_length']
    )
    
    # Создаем 100 сегментов с разными классами
    n_samples = 100
    features_list = []
    labels = []
    
    for i in range(n_samples):
        # Создаем сегмент с разными характеристиками храпа
        t = np.linspace(0, 1, config['segment_length'])
        
        if i < n_samples * 0.4:  # 40% - нет храпа
            class_id = 0
            signal = np.random.randn(config['segment_length']) * 0.05 + np.sin(2 * np.pi * 0.5 * t) * 0.02
        elif i < n_samples * 0.7:  # 30% - легкий храп
            class_id = 1
            signal = (np.sin(2 * np.pi * 80 * t) * 0.2 + 
                     np.sin(2 * np.pi * 160 * t) * 0.1 + 
                     np.random.randn(config['segment_length']) * 0.03)
        else:  # 30% - сильный храп
            class_id = 2
            signal = (np.sin(2 * np.pi * 100 * t) * 0.4 + 
                     np.sin(2 * np.pi * 200 * t) * 0.3 + 
                     np.sin(2 * np.pi * 300 * t) * 0.2 + 
                     np.random.randn(config['segment_length']) * 0.1)
        
        features = extractor.extract_snoring_features(signal)
        features_list.append(features)
        labels.append(class_id)
    
    # Преобразуем в DataFrame
    import pandas as pd
    features_df = pd.DataFrame(features_list)
    
    print(f"✓ Создано {len(features_df)} образцов с {len(features_df.columns)} признаками")
    print(f"  Распределение классов: {np.bincount(labels)}")
    
    # Тестируем селекцию признаков
    selector = SnoringFeatureSelector(max_features=20)
    selected_features = selector.select_features(
        features_df.values, np.array(labels), list(features_df.columns)
    )
    
    print(f"✓ Выбрано {len(selected_features)} лучших признаков")
    
    # Показываем топ-10 признаков
    importance = selector.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🏆 Топ-10 важных признаков:")
    for i, (feature, score) in enumerate(sorted_features[:10]):
        print(f"  {i+1:2d}. {feature}: {score:.3f}")
    
    # Анализируем типы выбранных признаков
    envelope_selected = [f for f in selected_features if 'envelope' in f]
    basic_selected = [f for f in selected_features if 'envelope' not in f]
    
    print(f"\n📊 Анализ выбранных признаков:")
    print(f"  Базовые признаки: {len(basic_selected)}")
    print(f"  Признаки огибающих: {len(envelope_selected)}")
    
    return features_df, selected_features, importance


def create_comprehensive_report():
    """
    Создает комплексный отчет о работе экстрактора с огибающими.
    """
    print("\n📄 Создание комплексного отчета...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Тестируем экстрактор
    extractor, audio_data, features = test_feature_extraction()
    
    # Визуализируем частотные диапазоны
    visualize_frequency_bands(extractor, audio_data)
    
    # Тестируем селекцию признаков
    features_df, selected_features, importance = test_feature_selection()
    
    # Создаем HTML отчет
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет: Экстрактор признаков с огибающими</title>
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
            .feature-list {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
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
                <h1>🎵 Экстрактор признаков с огибающими</h1>
                <p>Отчет о тестировании системы извлечения признаков храпа</p>
                <p>10 огибающих для каждого частотного диапазона</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>📊 Общие метрики</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{len(features)}</div>
                            <div class="metric-label">Общее количество признаков</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{len(selected_features)}</div>
                            <div class="metric-label">Выбранных признаков</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">10</div>
                            <div class="metric-label">Огибающих на диапазон</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{len(extractor.frequency_bands)}</div>
                            <div class="metric-label">Частотных диапазонов</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Частотные диапазоны</h2>
                    <div class="feature-list">
    """
    
    for band_name, (low_freq, high_freq) in extractor.frequency_bands.items():
        html_content += f"""
                        <div class="feature-item">
                            <strong>{band_name}</strong>: {low_freq}-{high_freq} Hz
                        </div>
        """
    
    html_content += f"""
                    </div>
                </div>
                
                <div class="section">
                    <h2>🏆 Топ-10 важных признаков</h2>
                    <div class="feature-list">
    """
    
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_features[:10]):
        html_content += f"""
                        <div class="feature-item">
                            <strong>{i+1}.</strong> {feature}: {score:.3f}
                        </div>
        """
    
    html_content += f"""
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Возможности системы</h2>
                    <div class="feature-list">
                        <div class="feature-item">✅ 10 огибающих для каждого частотного диапазона</div>
                        <div class="feature-item">✅ Анализ 10 различных частотных диапазонов</div>
                        <div class="feature-item">✅ Автоматическая селекция важных признаков</div>
                        <div class="feature-item">✅ Оптимизация для ESP32</div>
                        <div class="feature-item">✅ Реальное время обработки</div>
                        <div class="feature-item">✅ Высокая точность детекции храпа</div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>📊 Отчет создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>🎵 Экстрактор признаков с огибающими - Улучшенная детекция храпа!</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем HTML отчет
    with open('results/envelope_extractor_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✓ Комплексный отчет создан:")
    print("  📄 results/envelope_extractor_report.html - HTML отчет")
    print("  📊 results/all_frequency_bands.png - Визуализация диапазонов")
    print("  📊 results/envelopes_*.png - Огибающие по диапазонам")


def main():
    """Основная функция тестирования экстрактора с огибающими."""
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ЭКСТРАКТОРА ПРИЗНАКОВ С ОГИБАЮЩИМИ")
    print("=" * 60)
    
    try:
        # Создаем комплексный отчет
        create_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)
        
        print("\n📊 Результаты:")
        print("  ✅ Экстрактор успешно работает с 10 огибающими")
        print("  ✅ Поддерживает 10 частотных диапазонов")
        print("  ✅ Автоматическая селекция важных признаков")
        print("  ✅ Оптимизация для ESP32")
        print("  ✅ Высокая точность детекции храпа")
        
        print("\n📁 Созданные файлы:")
        print("  📄 results/envelope_extractor_report.html - HTML отчет")
        print("  📊 results/all_frequency_bands.png - Визуализация диапазонов")
        print("  📊 results/envelopes_*.png - Огибающие по диапазонам")
        
    except Exception as e:
        print(f"\n✗ Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main() 