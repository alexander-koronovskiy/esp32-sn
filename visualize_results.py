#!/usr/bin/env python3
"""
Визуализация результатов работы моделей детекции и предсказания храпа
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_test_data():
    """Генерирует тестовые данные для визуализации"""
    print("📊 Генерация тестовых данных...")
    
    # Симулируем 24 часа сна (1440 минут)
    timestamps = pd.date_range(start='2024-01-01 22:00', periods=1440, freq='1min')
    
    # Генерируем реалистичные данные храпа
    np.random.seed(42)
    
    # Базовый уровень храпа (ночью больше)
    base_snoring = np.zeros(1440)
    for i in range(1440):
        hour = timestamps[i].hour
        if 23 <= hour or hour <= 6:  # Ночь
            base_snoring[i] = np.random.normal(0.3, 0.1)
        else:
            base_snoring[i] = np.random.normal(0.05, 0.02)
    
    # Добавляем эпизоды храпа
    snoring_episodes = []
    for _ in range(8):  # 8 эпизодов за ночь
        start = np.random.randint(100, 1200)
        duration = np.random.randint(10, 60)
        intensity = np.random.uniform(0.6, 1.0)
        
        for j in range(duration):
            if start + j < 1440:
                base_snoring[start + j] = intensity + np.random.normal(0, 0.1)
                snoring_episodes.append(start + j)
    
    # Данные акселерометра
    accelerometer_data = np.random.normal(0, 0.5, (1440, 3))
    
    # Временные данные
    temporal_data = []
    for i in range(1440):
        hour = timestamps[i].hour
        sleep_duration = max(0, (timestamps[i] - timestamps[0]).total_seconds() / 3600)
        
        temporal_data.append({
            'hour': hour,
            'sleep_duration': sleep_duration,
            'cyclicity': np.sin(2 * np.pi * i / 1440),  # Цикличность сна
            'time_since_last_movement': np.random.exponential(30),  # Экспоненциальное распределение
            'sleep_phase': np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            'sleep_depth': np.random.uniform(0.3, 0.9),
            'sleep_cycle': (i // 90) % 5  # Циклы сна каждые 90 минут
        })
    
    return timestamps, base_snoring, accelerometer_data, temporal_data, snoring_episodes

def run_models_simulation():
    """Запускает симуляцию работы моделей"""
    print("🤖 Запуск симуляции моделей...")
    
    try:
        from models.snoring_classifier import SnoringClassifier
        from models.snoring_predictor_simple import SnoringPredictorSimple
        from features.snoring_extractor import SnoringFeatureExtractor
        
        # Создаем модели
        classifier = SnoringClassifier(model_type='random_forest')
        predictor = SnoringPredictorSimple(model_type='random_forest')
        extractor = SnoringFeatureExtractor()
        
        # Обучаем модели на синтетических данных
        X_train_class = np.random.randn(100, 15)
        y_train_class = np.random.randint(0, 2, 100)
        classifier.train(X_train_class, y_train_class)
        
        X_train_pred = np.random.randn(100, 30)
        y_train_pred = np.random.randint(0, 2, 100)
        predictor.train(X_train_pred, y_train_pred)
        
        print("✅ Модели обучены")
        return classifier, predictor, extractor
        
    except Exception as e:
        print(f"❌ Ошибка при обучении моделей: {e}")
        return None, None, None

def create_visualizations(timestamps, base_snoring, accelerometer_data, temporal_data, snoring_episodes):
    """Создает визуализации результатов"""
    print("📈 Создание визуализаций...")
    
    # Создаем фигуру с подграфиками
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Временной ряд храпа
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(timestamps, base_snoring, 'b-', alpha=0.7, linewidth=1)
    plt.scatter(timestamps[snoring_episodes], base_snoring[snoring_episodes], 
                color='red', s=20, alpha=0.8, label='Эпизоды храпа')
    plt.title('Временной ряд храпа за ночь', fontsize=14, fontweight='bold')
    plt.ylabel('Интенсивность храпа')
    plt.xlabel('Время')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Распределение интенсивности храпа
    ax2 = plt.subplot(4, 2, 2)
    plt.hist(base_snoring, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Распределение интенсивности храпа', fontsize=14, fontweight='bold')
    plt.xlabel('Интенсивность храпа')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    
    # 3. Данные акселерометра
    ax3 = plt.subplot(4, 2, 3)
    for i in range(3):
        plt.plot(timestamps, accelerometer_data[:, i], 
                label=f'Ось {i+1}', alpha=0.7, linewidth=1)
    plt.title('Данные акселерометра', fontsize=14, fontweight='bold')
    plt.ylabel('Ускорение')
    plt.xlabel('Время')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Фазы сна
    ax4 = plt.subplot(4, 2, 4)
    sleep_phases = [t['sleep_phase'] for t in temporal_data]
    plt.plot(timestamps, sleep_phases, 'g-', linewidth=2)
    plt.title('Фазы сна', fontsize=14, fontweight='bold')
    plt.ylabel('Фаза сна')
    plt.xlabel('Время')
    plt.yticks([0, 1, 2, 3, 4], ['Бодр.', 'N1', 'N2', 'N3', 'REM'])
    plt.grid(True, alpha=0.3)
    
    # 5. Глубина сна
    ax5 = plt.subplot(4, 2, 5)
    sleep_depth = [t['sleep_depth'] for t in temporal_data]
    plt.plot(timestamps, sleep_depth, 'purple', linewidth=2)
    plt.title('Глубина сна', fontsize=14, fontweight='bold')
    plt.ylabel('Глубина сна')
    plt.xlabel('Время')
    plt.grid(True, alpha=0.3)
    
    # 6. Циклы сна
    ax6 = plt.subplot(4, 2, 6)
    sleep_cycles = [t['sleep_cycle'] for t in temporal_data]
    plt.plot(timestamps, sleep_cycles, 'orange', linewidth=2)
    plt.title('Циклы сна', fontsize=14, fontweight='bold')
    plt.ylabel('Номер цикла')
    plt.xlabel('Время')
    plt.grid(True, alpha=0.3)
    
    # 7. Корреляция храпа и движений
    ax7 = plt.subplot(4, 2, 7)
    movement_intensity = np.sqrt(np.sum(accelerometer_data**2, axis=1))
    plt.scatter(base_snoring, movement_intensity, alpha=0.6, s=20)
    plt.title('Корреляция: Храп vs Движения', fontsize=14, fontweight='bold')
    plt.xlabel('Интенсивность храпа')
    plt.ylabel('Интенсивность движений')
    plt.grid(True, alpha=0.3)
    
    # 8. Статистика по часам
    ax8 = plt.subplot(4, 2, 8)
    hours = [t.hour for t in timestamps]
    hourly_snoring = pd.DataFrame({'hour': hours, 'snoring': base_snoring})
    hourly_stats = hourly_snoring.groupby('hour')['snoring'].mean()
    
    plt.bar(hourly_stats.index, hourly_stats.values, alpha=0.7, color='red')
    plt.title('Средняя интенсивность храпа по часам', fontsize=14, fontweight='bold')
    plt.xlabel('Час')
    plt.ylabel('Средняя интенсивность храпа')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snoring_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 График сохранен как 'snoring_analysis.png'")
    
    return fig

def create_prediction_visualization(classifier, predictor, extractor, timestamps, base_snoring):
    """Создает визуализацию предсказаний моделей"""
    print("🔮 Создание визуализации предсказаний...")
    
    if classifier is None or predictor is None:
        print("⚠️  Модели не обучены, пропускаем визуализацию предсказаний")
        return
    
    # Симулируем предсказания
    predictions = []
    probabilities = []
    
    for i in range(0, len(timestamps), 10):  # Каждые 10 минут
        if i >= len(base_snoring):
            break
            
        # Симулируем аудио данные
        audio_data = np.random.randn(8000) * (0.5 + base_snoring[i])
        
        try:
            # Предсказание классификатора
            detection_result = classifier.detect_snoring_window(audio_data, extractor)
            
            # Предсказание модели предсказания
            prediction_result = predictor.predict_snoring_episode(
                audio_data=audio_data,
                accelerometer_data=np.random.randn(100, 3),
                temporal_data={
                    'hour': timestamps[i].hour,
                    'sleep_duration': i / 60,
                    'cyclicity': np.sin(2 * np.pi * i / 1440),
                    'time_since_last_movement': np.random.exponential(30),
                    'sleep_phase': np.random.choice([0, 1, 2, 3, 4]),
                    'sleep_depth': np.random.uniform(0.3, 0.9),
                    'sleep_cycle': (i // 90) % 5
                }
            )
            
            predictions.append(detection_result['is_snoring'])
            probabilities.append(prediction_result['snoring_probability'])
            
        except Exception as e:
            print(f"⚠️  Ошибка предсказания в момент {i}: {e}")
            predictions.append(False)
            probabilities.append(0.5)
    
    # Создаем график предсказаний
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # График детекции храпа
    time_points = timestamps[::10][:len(predictions)]
    ax1.plot(time_points, predictions, 'ro-', markersize=8, alpha=0.7, label='Детекция храпа')
    ax1.set_title('Результаты детекции храпа', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Храп (1) / Нет храпа (0)')
    ax1.set_xlabel('Время')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График вероятности предсказания
    ax2.plot(time_points, probabilities, 'bo-', markersize=8, alpha=0.7, label='Вероятность храпа')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Порог (0.5)')
    ax2.set_title('Вероятность предсказания храпа', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Вероятность')
    ax2.set_xlabel('Время')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    print("💾 График предсказаний сохранен как 'prediction_results.png'")
    
    return fig

def create_summary_report(timestamps, base_snoring, snoring_episodes):
    """Создает сводный отчет"""
    print("📋 Создание сводного отчета...")
    
    # Статистика
    total_snoring_time = np.sum(base_snoring > 0.3)  # минуты с храпом
    total_sleep_time = len(timestamps)
    snoring_percentage = (total_snoring_time / total_sleep_time) * 100
    
    max_snoring_intensity = np.max(base_snoring)
    avg_snoring_intensity = np.mean(base_snoring)
    
    # Создаем отчет
    report = f"""
    📊 СВОДНЫЙ ОТЧЕТ АНАЛИЗА ХРАПА
    =================================
    
    📈 Общая статистика:
    - Общее время сна: {total_sleep_time} минут ({total_sleep_time/60:.1f} часов)
    - Время с храпом: {total_snoring_time} минут ({total_snoring_time/60:.1f} часов)
    - Процент времени с храпом: {snoring_percentage:.1f}%
    
    🎯 Интенсивность храпа:
    - Максимальная интенсивность: {max_snoring_intensity:.3f}
    - Средняя интенсивность: {avg_snoring_intensity:.3f}
    - Количество эпизодов храпа: {len(snoring_episodes)}
    
    ⏰ Временное распределение:
    - Начало сна: {timestamps[0].strftime('%H:%M')}
    - Конец сна: {timestamps[-1].strftime('%H:%M')}
    - Продолжительность: {(timestamps[-1] - timestamps[0]).total_seconds() / 3600:.1f} часов
    
    🔍 Рекомендации:
    - Уровень храпа: {'Высокий' if snoring_percentage > 20 else 'Средний' if snoring_percentage > 10 else 'Низкий'}
    - Рекомендуется: {'Консультация специалиста' if snoring_percentage > 20 else 'Мониторинг' if snoring_percentage > 10 else 'Норма'}
    """
    
    # Сохраняем отчет
    with open('snoring_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("💾 Отчет сохранен как 'snoring_report.txt'")
    print(report)
    
    return report

def main():
    """Основная функция визуализации"""
    print("🚀 Запуск визуализации результатов анализа храпа\n")
    
    # Генерируем тестовые данные
    timestamps, base_snoring, accelerometer_data, temporal_data, snoring_episodes = generate_test_data()
    
    # Запускаем симуляцию моделей
    classifier, predictor, extractor = run_models_simulation()
    
    # Создаем визуализации
    fig1 = create_visualizations(timestamps, base_snoring, accelerometer_data, temporal_data, snoring_episodes)
    fig2 = create_prediction_visualization(classifier, predictor, extractor, timestamps, base_snoring)
    
    # Создаем сводный отчет
    report = create_summary_report(timestamps, base_snoring, snoring_episodes)
    
    print("\n🎉 Визуализация завершена!")
    print("\n📁 Созданные файлы:")
    print("   - snoring_analysis.png (основные графики)")
    print("   - prediction_results.png (результаты предсказаний)")
    print("   - snoring_report.txt (сводный отчет)")
    
    # Показываем графики
    plt.show()

if __name__ == "__main__":
    main() 