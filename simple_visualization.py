#!/usr/bin/env python3
"""
Упрощенная визуализация результатов анализа храпа
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def generate_snoring_data():
    """Генерирует данные храпа для визуализации"""
    print("📊 Генерация данных храпа...")
    
    # Симулируем 8 часов сна (480 минут)
    timestamps = pd.date_range(start='2024-01-01 22:00', periods=480, freq='1min')
    
    # Генерируем реалистичные данные храпа
    np.random.seed(42)
    
    # Базовый уровень храпа
    base_snoring = np.zeros(480)
    
    # Добавляем эпизоды храпа
    snoring_episodes = []
    
    # 5 основных эпизодов храпа за ночь
    episode_starts = [60, 120, 180, 240, 300]  # минуты от начала сна
    episode_durations = [30, 45, 25, 40, 35]   # длительность в минутах
    episode_intensities = [0.8, 0.9, 0.7, 0.85, 0.75]  # интенсивность
    
    for start, duration, intensity in zip(episode_starts, episode_durations, episode_intensities):
        for i in range(duration):
            if start + i < 480:
                # Добавляем шум к интенсивности
                noise = np.random.normal(0, 0.1)
                base_snoring[start + i] = max(0, intensity + noise)
                snoring_episodes.append(start + i)
    
    # Добавляем фоновый храп
    for i in range(480):
        if base_snoring[i] == 0:  # Если нет активного эпизода
            # Фоновый храп ночью
            if 22 <= timestamps[i].hour or timestamps[i].hour <= 6:
                base_snoring[i] = np.random.normal(0.2, 0.05)
            else:
                base_snoring[i] = np.random.normal(0.05, 0.02)
    
    return timestamps, base_snoring, snoring_episodes

def create_snoring_visualization(timestamps, base_snoring, snoring_episodes):
    """Создает визуализацию храпа"""
    print("📈 Создание визуализации храпа...")
    
    # Создаем фигуру с подграфиками
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Временной ряд храпа
    ax1.plot(timestamps, base_snoring, 'b-', alpha=0.7, linewidth=1.5)
    ax1.scatter(timestamps[snoring_episodes], base_snoring[snoring_episodes], 
                color='red', s=30, alpha=0.8, label='Эпизоды храпа', zorder=5)
    ax1.set_title('Временной ряд храпа за ночь', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Интенсивность храпа')
    ax1.set_xlabel('Время')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Распределение интенсивности храпа
    ax2.hist(base_snoring, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Распределение интенсивности храпа', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Интенсивность храпа')
    ax2.set_ylabel('Частота')
    ax2.grid(True, alpha=0.3)
    
    # 3. Статистика по часам
    hours = [t.hour for t in timestamps]
    hourly_snoring = pd.DataFrame({'hour': hours, 'snoring': base_snoring})
    hourly_stats = hourly_snoring.groupby('hour')['snoring'].mean()
    
    bars = ax3.bar(hourly_stats.index, hourly_stats.values, alpha=0.7, color='red')
    ax3.set_title('Средняя интенсивность храпа по часам', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Час')
    ax3.set_ylabel('Средняя интенсивность храпа')
    ax3.grid(True, alpha=0.3)
    
    # Подсвечиваем ночные часы
    for i, bar in enumerate(bars):
        if 22 <= hourly_stats.index[i] or hourly_stats.index[i] <= 6:
            bar.set_color('darkred')
    
    # 4. Длительность эпизодов храпа
    episode_lengths = []
    current_episode = 0
    
    for i in range(len(base_snoring)):
        if base_snoring[i] > 0.3:  # Порог храпа
            current_episode += 1
        elif current_episode > 0:
            episode_lengths.append(current_episode)
            current_episode = 0
    
    if episode_lengths:
        ax4.hist(episode_lengths, bins=10, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('Распределение длительности эпизодов храпа', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Длительность эпизода (минуты)')
        ax4.set_ylabel('Количество эпизодов')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Нет эпизодов храпа', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Распределение длительности эпизодов храпа', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('snoring_visualization.png', dpi=300, bbox_inches='tight')
    print("💾 График сохранен как 'snoring_visualization.png'")
    
    return fig

def create_summary_statistics(timestamps, base_snoring, snoring_episodes):
    """Создает сводную статистику"""
    print("📋 Создание сводной статистики...")
    
    # Статистика
    total_snoring_time = np.sum(base_snoring > 0.3)  # минуты с храпом
    total_sleep_time = len(timestamps)
    snoring_percentage = (total_snoring_time / total_sleep_time) * 100
    
    max_snoring_intensity = np.max(base_snoring)
    avg_snoring_intensity = np.mean(base_snoring)
    
    # Находим эпизоды храпа
    episodes = []
    current_episode = {'start': None, 'duration': 0, 'max_intensity': 0}
    
    for i, intensity in enumerate(base_snoring):
        if intensity > 0.3:  # Порог храпа
            if current_episode['start'] is None:
                current_episode['start'] = timestamps[i]
            current_episode['duration'] += 1
            current_episode['max_intensity'] = max(current_episode['max_intensity'], intensity)
        elif current_episode['start'] is not None:
            episodes.append(current_episode.copy())
            current_episode = {'start': None, 'duration': 0, 'max_intensity': 0}
    
    # Добавляем последний эпизод если есть
    if current_episode['start'] is not None:
        episodes.append(current_episode)
    
    # Создаем отчет
    report = f"""
    📊 СВОДНАЯ СТАТИСТИКА АНАЛИЗА ХРАПА
    =====================================
    
    📈 Общая статистика:
    - Общее время сна: {total_sleep_time} минут ({total_sleep_time/60:.1f} часов)
    - Время с храпом: {total_snoring_time} минут ({total_snoring_time/60:.1f} часов)
    - Процент времени с храпом: {snoring_percentage:.1f}%
    
    🎯 Интенсивность храпа:
    - Максимальная интенсивность: {max_snoring_intensity:.3f}
    - Средняя интенсивность: {avg_snoring_intensity:.3f}
    - Количество эпизодов храпа: {len(episodes)}
    
    ⏰ Временное распределение:
    - Начало сна: {timestamps[0].strftime('%H:%M')}
    - Конец сна: {timestamps[-1].strftime('%H:%M')}
    - Продолжительность: {(timestamps[-1] - timestamps[0]).total_seconds() / 3600:.1f} часов
    
    🔍 Детали эпизодов храпа:
    """
    
    for i, episode in enumerate(episodes, 1):
        report += f"""
    Эпизод {i}:
    - Начало: {episode['start'].strftime('%H:%M')}
    - Длительность: {episode['duration']} минут
    - Максимальная интенсивность: {episode['max_intensity']:.3f}
    """
    
    # Рекомендации
    if snoring_percentage > 20:
        recommendation = "Высокий уровень храпа - рекомендуется консультация специалиста"
    elif snoring_percentage > 10:
        recommendation = "Средний уровень храпа - рекомендуется мониторинг"
    else:
        recommendation = "Низкий уровень храпа - в пределах нормы"
    
    report += f"""
    🔍 Рекомендации:
    - Уровень храпа: {'Высокий' if snoring_percentage > 20 else 'Средний' if snoring_percentage > 10 else 'Низкий'}
    - Рекомендуется: {recommendation}
    """
    
    # Сохраняем отчет
    with open('snoring_statistics.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("💾 Статистика сохранена как 'snoring_statistics.txt'")
    print(report)
    
    return report

def create_prediction_simulation():
    """Создает симуляцию предсказаний"""
    print("🔮 Создание симуляции предсказаний...")
    
    # Генерируем данные
    timestamps = pd.date_range(start='2024-01-01 22:00', periods=480, freq='1min')
    np.random.seed(42)
    
    # Симулируем реальные данные храпа
    base_snoring = np.zeros(480)
    
    # Добавляем эпизоды храпа
    episode_starts = [60, 120, 180, 240, 300]
    episode_durations = [30, 45, 25, 40, 35]
    episode_intensities = [0.8, 0.9, 0.7, 0.85, 0.75]
    
    for start, duration, intensity in zip(episode_starts, episode_durations, episode_intensities):
        for i in range(duration):
            if start + i < 480:
                noise = np.random.normal(0, 0.1)
                base_snoring[start + i] = max(0, intensity + noise)
    
    # Симулируем предсказания моделей
    predictions = []
    probabilities = []
    
    for i in range(0, 480, 5):  # Каждые 5 минут
        # Симулируем детекцию (на основе реальных данных + шум)
        detection_noise = np.random.normal(0, 0.1)
        detection = 1 if base_snoring[i] > 0.3 + detection_noise else 0
        predictions.append(detection)
        
        # Симулируем вероятность предсказания
        prediction_noise = np.random.normal(0, 0.1)
        probability = min(1.0, max(0.0, base_snoring[i] + prediction_noise))
        probabilities.append(probability)
    
    # Создаем график предсказаний
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # График детекции
    time_points = timestamps[::5][:len(predictions)]
    ax1.plot(time_points, predictions, 'ro-', markersize=6, alpha=0.7, label='Детекция храпа')
    ax1.set_title('Симуляция детекции храпа', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Храп (1) / Нет храпа (0)')
    ax1.set_xlabel('Время')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График вероятности предсказания
    ax2.plot(time_points, probabilities, 'bo-', markersize=6, alpha=0.7, label='Вероятность храпа')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Порог (0.5)')
    ax2.set_title('Симуляция вероятности предсказания храпа', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Вероятность')
    ax2.set_xlabel('Время')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_simulation.png', dpi=300, bbox_inches='tight')
    print("💾 График предсказаний сохранен как 'prediction_simulation.png'")
    
    return fig

def main():
    """Основная функция визуализации"""
    print("🚀 Запуск упрощенной визуализации результатов анализа храпа\n")
    
    # Генерируем данные
    timestamps, base_snoring, snoring_episodes = generate_snoring_data()
    
    # Создаем визуализации
    fig1 = create_snoring_visualization(timestamps, base_snoring, snoring_episodes)
    fig2 = create_prediction_simulation()
    
    # Создаем сводную статистику
    report = create_summary_statistics(timestamps, base_snoring, snoring_episodes)
    
    print("\n🎉 Визуализация завершена!")
    print("\n📁 Созданные файлы:")
    print("   - snoring_visualization.png (основные графики)")
    print("   - prediction_simulation.png (симуляция предсказаний)")
    print("   - snoring_statistics.txt (сводная статистика)")
    
    # Показываем графики
    plt.show()

if __name__ == "__main__":
    main() 