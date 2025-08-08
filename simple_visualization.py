#!/usr/bin/env python3
"""
Упрощенная визуализация результатов анализа храпа
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os

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
        ax4.hist(episode_lengths, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_title('Длительность эпизодов храпа', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Длительность (минуты)')
        ax4.set_ylabel('Количество эпизодов')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Нет эпизодов храпа', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Длительность эпизодов храпа', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Создаем папку results если её нет
    os.makedirs('results', exist_ok=True)
    
    # Сохраняем график
    plt.savefig('results/simple_snoring_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(timestamps, base_snoring, snoring_episodes):
    """Создает сводную статистику"""
    print("📊 Создание сводной статистики...")
    
    # Базовая статистика
    total_snoring_time = len(snoring_episodes)
    total_sleep_time = len(timestamps)
    snoring_percentage = (total_snoring_time / total_sleep_time) * 100
    
    # Статистика по интенсивности
    mean_intensity = np.mean(base_snoring)
    max_intensity = np.max(base_snoring)
    std_intensity = np.std(base_snoring)
    
    # Статистика по часам
    hours = [t.hour for t in timestamps]
    hourly_snoring = pd.DataFrame({'hour': hours, 'snoring': base_snoring})
    hourly_stats = hourly_snoring.groupby('hour')['snoring'].mean()
    
    worst_hour = hourly_stats.idxmax()
    worst_intensity = hourly_stats.max()
    
    print(f"\n📋 СВОДНАЯ СТАТИСТИКА ХРАПА")
    print("=" * 50)
    print(f"Общее время сна: {total_sleep_time} минут")
    print(f"Время храпа: {total_snoring_time} минут")
    print(f"Процент времени с храпом: {snoring_percentage:.1f}%")
    print(f"Средняя интенсивность храпа: {mean_intensity:.3f}")
    print(f"Максимальная интенсивность: {max_intensity:.3f}")
    print(f"Стандартное отклонение: {std_intensity:.3f}")
    print(f"Худший час: {worst_hour}:00 (интенсивность: {worst_intensity:.3f})")
    
    return {
        'total_sleep_time': total_sleep_time,
        'total_snoring_time': total_snoring_time,
        'snoring_percentage': snoring_percentage,
        'mean_intensity': mean_intensity,
        'max_intensity': max_intensity,
        'worst_hour': worst_hour,
        'worst_intensity': worst_intensity
    }

def create_prediction_simulation():
    """Создает симуляцию предсказания храпа"""
    print("🔮 Создание симуляции предсказания...")
    
    # Генерируем данные для симуляции
    np.random.seed(42)
    n_samples = 100
    
    # Симулируем признаки
    features = np.random.randn(n_samples, 10)
    
    # Симулируем предсказания
    predictions = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    probabilities = np.random.uniform(0, 1, n_samples)
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Распределение предсказаний
    ax1.hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Распределение вероятностей храпа', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Вероятность храпа')
    ax1.set_ylabel('Количество образцов')
    ax1.grid(True, alpha=0.3)
    
    # 2. Точность предсказаний
    correct_predictions = np.sum(predictions == np.random.choice([0, 1], n_samples))
    accuracy = correct_predictions / n_samples
    
    labels = ['Правильные', 'Неправильные']
    sizes = [accuracy, 1 - accuracy]
    colors = ['lightgreen', 'lightcoral']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Точность предсказаний: {accuracy:.1%}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('results/prediction_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def main():
    """Основная функция"""
    print("🚀 Запуск упрощенной визуализации храпа...")
    
    # Создаем папку results если её нет
    os.makedirs('results', exist_ok=True)
    
    # Генерируем данные
    timestamps, base_snoring, snoring_episodes = generate_snoring_data()
    
    # Создаем визуализацию
    create_snoring_visualization(timestamps, base_snoring, snoring_episodes)
    
    # Создаем статистику
    stats = create_summary_statistics(timestamps, base_snoring, snoring_episodes)
    
    # Создаем симуляцию предсказания
    accuracy = create_prediction_simulation()
    
    print(f"\n✅ Визуализация завершена!")
    print(f"📁 Графики сохранены в папке 'results/':")
    print(f"   • simple_snoring_visualization.png")
    print(f"   • prediction_simulation.png")
    print(f"📊 Точность симуляции: {accuracy:.1%}")

if __name__ == "__main__":
    main() 