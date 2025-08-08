#!/usr/bin/env python3
"""
Демонстрация системы ML Snoring.
Показывает детекцию храпа и предсказание будущего храпа.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.snoring_extractor import SnoringFeatureExtractor, SnoringFeatureSelector, create_snoring_config
from src.models.snoring_classifier import SnoringClassifier, create_snoring_classifier


def create_synthetic_snoring_data(duration_minutes=5, sampling_rate=8000, n_segments=60):
    """
    Создает синтетические данные храпа для демонстрации.
    
    Args:
        duration_minutes: Длительность в минутах
        sampling_rate: Частота дискретизации аудио
        n_segments: Количество сегментов
        
    Returns:
        segments, labels: Аудио сегменты и метки
    """
    segment_length = sampling_rate  # 1 секунда
    segments = []
    labels = []
    
    # Классы храпа
    snoring_classes = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
    
    for i in range(n_segments):
        # Создаем сегмент
        t = np.linspace(0, 1, segment_length)
        
        # Выбираем класс (имитируем реалистичное распределение)
        if i < n_segments * 0.5:  # 50% - нет храпа
            class_idx = 0
            # Тихий фоновый шум
            signal = np.random.randn(segment_length) * 0.05
        elif i < n_segments * 0.7:  # 20% - легкий храп
            class_idx = 1
            # Легкий храп (низкочастотный)
            signal = (np.sin(2 * np.pi * 50 * t) * 0.2 +
                     np.sin(2 * np.pi * 100 * t) * 0.15 +
                     np.random.randn(segment_length) * 0.1)
        else:  # 30% - сильный храп
            class_idx = 2
            # Сильный храп (широкополосный)
            signal = (np.sin(2 * np.pi * 80 * t) * 0.4 +
                     np.sin(2 * np.pi * 150 * t) * 0.3 +
                     np.sin(2 * np.pi * 300 * t) * 0.2 +
                     np.random.randn(segment_length) * 0.2)
        
        segments.append(signal)
        labels.append(class_idx)
    
    return np.array(segments), np.array(labels)


def visualize_snoring_results(segments, labels, predictions, features_df, test_metrics):
    """
    Создает визуализацию результатов детекции храпа.
    
    Args:
        segments: Аудио сегменты
        labels: Истинные метки
        predictions: Предсказания модели
        features_df: DataFrame с признаками
        test_metrics: Метрики тестирования
    """
    print("\n📊 Создание визуализации результатов...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Настройка стиля графиков
    plt.style.use('default')
    fig_width = 12
    fig_height = 8
    
    # 1. Временной ряд аудио с метками храпа
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), height_ratios=[2, 1])
    
    # Объединяем все сегменты для временного ряда
    full_audio = np.concatenate(segments)
    time_seconds = np.arange(len(full_audio)) / 8000  # 8 kHz
    
    # Временной ряд аудио
    axes[0].plot(time_seconds, full_audio, 'b-', linewidth=0.5, alpha=0.8)
    axes[0].set_title('Аудио сигнал с метками храпа', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Амплитуда', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Цветовая карта классов храпа
    class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
    colors = ['#2E8B57', '#FFD700', '#FF4500', '#FF69B4', '#9370DB']
    
    # Создаем временные метки для каждого сегмента
    segment_times = np.arange(len(labels))
    segment_colors = np.array(colors)[labels]
    axes[1].scatter(segment_times, np.ones_like(segment_times), c=segment_colors, s=50, alpha=0.7)
    axes[1].set_title('Классы храпа по времени', fontsize=12)
    axes[1].set_xlabel('Номер сегмента', fontsize=12)
    axes[1].set_yticks([])
    
    # Добавляем легенду
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=10, label=class_name) 
                      for color, class_name in zip(colors, class_names)]
    axes[1].legend(handles=legend_elements, loc='upper right', ncol=5)
    
    plt.tight_layout()
    plt.savefig('results/snoring_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(fig_width//2, fig_height//2))
    
    cm = np.array(test_metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion Matrix - Детекция храпа', fontsize=14, fontweight='bold')
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('results/snoring_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Метрики по классам
    fig, ax = plt.subplots(figsize=(fig_width//2, fig_height//2))
    
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
    
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_title('Метрики по классам храпа', fontsize=14, fontweight='bold')
    ax.set_xlabel('Классы')
    ax.set_ylabel('Значение')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/snoring_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Распределение признаков
    if len(features_df.columns) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        axes = axes.flatten()
        
        # Выбираем первые 4 признака для визуализации
        feature_cols = list(features_df.columns)[:4]
        
        for i, feature in enumerate(feature_cols):
            if i < len(axes):
                axes[i].hist(features_df[feature].dropna(), bins=20, alpha=0.7, 
                           color='skyblue', edgecolor='black')
                axes[i].set_title(f'Распределение: {feature}', fontsize=11)
                axes[i].set_xlabel('Значение')
                axes[i].set_ylabel('Частота')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/snoring_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Визуализация создана:")
    print("  📊 results/snoring_timeline.png - Временной ряд аудио")
    print("  📊 results/snoring_confusion_matrix.png - Матрица ошибок")
    print("  📊 results/snoring_class_metrics.png - Метрики по классам")
    print("  📊 results/snoring_feature_distributions.png - Распределения признаков")


def main():
    """Основная функция демонстрации ML Snoring."""
    
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ML SNORING")
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
            duration_minutes=5,
            sampling_rate=config['sampling_rate'],
            n_segments=60
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
            
            if (i + 1) % 15 == 0:
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
        
        # 8. Демонстрация предсказания
        print("\n8. Демонстрация предсказания")
        
        # Выбираем несколько тестовых сегментов для демонстрации
        demo_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
        
        for i, idx in enumerate(demo_indices):
            features_demo = X_test_scaled[idx:idx+1]
            prediction = classifier.predict(features_demo)[0]
            proba = classifier.predict_proba(features_demo)[0]
            
            # Предсказание будущего храпа
            future_pred = classifier.predict_snoring_future(
                features_demo[0], time_horizon=30
            )
            
            print(f"\n  Сегмент {i+1}:")
            print(f"    Предсказанный класс: {classifier.class_names[prediction]}")
            print(f"    Уверенность: {max(proba):.3f}")
            print(f"    Вероятность храпа в будущем: {future_pred['snoring_probability']:.3f}")
            print(f"    Уровень риска: {future_pred['risk_level']}")
        
        # 9. Создание визуализации
        print("\n9. Создание визуализации")
        visualize_snoring_results(segments, labels, test_metrics['predictions'], 
                                features_selected, test_metrics)
        
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
        print(f"  📄 esp32_code/snoring_detector.c - ESP32 код")
        
        print(f"\n✓ Демонстрация ML Snoring завершена успешно!")
        
    except Exception as e:
        print(f"\n✗ Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main() 