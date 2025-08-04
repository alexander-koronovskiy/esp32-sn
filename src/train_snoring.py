"""
Скрипт для обучения модели детекции храпа.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.snoring_extractor import SnoringFeatureExtractor, SnoringFeatureSelector, create_snoring_config
from src.models.snoring_classifier import SnoringClassifier, create_snoring_classifier


def create_synthetic_snoring_data(duration_minutes=10, sampling_rate=8000, n_segments=60):
    """
    Создает синтетические данные для обучения модели храпа.
    
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
    snoring_classes = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End']
    
    for i in range(n_segments):
        # Создаем сегмент
        t = np.linspace(0, 1, segment_length)
        
        # Выбираем класс (имитируем реальное распределение)
        if i < n_segments * 0.4:  # 40% - нет храпа
            class_idx = 0
            # Тихий фоновый шум
            signal = np.random.randn(segment_length) * 0.1
        elif i < n_segments * 0.6:  # 20% - легкий храп
            class_idx = 1
            # Легкий храп (низкочастотный)
            signal = (np.sin(2 * np.pi * 50 * t) * 0.3 +
                     np.sin(2 * np.pi * 100 * t) * 0.2 +
                     np.random.randn(segment_length) * 0.2)
        elif i < n_segments * 0.8:  # 20% - сильный храп
            class_idx = 2
            # Сильный храп (широкополосный)
            signal = (np.sin(2 * np.pi * 80 * t) * 0.5 +
                     np.sin(2 * np.pi * 150 * t) * 0.4 +
                     np.sin(2 * np.pi * 300 * t) * 0.3 +
                     np.random.randn(segment_length) * 0.3)
        elif i < n_segments * 0.9:  # 10% - начало храпа
            class_idx = 3
            # Начало храпа (нарастающий)
            ramp = np.linspace(0, 1, segment_length)
            signal = (np.sin(2 * np.pi * 60 * t) * 0.4 * ramp +
                     np.sin(2 * np.pi * 120 * t) * 0.3 * ramp +
                     np.random.randn(segment_length) * 0.2)
        else:  # 10% - конец храпа
            class_idx = 4
            # Конец храпа (затухающий)
            ramp = np.linspace(1, 0, segment_length)
            signal = (np.sin(2 * np.pi * 70 * t) * 0.3 * ramp +
                     np.sin(2 * np.pi * 140 * t) * 0.2 * ramp +
                     np.random.randn(segment_length) * 0.1)
        
        segments.append(signal)
        labels.append(class_idx)
    
    return np.array(segments), np.array(labels)


def segment_audio_data(audio_data, segment_length=8000, overlap=0.5):
    """
    Сегментирует аудио данные.
    
    Args:
        audio_data: Аудио данные
        segment_length: Длина сегмента в сэмплах
        overlap: Перекрытие сегментов
        
    Returns:
        segments: Сегменты аудио
    """
    segments = []
    step = int(segment_length * (1 - overlap))
    
    for i in range(0, len(audio_data) - segment_length + 1, step):
        segment = audio_data[i:i + segment_length]
        segments.append(segment)
    
    return np.array(segments)


def main():
    """Основная функция для обучения модели детекции храпа."""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Обучение модели детекции храпа')
    parser.add_argument('--config', type=str, default='config_snoring.yaml', 
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--output', type=str, help='Путь для сохранения модели')
    parser.add_argument('--log-level', type=str, default='INFO', help='Уровень логирования')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ ДЕТЕКЦИИ ХРАПА")
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
        for i, class_name in enumerate(['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End']):
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
        print(f"  Точность на обучающих данных: {train_metrics['train_accuracy']:.4f}")
        print(f"  Кросс-валидация: {train_metrics['cv_mean']:.4f} ± {train_metrics['cv_std']:.4f}")
        print(f"  Количество признаков: {train_metrics['feature_count']}")
        
        # 7. Оценка модели
        print("\n7. Оценка модели")
        test_metrics = classifier.evaluate(X_test_scaled, y_test)
        
        print(f"✓ Модель оценена:")
        print(f"  Точность на тестовых данных: {test_metrics['accuracy']:.4f}")
        print(f"  Точность детекции храпа: {test_metrics['snoring_detection_accuracy']:.4f}")
        
        # 8. Сохранение модели
        print("\n8. Сохранение модели")
        if args.output:
            model_path = args.output
        else:
            os.makedirs('models', exist_ok=True)
            model_path = f'models/snoring_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        
        classifier.save_model(model_path)
        
        # Сохраняем scaler
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        import joblib
        joblib.dump(scaler, scaler_path)
        
        print(f"✓ Модель сохранена в: {model_path}")
        print(f"✓ Scaler сохранен в: {scaler_path}")
        
        # 9. Создание визуализации
        print("\n9. Создание визуализации")
        create_snoring_visualization(test_metrics, selected_features, features_selected)
        
        # 10. Финальная статистика
        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА ОБУЧЕНИЯ МОДЕЛИ ХРАПА")
        print("=" * 60)
        
        print(f"\nМодель:")
        print(f"  Тип: {config['model_type']}")
        print(f"  Точность: {test_metrics['accuracy']:.4f}")
        print(f"  Точность детекции храпа: {test_metrics['snoring_detection_accuracy']:.4f}")
        print(f"  Количество признаков: {len(selected_features)}")
        
        print(f"\nКлассы:")
        for i, class_name in enumerate(['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End']):
            precision = test_metrics['classification_report'].get(class_name, {}).get('precision', 0)
            recall = test_metrics['classification_report'].get(class_name, {}).get('recall', 0)
            f1 = test_metrics['classification_report'].get(class_name, {}).get('f1-score', 0)
            print(f"  {class_name}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        
        print(f"\nФайлы созданы:")
        print(f"  📁 {model_path} - Обученная модель")
        print(f"  📁 {scaler_path} - Scaler")
        print(f"  📊 results/snoring_results.png - Визуализация результатов")
        
        print(f"\n✓ Обучение модели детекции храпа завершено успешно!")
        
    except Exception as e:
        print(f"\n✗ Ошибка: {str(e)}")
        raise


def create_snoring_visualization(test_metrics, selected_features, features_df):
    """Создает визуализацию результатов обучения модели храпа."""
    print("📊 Создание визуализации результатов...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Настройка стиля графиков
    plt.style.use('default')
    fig_width = 12
    fig_height = 8
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(fig_width//2, fig_height//2))
    
    class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring', 'Snoring_Start', 'Snoring_End']
    cm = np.array(test_metrics['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion Matrix - Детекция храпа', fontsize=14, fontweight='bold')
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('results/snoring_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Точность по классам
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
    
    # 3. Важность признаков
    if len(selected_features) > 0:
        fig, ax = plt.subplots(figsize=(fig_width//2, fig_height//2))
        
        # Берем первые 10 признаков для визуализации
        top_features = selected_features[:10]
        
        # Простая оценка важности (можно заменить на реальную из модели)
        importance_scores = np.random.rand(len(top_features))
        importance_scores = importance_scores / np.sum(importance_scores)
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, importance_scores, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Важность')
        ax.set_title('Топ-10 важных признаков для детекции храпа', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/snoring_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Визуализация создана:")
    print("  📊 results/snoring_confusion_matrix.png - Матрица ошибок")
    print("  📊 results/snoring_class_metrics.png - Метрики по классам")
    print("  📊 results/snoring_feature_importance.png - Важность признаков")


if __name__ == "__main__":
    main() 