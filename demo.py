#!/usr/bin/env python3
"""
Демонстрационный скрипт для проекта классификации стадий сна.

Этот скрипт показывает полный цикл работы проекта:
1. Создание синтетических данных
2. Обработка и извлечение признаков
3. Обучение модели
4. Предсказание и оценка
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

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.loader import EEGDataLoader
from src.features.extractor import FeatureExtractor
from src.models.classifiers import ModelFactory


def create_synthetic_data(duration_minutes=10, sampling_rate=256, n_channels=2):
    """Создает синтетические данные ЭЭГ с метками стадий сна."""
    duration_seconds = duration_minutes * 60
    n_samples = duration_seconds * sampling_rate
    
    t = np.linspace(0, duration_seconds, n_samples)
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_durations = [90, 90, 120, 90, 90]  # секунды - увеличены для обеспечения минимум 2 сегментов на класс
    
    data = np.zeros((n_channels, n_samples))
    labels = np.zeros(n_samples, dtype=int)
    
    current_time = 0
    stage_idx = 0
    
    for stage, duration in zip(stages, stage_durations):
        if current_time >= duration_seconds:
            break
            
        start_idx = int(current_time * sampling_rate)
        end_idx = min(int((current_time + duration) * sampling_rate), n_samples)
        
        # Генерируем сигнал в зависимости от стадии
        if stage == 'Wake':
            signal = (np.sin(2 * np.pi * 10 * t[start_idx:end_idx]) * 0.5 +
                     np.sin(2 * np.pi * 20 * t[start_idx:end_idx]) * 0.3 +
                     np.random.randn(end_idx - start_idx) * 0.1)
        elif stage == 'N1':
            signal = (np.sin(2 * np.pi * 5 * t[start_idx:end_idx]) * 0.7 +
                     np.random.randn(end_idx - start_idx) * 0.2)
        elif stage == 'N2':
            signal = (np.sin(2 * np.pi * 12 * t[start_idx:end_idx]) * 0.6 +
                     np.sin(2 * np.pi * 3 * t[start_idx:end_idx]) * 0.4 +
                     np.random.randn(end_idx - start_idx) * 0.1)
        elif stage == 'N3':
            signal = (np.sin(2 * np.pi * 2 * t[start_idx:end_idx]) * 0.8 +
                     np.random.randn(end_idx - start_idx) * 0.1)
        else:  # REM
            signal = (np.sin(2 * np.pi * 6 * t[start_idx:end_idx]) * 0.5 +
                     np.sin(2 * np.pi * 8 * t[start_idx:end_idx]) * 0.4 +
                     np.random.randn(end_idx - start_idx) * 0.2)
        
        # Добавляем сигнал к каналам
        for ch in range(n_channels):
            data[ch, start_idx:end_idx] = signal + np.random.randn(len(signal)) * 0.05
        
        # Создаем метки
        labels[start_idx:end_idx] = stage_idx
        
        current_time += duration
        stage_idx += 1
    
    return data, labels


def main():
    """Основная функция демонстрации."""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРОЕКТА КЛАССИФИКАЦИИ СТАДИЙ СНА")
    print("=" * 60)
    
    # Настраиваем логирование
    logger = setup_logger(name='demo', level='INFO')
    logger.info("Начало демонстрации проекта")
    
    try:
        # 1. Загружаем конфигурацию
        logger.info("1. Загрузка конфигурации")
        config = load_config('config.yaml')
        print("✓ Конфигурация загружена")
        
        # 2. Создаем синтетические данные
        logger.info("2. Создание синтетических данных")
        eeg_data, labels = create_synthetic_data(duration_minutes=15, sampling_rate=256, n_channels=2)
        print(f"✓ Созданы данные ЭЭГ: {eeg_data.shape}")
        print(f"  Длительность: {eeg_data.shape[1] / 256 / 60:.1f} минут")
        print(f"  Количество каналов: {eeg_data.shape[0]}")
        
        # 3. Обработка данных
        logger.info("3. Обработка данных")
        data_loader = EEGDataLoader(config)
        
        # Создаем объект Raw для сегментации
        import mne
        info = mne.create_info(ch_names=['EEG1', 'EEG2'], sfreq=256, ch_types=['eeg'] * 2)
        raw = mne.io.RawArray(eeg_data, info)
        
        # Сегментируем данные
        segments, timestamps = data_loader.segment_data(raw)
        print(f"✓ Создано {len(segments)} сегментов")
        
        # Создаем метки для сегментов
        segment_labels = []
        for i, timestamp in enumerate(timestamps):
            start_sample = int(timestamp * 256)
            end_sample = start_sample + int(30 * 256)  # 30 секунд
            segment_label_samples = labels[start_sample:end_sample]
            if len(segment_label_samples) > 0:
                segment_label = np.bincount(segment_label_samples).argmax()
            else:
                segment_label = 0
            segment_labels.append(segment_label)
        
        segment_labels = np.array(segment_labels)
        print(f"✓ Создано {len(segment_labels)} меток для сегментов")
        
        # 4. Извлечение признаков
        logger.info("4. Извлечение признаков")
        feature_extractor = FeatureExtractor(config)
        features_df = feature_extractor.extract_features_from_segments(segments)
        
        # Очищаем данные
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols].dropna()
        if 'segment_id' in features_df.columns:
            features_df = features_df.drop('segment_id', axis=1)
        
        segment_labels = segment_labels[:len(features_df)]
        print(f"✓ Извлечено {features_df.shape[1]} признаков")
        print(f"  Размер данных: {features_df.shape}")
        
        # 5. Разделение данных
        logger.info("5. Разделение данных")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df.values, segment_labels,
            test_size=0.2, random_state=42, stratify=segment_labels
        )
        
        # Стандартизация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Данные разделены:")
        print(f"  Обучающая выборка: {X_train.shape}")
        print(f"  Тестовая выборка: {X_test.shape}")
        
        # 6. Обучение модели
        logger.info("6. Обучение модели")
        classifier = ModelFactory.create_classifier(config)
        train_metrics = classifier.train(X_train_scaled, y_train)
        
        print(f"✓ Модель обучена:")
        print(f"  Точность на обучающих данных: {train_metrics['train_accuracy']:.4f}")
        print(f"  Кросс-валидация: {train_metrics['cv_mean']:.4f} ± {train_metrics['cv_std']:.4f}")
        
        # 7. Оценка модели
        logger.info("7. Оценка модели")
        test_metrics = classifier.evaluate(X_test_scaled, y_test)
        
        print(f"✓ Модель оценена:")
        print(f"  Точность на тестовых данных: {test_metrics['accuracy']:.4f}")
        
        # 8. Визуализация результатов
        logger.info("8. Создание визуализаций")
        
        # Матрица ошибок
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        cm = np.array(test_metrics['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные классы')
        plt.ylabel('Истинные классы')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Распределение классов
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Истинные классы
        true_counts = pd.Series(y_test).value_counts().sort_index()
        ax1.bar(range(len(true_counts)), true_counts.values)
        ax1.set_title('Распределение истинных классов')
        ax1.set_xlabel('Класс')
        ax1.set_ylabel('Количество')
        ax1.set_xticks(range(len(true_counts)))
        ax1.set_xticklabels(class_names)
        
        # Предсказанные классы
        pred_counts = pd.Series(test_metrics['predictions']).value_counts().sort_index()
        ax2.bar(range(len(pred_counts)), pred_counts.values)
        ax2.set_title('Распределение предсказанных классов')
        ax2.set_xlabel('Класс')
        ax2.set_ylabel('Количество')
        ax2.set_xticks(range(len(pred_counts)))
        ax2.set_xticklabels(class_names)
        
        plt.tight_layout()
        plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Визуализации сохранены в папку results/")
        
        # 9. Сохранение модели
        logger.info("9. Сохранение модели")
        import joblib
        
        os.makedirs('models', exist_ok=True)
        model_path = 'models/demo_model.pkl'
        scaler_path = 'models/demo_scaler.pkl'
        
        classifier.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✓ Модель сохранена в: {model_path}")
        print(f"✓ Scaler сохранен в: {scaler_path}")
        
        # 10. Демонстрация предсказания
        logger.info("10. Демонстрация предсказания")
        
        # Загружаем модель
        new_classifier = ModelFactory.create_classifier(config)
        new_classifier.load_model(model_path)
        
        # Делаем предсказания на тестовых данных
        predictions = new_classifier.predict(X_test_scaled)
        probabilities = new_classifier.predict_proba(X_test_scaled)
        
        print("✓ Предсказания выполнены")
        print(f"  Количество предсказаний: {len(predictions)}")
        print(f"  Средняя уверенность: {np.mean(np.max(probabilities, axis=1)):.3f}")
        
        # Показываем несколько примеров
        print("\nПримеры предсказаний:")
        for i in range(min(5, len(predictions))):
            true_class = class_names[y_test[i]]
            pred_class = class_names[predictions[i]]
            confidence = np.max(probabilities[i])
            print(f"  Сегмент {i}: {true_class} → {pred_class} (уверенность: {confidence:.3f})")
        
        # 11. Финальная статистика
        logger.info("11. Финальная статистика")
        
        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        
        # Статистика по классам
        print("\nРаспределение классов:")
        for i, class_name in enumerate(class_names):
            true_count = np.sum(y_test == i)
            pred_count = np.sum(predictions == i)
            print(f"  {class_name}: истинных={true_count}, предсказанных={pred_count}")
        
        # Детальный отчет
        from sklearn.metrics import classification_report
        report = classification_report(y_test, predictions, target_names=class_names, output_dict=True)
        
        print("\nДетальный отчет по классам:")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-score: {metrics['f1-score']:.3f}")
        
        print(f"\nОбщая точность: {test_metrics['accuracy']:.4f}")
        print(f"Время выполнения: {datetime.now().strftime('%H:%M:%S')}")
        
        logger.info("Демонстрация завершена успешно!")
        print("\n✓ Демонстрация завершена успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {str(e)}")
        print(f"\n✗ Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main() 