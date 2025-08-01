"""
Скрипт для обучения модели классификации стадий сна.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.loader import EEGDataLoader
from src.features.extractor import FeatureExtractor
from src.models.classifiers import ModelFactory


def main():
    """Основная функция для обучения модели."""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Обучение модели классификации стадий сна')
    parser.add_argument('--config', type=str, default='config.yaml', help='Путь к конфигурационному файлу')
    parser.add_argument('--data', type=str, help='Путь к данным (переопределяет конфигурацию)')
    parser.add_argument('--annotations', type=str, help='Путь к аннотациям (переопределяет конфигурацию)')
    parser.add_argument('--output', type=str, help='Путь для сохранения модели (переопределяет конфигурацию)')
    parser.add_argument('--log-level', type=str, default='INFO', help='Уровень логирования')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Настраиваем логирование
    log_file = os.path.join(config.get_output_paths().get('logs_path', 'logs'), 
                           f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logger(
        name='training',
        level=args.log_level,
        log_file=log_file
    )
    
    logger.info("Начало обучения модели классификации стадий сна")
    
    try:
        # Создаем загрузчик данных
        data_loader = EEGDataLoader(config)
        
        # Определяем пути к данным
        if args.data:
            data_path = args.data
        else:
            data_path = config.get_data_path('train')
        
        if args.annotations:
            annotations_path = args.annotations
        else:
            annotations_path = None
        
        # Загружаем и предобрабатываем данные
        logger.info("Загрузка и предобработка данных")
        segments, labels = data_loader.load_and_preprocess(data_path, annotations_path)
        
        if labels is None:
            raise ValueError("Не удалось загрузить метки классов. Проверьте файл аннотаций.")
        
        # Извлекаем признаки
        logger.info("Извлечение признаков")
        feature_extractor = FeatureExtractor(config)
        features_df = feature_extractor.extract_features_from_segments(segments)
        
        # Убираем нечисловые колонки и NaN значения
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols].dropna()
        
        # Убираем колонку segment_id если она есть
        if 'segment_id' in features_df.columns:
            features_df = features_df.drop('segment_id', axis=1)
        
        # Обновляем метки в соответствии с обработанными признаками
        labels = labels[:len(features_df)]
        
        logger.info(f"Размер данных: {features_df.shape}")
        logger.info(f"Количество классов: {len(np.unique(labels))}")
        
        # Разделяем данные на обучающую и тестовую выборки
        test_size = config.get("model.test_size", 0.2)
        random_state = config.get("model.random_state", 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df.values, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        logger.info(f"Размер обучающей выборки: {X_train.shape}")
        logger.info(f"Размер тестовой выборки: {X_test.shape}")
        
        # Стандартизация признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Создаем и обучаем модель
        logger.info("Создание и обучение модели")
        classifier = ModelFactory.create_classifier(config)
        
        # Обучаем модель
        train_metrics = classifier.train(X_train_scaled, y_train)
        
        # Оцениваем модель на тестовых данных
        test_metrics = classifier.evaluate(X_test_scaled, y_test)
        
        # Сохраняем результаты
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_type': config.get("model.type"),
            'data_shape': features_df.shape,
            'training_date': datetime.now().isoformat(),
            'config': {
                'model_params': config.get_model_params(),
                'preprocessing_params': config.get_preprocessing_params(),
                'features_params': config.get_features_params()
            }
        }
        
        # Определяем путь для сохранения модели
        if args.output:
            model_path = args.output
        else:
            model_path = os.path.join(
                config.get_output_paths().get('model_path', 'models'),
                f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            )
        
        # Сохраняем модель
        classifier.save_model(model_path)
        
        # Сохраняем результаты
        results_path = os.path.join(
            config.get_output_paths().get('results_path', 'results'),
            f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Сохраняем scaler
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        import joblib
        joblib.dump(scaler, scaler_path)
        
        # Выводим результаты
        logger.info("Результаты обучения:")
        logger.info(f"Точность на обучающих данных: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"Точность на тестовых данных: {test_metrics['accuracy']:.4f}")
        logger.info(f"Модель сохранена в: {model_path}")
        logger.info(f"Результаты сохранены в: {results_path}")
        
        # Выводим детальный отчет
        logger.info("\nДетальный отчет по классам:")
        for class_name, metrics in test_metrics['classification_report'].items():
            if isinstance(metrics, dict):
                logger.info(f"{class_name}: precision={metrics.get('precision', 0):.3f}, "
                          f"recall={metrics.get('recall', 0):.3f}, "
                          f"f1-score={metrics.get('f1-score', 0):.3f}")
        
        logger.info("Обучение завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise


if __name__ == "__main__":
    main() 