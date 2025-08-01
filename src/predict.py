"""
Скрипт для предсказания стадий сна.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.loader import EEGDataLoader
from src.features.extractor import FeatureExtractor
from src.models.classifiers import SleepStageClassifier


def main():
    """Основная функция для предсказания стадий сна."""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Предсказание стадий сна')
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели')
    parser.add_argument('--data', type=str, required=True, help='Путь к данным ЭЭГ')
    parser.add_argument('--output', type=str, help='Путь для сохранения результатов')
    parser.add_argument('--config', type=str, default='config.yaml', help='Путь к конфигурационному файлу')
    parser.add_argument('--log-level', type=str, default='INFO', help='Уровень логирования')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Настраиваем логирование
    log_file = os.path.join(config.get_output_paths().get('logs_path', 'logs'), 
                           f'prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logger(
        name='prediction',
        level=args.log_level,
        log_file=log_file
    )
    
    logger.info("Начало предсказания стадий сна")
    
    try:
        # Создаем классификатор и загружаем модель
        classifier = SleepStageClassifier(config)
        classifier.load_model(args.model)
        
        # Загружаем scaler если есть
        scaler_path = args.model.replace('.pkl', '_scaler.pkl')
        scaler = None
        if os.path.exists(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)
            logger.info("Загружен scaler для нормализации признаков")
        
        # Создаем загрузчик данных
        data_loader = EEGDataLoader(config)
        
        # Загружаем и предобрабатываем данные
        logger.info("Загрузка и предобработка данных")
        segments, _ = data_loader.load_and_preprocess(args.data)
        
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
        
        logger.info(f"Размер данных для предсказания: {features_df.shape}")
        
        # Нормализуем признаки если есть scaler
        if scaler is not None:
            features_scaled = scaler.transform(features_df.values)
        else:
            features_scaled = features_df.values
        
        # Делаем предсказания
        logger.info("Выполнение предсказаний")
        predictions = classifier.predict(features_scaled)
        probabilities = classifier.predict_proba(features_scaled)
        
        # Создаем результаты
        results = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'segment_ids': list(range(len(predictions))),
            'prediction_date': datetime.now().isoformat(),
            'model_path': args.model,
            'data_path': args.data
        }
        
        # Определяем путь для сохранения результатов
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                config.get_output_paths().get('results_path', 'results'),
                f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
        
        # Сохраняем результаты
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Создаем DataFrame с результатами
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        results_df = pd.DataFrame({
            'segment_id': range(len(predictions)),
            'predicted_stage': [stage_names[p] for p in predictions],
            'predicted_stage_id': predictions,
            'confidence': np.max(probabilities, axis=1)
        })
        
        # Добавляем вероятности для каждого класса
        for i, stage in enumerate(stage_names):
            results_df[f'prob_{stage}'] = probabilities[:, i]
        
        # Сохраняем CSV файл
        csv_path = output_path.replace('.json', '.csv')
        results_df.to_csv(csv_path, index=False)
        
        # Выводим статистику
        logger.info("Статистика предсказаний:")
        stage_counts = results_df['predicted_stage'].value_counts()
        for stage, count in stage_counts.items():
            percentage = (count / len(results_df)) * 100
            logger.info(f"{stage}: {count} ({percentage:.1f}%)")
        
        logger.info(f"Результаты сохранены в: {output_path}")
        logger.info(f"CSV файл сохранен в: {csv_path}")
        
        # Выводим несколько примеров предсказаний
        logger.info("\nПримеры предсказаний:")
        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            logger.info(f"Сегмент {row['segment_id']}: {row['predicted_stage']} "
                       f"(уверенность: {row['confidence']:.3f})")
        
        logger.info("Предсказание завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise


if __name__ == "__main__":
    main() 