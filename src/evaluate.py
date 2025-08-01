"""
Скрипт для оценки модели классификации стадий сна.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.loader import EEGDataLoader
from src.features.extractor import FeatureExtractor
from src.models.classifiers import SleepStageClassifier


def plot_confusion_matrix(conf_matrix, class_names, output_path):
    """Создает и сохраняет матрицу ошибок."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(y_true, y_pred, output_path):
    """Создает и сохраняет график распределения классов."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Истинные классы
    true_counts = pd.Series(y_true).value_counts().sort_index()
    ax1.bar(range(len(true_counts)), true_counts.values)
    ax1.set_title('Распределение истинных классов')
    ax1.set_xlabel('Класс')
    ax1.set_ylabel('Количество')
    ax1.set_xticks(range(len(true_counts)))
    ax1.set_xticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    
    # Предсказанные классы
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    ax2.bar(range(len(pred_counts)), pred_counts.values)
    ax2.set_title('Распределение предсказанных классов')
    ax2.set_xlabel('Класс')
    ax2.set_ylabel('Количество')
    ax2.set_xticks(range(len(pred_counts)))
    ax2.set_xticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Основная функция для оценки модели."""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Оценка модели классификации стадий сна')
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели')
    parser.add_argument('--data', type=str, required=True, help='Путь к данным ЭЭГ')
    parser.add_argument('--annotations', type=str, required=True, help='Путь к аннотациям')
    parser.add_argument('--output', type=str, help='Путь для сохранения результатов')
    parser.add_argument('--config', type=str, default='config.yaml', help='Путь к конфигурационному файлу')
    parser.add_argument('--log-level', type=str, default='INFO', help='Уровень логирования')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Настраиваем логирование
    log_file = os.path.join(config.get_output_paths().get('logs_path', 'logs'), 
                           f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logger(
        name='evaluation',
        level=args.log_level,
        log_file=log_file
    )
    
    logger.info("Начало оценки модели классификации стадий сна")
    
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
        segments, labels = data_loader.load_and_preprocess(args.data, args.annotations)
        
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
        
        logger.info(f"Размер данных для оценки: {features_df.shape}")
        logger.info(f"Количество классов: {len(np.unique(labels))}")
        
        # Нормализуем признаки если есть scaler
        if scaler is not None:
            features_scaled = scaler.transform(features_df.values)
        else:
            features_scaled = features_df.values
        
        # Оцениваем модель
        logger.info("Оценка модели")
        evaluation_results = classifier.evaluate(features_scaled, labels)
        
        # Создаем результаты оценки
        results = {
            'evaluation_metrics': evaluation_results,
            'evaluation_date': datetime.now().isoformat(),
            'model_path': args.model,
            'data_path': args.data,
            'annotations_path': args.annotations,
            'data_shape': features_df.shape,
            'class_distribution': {
                'true': pd.Series(labels).value_counts().to_dict(),
                'predicted': pd.Series(evaluation_results['predictions']).value_counts().to_dict()
            }
        }
        
        # Определяем путь для сохранения результатов
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                config.get_output_paths().get('results_path', 'results'),
                f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
        
        # Сохраняем результаты
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Создаем визуализации
        plots_dir = os.path.join(os.path.dirname(output_path), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Матрица ошибок
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        conf_matrix = np.array(evaluation_results['confusion_matrix'])
        plot_confusion_matrix(conf_matrix, class_names, 
                            os.path.join(plots_dir, 'confusion_matrix.png'))
        
        # Распределение классов
        plot_class_distribution(labels, evaluation_results['predictions'],
                              os.path.join(plots_dir, 'class_distribution.png'))
        
        # Создаем DataFrame с результатами
        results_df = pd.DataFrame({
            'segment_id': range(len(labels)),
            'true_stage': [class_names[l] for l in labels],
            'predicted_stage': [class_names[p] for p in evaluation_results['predictions']],
            'correct': [t == p for t, p in zip(labels, evaluation_results['predictions'])],
            'confidence': np.max(evaluation_results['probabilities'], axis=1)
        })
        
        # Добавляем вероятности для каждого класса
        for i, stage in enumerate(class_names):
            results_df[f'prob_{stage}'] = evaluation_results['probabilities'][:, i]
        
        # Сохраняем CSV файл
        csv_path = output_path.replace('.json', '.csv')
        results_df.to_csv(csv_path, index=False)
        
        # Выводим результаты
        logger.info("Результаты оценки:")
        logger.info(f"Общая точность: {evaluation_results['accuracy']:.4f}")
        
        # Детальный отчет по классам
        logger.info("\nДетальный отчет по классам:")
        for class_name, metrics in evaluation_results['classification_report'].items():
            if isinstance(metrics, dict):
                logger.info(f"{class_name}: precision={metrics.get('precision', 0):.3f}, "
                          f"recall={metrics.get('recall', 0):.3f}, "
                          f"f1-score={metrics.get('f1-score', 0):.3f}")
        
        # Статистика по классам
        logger.info("\nРаспределение классов:")
        true_dist = pd.Series(labels).value_counts().sort_index()
        pred_dist = pd.Series(evaluation_results['predictions']).value_counts().sort_index()
        
        for i, stage in enumerate(class_names):
            true_count = true_dist.get(i, 0)
            pred_count = pred_dist.get(i, 0)
            logger.info(f"{stage}: истинных={true_count}, предсказанных={pred_count}")
        
        # Анализ ошибок
        errors = results_df[~results_df['correct']]
        logger.info(f"\nКоличество ошибок: {len(errors)} ({len(errors)/len(results_df)*100:.1f}%)")
        
        if len(errors) > 0:
            logger.info("Топ-5 ошибок по уверенности:")
            top_errors = errors.nlargest(5, 'confidence')
            for _, row in top_errors.iterrows():
                logger.info(f"Сегмент {row['segment_id']}: {row['true_stage']} -> {row['predicted_stage']} "
                           f"(уверенность: {row['confidence']:.3f})")
        
        logger.info(f"Результаты сохранены в: {output_path}")
        logger.info(f"CSV файл сохранен в: {csv_path}")
        logger.info(f"Графики сохранены в: {plots_dir}")
        
        logger.info("Оценка завершена успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при оценке модели: {str(e)}")
        raise


if __name__ == "__main__":
    main() 