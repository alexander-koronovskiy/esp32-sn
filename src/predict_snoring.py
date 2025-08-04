"""
Скрипт для предсказания храпа в реальном времени.
Поддерживает детекцию текущего храпа и предсказание будущего храпа.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.snoring_extractor import SnoringFeatureExtractor
from src.models.snoring_classifier import SnoringClassifier


def load_audio_data(file_path: str, sampling_rate: int = 8000) -> np.ndarray:
    """
    Загружает аудио данные из файла.
    
    Args:
        file_path: Путь к аудио файлу
        sampling_rate: Частота дискретизации
        
    Returns:
        Аудио данные
    """
    # Простая загрузка аудио (можно расширить для разных форматов)
    try:
        # Пытаемся загрузить как numpy массив
        audio_data = np.load(file_path)
    except:
        try:
            # Пытаемся загрузить как CSV
            audio_data = pd.read_csv(file_path).values.flatten()
        except:
            # Создаем синтетические данные для демонстрации
            print(f"⚠️ Не удалось загрузить {file_path}, создаем синтетические данные")
            duration_seconds = 30
            t = np.linspace(0, duration_seconds, duration_seconds * sampling_rate)
            
            # Создаем синтетический храп
            audio_data = (np.sin(2 * np.pi * 80 * t) * 0.3 +
                         np.sin(2 * np.pi * 150 * t) * 0.2 +
                         np.random.randn(len(t)) * 0.1)
    
    return audio_data


def segment_audio_realtime(audio_data: np.ndarray, segment_length: int = 8000, 
                          overlap: float = 0.5) -> List[np.ndarray]:
    """
    Сегментирует аудио данные для обработки в реальном времени.
    
    Args:
        audio_data: Аудио данные
        segment_length: Длина сегмента в сэмплах
        overlap: Перекрытие сегментов
        
    Returns:
        Список сегментов
    """
    segments = []
    step = int(segment_length * (1 - overlap))
    
    for i in range(0, len(audio_data) - segment_length + 1, step):
        segment = audio_data[i:i + segment_length]
        segments.append(segment)
    
    return segments


def predict_snoring_realtime(audio_segments: List[np.ndarray], 
                           classifier: SnoringClassifier,
                           feature_extractor: SnoringFeatureExtractor,
                           time_horizon: int = 30) -> List[Dict]:
    """
    Предсказывает храп в реальном времени.
    
    Args:
        audio_segments: Сегменты аудио
        classifier: Обученный классификатор
        feature_extractor: Экстрактор признаков
        time_horizon: Горизонт предсказания в секундах
        
    Returns:
        Список результатов предсказания
    """
    results = []
    
    for i, segment in enumerate(audio_segments):
        # Детекция в текущем окне
        current_result = classifier.detect_snoring_window(segment, feature_extractor)
        
        # Предсказание будущего храпа
        features = feature_extractor.extract_snoring_features(segment)
        feature_vector = np.array(list(features.values()))
        
        # Используем историю для предсказания (последние 5 сегментов)
        history_features = None
        if i >= 5:
            history_segments = audio_segments[i-5:i]
            history_features_list = []
            for hist_segment in history_segments:
                hist_features = feature_extractor.extract_snoring_features(hist_segment)
                history_features_list.append(list(hist_features.values()))
            history_features = np.array(history_features_list)
        
        future_prediction = classifier.predict_snoring_future(
            feature_vector, history_features, time_horizon
        )
        
        # Объединяем результаты
        result = {
            'segment_id': i,
            'timestamp_seconds': i,  # Время в секундах
            'current_detection': current_result,
            'future_prediction': future_prediction,
            'features': features
        }
        
        results.append(result)
    
    return results


def main():
    """Основная функция для предсказания храпа."""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Предсказание храпа в реальном времени')
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели')
    parser.add_argument('--audio', type=str, help='Путь к аудио файлу')
    parser.add_argument('--output', type=str, help='Путь для сохранения результатов')
    parser.add_argument('--time-horizon', type=int, default=30, 
                       help='Горизонт предсказания в секундах')
    parser.add_argument('--sampling-rate', type=int, default=8000, 
                       help='Частота дискретизации аудио')
    parser.add_argument('--segment-length', type=int, default=8000, 
                       help='Длина сегмента в сэмплах')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ПРЕДСКАЗАНИЕ ХРАПА В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 60)
    
    try:
        # 1. Загружаем модель
        print("1. Загрузка модели")
        classifier = SnoringClassifier()
        classifier.load_model(args.model)
        print(f"✓ Модель загружена: {args.model}")
        print(f"  Тип модели: {classifier.model_type}")
        print(f"  Классы: {classifier.class_names}")
        
        # 2. Создаем экстрактор признаков
        print("\n2. Создание экстрактора признаков")
        feature_extractor = SnoringFeatureExtractor(
            sampling_rate=args.sampling_rate,
            segment_length=args.segment_length
        )
        print(f"✓ Экстрактор создан:")
        print(f"  Частота дискретизации: {args.sampling_rate} Hz")
        print(f"  Длина сегмента: {args.segment_length} сэмплов")
        
        # 3. Загружаем аудио данные
        print("\n3. Загрузка аудио данных")
        if args.audio:
            audio_data = load_audio_data(args.audio, args.sampling_rate)
        else:
            # Создаем синтетические данные для демонстрации
            print("⚠️ Аудио файл не указан, создаем синтетические данные")
            duration_seconds = 60
            t = np.linspace(0, duration_seconds, duration_seconds * args.sampling_rate)
            
            # Создаем реалистичный храп
            audio_data = np.zeros_like(t)
            
            # Первые 20 секунд - тишина
            audio_data[:20 * args.sampling_rate] = np.random.randn(20 * args.sampling_rate) * 0.05
            
            # 20-40 секунд - легкий храп
            start_idx = 20 * args.sampling_rate
            end_idx = 40 * args.sampling_rate
            audio_data[start_idx:end_idx] = (
                np.sin(2 * np.pi * 50 * t[:end_idx-start_idx]) * 0.2 +
                np.sin(2 * np.pi * 100 * t[:end_idx-start_idx]) * 0.15 +
                np.random.randn(end_idx-start_idx) * 0.1
            )
            
            # 40-60 секунд - сильный храп
            start_idx = 40 * args.sampling_rate
            end_idx = 60 * args.sampling_rate
            audio_data[start_idx:end_idx] = (
                np.sin(2 * np.pi * 80 * t[:end_idx-start_idx]) * 0.4 +
                np.sin(2 * np.pi * 150 * t[:end_idx-start_idx]) * 0.3 +
                np.sin(2 * np.pi * 300 * t[:end_idx-start_idx]) * 0.2 +
                np.random.randn(end_idx-start_idx) * 0.15
            )
        
        print(f"✓ Аудио данные загружены:")
        print(f"  Длительность: {len(audio_data) / args.sampling_rate:.1f} секунд")
        print(f"  Количество сэмплов: {len(audio_data)}")
        
        # 4. Сегментация аудио
        print("\n4. Сегментация аудио")
        segments = segment_audio_realtime(
            audio_data, 
            segment_length=args.segment_length,
            overlap=0.5
        )
        print(f"✓ Создано {len(segments)} сегментов")
        
        # 5. Предсказание храпа
        print("\n5. Предсказание храпа")
        results = predict_snoring_realtime(
            segments, classifier, feature_extractor, args.time_horizon
        )
        print(f"✓ Предсказания выполнены для {len(results)} сегментов")
        
        # 6. Анализ результатов
        print("\n6. Анализ результатов")
        snoring_detections = [r['current_detection']['is_snoring'] for r in results]
        snoring_percentage = np.mean(snoring_detections) * 100
        
        risk_levels = [r['future_prediction']['risk_level'] for r in results]
        high_risk_count = risk_levels.count('high')
        medium_risk_count = risk_levels.count('medium')
        
        print(f"✓ Анализ завершен:")
        print(f"  Детекция храпа: {snoring_percentage:.1f}% времени")
        print(f"  Высокий риск: {high_risk_count} сегментов")
        print(f"  Средний риск: {medium_risk_count} сегментов")
        
        # 7. Сохранение результатов
        print("\n7. Сохранение результатов")
        if args.output:
            output_path = args.output
        else:
            os.makedirs('results', exist_ok=True)
            output_path = f'results/snoring_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Подготавливаем результаты для сохранения
        save_results = []
        for result in results:
            save_result = {
                'segment_id': result['segment_id'],
                'timestamp_seconds': result['timestamp_seconds'],
                'current_detection': {
                    'is_snoring': result['current_detection']['is_snoring'],
                    'snoring_intensity': result['current_detection']['snoring_intensity'],
                    'predicted_class': result['current_detection']['predicted_class'],
                    'confidence': result['current_detection']['confidence']
                },
                'future_prediction': {
                    'snoring_probability': result['future_prediction']['snoring_probability'],
                    'risk_level': result['future_prediction']['risk_level'],
                    'time_horizon_seconds': result['future_prediction']['time_horizon_seconds'],
                    'prediction_confidence': result['future_prediction']['prediction_confidence']
                }
            }
            save_results.append(save_result)
        
        # Сохраняем результаты
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        # Создаем CSV отчет
        csv_path = output_path.replace('.json', '.csv')
        df_results = pd.DataFrame(save_results)
        df_results.to_csv(csv_path, index=False)
        
        print(f"✓ Результаты сохранены:")
        print(f"  📄 {output_path} - JSON результаты")
        print(f"  📄 {csv_path} - CSV отчет")
        
        # 8. Вывод детального отчета
        print("\n8. Детальный отчет")
        print("\n📊 СТАТИСТИКА ДЕТЕКЦИИ ХРАПА:")
        print("=" * 40)
        
        # Статистика по классам
        class_counts = {}
        for result in results:
            class_name = result['current_detection']['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {class_name}: {count} сегментов ({percentage:.1f}%)")
        
        # Статистика по риску
        print(f"\n📊 СТАТИСТИКА РИСКА ХРАПА:")
        print("=" * 40)
        
        risk_counts = {}
        for result in results:
            risk_level = result['future_prediction']['risk_level']
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        for risk_level, count in risk_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {risk_level.upper()}: {count} сегментов ({percentage:.1f}%)")
        
        # Средние вероятности
        avg_snoring_prob = np.mean([r['future_prediction']['snoring_probability'] for r in results])
        avg_confidence = np.mean([r['current_detection']['confidence'] for r in results])
        
        print(f"\n📊 СРЕДНИЕ ПОКАЗАТЕЛИ:")
        print("=" * 40)
        print(f"  Средняя вероятность храпа: {avg_snoring_prob:.3f}")
        print(f"  Средняя уверенность детекции: {avg_confidence:.3f}")
        print(f"  Горизонт предсказания: {args.time_horizon} секунд")
        
        print(f"\n✓ Предсказание храпа завершено успешно!")
        
    except Exception as e:
        print(f"\n✗ Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main() 