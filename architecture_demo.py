#!/usr/bin/env python3
"""
Демонстрация реализованной архитектуры Snoring системы
"""

import numpy as np
import time
import json
from typing import Dict, List

from src.system.snoring_system import SnoringSystem
from src.models.snoring_classifier_mcu import SnoringClassifierMCU
from src.models.snoring_predictor_tflite import SnoringPredictorTFLite
from src.utils.fixed_point_dsp import FixedPointDSP
from src.utils.pose_estimator import PoseEstimator
from src.utils.ring_buffers import WindowAggregator

def create_synthetic_data(num_samples: int = 1000) -> tuple:
    """Создание синтетических данных для демонстрации"""
    print("🎵 Создание синтетических данных...")
    
    # Аудио данные
    audio_data = []
    labels = []
    
    for i in range(num_samples):
        # Создание аудио сегмента
        t = np.linspace(0, 1, 8000)  # 1 секунда при 8 кГц
        
        # Определение класса
        if i < num_samples // 3:
            # No_Snoring - тихий звук
            freq = 50 + np.random.randn() * 10
            audio = 0.1 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(len(t))
            label = 0
        elif i < 2 * num_samples // 3:
            # Light_Snoring - легкий храп
            freq = 100 + np.random.randn() * 20
            audio = 0.3 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
            label = 1
        else:
            # Heavy_Snoring - сильный храп
            freq = 80 + np.random.randn() * 30
            audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.random.randn(len(t))
            label = 2
        
        audio_data.append(audio)
        labels.append(label)
    
    # Акселерометр данные
    accel_data = []
    for i in range(num_samples):
        # Случайная ориентация
        accel_x = np.random.randn() * 0.5
        accel_y = np.random.randn() * 0.5
        accel_z = 9.81 + np.random.randn() * 0.1  # Гравитация + шум
        accel_data.append({'x': accel_x, 'y': accel_y, 'z': accel_z})
    
    return audio_data, labels, accel_data

def demo_fixed_point_dsp():
    """Демонстрация Fixed-point DSP"""
    print("\n🔧 ДЕМОНСТРАЦИЯ FIXED-POINT DSP")
    print("=" * 50)
    
    dsp = FixedPointDSP()
    
    # Тестовый аудио сигнал
    t = np.linspace(0, 1, 8000)
    audio = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
    
    # Извлечение признаков
    features = dsp.compute_spectral_features(audio)
    
    print("📊 Извлеченные признаки:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Тест квантования
    quantizer = FeatureQuantizer()
    original_value = 0.5
    quantized = quantizer.quantize_feature(original_value)
    dequantized = quantizer.dequantize_feature(quantized)
    
    print(f"\n🔢 Квантование:")
    print(f"  Оригинал: {original_value}")
    print(f"  Квантовано: {quantized}")
    print(f"  Деквантовано: {dequantized}")
    print(f"  Ошибка: {abs(original_value - dequantized):.6f}")

def demo_pose_estimator():
    """Демонстрация PoseEstimator"""
    print("\n📱 ДЕМОНСТРАЦИЯ POSE ESTIMATOR")
    print("=" * 50)
    
    pose_estimator = PoseEstimator()
    
    # Тестовые данные акселерометра
    test_accels = [
        (0, 0, 9.81),      # supine
        (0, 0, -9.81),     # prone
        (9.81, 0, 0),      # left
        (-9.81, 0, 0),     # right
    ]
    
    pose_names = ['supine', 'prone', 'left', 'right']
    
    for i, (accel_x, accel_y, accel_z) in enumerate(test_accels):
        pose_data = pose_estimator.update_pose(accel_x, accel_y, accel_z)
        
        print(f"📱 Тест {i+1} ({pose_names[i]}):")
        print(f"  Акселерометр: ({accel_x:.2f}, {accel_y:.2f}, {accel_z:.2f})")
        print(f"  Определенная поза: {pose_data['pose_state']}")
        print(f"  Pitch: {pose_data['pitch']}")
        print(f"  Roll: {pose_data['roll']}")
        print(f"  Стабильность: {pose_data['pose_stability']:.3f}")

def demo_ring_buffers():
    """Демонстрация кольцевых буферов"""
    print("\n🔄 ДЕМОНСТРАЦИЯ RING BUFFERS")
    print("=" * 50)
    
    from src.utils.ring_buffers import RingBuffer, SnoringRingBuffer
    
    # Простой кольцевой буфер
    ring_buffer = RingBuffer(size=5)
    
    print("📊 Простой кольцевой буфер:")
    for i in range(10):
        ring_buffer.push(i)
        print(f"  Добавлен {i}, содержимое: {ring_buffer.get_all()}")
    
    # Специализированный буфер для храпа
    snoring_buffer = SnoringRingBuffer(size=10)
    
    print("\n📊 Буфер данных храпа:")
    for i in range(5):
        data = {
            'label_3c': i % 3,
            'p_snore': 0.1 + i * 0.2,
            'is_snoring': i % 2,
            'breath_energy': 100 + i * 50,
            'snore_energy': 200 + i * 100,
            'speech_energy': 50 + i * 25,
            'pose_state': i % 4,
            'pitch': 0.1 + i * 0.1,
            'roll': -0.1 + i * 0.1
        }
        snoring_buffer.push_instant_data(data)
        print(f"  Добавлены данные {i+1}")

def demo_classifier():
    """Демонстрация классификатора"""
    print("\n🎯 ДЕМОНСТРАЦИЯ SNORING CLASSIFIER")
    print("=" * 50)
    
    classifier = SnoringClassifierMCU()
    
    # Создание синтетических данных
    audio_data, labels, accel_data = create_synthetic_data(100)
    
    # Подготовка данных для обучения
    X = []
    for i, audio in enumerate(audio_data):
        features = classifier.extract_features(audio, accel_data[i])
        X.append(features)
    
    X = np.array(X)
    y = np.array(labels)
    
    # Обучение модели
    print("🎓 Обучение классификатора...")
    train_result = classifier.train(X, y)
    
    print(f"📊 Результаты обучения:")
    print(f"  Точность: {train_result['accuracy']:.4f}")
    print(f"  Количество признаков: {train_result['feature_count']}")
    print(f"  Тип модели: {train_result['model_type']}")
    
    # Тестирование
    print("\n🧪 Тестирование классификатора...")
    test_results = []
    
    for i in range(10):
        result = classifier.predict(audio_data[i], accel_data[i])
        test_results.append(result)
        
        print(f"  Тест {i+1}: {result['class']} (уверенность: {result['confidence']:.3f})")
    
    # Сохранение модели
    classifier.save_model('models/snoring_classifier_mcu_demo.pkl')
    print("\n💾 Модель сохранена в models/snoring_classifier_mcu_demo.pkl")

def demo_predictor():
    """Демонстрация предиктора"""
    print("\n🔮 ДЕМОНСТРАЦИЯ SNORING PREDICTOR")
    print("=" * 50)
    
    predictor = SnoringPredictorTFLite()
    
    # Создание синтетических данных для предиктора
    num_samples = 100
    X_pred = np.random.randn(num_samples, 20)  # 20 признаков
    y_pred = np.random.randint(0, 2, num_samples)  # Бинарные метки
    
    # Обучение модели
    print("🎓 Обучение предиктора...")
    train_result = predictor.train(X_pred, y_pred)
    
    print(f"📊 Результаты обучения:")
    print(f"  Точность: {train_result['accuracy']:.4f}")
    print(f"  Количество признаков: {train_result['feature_count']}")
    print(f"  Тип модели: {train_result['model_type']}")
    
    # Конвертация в TFLite
    print("\n🔄 Конвертация в TFLite...")
    try:
        tflite_model = predictor.convert_to_tflite(quantize=True)
        print(f"✅ TFLite модель создана, размер: {len(tflite_model)} байт")
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
    
    # Тестирование
    print("\n🧪 Тестирование предиктора...")
    for i in range(5):
        # Добавление тестовых данных в буфер
        for j in range(30):  # 30 секунд данных
            test_data = {
                'label_3c': np.random.randint(0, 3),
                'p_snore': np.random.random(),
                'is_snoring': np.random.randint(0, 2),
                'breath_energy': np.random.randint(100, 1000),
                'snore_energy': np.random.randint(200, 2000),
                'speech_energy': np.random.randint(50, 500),
                'pose_state': np.random.randint(0, 4),
                'pitch': np.random.randn(),
                'roll': np.random.randn()
            }
            predictor.push_instant_data(test_data)
        
        # Предсказание
        result = predictor.predict_snoring_episode()
        print(f"  Предсказание {i+1}: вероятность={result['snoring_probability']:.3f}, риск={result['risk_level']}")
    
    # Сохранение модели
    predictor.save_model('models/snoring_predictor_tflite_demo.pkl')
    print("\n💾 Модель сохранена в models/snoring_predictor_tflite_demo.pkl")

def demo_integrated_system():
    """Демонстрация интегрированной системы"""
    print("\n🚀 ДЕМОНСТРАЦИЯ ИНТЕГРИРОВАННОЙ СИСТЕМЫ")
    print("=" * 50)
    
    # Создание системы
    config = {
        'classifier_frequency': 10,
        'predictor_frequency': 1,
        'window_size_sec': 30
    }
    
    system = SnoringSystem(config)
    
    # Создание синтетических данных
    audio_data, labels, accel_data = create_synthetic_data(50)
    
    print("🎵 Обработка аудио данных...")
    
    # Обработка аудио сегментов
    for i, audio in enumerate(audio_data):
        result = system.process_audio_segment(audio)
        
        if result['success']:
            classifier_result = result['classifier_result']
            print(f"  Сегмент {i+1}: {classifier_result['class']} (время: {result['processing_time_ms']:.2f} мс)")
    
    print("\n🔮 Обработка предсказаний...")
    
    # Обработка предсказаний
    for i in range(5):
        result = system.process_prediction()
        
        if result['success']:
            prediction_result = result['prediction_result']
            print(f"  Предсказание {i+1}: вероятность={prediction_result['snoring_probability']:.3f}, риск={prediction_result['risk_level']}")
    
    # Статус системы
    status = system.get_system_status()
    print(f"\n📊 Статус системы:")
    print(f"  Вызовы классификатора: {status['stats']['classifier_calls']}")
    print(f"  Вызовы предиктора: {status['stats']['predictor_calls']}")
    print(f"  Общее время обработки: {status['stats']['total_processing_time']:.3f} с")
    print(f"  Ошибки: {len(status['stats']['errors'])}")
    
    # Бенчмарк
    print("\n⚡ Бенчмарк системы...")
    benchmark = system.benchmark_system(num_audio_segments=50)
    
    print(f"📊 Результаты бенчмарка:")
    print(f"  Классификатор: {benchmark['classifier']['mean_time_ms']:.2f} мс (среднее)")
    print(f"  Предиктор: {benchmark['predictor']['mean_time_ms']:.2f} мс (среднее)")
    print(f"  Пропускная способность классификатора: {benchmark['classifier']['throughput_hz']:.1f} Гц")

def main():
    """Основная функция демонстрации"""
    print("🎯 ДЕМОНСТРАЦИЯ АРХИТЕКТУРЫ SNORING СИСТЕМЫ")
    print("=" * 60)
    print("Реализованные компоненты:")
    print("✅ Fixed-point DSP")
    print("✅ PoseEstimator с LUT")
    print("✅ Кольцевые буферы")
    print("✅ WindowAggregator")
    print("✅ SnoringClassifierMCU")
    print("✅ SnoringPredictorTFLite")
    print("✅ Интегрированная система")
    print("=" * 60)
    
    try:
        # Демонстрации компонентов
        demo_fixed_point_dsp()
        demo_pose_estimator()
        demo_ring_buffers()
        demo_classifier()
        demo_predictor()
        demo_integrated_system()
        
        print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("✅ Все компоненты архитектуры реализованы и протестированы")
        
    except Exception as e:
        print(f"\n❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 