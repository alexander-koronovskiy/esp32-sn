#!/usr/bin/env python3
"""
Интегрированная система детекции храпа
Реализует архитектуру: Fixed-point DSP → Классификатор → Буфер → Предиктор
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
import json
import logging

from src.utils.fixed_point_dsp import FixedPointDSP, FeatureQuantizer
from src.utils.pose_estimator import PoseEstimator
from src.utils.ring_buffers import WindowAggregator
from src.models.snoring_classifier_mcu import SnoringClassifierMCU
from src.models.snoring_predictor_tflite import SnoringPredictorTFLite

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnoringSystem:
    """Интегрированная система детекции храпа"""
    
    def __init__(self, config: Optional[Dict] = None):
        # Конфигурация по умолчанию
        self.config = {
            'classifier_frequency': 10,  # Гц
            'predictor_frequency': 1,    # Гц
            'audio_sample_rate': 8000,
            'audio_segment_length': 8000,  # 1 секунда
            'window_size_sec': 30,
            'model_paths': {
                'classifier': 'models/snoring_classifier_mcu.pkl',
                'predictor': 'models/snoring_predictor_tflite.pkl'
            }
        }
        
        if config:
            self.config.update(config)
        
        # Компоненты системы
        self.dsp = FixedPointDSP(sample_rate=self.config['audio_sample_rate'])
        self.pose_estimator = PoseEstimator()
        self.window_aggregator = WindowAggregator()
        
        # Модели
        self.classifier = SnoringClassifierMCU()
        self.predictor = SnoringPredictorTFLite()
        
        # Состояние системы
        self.is_running = False
        self.last_classifier_time = 0
        self.last_predictor_time = 0
        
        # Статистика
        self.stats = {
            'classifier_calls': 0,
            'predictor_calls': 0,
            'total_processing_time': 0,
            'errors': []
        }
        
        # Поток обработки
        self.processing_thread = None
        self.audio_buffer = []
        self.accel_buffer = []
    
    def load_models(self) -> bool:
        """Загрузка обученных моделей"""
        try:
            # Загрузка классификатора
            if self.config['model_paths']['classifier']:
                self.classifier.load_model(self.config['model_paths']['classifier'])
                logger.info("Классификатор загружен")
            
            # Загрузка предиктора
            if self.config['model_paths']['predictor']:
                self.predictor.load_model(self.config['model_paths']['predictor'])
                logger.info("Предиктор загружен")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            self.stats['errors'].append(str(e))
            return False
    
    def process_audio_segment(self, audio_data: np.ndarray) -> Dict:
        """Обработка аудио сегмента"""
        start_time = time.time()
        
        try:
            # Извлечение аудио признаков
            audio_features = self.dsp.compute_spectral_features(audio_data)
            
            # Создание заглушки для акселерометра (в реальной системе будет реальный акселерометр)
            accel_data = {
                'x': 0.0,
                'y': 0.0,
                'z': 9.81  # Гравитация
            }
            
            # Предсказание классификатора
            classifier_result = self.classifier.predict(audio_data, accel_data)
            
            # Создание мгновенных данных для буфера
            instant_data = {
                'label_3c': classifier_result['class_id'],
                'p_snore': classifier_result['p_snore'],
                'is_snoring': int(classifier_result['is_snoring']),
                'breath_energy': audio_features['breath_energy'],
                'snore_energy': audio_features['snore_energy'],
                'speech_energy': audio_features['speech_energy'],
                'pose_state': 0,  # Заглушка
                'pitch': 0,       # Заглушка
                'roll': 0         # Заглушка
            }
            
            # Добавление в буфер
            self.window_aggregator.push_instant_data(instant_data)
            
            processing_time = time.time() - start_time
            self.stats['classifier_calls'] += 1
            self.stats['total_processing_time'] += processing_time
            
            return {
                'success': True,
                'classifier_result': classifier_result,
                'processing_time_ms': processing_time * 1000
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки аудио: {e}")
            self.stats['errors'].append(str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_prediction(self) -> Dict:
        """Обработка предсказания"""
        start_time = time.time()
        
        try:
            # Предсказание эпизода храпа
            prediction_result = self.predictor.predict_snoring_episode(
                self.config['window_size_sec']
            )
            
            processing_time = time.time() - start_time
            self.stats['predictor_calls'] += 1
            
            return {
                'success': True,
                'prediction_result': prediction_result,
                'processing_time_ms': processing_time * 1000
            }
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            self.stats['errors'].append(str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_audio_data(self, audio_data: np.ndarray) -> None:
        """Добавление аудио данных в буфер"""
        self.audio_buffer.extend(audio_data)
        
        # Обработка полных сегментов
        segment_length = self.config['audio_segment_length']
        while len(self.audio_buffer) >= segment_length:
            segment = np.array(self.audio_buffer[:segment_length])
            self.audio_buffer = self.audio_buffer[segment_length:]
            
            # Обработка сегмента
            result = self.process_audio_segment(segment)
            
            if result['success']:
                logger.debug(f"Обработан аудио сегмент: {result['classifier_result']['class']}")
    
    def add_accelerometer_data(self, accel_x: float, accel_y: float, accel_z: float) -> None:
        """Добавление данных акселерометра"""
        # Обновление позы
        pose_data = self.pose_estimator.update_pose(accel_x, accel_y, accel_z)
        
        # Сохранение для использования в классификаторе
        self.accel_buffer.append({
            'x': accel_x,
            'y': accel_y,
            'z': accel_z,
            'pose': pose_data
        })
    
    def start_processing(self) -> None:
        """Запуск обработки в отдельном потоке"""
        if self.is_running:
            logger.warning("Система уже запущена")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Система обработки запущена")
    
    def stop_processing(self) -> None:
        """Остановка обработки"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        logger.info("Система обработки остановлена")
    
    def _processing_loop(self) -> None:
        """Основной цикл обработки"""
        while self.is_running:
            current_time = time.time()
            
            # Проверка необходимости запуска предиктора
            if (current_time - self.last_predictor_time) >= (1.0 / self.config['predictor_frequency']):
                result = self.process_prediction()
                if result['success']:
                    logger.info(f"Предсказание: {result['prediction_result']}")
                
                self.last_predictor_time = current_time
            
            # Небольшая задержка
            time.sleep(0.01)
    
    def get_system_status(self) -> Dict:
        """Получение статуса системы"""
        return {
            'is_running': self.is_running,
            'config': self.config,
            'stats': self.stats,
            'buffer_sizes': {
                'audio': len(self.audio_buffer),
                'accel': len(self.accel_buffer)
            },
            'model_info': {
                'classifier': self.classifier.get_model_info(),
                'predictor': self.predictor.get_model_info()
            }
        }
    
    def get_recent_results(self, num_results: int = 10) -> Dict:
        """Получение последних результатов"""
        # В реальной системе здесь будет кэш результатов
        return {
            'classifier_results': [],
            'predictor_results': [],
            'system_stats': self.stats
        }
    
    def save_system_state(self, filepath: str) -> None:
        """Сохранение состояния системы"""
        state = {
            'config': self.config,
            'stats': self.stats,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_system_state(self, filepath: str) -> None:
        """Загрузка состояния системы"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config.update(state.get('config', {}))
        self.stats.update(state.get('stats', {}))
    
    def benchmark_system(self, num_audio_segments: int = 100) -> Dict:
        """Бенчмарк системы"""
        logger.info("Запуск бенчмарка системы...")
        
        # Генерация тестовых данных
        audio_segments = []
        for i in range(num_audio_segments):
            # Синусоида с шумом
            t = np.linspace(0, 1, self.config['audio_segment_length'])
            freq = 100 + i * 10  # Разные частоты
            audio = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
            audio_segments.append(audio)
        
        # Бенчмарк классификатора
        classifier_times = []
        for audio in audio_segments:
            start_time = time.time()
            result = self.process_audio_segment(audio)
            end_time = time.time()
            
            if result['success']:
                classifier_times.append((end_time - start_time) * 1000)
        
        # Бенчмарк предиктора
        predictor_times = []
        for _ in range(num_audio_segments // 10):  # Меньше вызовов предиктора
            start_time = time.time()
            result = self.process_prediction()
            end_time = time.time()
            
            if result['success']:
                predictor_times.append((end_time - start_time) * 1000)
        
        return {
            'classifier': {
                'mean_time_ms': np.mean(classifier_times),
                'std_time_ms': np.std(classifier_times),
                'min_time_ms': np.min(classifier_times),
                'max_time_ms': np.max(classifier_times),
                'throughput_hz': 1000 / np.mean(classifier_times)
            },
            'predictor': {
                'mean_time_ms': np.mean(predictor_times),
                'std_time_ms': np.std(predictor_times),
                'min_time_ms': np.min(predictor_times),
                'max_time_ms': np.max(predictor_times),
                'throughput_hz': 1000 / np.mean(predictor_times)
            },
            'system': {
                'total_processing_time': self.stats['total_processing_time'],
                'classifier_calls': self.stats['classifier_calls'],
                'predictor_calls': self.stats['predictor_calls'],
                'errors': len(self.stats['errors'])
            }
        } 