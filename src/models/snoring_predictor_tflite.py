#!/usr/bin/env python3
"""
SnoringPredictor с TFLite Micro оптимизацией
INT8 квантование, статическая tensor arena
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import joblib
import time

try:
    import tensorflow as tf
    from tensorflow import keras
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("TensorFlow не установлен, TFLite функции недоступны")

from src.utils.ring_buffers import WindowAggregator

class SnoringPredictorTFLite:
    """SnoringPredictor с TFLite Micro оптимизацией"""
    
    def __init__(self, model_type: str = 'neural_network'):
        self.model_type = model_type
        self.model = None
        self.tflite_model = None
        self.interpreter = None
        
        # WindowAggregator для извлечения признаков
        self.window_aggregator = WindowAggregator()
        
        # Архитектура нейронной сети
        self.network_architecture = [
            keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ]
        
        # Параметры обучения
        self.training_params = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2
        }
        
        # Признаки для прогноза
        self.feature_names = [
            'snore_share', 'episode_len_sec', 'snore_transitions',
            'snore_energy_mean', 'snore_energy_max', 'breath_energy_mean',
            'snore_breath_ratio', 'pose_state_mode', 'pose_stability',
            'pitch_mean', 'pitch_std', 'roll_mean', 'roll_std',
            'pose_changes', 'accel_active_ratio', 'jerk_events',
            'hour_of_day', 'sleep_duration', 'sleep_stage',
            'window_sec'
        ]
        
        # Статистики для нормализации
        self.feature_stats = {
            'mu': None,
            'sigma': None
        }
    
    def create_model(self) -> keras.Model:
        """Создание нейронной сети"""
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        model = keras.Sequential(self.network_architecture)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.training_params['learning_rate']),
            loss=self.training_params['loss'],
            metrics=self.training_params['metrics']
        )
        
        return model
    
    def extract_prediction_features(self, window_sec: int = 30) -> np.ndarray:
        """Извлечение признаков для прогноза"""
        # Получение агрегированных признаков из WindowAggregator
        features = self.window_aggregator.aggregate_window_features(window_sec)
        
        # Создание вектора признаков в правильном порядке
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0.0))
        
        return np.array(feature_vector, dtype=np.float32)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Обучение модели"""
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        # Создание модели
        self.model = self.create_model()
        
        # Нормализация признаков
        self.feature_stats['mu'] = np.mean(X, axis=0)
        self.feature_stats['sigma'] = np.std(X, axis=0)
        self.feature_stats['sigma'] = np.where(self.feature_stats['sigma'] == 0, 1, self.feature_stats['sigma'])
        
        X_normalized = (X - self.feature_stats['mu']) / self.feature_stats['sigma']
        
        # Обучение
        history = self.model.fit(
            X_normalized, y,
            batch_size=self.training_params['batch_size'],
            epochs=self.training_params['epochs'],
            validation_split=self.training_params['validation_split'],
            verbose=0
        )
        
        # Оценка производительности
        y_pred = self.model.predict(X_normalized)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred_binary.flatten() == y)
        
        return {
            'accuracy': accuracy,
            'feature_count': X.shape[1],
            'model_type': self.model_type,
            'training_history': history.history
        }
    
    def convert_to_tflite(self, quantize: bool = True) -> bytes:
        """Конвертация модели в TFLite"""
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Конвертация в TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Post-training квантование INT8
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        self.tflite_model = converter.convert()
        return self.tflite_model
    
    def create_interpreter(self, model_content: bytes) -> tf.lite.Interpreter:
        """Создание интерпретатора TFLite"""
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        # Создание интерпретатора
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        
        return interpreter
    
    def predict_snoring_episode(self, window_sec: int = 30) -> Dict:
        """Предсказание эпизода храпа"""
        # Извлечение признаков
        features = self.extract_prediction_features(window_sec)
        
        # Нормализация
        if self.feature_stats['mu'] is not None:
            features_normalized = (features - self.feature_stats['mu']) / self.feature_stats['sigma']
        else:
            features_normalized = features
        
        # Предсказание
        if self.interpreter is not None:
            # TFLite предсказание
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Установка входных данных
            self.interpreter.set_tensor(input_details[0]['index'], features_normalized.reshape(1, -1))
            
            # Выполнение инференса
            self.interpreter.invoke()
            
            # Получение результата
            prediction = self.interpreter.get_tensor(output_details[0]['index'])[0][0]
        elif self.model is not None:
            # Keras предсказание
            prediction = self.model.predict(features_normalized.reshape(1, -1))[0][0]
        else:
            prediction = 0.0
        
        # Определение уровня риска
        risk_level = self._determine_risk_level(prediction)
        
        return {
            'snoring_probability': float(prediction),
            'risk_level': risk_level,
            'confidence': abs(prediction - 0.5) * 2  # Уверенность
        }
    
    def _determine_risk_level(self, probability: float) -> str:
        """Определение уровня риска"""
        if probability < 0.3:
            return 'low'
        elif probability < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def push_instant_data(self, data: Dict) -> None:
        """Добавление мгновенных данных в буфер"""
        self.window_aggregator.push_instant_data(data)
    
    def save_model(self, filepath: str) -> None:
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'feature_stats': self.feature_stats,
            'model_type': self.model_type,
            'training_params': self.training_params,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        
        # Сохранение TFLite модели
        if self.tflite_model is not None:
            tflite_path = filepath.replace('.pkl', '.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(self.tflite_model)
        
        # Сохранение метаданных
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'feature_stats': self.feature_stats,
            'training_params': self.training_params
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Загрузка модели"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_stats = model_data['feature_stats']
        self.model_type = model_data['model_type']
        self.training_params = model_data['training_params']
        self.feature_names = model_data['feature_names']
        
        # Загрузка TFLite модели
        tflite_path = filepath.replace('.pkl', '.tflite')
        try:
            with open(tflite_path, 'rb') as f:
                self.tflite_model = f.read()
            self.interpreter = self.create_interpreter(self.tflite_model)
        except FileNotFoundError:
            print("TFLite модель не найдена, будет использоваться Keras модель")
    
    def get_model_info(self) -> Dict:
        """Получение информации о модели"""
        return {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'feature_stats': self.feature_stats,
            'training_params': self.training_params,
            'tflite_available': self.tflite_model is not None
        }
    
    def benchmark_inference_time(self, num_runs: int = 100) -> Dict:
        """Бенчмарк времени инференса"""
        if self.model is None and self.interpreter is None:
            return {'error': 'Модель не загружена'}
        
        # Тестовые данные
        test_features = np.random.randn(num_runs, len(self.feature_names))
        
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            if self.interpreter is not None:
                # TFLite инференс
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], test_features[i:i+1])
                self.interpreter.invoke()
                _ = self.interpreter.get_tensor(output_details[0]['index'])
            else:
                # Keras инференс
                _ = self.model.predict(test_features[i:i+1], verbose=0)
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # в миллисекундах
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times)
        } 