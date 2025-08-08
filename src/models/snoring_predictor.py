"""
Специализированная модель для предсказания вероятности эпизода храпа.
Использует нейронную сеть TensorFlow Lite для интегративного анализа.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import json
from datetime import datetime


class SnoringPredictor:
    """Модель для предсказания вероятности эпизода храпа."""
    
    def __init__(self, model_type: str = 'neural_network', input_dim: int = 30, 
                 hidden_layers: List[int] = [64, 32, 16], dropout_rate: float = 0.2, 
                 random_state: int = 42):
        """
        Инициализация модели предсказания храпа.
        
        Args:
            model_type: Тип модели ('neural_network', 'random_forest', 'svm')
            input_dim: Размерность входных признаков
            hidden_layers: Архитектура скрытых слоев
            dropout_rate: Коэффициент dropout
            random_state: Случайное состояние
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_tflite = False
        self.tflite_model = None
        
        # Пороги риска
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7,
            'high': 0.9
        }
        
    def create_neural_network(self) -> tf.keras.Model:
        """
        Создает нейронную сеть TensorFlow.
        
        Returns:
            Модель нейронной сети
        """
        model = tf.keras.Sequential([
            # Входной слой
            tf.keras.layers.Dense(self.hidden_layers[0], activation='relu', 
                                 input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # Скрытые слои
            tf.keras.layers.Dense(self.hidden_layers[1], activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            tf.keras.layers.Dense(self.hidden_layers[2], activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # Выходной слой
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Компиляция
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_model(self) -> Any:
        """
        Создает модель в зависимости от типа.
        
        Returns:
            Модель классификатора
        """
        if self.model_type == 'neural_network':
            return self.create_neural_network()
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
    
    def extract_prediction_features(self, audio_data: np.ndarray, 
                                  accelerometer_data: Optional[np.ndarray] = None,
                                  temporal_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Извлекает признаки для предсказания храпа.
        
        Args:
            audio_data: Аудио данные
            accelerometer_data: Данные акселерометра (опционально)
            temporal_data: Временные данные (опционально)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        # Аудио-признаки (15 признаков)
        audio_features = self._extract_audio_features(audio_data)
        features.update(audio_features)
        
        # Данные акселерометра (5-8 признаков)
        if accelerometer_data is not None:
            accel_features = self._extract_accelerometer_features(accelerometer_data)
            features.update(accel_features)
        
        # Временные признаки (5-7 признаков)
        if temporal_data is not None:
            temporal_features = self._extract_temporal_features(temporal_data)
            features.update(temporal_features)
        
        return features
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Извлекает аудио-признаки."""
        features = {}
        
        # Базовые признаки
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        features['avg_power'] = np.mean(audio_data**2)
        features['log_energy'] = np.log(np.sum(audio_data**2) + 1e-10)
        features['signal_range'] = np.max(audio_data) - np.min(audio_data)
        
        # Статистические признаки
        features['std'] = np.std(audio_data)
        features['skewness'] = self._calculate_skewness(audio_data)
        features['kurtosis'] = self._calculate_kurtosis(audio_data)
        
        # Спектральные признаки
        features['spectral_centroid'] = self._calculate_spectral_centroid(audio_data)
        features['spectral_bandwidth'] = self._calculate_spectral_bandwidth(audio_data)
        features['spectral_entropy'] = self._calculate_spectral_entropy(audio_data)
        
        # Дополнительные признаки
        features['zero_crossing_rate'] = self._calculate_zero_crossing_rate(audio_data)
        features['peak_count'] = len(self._find_peaks(audio_data)[0])
        features['abs_mean'] = np.mean(np.abs(audio_data))
        features['energy'] = np.sum(audio_data**2)
        features['temporal_entropy'] = self._calculate_temporal_entropy(audio_data)
        features['autocorrelation_peak'] = self._calculate_autocorrelation_peak(audio_data)
        features['autocorrelation_decay'] = self._calculate_autocorrelation_decay(audio_data)
        
        return features
    
    def _extract_accelerometer_features(self, accelerometer_data: np.ndarray) -> Dict[str, float]:
        """Извлекает признаки акселерометра."""
        features = {}
        
        # Базовые признаки движения
        features['avg_activity'] = np.mean(np.abs(accelerometer_data))
        features['acceleration'] = np.std(accelerometer_data)
        features['jitter'] = self._calculate_jitter(accelerometer_data)
        features['movement_intensity'] = self._calculate_movement_intensity(accelerometer_data)
        features['position_stability'] = self._calculate_position_stability(accelerometer_data)
        
        return features
    
    def _extract_temporal_features(self, temporal_data: Dict) -> Dict[str, float]:
        """Извлекает временные признаки."""
        features = {}
        
        # Временные признаки
        features['hour_of_day'] = temporal_data.get('hour', 0) / 24.0  # Нормализация 0-1
        features['sleep_duration'] = temporal_data.get('sleep_duration', 0) / 3600.0  # Часы
        features['cyclicity'] = temporal_data.get('cyclicity', 0)
        features['time_since_last_movement'] = temporal_data.get('time_since_last_movement', 0) / 60.0  # Минуты
        features['sleep_phase'] = temporal_data.get('sleep_phase', 0) / 4.0  # Нормализация 0-1
        features['sleep_depth'] = temporal_data.get('sleep_depth', 0)
        features['sleep_cycle'] = temporal_data.get('sleep_cycle', 0) / 10.0  # Нормализация
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Вычисляет асимметрию."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Вычисляет эксцесс."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_spectral_centroid(self, data: np.ndarray) -> float:
        """Вычисляет спектральный центроид."""
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        magnitude = np.abs(fft)
        if np.sum(magnitude) == 0:
            return 0
        return np.sum(freqs * magnitude) / np.sum(magnitude)
    
    def _calculate_spectral_bandwidth(self, data: np.ndarray) -> float:
        """Вычисляет спектральную ширину."""
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        magnitude = np.abs(fft)
        centroid = self._calculate_spectral_centroid(data)
        if np.sum(magnitude) == 0:
            return 0
        return np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
    
    def _calculate_spectral_entropy(self, data: np.ndarray) -> float:
        """Вычисляет спектральную энтропию."""
        fft = np.fft.fft(data)
        magnitude = np.abs(fft)
        magnitude = magnitude / np.sum(magnitude)
        magnitude = magnitude[magnitude > 0]
        if len(magnitude) == 0:
            return 0
        return -np.sum(magnitude * np.log2(magnitude))
    
    def _calculate_zero_crossing_rate(self, data: np.ndarray) -> float:
        """Вычисляет частоту пересечения нуля."""
        return np.sum(np.diff(np.signbit(data))) / (2 * len(data))
    
    def _find_peaks(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Находит пики в сигнале."""
        from scipy.signal import find_peaks
        return find_peaks(data, height=0.1*np.max(data))
    
    def _calculate_temporal_entropy(self, data: np.ndarray) -> float:
        """Вычисляет временную энтропию."""
        hist, _ = np.histogram(data, bins=50)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_autocorrelation_peak(self, data: np.ndarray) -> float:
        """Вычисляет пик автокорреляции."""
        autocorr = np.correlate(data, data, mode='full')
        return np.max(autocorr[len(data)-1:])
    
    def _calculate_autocorrelation_decay(self, data: np.ndarray) -> float:
        """Вычисляет спад автокорреляции."""
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(data)-1:]
        if len(autocorr) < 2:
            return 0
        return autocorr[1] / autocorr[0] if autocorr[0] != 0 else 0
    
    def _calculate_jitter(self, data: np.ndarray) -> float:
        """Вычисляет дрожание акселерометра."""
        return np.std(np.diff(data))
    
    def _calculate_movement_intensity(self, data: np.ndarray) -> float:
        """Вычисляет интенсивность движения."""
        return np.mean(np.abs(np.diff(data)))
    
    def _calculate_position_stability(self, data: np.ndarray) -> float:
        """Вычисляет стабильность положения."""
        return 1.0 / (1.0 + np.std(data))
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None,
              validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """
        Обучает модель предсказания храпа.
        
        Args:
            X: Признаки для обучения
            y: Метки классов
            feature_names: Названия признаков
            validation_split: Доля данных для валидации
            epochs: Количество эпох
            batch_size: Размер батча
            
        Returns:
            Словарь с метриками обучения
        """
        # Сохраняем названия признаков
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Предобработка данных
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=self.random_state
        )
        
        # Создаем модель
        self.model = self.create_model()
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Оценка на валидационной выборке
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(X_val, y_val, verbose=0)
        
        return {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'final_epoch': len(history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Предсказанные вероятности
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def predict_snoring_episode(self, audio_data: np.ndarray, 
                               accelerometer_data: Optional[np.ndarray] = None,
                               temporal_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Предсказание вероятности эпизода храпа.
        
        Args:
            audio_data: Аудио данные
            accelerometer_data: Данные акселерометра
            temporal_data: Временные данные
            
        Returns:
            Словарь с результатами предсказания
        """
        # Извлечение признаков
        features = self.extract_prediction_features(audio_data, accelerometer_data, temporal_data)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Предсказание
        probability = self.predict(feature_vector)[0]
        
        # Определение уровня риска
        risk_level = self._determine_risk_level(probability)
        
        # Расчет временного горизонта
        time_horizon = self._calculate_time_horizon(risk_level)
        
        return {
            'snoring_probability': float(probability),
            'risk_level': risk_level,
            'time_horizon_seconds': time_horizon,
            'confidence': float(probability if probability > 0.5 else 1 - probability),
            'features': features
        }
    
    def _determine_risk_level(self, probability: float) -> str:
        """Определяет уровень риска храпа."""
        if probability >= self.risk_thresholds['high']:
            return 'high'
        elif probability >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_time_horizon(self, risk_level: str) -> int:
        """Рассчитывает временной горизонт предсказания."""
        horizons = {
            'low': 30,
            'medium': 20,
            'high': 10
        }
        return horizons.get(risk_level, 30)
    
    def convert_to_tflite(self) -> bytes:
        """
        Конвертирует модель в TensorFlow Lite формат.
        
        Returns:
            Байты TFLite модели
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        # Конвертация в TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Пост-тренировочная квантование (int8)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # Конвертация
        tflite_model = converter.convert()
        self.tflite_model = tflite_model
        self.is_tflite = True
        
        return tflite_model
    
    def save_model(self, file_path: str):
        """
        Сохраняет модель.
        
        Args:
            file_path: Путь для сохранения
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Сохраняем модель
        if self.is_tflite and self.tflite_model is not None:
            with open(file_path, 'wb') as f:
                f.write(self.tflite_model)
        else:
            self.model.save(file_path)
        
        # Сохраняем скалер
        scaler_path = file_path.replace('.h5', '_scaler.pkl').replace('.tflite', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Сохраняем метаданные
        metadata = {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'feature_names': self.feature_names,
            'risk_thresholds': self.risk_thresholds,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = file_path.replace('.h5', '_metadata.json').replace('.tflite', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, file_path: str):
        """
        Загружает модель.
        
        Args:
            file_path: Путь к модели
        """
        # Загружаем модель
        if file_path.endswith('.tflite'):
            self.tflite_model = tf.io.read_file(file_path)
            self.is_tflite = True
            # Для TFLite модели нужно использовать интерпретатор
            self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
            self.interpreter.allocate_tensors()
        else:
            self.model = tf.keras.models.load_model(file_path)
            self.is_tflite = False
        
        # Загружаем скалер
        scaler_path = file_path.replace('.h5', '_scaler.pkl').replace('.tflite', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Загружаем метаданные
        metadata_path = file_path.replace('.h5', '_metadata.json').replace('.tflite', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_type = metadata.get('model_type', self.model_type)
                self.input_dim = metadata.get('input_dim', self.input_dim)
                self.hidden_layers = metadata.get('hidden_layers', self.hidden_layers)
                self.dropout_rate = metadata.get('dropout_rate', self.dropout_rate)
                self.feature_names = metadata.get('feature_names', self.feature_names)
                self.risk_thresholds = metadata.get('risk_thresholds', self.risk_thresholds)


def create_snoring_predictor(config: Dict) -> SnoringPredictor:
    """
    Создает экземпляр SnoringPredictor с заданной конфигурацией.
    
    Args:
        config: Конфигурация модели
        
    Returns:
        Экземпляр SnoringPredictor
    """
    return SnoringPredictor(
        model_type=config.get('model_type', 'neural_network'),
        input_dim=config.get('input_dim', 30),
        hidden_layers=config.get('hidden_layers', [64, 32, 16]),
        dropout_rate=config.get('dropout_rate', 0.2),
        random_state=config.get('random_state', 42)
    ) 