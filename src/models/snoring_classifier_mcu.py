#!/usr/bin/env python3
"""
SnoringClassifier оптимизированный для MCU
Fixed-point вычисления, INT16 признаки и пороги
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.fixed_point_dsp import FixedPointDSP, FeatureQuantizer
from src.utils.pose_estimator import PoseEstimator

class SnoringClassifierMCU:
    """SnoringClassifier оптимизированный для ESP32"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_quantizer = None
        
        # Fixed-point компоненты
        self.dsp = FixedPointDSP()
        self.pose_estimator = PoseEstimator()
        
        # Классы
        self.class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
        self.class_count = 3
        
        # Параметры модели
        self.model_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 6,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'random_state': 42
            }
        }
        
        # Статистики для квантования
        self.feature_stats = {
            'mu': None,
            'sigma': None,
            'scale': 1.0
        }
    
    def extract_audio_features(self, audio_data: np.ndarray) -> Dict:
        """Извлечение аудио признаков в fixed-point"""
        # Спектральные признаки
        spectral_features = self.dsp.compute_spectral_features(audio_data)
        
        # Дополнительные признаки
        features = {}
        
        # RMS и энергетические признаки
        features['rms'] = spectral_features['rms']
        features['zero_crossing_rate'] = spectral_features['zero_crossing_rate']
        features['peak_count'] = spectral_features['peak_count']
        features['spectral_centroid'] = spectral_features['spectral_centroid']
        
        # Полосные энергии
        features['breath_energy'] = spectral_features['breath_energy']
        features['snore_energy'] = spectral_features['snore_energy']
        features['speech_energy'] = spectral_features['speech_energy']
        
        # Дополнительные спектральные признаки
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/8000)
        
        # Спектральная ширина
        spectral_bandwidth = np.sqrt(np.sum(np.abs(fft)**2 * freqs**2) / (np.sum(np.abs(fft)**2) + 1e-10))
        features['spectral_bandwidth'] = self.dsp.float_to_fixed(spectral_bandwidth)
        
        # Спектральная энтропия
        power_spectrum = np.abs(fft)**2
        power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        spectral_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        features['spectral_entropy'] = self.dsp.float_to_fixed(spectral_entropy)
        
        # Статистические признаки
        features['mean'] = self.dsp.float_to_fixed(np.mean(audio_data))
        features['std'] = self.dsp.float_to_fixed(np.std(audio_data))
        features['skewness'] = self.dsp.float_to_fixed(self._compute_skewness(audio_data))
        features['kurtosis'] = self.dsp.float_to_fixed(self._compute_kurtosis(audio_data))
        
        # Временные признаки
        features['temporal_entropy'] = self.dsp.float_to_fixed(self._compute_temporal_entropy(audio_data))
        features['abs_mean'] = self.dsp.float_to_fixed(np.mean(np.abs(audio_data)))
        
        return features
    
    def extract_accelerometer_features(self, accel_x: float, accel_y: float, accel_z: float) -> Dict:
        """Извлечение признаков акселерометра"""
        # Обновление позы
        pose_data = self.pose_estimator.update_pose(accel_x, accel_y, accel_z)
        
        features = {
            'pitch': pose_data['pitch'],
            'roll': pose_data['roll'],
            'pose_state': self._pose_to_int(pose_data['pose_state']),
            'pose_stability': self.dsp.float_to_fixed(pose_data['pose_stability']),
            'pose_changes': pose_data['pose_changes']
        }
        
        # Дополнительные признаки акселерометра
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        features['accel_magnitude'] = self.dsp.float_to_fixed(accel_magnitude)
        
        # Активность акселерометра
        features['accel_variance'] = self.dsp.float_to_fixed(
            np.var([accel_x, accel_y, accel_z])
        )
        
        return features
    
    def _pose_to_int(self, pose: str) -> int:
        """Конвертация позы в целое число"""
        pose_map = {
            'supine': 0,
            'prone': 1,
            'left': 2,
            'right': 3
        }
        return pose_map.get(pose, 0)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Вычисление асимметрии"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Вычисление эксцесса"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_temporal_entropy(self, data: np.ndarray) -> float:
        """Вычисление временной энтропии"""
        # Гистограмма амплитуд
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0
        return -np.sum(hist * np.log(hist))
    
    def extract_features(self, audio_data: np.ndarray, accel_data: Optional[Dict] = None) -> np.ndarray:
        """Извлечение всех признаков"""
        # Аудио признаки
        audio_features = self.extract_audio_features(audio_data)
        
        # Признаки акселерометра
        if accel_data is not None:
            accel_features = self.extract_accelerometer_features(
                accel_data.get('x', 0),
                accel_data.get('y', 0),
                accel_data.get('z', 0)
            )
        else:
            # Заглушки для акселерометра
            accel_features = {
                'pitch': 0,
                'roll': 0,
                'pose_state': 0,
                'pose_stability': 0,
                'pose_changes': 0,
                'accel_magnitude': 0,
                'accel_variance': 0
            }
        
        # Объединение признаков
        all_features = {**audio_features, **accel_features}
        
        # Сортировка по ключам для стабильного порядка
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        return feature_vector
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Обучение модели"""
        # Создание модели
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params['random_forest'])
        elif self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(**self.model_params['lightgbm'])
            except ImportError:
                print("LightGBM не установлен, используем RandomForest")
                self.model = RandomForestClassifier(**self.model_params['random_forest'])
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        # Обучение
        self.model.fit(X, y)
        
        # Создание scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Вычисление статистик для квантования
        self.feature_stats['mu'] = self.scaler.mean_
        self.feature_stats['sigma'] = self.scaler.scale_
        
        # Создание квантователя
        self.feature_quantizer = FeatureQuantizer(
            mu=self.feature_stats['mu'][0],
            sigma=self.feature_stats['sigma'][0],
            scale=self.feature_stats['scale']
        )
        
        # Оценка производительности
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        
        return {
            'accuracy': accuracy,
            'feature_count': X.shape[1],
            'model_type': self.model_type
        }
    
    def predict(self, audio_data: np.ndarray, accel_data: Optional[Dict] = None) -> Dict:
        """Предсказание класса храпа"""
        # Извлечение признаков
        features = self.extract_features(audio_data, accel_data)
        
        # Нормализация
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Предсказание
        if self.model is not None:
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
        else:
            prediction = 0
            probabilities = [1.0, 0.0, 0.0]
        
        # Определение класса
        class_name = self.class_names[prediction]
        
        # Вероятность храпа
        p_snore = probabilities[1] + probabilities[2]  # Light + Heavy
        
        # Бинарный индикатор
        is_snoring = prediction > 0
        
        return {
            'class': class_name,
            'class_id': prediction,
            'probabilities': probabilities.tolist(),
            'p_snore': p_snore,
            'is_snoring': is_snoring,
            'confidence': max(probabilities)
        }
    
    def save_model(self, filepath: str) -> None:
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_stats': self.feature_stats,
            'class_names': self.class_names,
            'model_type': self.model_type,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, filepath)
        
        # Сохранение метаданных
        metadata = {
            'class_names': self.class_names,
            'class_count': self.class_count,
            'model_type': self.model_type,
            'feature_stats': self.feature_stats,
            'classes': self.class_names
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Загрузка модели"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_stats = model_data['feature_stats']
        self.class_names = model_data['class_names']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        
        # Создание квантователя
        if self.feature_stats['mu'] is not None:
            self.feature_quantizer = FeatureQuantizer(
                mu=self.feature_stats['mu'][0],
                sigma=self.feature_stats['sigma'][0],
                scale=self.feature_stats['scale']
            )
    
    def get_model_info(self) -> Dict:
        """Получение информации о модели"""
        return {
            'model_type': self.model_type,
            'class_names': self.class_names,
            'class_count': self.class_count,
            'feature_stats': self.feature_stats,
            'model_params': self.model_params
        } 