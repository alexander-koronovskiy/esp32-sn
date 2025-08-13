#!/usr/bin/env python3
"""
SnoringClassifier адаптированный для 2-классовой классификации
Работает с реальными данными: W (No Snoring) и отсутствие метки (Snoring)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Временно отключаем проблемные импорты
# from src.utils.fixed_point_dsp import FixedPointDSP, FeatureQuantizer
# from src.utils.pose_estimator import PoseEstimator

class SnoringClassifier2Class:
    """SnoringClassifier для 2-классовой классификации (адаптирован под реальные данные)"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_quantizer = None
        
        # Временно отключаем fixed-point компоненты
        # self.dsp = FixedPointDSP()
        # self.pose_estimator = PoseEstimator()
        
        # 2 класса для реальных данных
        self.class_names = ['No_Snoring', 'Snoring']  # W = No_Snoring, отсутствие метки = Snoring
        self.class_count = 2
        
        # Параметры модели (оптимизированы для 2 классов)
        self.model_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 6,
                'random_state': 42,
                'class_weight': 'balanced'  # Для несбалансированных классов
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        }
        
        # Статистики для квантования
        self.feature_stats = {
            'mu': None,
            'sigma': None,
            'scale': 1.0
        }
        
        # Маппинг для реальных данных
        self.real_data_mapping = {
            'W': 0,  # Wake/No Snoring
            '': 1,   # Отсутствие метки = Snoring
            None: 1  # None = Snoring
        }
    
    def float_to_fixed(self, value: float) -> int:
        """Простая конвертация float в int (замена FixedPointDSP)"""
        return int(value * 1000)  # Масштабирование на 1000
    
    def extract_real_data_features(self, csv_data: np.ndarray) -> Dict:
        """Извлечение признаков из реальных CSV данных"""
        features = {}
        
        # Базовые признаки из CSV
        if csv_data.shape[1] >= 25:  # Проверяем количество колонок
            # Акселерометр (первые 3 колонки после time)
            features['accel_x'] = self.float_to_fixed(float(csv_data[0, 1]))  # x
            features['accel_y'] = self.float_to_fixed(float(csv_data[0, 2]))  # y
            features['accel_z'] = self.float_to_fixed(float(csv_data[0, 3]))  # z
            
            # Пульс и физиологические данные
            features['bpm'] = self.float_to_fixed(float(csv_data[0, 4]))  # bpm
            features['bpm_by_nn'] = self.float_to_fixed(float(csv_data[0, 5]))  # bpm_by_nn
            features['bpm_by_acf'] = self.float_to_fixed(float(csv_data[0, 6]))  # bpm_by_acf
            
            # Нейронные сети
            features['breath_nn'] = self.float_to_fixed(float(csv_data[0, 16]))  # breath_nn
            features['snore_nn'] = self.float_to_fixed(float(csv_data[0, 17]))  # snore_nn
            features['signal_nn'] = self.float_to_fixed(float(csv_data[0, 18]))  # signal_nn
            features['splash_nn'] = self.float_to_fixed(float(csv_data[0, 19]))  # splash_nn
            
            # Полосовые фильтры
            features['b100'] = self.float_to_fixed(float(csv_data[0, 21]))  # b100
            features['b400'] = self.float_to_fixed(float(csv_data[0, 22]))  # b400
            features['b1000'] = self.float_to_fixed(float(csv_data[0, 23]))  # b1000
            
            # Огибающая
            features['env'] = self.float_to_fixed(float(csv_data[0, 24]))  # env
            
            # Дополнительные признаки
            features['mf'] = self.float_to_fixed(float(csv_data[0, 7]))  # mf
            features['af'] = self.float_to_fixed(float(csv_data[0, 8]))  # af
            features['arf'] = self.float_to_fixed(float(csv_data[0, 9]))  # arf
            features['sf'] = self.float_to_fixed(float(csv_data[0, 10]))  # sf
            features['r_th'] = self.float_to_fixed(float(csv_data[0, 11]))  # r_th
            
            # Статусы
            features['status_1'] = self.float_to_fixed(float(csv_data[0, 12]))  # status_1
            features['status_2'] = self.float_to_fixed(float(csv_data[0, 13]))  # status_2
            features['status_3'] = self.float_to_fixed(float(csv_data[0, 14]))  # status_3
            
            # Дополнительные метки
            features['ml'] = self.float_to_fixed(float(csv_data[0, 15]))  # ml
            features['eog'] = self.float_to_fixed(float(csv_data[0, 20]))  # eog
            
            # Вычисленные признаки
            features['accel_magnitude'] = self.float_to_fixed(
                np.sqrt(float(csv_data[0, 1])**2 + float(csv_data[0, 2])**2 + float(csv_data[0, 3])**2)
            )
            
            features['snore_breath_ratio'] = self.float_to_fixed(
                float(csv_data[0, 17]) / (float(csv_data[0, 16]) + 1e-10)  # snore_nn / breath_nn
            )
            
            features['signal_activity'] = self.float_to_fixed(
                float(csv_data[0, 18]) + float(csv_data[0, 19])  # signal_nn + splash_nn
            )
            
            features['band_energy_ratio'] = self.float_to_fixed(
                float(csv_data[0, 23]) / (float(csv_data[0, 21]) + 1e-10)  # b1000 / b100
            )
        
        return features
    
    def extract_audio_features(self, audio_data: np.ndarray) -> Dict:
        """Извлечение аудио признаков (для совместимости)"""
        features = {}
        
        # Простые признаки
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        features['mean'] = np.mean(audio_data)
        features['std'] = np.std(audio_data)
        features['max'] = np.max(audio_data)
        features['min'] = np.min(audio_data)
        
        # Дополнительные признаки
        features['zero_crossing_rate'] = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)
        features['peak_count'] = np.sum(np.diff(audio_data) > 0)
        
        # Полосовые энергии (упрощенно)
        features['breath_energy'] = np.sum(audio_data**2)
        features['snore_energy'] = np.sum(audio_data**2)
        features['speech_energy'] = np.sum(audio_data**2)
        
        return features
    
    def extract_accelerometer_features(self, accel_x: float, accel_y: float, accel_z: float) -> Dict:
        """Извлечение признаков акселерометра"""
        features = {}
        
        # Простые признаки
        features['pitch'] = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2))
        features['roll'] = np.arctan2(-accel_x, accel_z)
        features['pose_state'] = 0  # По умолчанию
        features['pose_stability'] = 1.0
        features['pose_changes'] = 0
        
        # Дополнительные признаки
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        features['accel_magnitude'] = accel_magnitude
        features['accel_variance'] = np.var([accel_x, accel_y, accel_z])
        
        return features
    
    def extract_features(self, data: np.ndarray, data_type: str = 'csv', accel_data: Optional[Dict] = None) -> np.ndarray:
        """Извлечение всех признаков"""
        if data_type == 'csv':
            # Реальные данные из CSV
            features = self.extract_real_data_features(data)
        elif data_type == 'audio':
            # Аудио данные
            features = self.extract_audio_features(data)
            
            # Признаки акселерометра
            if accel_data is not None:
                accel_features = self.extract_accelerometer_features(
                    accel_data.get('x', 0),
                    accel_data.get('y', 0),
                    accel_data.get('z', 0)
                )
                features.update(accel_features)
        else:
            raise ValueError(f"Неизвестный тип данных: {data_type}")
        
        # Объединение признаков
        all_features = {**features}
        
        # Сортировка по ключам для стабильного порядка
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        return feature_vector
    
    def convert_real_data_labels(self, labels: List[str]) -> np.ndarray:
        """Конвертация меток реальных данных в числовые"""
        converted_labels = []
        for label in labels:
            if label in self.real_data_mapping:
                converted_labels.append(self.real_data_mapping[label])
            else:
                # По умолчанию считаем отсутствие метки как Snoring
                converted_labels.append(1)
        
        return np.array(converted_labels)
    
    def train(self, X: np.ndarray, y: np.ndarray, data_type: str = 'csv') -> Dict:
        """Обучение модели"""
        # Конвертация меток для реальных данных
        if data_type == 'csv':
            y = self.convert_real_data_labels(y)
        
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
        
        # Оценка производительности
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        
        return {
            'accuracy': accuracy,
            'feature_count': X.shape[1],
            'model_type': self.model_type,
            'class_distribution': np.bincount(y)
        }
    
    def predict(self, data: np.ndarray, data_type: str = 'csv', accel_data: Optional[Dict] = None) -> Dict:
        """Предсказание класса храпа"""
        # Извлечение признаков
        features = self.extract_features(data, data_type, accel_data)
        
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
            probabilities = [1.0, 0.0]
        
        # Определение класса
        class_name = self.class_names[prediction]
        
        # Вероятность храпа
        p_snore = probabilities[1]  # Snoring класс
        
        # Бинарный индикатор
        is_snoring = prediction == 1
        
        return {
            'class': class_name,
            'class_id': prediction,
            'probabilities': probabilities.tolist(),
            'p_snore': p_snore,
            'is_snoring': is_snoring,
            'confidence': max(probabilities),
            'features': features.tolist()
        }
    
    def save_model(self, filepath: str) -> None:
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_stats': self.feature_stats,
            'class_names': self.class_names,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'real_data_mapping': self.real_data_mapping
        }
        
        joblib.dump(model_data, filepath)
        
        # Сохранение метаданных
        metadata = {
            'class_names': self.class_names,
            'class_count': self.class_count,
            'model_type': self.model_type,
            'feature_stats': self._convert_numpy_to_list(self.feature_stats),
            'real_data_mapping': self.real_data_mapping,
            'description': '2-class SnoringClassifier adapted for real data'
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _convert_numpy_to_list(self, obj):
        """Конвертация numpy объектов в списки для JSON сериализации"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def load_model(self, filepath: str) -> None:
        """Загрузка модели"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_stats = model_data['feature_stats']
        self.class_names = model_data['class_names']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        
        if 'real_data_mapping' in model_data:
            self.real_data_mapping = model_data['real_data_mapping']
    
    def get_model_info(self) -> Dict:
        """Получение информации о модели"""
        return {
            'model_type': self.model_type,
            'class_names': self.class_names,
            'class_count': self.class_count,
            'feature_stats': self.feature_stats,
            'model_params': self.model_params,
            'real_data_mapping': self.real_data_mapping,
            'description': '2-class SnoringClassifier adapted for real data'
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, data_type: str = 'csv') -> Dict:
        """Оценка модели"""
        if data_type == 'csv':
            y = self.convert_real_data_labels(y)
        
        if self.model is None:
            return {'error': 'Модель не обучена'}
        
        # Предсказания
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Метрики
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, target_names=self.class_names, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'class_distribution': np.bincount(y).tolist()
        } 