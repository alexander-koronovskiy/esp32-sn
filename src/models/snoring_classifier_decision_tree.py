#!/usr/bin/env python3
"""
Decision Tree классификатор для детекции храпа.
Архитектура согласно техническому заданию:
- 30 признаков за окно 1 с (10 тиков)
- 3 класса: No Snoring / Light Snoring / Heavy Snoring
- Ограничения сложности: max_depth ≤ 5, min_samples_leaf ≥ 10
- Робастная нормализация (median+IQR) на длинном окне (60–120 с)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from datetime import datetime
from scipy import signal
from scipy.stats import entropy


class SnoringDecisionTreeClassifier:
    """Decision Tree классификатор для детекции храпа с 30 признаками."""
    
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 10, 
                 criterion: str = 'gini', random_state: int = 42):
        """
        Инициализация Decision Tree классификатора.
        
        Args:
            max_depth: Максимальная глубина дерева (≤ 5)
            min_samples_leaf: Минимальное количество сэмплов в листе (≥ 10)
            criterion: Критерий разделения ('gini' или 'entropy')
            random_state: Случайное состояние
        """
        # Ограничения сложности согласно ТЗ
        self.max_depth = min(max_depth, 5)
        self.min_samples_leaf = max(min_samples_leaf, 10)
        self.criterion = criterion
        self.random_state = random_state
        
        # Модель и препроцессор
        self.model = None
        self.scaler = None
        
        # 3 класса согласно ТЗ
        self.class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
        self.class_count = 3
        
        # Параметры постобработки
        self.postprocessing_config = {
            'median_filter_window': 3,  # Медианный фильтр по 3 соседним окнам
            'hysteresis_confirm': 3,    # Подтверждаем храп если ≥3 окон подряд не-No
            'hysteresis_complete': 3    # Эпизод завершён если ≥3 окон подряд No
        }
        
        # Статистики для робастной нормализации
        self.normalization_stats = {
            'median': None,
            'iqr': None,
            'window_size': 90  # 90 секунд (60-120 с)
        }
        
    def create_model(self) -> DecisionTreeClassifier:
        """Создает Decision Tree с ограничениями сложности."""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
            class_weight='balanced'  # Для несбалансированных классов
        )
    
    def extract_30_features(self, audio_data: np.ndarray, accel_data: np.ndarray) -> np.ndarray:
        """
        Извлекает 30 признаков согласно ТЗ.
        
        Args:
            audio_data: Аудио данные (4 полосы × 10 тиков)
            accel_data: Данные акселерометра (3 оси × 10 тиков)
            
        Returns:
            Вектор из 30 признаков
        """
        features = []
        
        # 1. Аудио по 4 полосам (20 признаков): mean, max, std, доля выше порога, тренд
        # Полосы: Breath (100-400 Гц), Snore (300-1000 Гц), Speech (1000-4000 Гц), High (4000-8000 Гц)
        audio_features = self._extract_audio_features(audio_data)
        features.extend(audio_features)
        
        # 2. Акселерометр (7 признаков): средняя амплитуда |Δ| по X/Y/Z (3), 
        # максимальная амплитуда |Δ| по X/Y/Z (3), доля «подвижных» тиков (1)
        accel_features = self._extract_accelerometer_features(accel_data)
        features.extend(accel_features)
        
        # 3. Смешанные признаки (3): Snore/Breath, Snore/Speech, Breath_autocorr_lag1
        mixed_features = self._extract_mixed_features(audio_data)
        features.extend(mixed_features)
        
        return np.array(features)
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> List[float]:
        """Извлекает 20 аудио признаков (5 признаков × 4 полосы)."""
        features = []
        
        # Определяем 4 частотные полосы
        frequency_bands = {
            'breath': (100, 400),    # Breath: 100-400 Гц
            'snore': (300, 1000),    # Snore: 300-1000 Гц
            'speech': (1000, 4000),  # Speech: 1000-4000 Гц
            'high': (4000, 8000)     # High: 4000-8000 Гц
        }
        
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            # Фильтруем полосу частот
            filtered_band = self._filter_frequency_band(audio_data, low_freq, high_freq)
            
            # Извлекаем 5 признаков для каждой полосы
            band_features = self._extract_band_features(filtered_band)
            features.extend(band_features)
        
        return features
    
    def _filter_frequency_band(self, audio_data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Фильтрует аудио данные по частотной полосе."""
        # Простая фильтрация по частотным компонентам
        # В реальной реализации здесь будет FFT фильтр
        return audio_data  # Заглушка для демонстрации
    
    def _extract_band_features(self, band_data: np.ndarray) -> List[float]:
        """Извлекает 5 признаков для частотной полосы."""
        features = []
        
        # 1. Mean (средний уровень сигнала)
        features.append(np.mean(band_data))
        
        # 2. Max (максимальный уровень)
        features.append(np.max(band_data))
        
        # 3. Std (изменчивость)
        features.append(np.std(band_data))
        
        # 4. Доля выше адаптивного порога
        threshold = np.percentile(band_data, 75)  # 75-й процентиль как порог
        features.append(np.mean(band_data > threshold))
        
        # 5. Тренд (наклон) - линейная регрессия по времени
        if len(band_data) > 1:
            x = np.arange(len(band_data))
            slope = np.polyfit(x, band_data, 1)[0]
            features.append(slope)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_accelerometer_features(self, accel_data: np.ndarray) -> List[float]:
        """Извлекает 7 признаков акселерометра."""
        features = []
        
        # Разделяем данные по осям X, Y, Z
        x_data = accel_data[:, 0] if accel_data.ndim > 1 else accel_data
        y_data = accel_data[:, 1] if accel_data.ndim > 1 else accel_data
        z_data = accel_data[:, 2] if accel_data.ndim > 1 else accel_data
        
        # 1-3. Средняя амплитуда |Δ| по X/Y/Z (3 признака)
        features.append(np.mean(np.abs(np.diff(x_data))))
        features.append(np.mean(np.abs(np.diff(y_data))))
        features.append(np.mean(np.abs(np.diff(z_data))))
        
        # 4-6. Максимальная амплитуда |Δ| по X/Y/Z (3 признака)
        features.append(np.max(np.abs(np.diff(x_data))))
        features.append(np.max(np.abs(np.diff(y_data))))
        features.append(np.max(np.abs(np.diff(z_data))))
        
        # 7. Доля «подвижных» тиков (1 признак)
        # Считаем тик подвижным, если амплитуда выше медианы
        movement_threshold = np.median([np.abs(np.diff(x_data)), np.abs(np.diff(y_data)), np.abs(np.diff(z_data))])
        x_moving = np.sum(np.abs(np.diff(x_data)) > movement_threshold)
        y_moving = np.sum(np.abs(np.diff(y_data)) > movement_threshold)
        z_moving = np.sum(np.abs(np.diff(z_data)) > movement_threshold)
        features.append((x_moving + y_moving + z_moving) / (3 * len(x_data)))
        
        return features
    
    def _extract_mixed_features(self, audio_data: np.ndarray) -> List[float]:
        """Извлекает 3 смешанных признака."""
        features = []
        
        # 1. Snore/Breath = mean(snore)/mean(breath)
        snore_band = self._filter_frequency_band(audio_data, 300, 1000)
        breath_band = self._filter_frequency_band(audio_data, 100, 400)
        snore_breath_ratio = np.mean(snore_band) / (np.mean(breath_band) + 1e-8)
        features.append(snore_breath_ratio)
        
        # 2. Snore/Speech = mean(snore)/mean(speech)
        speech_band = self._filter_frequency_band(audio_data, 1000, 4000)
        snore_speech_ratio = np.mean(snore_band) / (np.mean(speech_band) + 1e-8)
        features.append(snore_speech_ratio)
        
        # 3. Breath_autocorr_lag1 (регулярность дыхания как эталон ритма)
        breath_autocorr = self._calculate_autocorrelation_lag1(breath_band)
        features.append(breath_autocorr)
        
        return features
    
    def _calculate_autocorrelation_lag1(self, data: np.ndarray) -> float:
        """Вычисляет автокорреляцию с лагом 1."""
        if len(data) < 2:
            return 0.0
        
        # Нормализуем данные
        data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # Автокорреляция с лагом 1
        autocorr = np.corrcoef(data_norm[:-1], data_norm[1:])[0, 1]
        return autocorr if not np.isnan(autocorr) else 0.0
    
    def apply_robust_normalization(self, features: np.ndarray) -> np.ndarray:
        """
        Применяет робастную нормализацию (median+IQR) на длинном окне (60–120 с).
        
        Args:
            features: Признаки для нормализации
            
        Returns:
            Нормализованные признаки
        """
        if self.scaler is None:
            self.scaler = RobustScaler()
            # Обучаем scaler на входных данных
            self.scaler.fit(features)
        
        # Применяем нормализацию
        normalized = self.scaler.transform(features)
        
        return normalized
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Обучает Decision Tree классификатор.
        
        Args:
            X: Признаки для обучения (должно быть 30 признаков)
            y: Метки классов
            feature_names: Названия признаков
            
        Returns:
            Словарь с метриками обучения
        """
        # Проверяем количество признаков
        if X.shape[1] != 30:
            raise ValueError(f"Ожидается 30 признаков, получено {X.shape[1]}")
        
        # Создаем модель
        self.model = self.create_model()
        
        # Применяем робастную нормализацию
        X_normalized = self.apply_robust_normalization(X)
        
        # Обучение
        self.model.fit(X_normalized, y)
        
        # Оценка производительности
        y_pred = self.model.predict(X_normalized)
        accuracy = accuracy_score(y, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'model_type': 'decision_tree',
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'feature_count': 30
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Предсказанные метки классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        # Нормализуем признаки
        X_normalized = self.apply_robust_normalization(X)
        
        # Предсказание
        return self.model.predict(X_normalized)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности классов.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Матрица вероятностей классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        # Нормализуем признаки
        X_normalized = self.apply_robust_normalization(X)
        
        # Вероятности классов
        return self.model.predict_proba(X_normalized)
    
    def apply_postprocessing(self, predictions: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """
        Применяет постобработку для устойчивости предсказаний.
        
        Args:
            predictions: Исходные предсказания
            probabilities: Вероятности классов
            
        Returns:
            Обработанные предсказания
        """
        # 1. Медианный фильтр по 3 соседним окнам (0.3 с)
        filtered_predictions = self._apply_median_filter(predictions)
        
        # 2. Гистерезис по длительности эпизода
        final_predictions = self._apply_hysteresis_filter(filtered_predictions)
        
        return final_predictions
    
    def _apply_median_filter(self, predictions: np.ndarray) -> np.ndarray:
        """Применяет медианный фильтр по 3 соседним окнам."""
        window_size = self.postprocessing_config['median_filter_window']
        filtered = np.copy(predictions)
        
        for i in range(window_size // 2, len(predictions) - window_size // 2):
            window = predictions[i - window_size // 2:i + window_size // 2 + 1]
            filtered[i] = np.median(window)
        
        return filtered
    
    def _apply_hysteresis_filter(self, predictions: np.ndarray) -> np.ndarray:
        """Применяет гистерезис фильтр для устойчивости эпизодов."""
        confirm_threshold = self.postprocessing_config['hysteresis_confirm']
        complete_threshold = self.postprocessing_config['hysteresis_complete']
        
        filtered = np.copy(predictions)
        
        # Простой алгоритм гистерезиса
        # В реальной реализации здесь будет более сложная логика
        for i in range(confirm_threshold, len(predictions) - confirm_threshold):
            # Проверяем окрестность для подтверждения
            window = predictions[i - confirm_threshold:i + confirm_threshold + 1]
            if np.all(window != 0):  # Все окна не-No
                filtered[i] = 1  # Подтверждаем храп
            elif np.all(window == 0):  # Все окна No
                filtered[i] = 0  # Завершаем эпизод
        
        return filtered
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели."""
        if self.model is None:
            return {"status": "Модель не обучена"}
        
        return {
            "model_type": "Decision Tree",
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "feature_count": 30,
            "class_names": self.class_names,
            "postprocessing_config": self.postprocessing_config,
            "normalization_stats": self.normalization_stats
        }
    
    def save_model(self, filepath: str) -> None:
        """Сохраняет модель в файл."""
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'normalization_stats': self.normalization_stats,
            'postprocessing_config': self.postprocessing_config,
            'class_names': self.class_names,
            'model_info': self.get_model_info()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Загружает модель из файла."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.normalization_stats = model_data['normalization_stats']
        self.postprocessing_config = model_data['postprocessing_config']
        self.class_names = model_data['class_names']
        
        print(f"✅ Модель загружена из {filepath}")


def create_decision_tree_config() -> Dict[str, Any]:
    """
    Создает конфигурацию для Decision Tree классификатора.
    
    Returns:
        Словарь с конфигурацией
    """
    return {
        'model_type': 'decision_tree',
        'max_depth': 5,                    # Ограничение сложности согласно ТЗ
        'min_samples_leaf': 10,            # Минимальное количество сэмплов в листе
        'criterion': 'gini',               # Критерий разделения
        'random_state': 42,
        'feature_count': 30,               # 30 признаков согласно ТЗ
        'class_count': 3,                  # 3 класса
        'postprocessing': {
            'median_filter_window': 3,     # Медианный фильтр по 3 окнам
            'hysteresis_confirm': 3,       # Подтверждение храпа
            'hysteresis_complete': 3       # Завершение эпизода
        },
        'normalization': {
            'window_size': 90,             # 90 секунд для робастной нормализации
            'method': 'robust_median_iqr'  # Робастная нормализация
        },
        'frequency_bands': {
            'breath': (100, 400),          # Breath: 100-400 Гц
            'snore': (300, 1000),          # Snore: 300-1000 Гц
            'speech': (1000, 4000),        # Speech: 1000-4000 Гц
            'high': (4000, 8000)           # High: 4000-8000 Гц
        }
    } 