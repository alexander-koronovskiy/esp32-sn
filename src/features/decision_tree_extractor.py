#!/usr/bin/env python3
"""
Экстрактор признаков для Decision Tree классификатора храпа.
Генерирует ровно 30 признаков согласно архитектуре:
- Аудио по 4 полосам (20 признаков): mean, max, std, доля выше порога, тренд
- Акселерометр (7 признаков): средняя и максимальная амплитуда |Δ| по X/Y/Z, доля подвижных тиков
- Смешанные (3 признака): Snore/Breath, Snore/Speech, Breath_autocorr_lag1
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class DecisionTreeFeatureExtractor:
    """Экстрактор 30 признаков для Decision Tree классификатора храпа."""
    
    def __init__(self, sampling_rate: int = 8000, window_size: int = 8000):
        """
        Инициализация экстрактора признаков.
        
        Args:
            sampling_rate: Частота дискретизации (Гц)
            window_size: Размер окна (сэмплов) - 1 секунда
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        
        # 4 частотные полосы согласно ТЗ (исправлены для корректной фильтрации)
        self.frequency_bands = {
            'breath': (100, 400),    # Breath: 100-400 Гц
            'snore': (400, 1000),    # Snore: 400-1000 Гц (исправлено)
            'speech': (1000, 4000),  # Speech: 1000-4000 Гц
            'high': (4000, 7000)     # High: 4000-7000 Гц (исправлено)
        }
        
        # Количество тиков в окне (10 тиков за секунду)
        self.ticks_per_window = 10
        self.samples_per_tick = self.window_size // self.ticks_per_window
        
    def extract_30_features(self, audio_data: np.ndarray, accel_data: np.ndarray) -> np.ndarray:
        """
        Извлекает ровно 30 признаков согласно архитектуре.
        
        Args:
            audio_data: Аудио данные (1 секунда, 8000 сэмплов)
            accel_data: Данные акселерометра (1 секунда, 8000 сэмплов × 3 оси)
            
        Returns:
            Вектор из 30 признаков
        """
        features = []
        
        # 1. Аудио по 4 полосам (20 признаков): 5 признаков × 4 полосы
        audio_features = self._extract_audio_features(audio_data)
        features.extend(audio_features)
        
        # 2. Акселерометр (7 признаков)
        accel_features = self._extract_accelerometer_features(accel_data)
        features.extend(accel_features)
        
        # 3. Смешанные признаки (3 признака)
        mixed_features = self._extract_mixed_features(audio_data)
        features.extend(mixed_features)
        
        # Проверяем, что получилось ровно 30 признаков
        if len(features) != 30:
            raise ValueError(f"Ожидается 30 признаков, получено {len(features)}")
        
        return np.array(features)
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> List[float]:
        """Извлекает 20 аудио признаков (5 признаков × 4 полосы)."""
        features = []
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Фильтруем полосу частот
            filtered_band = self._filter_frequency_band(audio_data, low_freq, high_freq)
            
            # Разделяем на 10 тиков
            tick_data = self._split_into_ticks(filtered_band)
            
            # Извлекаем 5 признаков для каждой полосы
            band_features = self._extract_band_features(tick_data)
            features.extend(band_features)
        
        return features
    
    def _filter_frequency_band(self, audio_data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Фильтрует аудио данные по частотной полосе."""
        nyquist = self.sampling_rate / 2
        
        # Нормализуем частоты и проверяем границы
        low_norm = max(0.001, low_freq / nyquist)  # Минимум 0.001
        high_norm = min(0.999, high_freq / nyquist)  # Максимум 0.999
        
        # Создаем полосовой фильтр Баттерворта
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Применяем фильтр
        filtered = signal.filtfilt(b, a, audio_data)
        
        return filtered
    
    def _split_into_ticks(self, data: np.ndarray) -> np.ndarray:
        """Разделяет данные на 10 тиков."""
        ticks = []
        for i in range(self.ticks_per_window):
            start_idx = i * self.samples_per_tick
            end_idx = start_idx + self.samples_per_tick
            if i == self.ticks_per_window - 1:  # Последний тик
                end_idx = len(data)
            tick = data[start_idx:end_idx]
            ticks.append(tick)
        return np.array(ticks)
    
    def _extract_band_features(self, tick_data: np.ndarray) -> List[float]:
        """Извлекает 5 признаков для частотной полосы."""
        features = []
        
        # 1. Mean (средний уровень сигнала)
        features.append(np.mean(tick_data))
        
        # 2. Max (максимальный уровень)
        features.append(np.max(tick_data))
        
        # 3. Std (изменчивость)
        features.append(np.std(tick_data))
        
        # 4. Доля выше адаптивного порога (75-й процентиль)
        threshold = np.percentile(tick_data, 75)
        features.append(np.mean(tick_data > threshold))
        
        # 5. Тренд (наклон) - линейная регрессия по времени
        if len(tick_data) > 1:
            x = np.arange(len(tick_data))
            slope = np.polyfit(x, tick_data, 1)[0]
            features.append(slope)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_accelerometer_features(self, accel_data: np.ndarray) -> List[float]:
        """Извлекает 7 признаков акселерометра."""
        features = []
        
        # Разделяем данные по осям X, Y, Z
        if accel_data.ndim == 1:
            # Если данные в одной колонке, предполагаем чередование X, Y, Z
            x_data = accel_data[::3]
            y_data = accel_data[1::3]
            z_data = accel_data[2::3]
        else:
            x_data = accel_data[:, 0]
            y_data = accel_data[:, 1]
            z_data = accel_data[:, 2]
        
        # Разделяем на тики
        x_ticks = self._split_into_ticks(x_data)
        y_ticks = self._split_into_ticks(y_data)
        z_ticks = self._split_into_ticks(z_data)
        
        # 1-3. Средняя амплитуда |Δ| по X/Y/Z (3 признака)
        features.append(np.mean([np.mean(np.abs(np.diff(tick))) for tick in x_ticks]))
        features.append(np.mean([np.mean(np.abs(np.diff(tick))) for tick in y_ticks]))
        features.append(np.mean([np.mean(np.abs(np.diff(tick))) for tick in z_ticks]))
        
        # 4-6. Максимальная амплитуда |Δ| по X/Y/Z (3 признака)
        features.append(np.max([np.max(np.abs(np.diff(tick))) for tick in x_ticks]))
        features.append(np.max([np.max(np.abs(np.diff(tick))) for tick in y_ticks]))
        features.append(np.max([np.max(np.abs(np.diff(tick))) for tick in z_ticks]))
        
        # 7. Доля «подвижных» тиков (1 признак)
        # Считаем тик подвижным, если средняя амплитуда выше медианы
        x_movements = [np.mean(np.abs(np.diff(tick))) for tick in x_ticks]
        y_movements = [np.mean(np.abs(np.diff(tick))) for tick in y_ticks]
        z_movements = [np.mean(np.abs(np.diff(tick))) for tick in z_ticks]
        
        movement_threshold = np.median(x_movements + y_movements + z_movements)
        
        x_moving = np.sum(np.array(x_movements) > movement_threshold)
        y_moving = np.sum(np.array(y_movements) > movement_threshold)
        z_moving = np.sum(np.array(z_movements) > movement_threshold)
        
        features.append((x_moving + y_moving + z_moving) / (3 * self.ticks_per_window))
        
        return features
    
    def _extract_mixed_features(self, audio_data: np.ndarray) -> List[float]:
        """Извлекает 3 смешанных признака."""
        features = []
        
        # 1. Snore/Breath = mean(snore)/mean(breath)
        snore_band = self._filter_frequency_band(audio_data, 300, 1000)
        breath_band = self._filter_frequency_band(audio_data, 100, 400)
        
        snore_mean = np.mean(snore_band)
        breath_mean = np.mean(breath_band)
        snore_breath_ratio = snore_mean / (breath_mean + 1e-8)
        features.append(snore_breath_ratio)
        
        # 2. Snore/Speech = mean(snore)/mean(speech)
        speech_band = self._filter_frequency_band(audio_data, 1000, 4000)
        speech_mean = np.mean(speech_band)
        snore_speech_ratio = snore_mean / (speech_mean + 1e-8)
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
    
    def extract_features_from_csv(self, csv_data: np.ndarray) -> np.ndarray:
        """
        Извлекает признаки из CSV данных (адаптация для реальных данных).
        
        Args:
            csv_data: CSV данные с колонками ['time', 'x', 'y', 'z', 'bpm', ...]
            
        Returns:
            Вектор из 30 признаков
        """
        features = []
        
        # Проверяем количество колонок
        if csv_data.shape[1] < 25:
            raise ValueError(f"CSV должен содержать минимум 25 колонок, получено {csv_data.shape[1]}")
        
        # Берем первый сэмпл (первую строку)
        sample = csv_data[0]
        
        # 1. Аудио признаки (20) - используем доступные колонки
        # Полосовые фильтры
        b100 = float(sample[21])  # b100
        b400 = float(sample[22])  # b400  
        b1000 = float(sample[23]) # b1000
        env = float(sample[24])   # env
        
        # Нейронные сети
        breath_nn = float(sample[16])  # breath_nn
        snore_nn = float(sample[17])   # snore_nn
        signal_nn = float(sample[18])  # signal_nn
        splash_nn = float(sample[19])  # splash_nn
        
        # Генерируем 20 аудио признаков на основе доступных данных
        audio_features = self._generate_audio_features_from_csv(
            b100, b400, b1000, env, breath_nn, snore_nn, signal_nn, splash_nn
        )
        features.extend(audio_features)
        
        # 2. Акселерометр (7 признаков)
        x = float(sample[1])
        y = float(sample[2]) 
        z = float(sample[3])
        
        accel_features = self._generate_accel_features_from_csv(x, y, z)
        features.extend(accel_features)
        
        # 3. Смешанные признаки (3)
        mixed_features = self._generate_mixed_features_from_csv(
            b100, b400, b1000, breath_nn, snore_nn, signal_nn
        )
        features.extend(mixed_features)
        
        # Проверяем количество признаков
        if len(features) != 30:
            raise ValueError(f"Ожидается 30 признаков, получено {len(features)}")
        
        return np.array(features)
    
    def _generate_audio_features_from_csv(self, b100: float, b400: float, b1000: float, 
                                        env: float, breath_nn: float, snore_nn: float, 
                                        signal_nn: float, splash_nn: float) -> List[float]:
        """Генерирует 20 аудио признаков из CSV данных."""
        features = []
        
        # 4 полосы × 5 признаков = 20 признаков
        bands = [
            ('breath', [b100, breath_nn]),      # Breath: b100 + breath_nn
            ('snore', [b400, snore_nn]),       # Snore: b400 + snore_nn  
            ('speech', [b1000, signal_nn]),    # Speech: b1000 + signal_nn
            ('high', [env, splash_nn])         # High: env + splash_nn
        ]
        
        for band_name, band_values in bands:
            # 5 признаков для каждой полосы
            features.append(np.mean(band_values))           # Mean
            features.append(np.max(band_values))            # Max
            features.append(np.std(band_values))            # Std
            features.append(np.mean(np.array(band_values) > np.median(band_values)))  # Доля выше порога
            features.append(0.0)  # Тренд (заглушка)
        
        return features
    
    def _generate_accel_features_from_csv(self, x: float, y: float, z: float) -> List[float]:
        """Генерирует 7 признаков акселерометра из CSV данных."""
        features = []
        
        # 1-3. Средняя амплитуда |Δ| по X/Y/Z (3 признака)
        features.append(abs(x))
        features.append(abs(y))
        features.append(abs(z))
        
        # 4-6. Максимальная амплитуда |Δ| по X/Y/Z (3 признака)
        features.append(abs(x))
        features.append(abs(y))
        features.append(abs(z))
        
        # 7. Доля «подвижных» тиков (1 признак)
        movement_threshold = np.median([abs(x), abs(y), abs(z)])
        features.append(np.mean([abs(x) > movement_threshold, abs(y) > movement_threshold, abs(z) > movement_threshold]))
        
        return features
    
    def _generate_mixed_features_from_csv(self, b100: float, b400: float, b1000: float,
                                        breath_nn: float, snore_nn: float, signal_nn: float) -> List[float]:
        """Генерирует 3 смешанных признака из CSV данных."""
        features = []
        
        # 1. Snore/Breath = mean(snore)/mean(breath)
        snore_breath_ratio = (b400 + snore_nn) / (b100 + breath_nn + 1e-8)
        features.append(snore_breath_ratio)
        
        # 2. Snore/Speech = mean(snore)/mean(speech)
        snore_speech_ratio = (b400 + snore_nn) / (b1000 + signal_nn + 1e-8)
        features.append(snore_speech_ratio)
        
        # 3. Breath_autocorr_lag1 (регулярность дыхания)
        # Используем breath_nn как индикатор регулярности
        features.append(breath_nn)
        
        return features


def create_feature_extractor_config() -> Dict[str, Any]:
    """
    Создает конфигурацию для экстрактора признаков.
    
    Returns:
        Словарь с конфигурацией
    """
    return {
        'sampling_rate': 8000,        # Частота дискретизации
        'window_size': 8000,          # Размер окна (1 секунда)
        'ticks_per_window': 10,       # 10 тиков за секунду
        'feature_count': 30,          # Ровно 30 признаков
        'frequency_bands': {
            'breath': (100, 400),     # Breath: 100-400 Гц
            'snore': (300, 1000),     # Snore: 300-1000 Гц
            'speech': (1000, 4000),   # Speech: 1000-4000 Гц
            'high': (4000, 8000)      # High: 4000-8000 Гц
        },
        'audio_features_per_band': 5,  # 5 признаков на полосу
        'accel_features': 7,           # 7 признаков акселерометра
        'mixed_features': 3            # 3 смешанных признака
    } 