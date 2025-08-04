"""
Специализированный экстрактор признаков для детекции храпа.
Извлекает признаки, специфичные для анализа храпа и дыхания.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import entropy, kurtosis, skew
import pandas as pd


class SnoringFeatureExtractor:
    """Экстрактор признаков для детекции храпа."""
    
    def __init__(self, sampling_rate: int = 8000, segment_length: int = 8000):
        """
        Инициализация экстрактора признаков храпа.
        
        Args:
            sampling_rate: Частота дискретизации аудио (по умолчанию 8 kHz)
            segment_length: Длина сегмента в сэмплах (по умолчанию 1 секунда)
        """
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        
        # Частотные диапазоны для анализа храпа
        self.freq_bands = {
            'snoring_low': (20, 100),    # Низкочастотный храп
            'snoring_mid': (100, 300),   # Среднечастотный храп
            'snoring_high': (300, 800),  # Высокочастотный храп
            'breathing': (0.1, 2.0),     # Дыхание
            'speech': (85, 255),         # Речь
        }
    
    def extract_snoring_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает признаки для детекции храпа.
        
        Args:
            segment: Сегмент аудио данных
            
        Returns:
            Словарь с признаками храпа
        """
        features = {}
        
        # 1. Базовые статистические признаки
        features.update(self._extract_basic_stats(segment))
        
        # 2. Частотные признаки храпа
        features.update(self._extract_frequency_features(segment))
        
        # 3. Признаки дыхания
        features.update(self._extract_breathing_features(segment))
        
        # 4. Признаки интенсивности храпа
        features.update(self._extract_intensity_features(segment))
        
        # 5. Признаки паттернов храпа
        features.update(self._extract_pattern_features(segment))
        
        # Проверяем на NaN значения
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features
    
    def _extract_basic_stats(self, segment: np.ndarray) -> Dict[str, float]:
        """Извлекает базовые статистические признаки."""
        features = {}
        
        # Базовые статистики
        features['rms'] = float(np.sqrt(np.mean(segment**2)))
        features['std'] = float(np.std(segment))
        features['mean'] = float(np.mean(segment))
        features['max'] = float(np.max(segment))
        features['min'] = float(np.min(segment))
        features['range'] = float(np.max(segment) - np.min(segment))
        features['skewness'] = float(skew(segment))
        features['kurtosis'] = float(kurtosis(segment))
        
        # Энергетические признаки
        features['energy'] = float(np.sum(segment**2))
        features['abs_mean'] = float(np.mean(np.abs(segment)))
        
        # Признаки сложности
        features['zero_crossings'] = float(np.sum(np.diff(np.sign(segment)) != 0))
        features['peak_count'] = float(len(signal.find_peaks(segment)[0]))
        
        return features
    
    def _extract_frequency_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Извлекает частотные признаки храпа."""
        features = {}
        
        # FFT для частотного анализа
        fft_data = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1.0 / self.sampling_rate)
        
        # Положительные частоты
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(freqs)//2])
        
        # Мощность в диапазонах храпа
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
            if np.any(mask):
                power = np.sum(positive_fft[mask])
                features[f'{band_name}_power'] = float(power)
            else:
                features[f'{band_name}_power'] = 0.0
        
        # Доминирующая частота
        if len(positive_fft) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            features['dominant_frequency'] = float(positive_freqs[dominant_freq_idx])
        else:
            features['dominant_frequency'] = 0.0
        
        # Спектральный центроид
        if np.sum(positive_fft) > 0:
            centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
            features['spectral_centroid'] = float(centroid)
        else:
            features['spectral_centroid'] = 0.0
        
        return features
    
    def _extract_breathing_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Извлекает признаки дыхания."""
        features = {}
        
        # Фильтр для дыхания (0.1-2 Hz)
        nyquist = self.sampling_rate / 2
        low_freq = 0.1 / nyquist
        high_freq = 2.0 / nyquist
        
        # Простой полосовой фильтр
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        breathing_signal = signal.filtfilt(b, a, segment)
        
        # Признаки дыхания
        features['breathing_rate'] = float(self._estimate_breathing_rate(breathing_signal))
        features['breathing_amplitude'] = float(np.std(breathing_signal))
        features['breathing_regularity'] = float(self._estimate_breathing_regularity(breathing_signal))
        
        return features
    
    def _extract_intensity_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Извлекает признаки интенсивности храпа."""
        features = {}
        
        # Интенсивность в разных частотных диапазонах
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            if band_name.startswith('snoring'):
                # Полосовой фильтр для диапазона храпа
                nyquist = self.sampling_rate / 2
                low_norm = low_freq / nyquist
                high_norm = high_freq / nyquist
                
                if low_norm < 1.0 and high_norm < 1.0:
                    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                    filtered_signal = signal.filtfilt(b, a, segment)
                    
                    features[f'{band_name}_intensity'] = float(np.sqrt(np.mean(filtered_signal**2)))
                    features[f'{band_name}_peak_intensity'] = float(np.max(np.abs(filtered_signal)))
                else:
                    features[f'{band_name}_intensity'] = 0.0
                    features[f'{band_name}_peak_intensity'] = 0.0
        
        # Общая интенсивность храпа
        snoring_powers = [features.get(f'{band}_power', 0) for band in self.freq_bands.keys() 
                         if band.startswith('snoring')]
        features['total_snoring_intensity'] = float(np.sum(snoring_powers))
        
        return features
    
    def _extract_pattern_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Извлекает признаки паттернов храпа."""
        features = {}
        
        # Автокорреляция для выявления периодичности
        autocorr = signal.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Периодичность храпа
        peaks = signal.find_peaks(autocorr[:len(autocorr)//4])[0]
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            features['snoring_periodicity'] = float(np.mean(intervals))
            features['snoring_regularity'] = float(1.0 / np.std(intervals) if np.std(intervals) > 0 else 0.0)
        else:
            features['snoring_periodicity'] = 0.0
            features['snoring_regularity'] = 0.0
        
        # Энтропия для оценки сложности паттерна
        hist, _ = np.histogram(segment, bins=50)
        hist = hist[hist > 0]
        if len(hist) > 0:
            features['pattern_entropy'] = float(entropy(hist))
        else:
            features['pattern_entropy'] = 0.0
        
        return features
    
    def _estimate_breathing_rate(self, breathing_signal: np.ndarray) -> float:
        """Оценивает частоту дыхания."""
        # Поиск пиков дыхания
        peaks = signal.find_peaks(breathing_signal, height=np.std(breathing_signal))[0]
        
        if len(peaks) > 1:
            # Интервалы между пиками
            intervals = np.diff(peaks)
            # Частота дыхания (вдохов в минуту)
            breathing_rate = (60.0 * self.sampling_rate) / np.mean(intervals)
            return breathing_rate
        else:
            return 0.0
    
    def _estimate_breathing_regularity(self, breathing_signal: np.ndarray) -> float:
        """Оценивает регулярность дыхания."""
        peaks = signal.find_peaks(breathing_signal, height=np.std(breathing_signal))[0]
        
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            # Коэффициент вариации (ниже = регулярнее)
            cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1.0
            return 1.0 / (1.0 + cv)  # Нормализуем к [0, 1]
        else:
            return 0.0
    
    def extract_features_batch(self, segments: np.ndarray) -> pd.DataFrame:
        """
        Извлекает признаки из батча сегментов.
        
        Args:
            segments: Массив сегментов (n_segments, n_samples)
            
        Returns:
            DataFrame с признаками
        """
        features_list = []
        
        for i, segment in enumerate(segments):
            features = self.extract_snoring_features(segment)
            features['segment_id'] = i
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список названий признаков."""
        # Создаем тестовый сегмент для получения названий признаков
        test_segment = np.random.randn(self.segment_length)
        features = self.extract_snoring_features(test_segment)
        return list(features.keys())


class SnoringFeatureSelector:
    """Селектор признаков для храпа."""
    
    def __init__(self, max_features: int = 15):
        """
        Инициализация селектора.
        
        Args:
            max_features: Максимальное количество признаков
        """
        self.max_features = max_features
        self.selected_features = []
        self.feature_importance = {}
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """
        Выбирает наиболее важные признаки для детекции храпа.
        
        Args:
            X: Матрица признаков
            y: Метки классов
            feature_names: Названия признаков
            
        Returns:
            Список выбранных признаков
        """
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Выбор k лучших признаков
        selector = SelectKBest(score_func=f_classif, k=min(self.max_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Получаем индексы выбранных признаков
        selected_indices = selector.get_support(indices=True)
        self.selected_features = [feature_names[i] for i in selected_indices]
        
        # Сохраняем важность признаков
        scores = selector.scores_
        for i, score in enumerate(scores):
            if i < len(feature_names):
                feature_name = feature_names[i]
                self.feature_importance[feature_name] = float(score)
        
        return self.selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Возвращает важность признаков."""
        return self.feature_importance


def create_snoring_config() -> Dict:
    """
    Создает конфигурацию для детекции храпа.
    
    Returns:
        Словарь с конфигурацией
    """
    return {
        'sampling_rate': 8000,  # Частота дискретизации аудио
        'segment_length': 8000,  # 1 секунда
        'max_features': 15,  # Максимум 15 признаков
        'model_type': 'random_forest',  # Модель для храпа
        'max_depth': 10,  # Глубина дерева
        'memory_limit': 100000,  # 100KB лимит памяти
    } 