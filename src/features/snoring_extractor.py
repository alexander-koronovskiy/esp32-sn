"""
Специализированный экстрактор признаков для детекции храпа.
Поддерживает работу с 10 огибающими для каждого диапазона частот.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


class SnoringFeatureExtractor:
    """Экстрактор признаков для детекции храпа с поддержкой 10 огибающих."""
    
    def __init__(self, sampling_rate: int = 8000, segment_length: int = 8000):
        """
        Инициализация экстрактора признаков храпа.
        
        Args:
            sampling_rate: Частота дискретизации (Гц)
            segment_length: Длина сегмента (сэмплов)
        """
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        
        # Определяем частотные диапазоны для храпа
        self.frequency_bands = {
            'snoring_low': (50, 150),      # Низкочастотный храп
            'snoring_mid': (150, 300),     # Среднечастотный храп
            'snoring_high': (300, 500),    # Высокочастотный храп
            'breathing_low': (0.5, 2),     # Низкочастотное дыхание
            'breathing_high': (2, 5),      # Высокочастотное дыхание
            'noise_low': (500, 1000),      # Низкочастотный шум
            'noise_high': (1000, 2000),    # Высокочастотный шум
            'harmonic_1': (100, 200),      # Первая гармоника
            'harmonic_2': (200, 400),      # Вторая гармоника
            'harmonic_3': (400, 800)       # Третья гармоника
        }
        
        # Количество огибающих для каждого диапазона
        self.envelope_count = 10
        
    def extract_envelopes(self, audio_data: np.ndarray, band_name: str) -> np.ndarray:
        """
        Извлекает 10 огибающих для заданного частотного диапазона.
        
        Args:
            audio_data: Аудио данные
            band_name: Название частотного диапазона
            
        Returns:
            Массив из 10 огибающих
        """
        low_freq, high_freq = self.frequency_bands[band_name]
        
        # Создаем полосовой фильтр
        nyquist = self.sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Проектирование фильтра Баттерворта
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Применяем фильтр
        filtered_signal = signal.filtfilt(b, a, audio_data)
        
        # Извлекаем огибающую с помощью детектора огибающей
        analytic_signal = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        
        # Разделяем огибающую на 10 сегментов и извлекаем признаки
        segment_length = len(envelope) // self.envelope_count
        envelopes = []
        
        for i in range(self.envelope_count):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            
            if i == self.envelope_count - 1:  # Последний сегмент
                end_idx = len(envelope)
            
            segment = envelope[start_idx:end_idx]
            
            # Извлекаем признаки для каждого сегмента огибающей
            segment_features = self._extract_envelope_features(segment)
            envelopes.extend(segment_features)
        
        return np.array(envelopes)
    
    def _extract_envelope_features(self, envelope_segment: np.ndarray) -> List[float]:
        """
        Извлекает признаки из сегмента огибающей.
        
        Args:
            envelope_segment: Сегмент огибающей
            
        Returns:
            Список признаков
        """
        if len(envelope_segment) == 0:
            return [0.0] * 5  # Возвращаем нули если сегмент пустой
        
        features = []
        
        # 1. Среднее значение огибающей
        features.append(np.mean(envelope_segment))
        
        # 2. Стандартное отклонение огибающей
        features.append(np.std(envelope_segment))
        
        # 3. Максимальное значение огибающей
        features.append(np.max(envelope_segment))
        
        # 4. Минимальное значение огибающей
        features.append(np.min(envelope_segment))
        
        # 5. Размах огибающей
        features.append(np.max(envelope_segment) - np.min(envelope_segment))
        
        return features
    
    def extract_snoring_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Извлекает комплексные признаки храпа с использованием 10 огибающих.
        
        Args:
            audio_data: Аудио данные (1 секунда при 8 kHz)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        # Базовые статистические признаки
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        features['std'] = np.std(audio_data)
        features['mean'] = np.mean(audio_data)
        features['max'] = np.max(audio_data)
        features['min'] = np.min(audio_data)
        features['range'] = np.max(audio_data) - np.min(audio_data)
        
        # Статистические моменты
        features['skewness'] = skew(audio_data)
        features['kurtosis'] = kurtosis(audio_data)
        
        # Энергетические признаки
        features['energy'] = np.sum(audio_data**2)
        features['abs_mean'] = np.mean(np.abs(audio_data))
        
        # Частотные признаки
        features['zero_crossings'] = np.sum(np.diff(np.signbit(audio_data)))
        features['peak_count'] = len(signal.find_peaks(audio_data)[0])
        
        # Извлекаем огибающие для каждого частотного диапазона
        for band_name in self.frequency_bands.keys():
            envelopes = self.extract_envelopes(audio_data, band_name)
            
            # Добавляем статистики огибающих
            features[f'{band_name}_envelope_mean'] = np.mean(envelopes)
            features[f'{band_name}_envelope_std'] = np.std(envelopes)
            features[f'{band_name}_envelope_max'] = np.max(envelopes)
            features[f'{band_name}_envelope_min'] = np.min(envelopes)
            features[f'{band_name}_envelope_range'] = np.max(envelopes) - np.min(envelopes)
            
            # Добавляем индивидуальные огибающие (10 для каждого диапазона)
            for i in range(self.envelope_count):
                features[f'{band_name}_envelope_{i+1}'] = envelopes[i] if i < len(envelopes) else 0.0
        
        # Специализированные признаки храпа
        features['snoring_intensity'] = self._calculate_snoring_intensity(audio_data)
        features['breathing_rate'] = self._estimate_breathing_rate(audio_data)
        features['breathing_amplitude'] = self._estimate_breathing_amplitude(audio_data)
        features['breathing_regularity'] = self._estimate_breathing_regularity(audio_data)
        
        # Признаки периодичности
        features['snoring_periodicity'] = self._calculate_snoring_periodicity(audio_data)
        features['snoring_regularity'] = self._calculate_snoring_regularity(audio_data)
        
        # Спектральные признаки
        features['spectral_centroid'] = self._calculate_spectral_centroid(audio_data)
        features['spectral_rolloff'] = self._calculate_spectral_rolloff(audio_data)
        features['spectral_bandwidth'] = self._calculate_spectral_bandwidth(audio_data)
        
        # Признаки энтропии
        features['spectral_entropy'] = self._calculate_spectral_entropy(audio_data)
        features['temporal_entropy'] = entropy(np.histogram(audio_data, bins=50)[0])
        
        # Признаки автокорреляции
        features['autocorrelation_peak'] = self._calculate_autocorrelation_peak(audio_data)
        features['autocorrelation_decay'] = self._calculate_autocorrelation_decay(audio_data)
        
        return features
    
    def _calculate_snoring_intensity(self, audio_data: np.ndarray) -> float:
        """Вычисляет интенсивность храпа."""
        # Фокусируемся на частотном диапазоне храпа (50-500 Гц)
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [50/nyquist, 500/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, audio_data)
        return np.sqrt(np.mean(filtered**2))
    
    def _estimate_breathing_rate(self, audio_data: np.ndarray) -> float:
        """Оценивает частоту дыхания."""
        # Фильтруем низкие частоты (0.5-5 Гц)
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [0.5/nyquist, 5/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, audio_data)
        
        # Находим пики для оценки частоты дыхания
        peaks, _ = signal.find_peaks(filtered, distance=int(self.sampling_rate/4))
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sampling_rate
            return 60 / np.mean(intervals)  # Дыханий в минуту
        return 12.0  # Значение по умолчанию
    
    def _estimate_breathing_amplitude(self, audio_data: np.ndarray) -> float:
        """Оценивает амплитуду дыхания."""
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [0.5/nyquist, 5/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, audio_data)
        return np.std(filtered)
    
    def _estimate_breathing_regularity(self, audio_data: np.ndarray) -> float:
        """Оценивает регулярность дыхания."""
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [0.5/nyquist, 5/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, audio_data)
        
        peaks, _ = signal.find_peaks(filtered, distance=int(self.sampling_rate/4))
        if len(peaks) > 2:
            intervals = np.diff(peaks) / self.sampling_rate
            return 1.0 - np.std(intervals) / np.mean(intervals)  # Нормализованная регулярность
        return 0.8  # Значение по умолчанию
    
    def _calculate_snoring_periodicity(self, audio_data: np.ndarray) -> float:
        """Вычисляет периодичность храпа."""
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [50/nyquist, 500/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, audio_data)
        
        # Автокорреляция для оценки периодичности
        autocorr = signal.correlate(filtered, filtered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Находим первый пик после нулевого лага
        peaks, _ = signal.find_peaks(autocorr[100:], height=np.max(autocorr)*0.5)
        if len(peaks) > 0:
            return peaks[0] / self.sampling_rate
        return 0.0
    
    def _calculate_snoring_regularity(self, audio_data: np.ndarray) -> float:
        """Вычисляет регулярность храпа."""
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [50/nyquist, 500/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, audio_data)
        
        # Анализ спектральной плотности
        freqs, psd = signal.welch(filtered, self.sampling_rate)
        return np.std(psd) / np.mean(psd)  # Нормализованная нерегулярность
    
    def _calculate_spectral_centroid(self, audio_data: np.ndarray) -> float:
        """Вычисляет спектральный центроид."""
        freqs, psd = signal.welch(audio_data, self.sampling_rate)
        return np.sum(freqs * psd) / np.sum(psd)
    
    def _calculate_spectral_rolloff(self, audio_data: np.ndarray) -> float:
        """Вычисляет спектральный rolloff."""
        freqs, psd = signal.welch(audio_data, self.sampling_rate)
        cumsum = np.cumsum(psd)
        threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= threshold)[0]
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    def _calculate_spectral_bandwidth(self, audio_data: np.ndarray) -> float:
        """Вычисляет спектральную ширину полосы."""
        freqs, psd = signal.welch(audio_data, self.sampling_rate)
        centroid = np.sum(freqs * psd) / np.sum(psd)
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * psd) / np.sum(psd))
        return bandwidth
    
    def _calculate_spectral_entropy(self, audio_data: np.ndarray) -> float:
        """Вычисляет спектральную энтропию."""
        freqs, psd = signal.welch(audio_data, self.sampling_rate)
        psd_norm = psd / np.sum(psd)
        return entropy(psd_norm)
    
    def _calculate_autocorrelation_peak(self, audio_data: np.ndarray) -> float:
        """Вычисляет пик автокорреляции."""
        autocorr = signal.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        return np.max(autocorr[1:]) / autocorr[0]  # Нормализованный пик
    
    def _calculate_autocorrelation_decay(self, audio_data: np.ndarray) -> float:
        """Вычисляет затухание автокорреляции."""
        autocorr = signal.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Находим время затухания до 50% от максимума
        threshold = np.max(autocorr) * 0.5
        decay_idx = np.where(autocorr < threshold)[0]
        return decay_idx[0] / self.sampling_rate if len(decay_idx) > 0 else 1.0


class SnoringFeatureSelector:
    """Селектор признаков для детекции храпа."""
    
    def __init__(self, max_features: int = 50):
        """
        Инициализация селектора признаков.
        
        Args:
            max_features: Максимальное количество признаков
        """
        self.max_features = max_features
        self.selector = SelectKBest(score_func=f_classif, k=max_features)
        self.feature_importance = {}
        
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Выбирает лучшие признаки для детекции храпа.
        
        Args:
            X: Матрица признаков
            y: Метки классов
            feature_names: Названия признаков
            
        Returns:
            Список выбранных признаков
        """
        # Применяем селектор
        X_selected = self.selector.fit_transform(X, y)
        
        # Получаем индексы выбранных признаков
        selected_indices = self.selector.get_support(indices=True)
        
        # Сохраняем важность признаков
        if feature_names is not None:
            scores = self.selector.scores_
            for i, score in enumerate(scores):
                if i < len(feature_names):
                    self.feature_importance[feature_names[i]] = score
        
        # Возвращаем названия выбранных признаков
        if feature_names is not None:
            return [feature_names[i] for i in selected_indices]
        else:
            return [f"feature_{i}" for i in selected_indices]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Возвращает важность признаков."""
        return self.feature_importance


def create_snoring_config() -> Dict[str, Any]:
    """
    Создает конфигурацию для системы детекции храпа.
    
    Returns:
        Словарь с конфигурацией
    """
    return {
        'sampling_rate': 8000,        # Частота дискретизации аудио
        'segment_length': 8000,       # 1 секунда
        'max_features': 50,           # Максимум 50 признаков (включая огибающие)
        'model_type': 'random_forest', # Модель для храпа
        'max_depth': 15,              # Глубина дерева
        'memory_limit': 200000,       # 200KB лимит памяти (увеличен для огибающих)
        'envelope_count': 10,         # Количество огибающих для каждого диапазона
        'frequency_bands': {          # Частотные диапазоны
            'snoring_low': (50, 150),
            'snoring_mid': (150, 300),
            'snoring_high': (300, 500),
            'breathing_low': (0.5, 2),
            'breathing_high': (2, 5),
            'noise_low': (500, 1000),
            'noise_high': (1000, 2000),
            'harmonic_1': (100, 200),
            'harmonic_2': (200, 400),
            'harmonic_3': (400, 800)
        }
    } 