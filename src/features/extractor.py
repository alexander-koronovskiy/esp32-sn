"""
Модуль для извлечения признаков из данных ЭЭГ.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy, kurtosis, skew
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class FeatureExtractor(LoggerMixin):
    """Класс для извлечения признаков из данных ЭЭГ."""
    
    def __init__(self, config: Config):
        """
        Инициализация экстрактора признаков.
        
        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self.features_config = config.get_features_params()
        self.sampling_rate = config.get("preprocessing.sampling_rate", 256)
        self.filter_bands = config.get("preprocessing.filter_bands", [
            [0.5, 4.0],   # Дельта
            [4.0, 8.0],   # Тета
            [8.0, 13.0],  # Альфа
            [13.0, 30.0], # Бета
            [30.0, 100.0] # Гамма
        ])
        
    def extract_time_domain_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает признаки во временной области.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        for ch_idx in range(segment.shape[0]):
            signal_data = segment[ch_idx, :]
            ch_name = f"ch_{ch_idx}"
            
            # Базовые статистические признаки
            features[f"{ch_name}_mean"] = np.mean(signal_data)
            features[f"{ch_name}_std"] = np.std(signal_data)
            features[f"{ch_name}_var"] = np.var(signal_data)
            features[f"{ch_name}_skewness"] = skew(signal_data)
            features[f"{ch_name}_kurtosis"] = kurtosis(signal_data)
            features[f"{ch_name}_min"] = np.min(signal_data)
            features[f"{ch_name}_max"] = np.max(signal_data)
            features[f"{ch_name}_range"] = np.max(signal_data) - np.min(signal_data)
            features[f"{ch_name}_median"] = np.median(signal_data)
            features[f"{ch_name}_q25"] = np.percentile(signal_data, 25)
            features[f"{ch_name}_q75"] = np.percentile(signal_data, 75)
            features[f"{ch_name}_iqr"] = np.percentile(signal_data, 75) - np.percentile(signal_data, 25)
            
            # Признаки формы сигнала
            features[f"{ch_name}_zero_crossings"] = np.sum(np.diff(np.sign(signal_data)) != 0)
            features[f"{ch_name}_peak_count"] = len(signal.find_peaks(signal_data)[0])
            features[f"{ch_name}_valley_count"] = len(signal.find_peaks(-signal_data)[0])
            
            # Энергетические признаки
            features[f"{ch_name}_rms"] = np.sqrt(np.mean(signal_data**2))
            features[f"{ch_name}_energy"] = np.sum(signal_data**2)
            features[f"{ch_name}_abs_mean"] = np.mean(np.abs(signal_data))
            
            # Признаки сложности
            hist, _ = np.histogram(signal_data, bins=20)
            hist = hist[hist > 0]  # Убираем нулевые значения
            if len(hist) > 0:
                features[f"{ch_name}_entropy"] = entropy(hist)
            else:
                features[f"{ch_name}_entropy"] = 0
            
        return features
    
    def extract_frequency_domain_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает признаки в частотной области.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        for ch_idx in range(segment.shape[0]):
            signal_data = segment[ch_idx, :]
            ch_name = f"ch_{ch_idx}"
            
            # Вычисляем спектр мощности
            freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)))
            
            # Общая мощность
            features[f"{ch_name}_total_power"] = np.sum(psd)
            
            # Мощность в различных частотных диапазонах
            for band_name, (low_freq, high_freq) in zip(['delta', 'theta', 'alpha', 'beta', 'gamma'], self.filter_bands):
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                features[f"{ch_name}_{band_name}_power"] = np.sum(psd[band_mask])
                features[f"{ch_name}_{band_name}_power_ratio"] = np.sum(psd[band_mask]) / features[f"{ch_name}_total_power"] if features[f"{ch_name}_total_power"] > 0 else 0
            
            # Доминирующая частота
            features[f"{ch_name}_dominant_freq"] = freqs[np.argmax(psd)]
            
            # Спектральная энтропия
            psd_nonzero = psd[psd > 0]
            if len(psd_nonzero) > 0:
                features[f"{ch_name}_spectral_entropy"] = entropy(psd_nonzero)
            else:
                features[f"{ch_name}_spectral_entropy"] = 0
            
            # Спектральная плотность
            total_power = np.sum(psd)
            if total_power > 0:
                features[f"{ch_name}_spectral_centroid"] = np.sum(freqs * psd) / total_power
                features[f"{ch_name}_spectral_bandwidth"] = np.sqrt(np.sum(((freqs - features[f"{ch_name}_spectral_centroid"])**2) * psd) / total_power)
                
                # Спектральная асимметрия и эксцесс
                if features[f"{ch_name}_spectral_bandwidth"] > 0:
                    features[f"{ch_name}_spectral_skewness"] = np.sum(((freqs - features[f"{ch_name}_spectral_centroid"])**3) * psd) / (features[f"{ch_name}_spectral_bandwidth"]**3 * total_power)
                    features[f"{ch_name}_spectral_kurtosis"] = np.sum(((freqs - features[f"{ch_name}_spectral_centroid"])**4) * psd) / (features[f"{ch_name}_spectral_bandwidth"]**4 * total_power)
                else:
                    features[f"{ch_name}_spectral_skewness"] = 0
                    features[f"{ch_name}_spectral_kurtosis"] = 0
            else:
                features[f"{ch_name}_spectral_centroid"] = 0
                features[f"{ch_name}_spectral_bandwidth"] = 0
                features[f"{ch_name}_spectral_skewness"] = 0
                features[f"{ch_name}_spectral_kurtosis"] = 0
        
        return features
    
    def extract_statistical_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает статистические признаки.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        # Признаки для всех каналов
        features['mean_across_channels'] = np.mean(segment, axis=0).mean()
        features['std_across_channels'] = np.std(segment, axis=0).mean()
        features['var_across_channels'] = np.var(segment, axis=0).mean()
        
        # Корреляции между каналами
        if segment.shape[0] > 1:
            corr_matrix = np.corrcoef(segment)
            # Убираем диагональ
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            features['mean_correlation'] = np.mean(upper_tri)
            features['std_correlation'] = np.std(upper_tri)
            features['max_correlation'] = np.max(upper_tri)
            features['min_correlation'] = np.min(upper_tri)
        
        # Признаки сложности
        complexity_values = []
        for ch in segment:
            hist, _ = np.histogram(ch, bins=20)
            hist = hist[hist > 0]  # Убираем нулевые значения
            if len(hist) > 0:
                complexity_values.append(entropy(hist))
            else:
                complexity_values.append(0)
        features['signal_complexity'] = np.mean(complexity_values)
        
        return features
    
    def extract_spectral_power_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает признаки спектральной мощности.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        for ch_idx in range(segment.shape[0]):
            signal_data = segment[ch_idx, :]
            ch_name = f"ch_{ch_idx}"
            
            # Вычисляем спектр мощности
            freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)))
            
            # Относительная мощность в каждом диапазоне
            total_power = np.sum(psd)
            
            for band_name, (low_freq, high_freq) in zip(['delta', 'theta', 'alpha', 'beta', 'gamma'], self.filter_bands):
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(psd[band_mask])
                
                features[f"{ch_name}_{band_name}_relative_power"] = band_power / total_power if total_power > 0 else 0
                features[f"{ch_name}_{band_name}_absolute_power"] = band_power
                
                # Логарифм мощности
                features[f"{ch_name}_{band_name}_log_power"] = np.log(band_power + 1e-10)
        
        return features
    
    def extract_entropy_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает признаки энтропии.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        for ch_idx in range(segment.shape[0]):
            signal_data = segment[ch_idx, :]
            ch_name = f"ch_{ch_idx}"
            
            # Энтропия Шеннона
            hist, _ = np.histogram(signal_data, bins=20)
            hist = hist[hist > 0]  # Убираем нулевые значения
            if len(hist) > 0:
                features[f"{ch_name}_shannon_entropy"] = entropy(hist)
            else:
                features[f"{ch_name}_shannon_entropy"] = 0
            
            # Энтропия Реньи (α=2)
            if len(hist) > 0:
                features[f"{ch_name}_renyi_entropy"] = -np.log(np.sum((hist / np.sum(hist))**2))
            else:
                features[f"{ch_name}_renyi_entropy"] = 0
            
            # Энтропия Тсаллиса (q=2)
            if len(hist) > 0:
                p = hist / np.sum(hist)
                features[f"{ch_name}_tsallis_entropy"] = (1 - np.sum(p**2)) / 1
            
        return features
    
    def extract_correlation_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает признаки корреляции между каналами.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        if segment.shape[0] < 2:
            return features
        
        # Корреляционная матрица
        corr_matrix = np.corrcoef(segment)
        
        # Убираем диагональ
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Статистики корреляций
        features['mean_correlation'] = np.mean(upper_tri)
        features['std_correlation'] = np.std(upper_tri)
        features['max_correlation'] = np.max(upper_tri)
        features['min_correlation'] = np.min(upper_tri)
        features['correlation_range'] = features['max_correlation'] - features['min_correlation']
        
        # Количество сильных корреляций
        features['strong_correlations'] = np.sum(np.abs(upper_tri) > 0.7)
        features['weak_correlations'] = np.sum(np.abs(upper_tri) < 0.3)
        
        return features
    
    def extract_all_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает все доступные признаки.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь со всеми признаками
        """
        features = {}
        
        # Извлекаем признаки в зависимости от конфигурации
        if self.features_config.get('time_domain', True):
            features.update(self.extract_time_domain_features(segment))
        
        if self.features_config.get('frequency_domain', True):
            features.update(self.extract_frequency_domain_features(segment))
        
        if self.features_config.get('statistical', True):
            features.update(self.extract_statistical_features(segment))
        
        if self.features_config.get('spectral_power', True):
            features.update(self.extract_spectral_power_features(segment))
        
        if self.features_config.get('entropy', True):
            features.update(self.extract_entropy_features(segment))
        
        if self.features_config.get('correlation', True):
            features.update(self.extract_correlation_features(segment))
        
        return features
    
    def extract_features_from_segments(self, segments: np.ndarray) -> pd.DataFrame:
        """
        Извлекает признаки из всех сегментов.
        
        Args:
            segments: Массив сегментов (сегменты x каналы x время)
            
        Returns:
            DataFrame с признаками
        """
        self.logger.info(f"Извлечение признаков из {len(segments)} сегментов")
        
        features_list = []
        
        for i, segment in enumerate(segments):
            features = self.extract_all_features(segment)
            features['segment_id'] = i
            features_list.append(features)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Обработано {i + 1} сегментов")
        
        features_df = pd.DataFrame(features_list)
        self.logger.info(f"Извлечено {features_df.shape[1] - 1} признаков")
        
        return features_df
    
    def reduce_dimensionality(self, features_df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """
        Уменьшает размерность признаков с помощью PCA.
        
        Args:
            features_df: DataFrame с признаками
            n_components: Количество компонент
            
        Returns:
            DataFrame с уменьшенной размерностью
        """
        self.logger.info(f"Уменьшение размерности до {n_components} компонент")
        
        # Убираем нечисловые колонки
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_numeric = features_df[numeric_cols].dropna()
        
        # Стандартизация
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_numeric)
        
        # PCA
        pca = PCA(n_components=min(n_components, features_scaled.shape[1]))
        features_reduced = pca.fit_transform(features_scaled)
        
        # Создаем новый DataFrame
        reduced_df = pd.DataFrame(
            features_reduced,
            columns=[f'pca_{i}' for i in range(features_reduced.shape[1])],
            index=features_df.index
        )
        
        # Добавляем обратно нечисловые колонки
        non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            reduced_df[col] = features_df[col]
        
        explained_variance = pca.explained_variance_ratio_.sum()
        self.logger.info(f"Объясненная дисперсия: {explained_variance:.3f}")
        
        return reduced_df 