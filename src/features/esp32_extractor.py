"""
Оптимизированный экстрактор признаков для ESP32.
Извлекает только самые важные признаки для экономии памяти и времени.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import signal


class ESP32FeatureExtractor:
    """Упрощенный экстрактор признаков для ESP32."""
    
    def __init__(self, sampling_rate: int = 64, segment_length: int = 640):
        """
        Инициализация экстрактора.
        
        Args:
            sampling_rate: Частота дискретизации (по умолчанию 64 Hz)
            segment_length: Длина сегмента в сэмплах (по умолчанию 640 = 10 сек)
        """
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        
        # Частотные диапазоны для стадий сна
        self.freq_bands = {
            'delta': (0.5, 4.0),    # Глубокий сон
            'theta': (4.0, 8.0),    # Легкий сон
            'alpha': (8.0, 13.0),   # Расслабленное бодрствование
            'beta': (13.0, 30.0),   # Активное бодрствование
        }
    
    def extract_minimal_features(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Извлекает минимальный набор признаков для ESP32.
        
        Args:
            segment: Сегмент данных ЭЭГ (каналы x время)
            
        Returns:
            Словарь с признаками (максимум 8-10 признаков)
        """
        features = {}
        
        # Обрабатываем каждый канал
        for ch_idx in range(segment.shape[0]):
            signal_data = segment[ch_idx, :]
            ch_name = f"ch_{ch_idx}"
            
            # 1. Базовые статистики (4 признака)
            features[f"{ch_name}_rms"] = float(np.sqrt(np.mean(signal_data**2)))
            features[f"{ch_name}_std"] = float(np.std(signal_data))
            features[f"{ch_name}_mean"] = float(np.mean(signal_data))
            features[f"{ch_name}_range"] = float(np.max(signal_data) - np.min(signal_data))
            
            # 2. Простые частотные признаки (2 признака)
            features[f"{ch_name}_zero_crossings"] = float(np.sum(np.diff(np.sign(signal_data)) != 0))
            features[f"{ch_name}_peak_count"] = float(len(self._find_peaks_simple(signal_data)))
            
            # 3. Простая спектральная мощность (2 признака)
            delta_power, theta_power = self._extract_simple_power_bands(signal_data)
            features[f"{ch_name}_delta_power"] = float(delta_power)
            features[f"{ch_name}_theta_power"] = float(theta_power)
        
        # 4. Межканальные признаки (если несколько каналов)
        if segment.shape[0] > 1:
            features['cross_correlation'] = float(self._calculate_cross_correlation(segment))
            features['channel_diff'] = float(np.mean(np.abs(segment[0] - segment[1])))
        
        # Проверяем на NaN значения и заменяем их на 0
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features
    
    def _find_peaks_simple(self, signal_data: np.ndarray) -> List[int]:
        """
        Простой поиск пиков без использования scipy.signal.
        
        Args:
            signal_data: Сигнал
            
        Returns:
            Индексы пиков
        """
        peaks = []
        for i in range(1, len(signal_data) - 1):
            if signal_data[i] > signal_data[i-1] and signal_data[i] > signal_data[i+1]:
                peaks.append(i)
        return peaks
    
    def _extract_simple_power_bands(self, signal_data: np.ndarray) -> Tuple[float, float]:
        """
        Извлекает простую спектральную мощность в дельта и тета диапазонах.
        
        Args:
            signal_data: Сигнал
            
        Returns:
            (delta_power, theta_power)
        """
        # Простое FFT (без оптимизаций scipy)
        fft_data = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1.0 / self.sampling_rate)
        
        # Положительные частоты
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(freqs)//2])
        
        # Дельта диапазон (0.5 - 4 Hz)
        delta_mask = (positive_freqs >= 0.5) & (positive_freqs <= 4.0)
        delta_power = np.sum(positive_fft[delta_mask]) if np.any(delta_mask) else 0.0
        
        # Тета диапазон (4 - 8 Hz)
        theta_mask = (positive_freqs >= 4.0) & (positive_freqs <= 8.0)
        theta_power = np.sum(positive_fft[theta_mask]) if np.any(theta_mask) else 0.0
        
        # Проверяем на NaN и бесконечность
        if np.isnan(delta_power) or np.isinf(delta_power):
            delta_power = 0.0
        if np.isnan(theta_power) or np.isinf(theta_power):
            theta_power = 0.0
        
        return delta_power, theta_power
    
    def _calculate_cross_correlation(self, segment: np.ndarray) -> float:
        """
        Вычисляет простую корреляцию между каналами.
        
        Args:
            segment: Сегмент с несколькими каналами
            
        Returns:
            Коэффициент корреляции
        """
        if segment.shape[0] < 2:
            return 0.0
        
        # Простая корреляция
        ch1 = segment[0]
        ch2 = segment[1]
        
        # Нормализация
        ch1_std = np.std(ch1)
        ch2_std = np.std(ch2)
        
        # Проверяем на нулевое стандартное отклонение
        if ch1_std < 1e-8:
            ch1_norm = np.zeros_like(ch1)
        else:
            ch1_norm = (ch1 - np.mean(ch1)) / ch1_std
            
        if ch2_std < 1e-8:
            ch2_norm = np.zeros_like(ch2)
        else:
            ch2_norm = (ch2 - np.mean(ch2)) / ch2_std
        
        # Корреляция
        correlation = np.mean(ch1_norm * ch2_norm)
        
        # Проверяем на NaN и бесконечность
        if np.isnan(correlation) or np.isinf(correlation):
            correlation = 0.0
        
        return correlation
    
    def quantize_features(self, features: Dict[str, float], scale: int = 100) -> Dict[str, int]:
        """
        Квантизует признаки в целые числа для экономии памяти.
        
        Args:
            features: Словарь с признаками
            scale: Масштаб квантизации
            
        Returns:
            Словарь с квантизованными признаками
        """
        quantized = {}
        for key, value in features.items():
            # Убеждаемся, что значение - число
            if not isinstance(value, (int, float)):
                value = float(value)
            # Ограничиваем значения для int8
            quantized_value = int(np.clip(value * scale, -128, 127))
            quantized[key] = quantized_value
        return quantized
    
    def extract_features_batch(self, segments: np.ndarray) -> np.ndarray:
        """
        Извлекает признаки из батча сегментов.
        
        Args:
            segments: Массив сегментов (сегменты x каналы x время)
            
        Returns:
            Матрица признаков (сегменты x признаки)
        """
        features_list = []
        
        for segment in segments:
            features = self.extract_minimal_features(segment)
            # Убеждаемся, что все значения - числа
            features_vector = [float(val) for val in features.values()]
            features_list.append(features_vector)
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        """
        Возвращает названия признаков.
        
        Returns:
            Список названий признаков
        """
        # Создаем тестовый сегмент для получения названий
        test_segment = np.random.randn(2, self.segment_length)
        features = self.extract_minimal_features(test_segment)
        return list(features.keys())
    
    def estimate_memory_usage(self) -> Dict[str, int]:
        """
        Оценивает использование памяти.
        
        Returns:
            Словарь с оценками памяти
        """
        feature_names = self.get_feature_names()
        num_features = len(feature_names)
        
        # Оценки памяти (в байтах)
        memory_estimates = {
            'feature_names': len(feature_names) * 20,  # ~20 байт на название
            'feature_values': num_features * 4,  # float32
            'quantized_values': num_features * 1,  # int8
            'segment_buffer': self.segment_length * 4,  # float32
            'fft_buffer': self.segment_length * 8,  # complex64
            'total_per_segment': num_features * 4 + self.segment_length * 12
        }
        
        return memory_estimates


class ESP32FeatureSelector:
    """Селектор признаков для ESP32."""
    
    def __init__(self, max_features: int = 8):
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
        Выбирает наиболее важные признаки.
        
        Args:
            X: Матрица признаков
            y: Метки классов
            feature_names: Названия признаков
            
        Returns:
            Список выбранных признаков
        """
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Проверяем, что feature_names содержит строки
        if not all(isinstance(name, str) for name in feature_names):
            raise ValueError("Все названия признаков должны быть строками")
        
        # Выбор k лучших признаков
        selector = SelectKBest(score_func=f_classif, k=min(self.max_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Получаем индексы выбранных признаков
        selected_indices = selector.get_support(indices=True)
        self.selected_features = [feature_names[i] for i in selected_indices]
        
        # Сохраняем важность признаков
        scores = selector.scores_
        for i, score in enumerate(scores):
            if isinstance(i, int) and i < len(feature_names):
                feature_name = feature_names[i]
                if isinstance(feature_name, str):
                    self.feature_importance[feature_name] = float(score)
        
        return self.selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Возвращает важность признаков.
        
        Returns:
            Словарь с важностью признаков
        """
        return self.feature_importance


def create_esp32_config() -> Dict:
    """
    Создает конфигурацию для ESP32.
    
    Returns:
        Словарь с конфигурацией
    """
    return {
        'sampling_rate': 64,  # Уменьшенная частота дискретизации
        'segment_length': 640,  # 10 секунд
        'max_features': 8,  # Максимум 8 признаков
        'quantization_scale': 100,  # Масштаб квантизации
        'model_type': 'decision_tree',  # Простая модель
        'max_depth': 5,  # Ограниченная глубина дерева
        'memory_limit': 50000,  # 50KB лимит памяти
    } 