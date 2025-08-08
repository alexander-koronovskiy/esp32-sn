#!/usr/bin/env python3
"""
Fixed-point DSP для ESP32 оптимизации
Реализует IIR/FIR фильтры, LUT для log/entropy, квантование
"""

import numpy as np
from typing import List, Tuple, Optional
import math

# Константы для fixed-point арифметики
FIXED_POINT_BITS = 16
FIXED_POINT_SCALE = 2 ** (FIXED_POINT_BITS - 1)
MAX_INT16 = 32767
MIN_INT16 = -32768

class FixedPointDSP:
    """Fixed-point DSP для аудио обработки на ESP32"""
    
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        self.fixed_point_scale = FIXED_POINT_SCALE
        
        # LUT для log и entropy вычислений
        self.log_lut = self._create_log_lut()
        self.entropy_lut = self._create_entropy_lut()
        
        # Фильтры
        self.lp_filter = self._create_lp_filter()
        self.bandpass_filters = self._create_bandpass_filters()
    
    def _create_log_lut(self) -> np.ndarray:
        """Создание LUT для логарифмических вычислений"""
        lut_size = 1024
        lut = np.zeros(lut_size, dtype=np.int16)
        
        for i in range(lut_size):
            if i > 0:
                # log(x) * scale, где x = i / lut_size
                log_val = math.log(i / lut_size + 1e-10)
                lut[i] = int(log_val * self.fixed_point_scale)
        
        return lut
    
    def _create_entropy_lut(self) -> np.ndarray:
        """Создание LUT для энтропийных вычислений"""
        lut_size = 256
        lut = np.zeros(lut_size, dtype=np.int16)
        
        for i in range(lut_size):
            if i > 0:
                # -p * log(p), где p = i / lut_size
                p = i / lut_size
                entropy_val = -p * math.log(p + 1e-10)
                lut[i] = int(entropy_val * self.fixed_point_scale)
        
        return lut
    
    def _create_lp_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Создание low-pass фильтра для гравитационной составляющей"""
        # Простой IIR фильтр: y[n] = 0.95*y[n-1] + 0.05*x[n]
        b = np.array([0.05], dtype=np.float32)
        a = np.array([1.0, -0.95], dtype=np.float32)
        
        # Конвертация в fixed-point
        b_fixed = (b * self.fixed_point_scale).astype(np.int16)
        a_fixed = (a * self.fixed_point_scale).astype(np.int16)
        
        return b_fixed, a_fixed
    
    def _create_bandpass_filters(self) -> dict:
        """Создание полосовых фильтров для частотных диапазонов"""
        filters = {}
        
        # Breath: 100-400 Hz
        breath_b, breath_a = self._design_bandpass(100, 400)
        filters['breath'] = (breath_b, breath_a)
        
        # Snore: 300-1000 Hz
        snore_b, snore_a = self._design_bandpass(300, 1000)
        filters['snore'] = (snore_b, snore_a)
        
        # Speech/Noise: 1000-4000 Hz
        speech_b, speech_a = self._design_bandpass(1000, 4000)
        filters['speech'] = (speech_b, speech_a)
        
        return filters
    
    def _design_bandpass(self, low_freq: float, high_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Проектирование полосового фильтра"""
        # Нормализованные частоты
        low_norm = low_freq / (self.sample_rate / 2)
        high_norm = high_freq / (self.sample_rate / 2)
        
        # Простой FIR фильтр (в реальности нужно более сложное проектирование)
        order = 32
        b = np.zeros(order + 1)
        
        for i in range(order + 1):
            if i == order // 2:
                b[i] = high_norm - low_norm
            else:
                b[i] = (np.sin(2 * np.pi * high_norm * (i - order // 2)) - 
                        np.sin(2 * np.pi * low_norm * (i - order // 2))) / (np.pi * (i - order // 2))
        
        a = np.array([1.0])
        
        # Конвертация в fixed-point
        b_fixed = (b * self.fixed_point_scale).astype(np.int16)
        a_fixed = (a * self.fixed_point_scale).astype(np.int16)
        
        return b_fixed, a_fixed
    
    def float_to_fixed(self, x: float) -> int:
        """Конвертация float в fixed-point INT16"""
        fixed = int(x * self.fixed_point_scale)
        return max(MIN_INT16, min(MAX_INT16, fixed))
    
    def fixed_to_float(self, x: int) -> float:
        """Конвертация fixed-point INT16 в float"""
        return x / self.fixed_point_scale
    
    def apply_iir_filter(self, signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Применение IIR фильтра в fixed-point"""
        # Простая реализация IIR фильтра
        output = np.zeros_like(signal, dtype=np.int16)
        
        for i in range(len(signal)):
            # Сумма входов
            input_sum = 0
            for j in range(min(len(b), i + 1)):
                input_sum += signal[i - j] * b[j]
            
            # Сумма выходов (с задержкой)
            output_sum = 0
            for j in range(1, min(len(a), i + 1)):
                output_sum += output[i - j] * a[j]
            
            # Результат
            output[i] = self.float_to_fixed(
                self.fixed_to_float(input_sum - output_sum)
            )
        
        return output
    
    def extract_band_energy(self, signal: np.ndarray, band_name: str) -> int:
        """Извлечение энергии в заданной полосе частот"""
        if band_name not in self.bandpass_filters:
            raise ValueError(f"Неизвестная полоса: {band_name}")
        
        b, a = self.bandpass_filters[band_name]
        
        # Применение фильтра
        filtered = self.apply_iir_filter(signal, b, a)
        
        # Вычисление энергии
        energy = np.sum(filtered.astype(np.int32) ** 2)
        
        return int(energy)
    
    def compute_spectral_features(self, signal: np.ndarray) -> dict:
        """Вычисление спектральных признаков в fixed-point"""
        # Конвертация в fixed-point
        signal_fixed = (signal * self.fixed_point_scale).astype(np.int16)
        
        # RMS
        rms = np.sqrt(np.mean(signal_fixed.astype(np.int32) ** 2))
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal_fixed)) != 0)
        
        # Peak count
        peaks = np.sum(np.diff(signal_fixed) > 0)
        
        # Spectral centroid (упрощенная версия)
        fft = np.fft.fft(signal_fixed)
        freqs = np.fft.fftfreq(len(signal_fixed), 1/self.sample_rate)
        spectral_centroid = np.sum(np.abs(fft) * np.abs(freqs)) / (np.sum(np.abs(fft)) + 1e-10)
        
        # Band energies
        breath_energy = self.extract_band_energy(signal_fixed, 'breath')
        snore_energy = self.extract_band_energy(signal_fixed, 'snore')
        speech_energy = self.extract_band_energy(signal_fixed, 'speech')
        
        return {
            'rms': self.float_to_fixed(rms),
            'zero_crossing_rate': self.float_to_fixed(zero_crossings / len(signal)),
            'peak_count': self.float_to_fixed(peaks / len(signal)),
            'spectral_centroid': self.float_to_fixed(spectral_centroid),
            'breath_energy': breath_energy,
            'snore_energy': snore_energy,
            'speech_energy': speech_energy
        }
    
    def apply_lp_filter(self, signal: np.ndarray) -> np.ndarray:
        """Применение low-pass фильтра для гравитационной составляющей"""
        b, a = self.lp_filter
        return self.apply_iir_filter(signal, b, a)

class FeatureQuantizer:
    """Квантователь признаков для INT16"""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, scale: float = 1.0):
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.fixed_point_scale = FIXED_POINT_SCALE
    
    def quantize_feature(self, value: float) -> int:
        """Квантование признака в INT16"""
        # Нормализация
        normalized = (value - self.mu) / (self.sigma + 1e-10)
        
        # Масштабирование
        scaled = normalized * self.scale
        
        # Конвертация в fixed-point
        fixed = int(scaled * self.fixed_point_scale)
        
        # Ограничение диапазона
        return max(MIN_INT16, min(MAX_INT16, fixed))
    
    def dequantize_feature(self, fixed_value: int) -> float:
        """Деквантование признака из INT16"""
        # Конвертация из fixed-point
        scaled = fixed_value / self.fixed_point_scale
        
        # Обратное масштабирование
        normalized = scaled / self.scale
        
        # Денормализация
        return normalized * self.sigma + self.mu 