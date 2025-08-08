#!/usr/bin/env python3
"""
PoseEstimator для определения ориентации головы
Реализует pitch/roll вычисления через LUT, гистерезис для устойчивости
"""

import numpy as np
from typing import Tuple, Dict, List
import math

# Константы для fixed-point арифметики
FIXED_POINT_BITS = 16
FIXED_POINT_SCALE = 2 ** (FIXED_POINT_BITS - 1)
MAX_INT16 = 32767
MIN_INT16 = -32768

# Константы для акселерометра
GRAVITY = 9.81
ACCEL_SCALE = 16384  # ±2g range

class PoseEstimator:
    """Оценщик позы с fixed-point вычислениями"""
    
    def __init__(self):
        self.fixed_point_scale = FIXED_POINT_SCALE
        
        # LUT для тригонометрических функций
        self.sin_lut = self._create_sin_lut()
        self.cos_lut = self._create_cos_lut()
        self.atan2_lut = self._create_atan2_lut()
        
        # Гистерезис для устойчивости позы
        self.pose_hysteresis = {
            'supine': {'threshold': 0.7, 'hysteresis': 0.1},
            'prone': {'threshold': -0.7, 'hysteresis': 0.1},
            'left': {'threshold': 0.5, 'hysteresis': 0.1},
            'right': {'threshold': -0.5, 'hysteresis': 0.1}
        }
        
        # История поз для стабильности
        self.pose_history = []
        self.max_history_size = 10
        
        # Текущее состояние
        self.current_pose = 'supine'
        self.pose_stability = 1.0
    
    def _create_sin_lut(self) -> np.ndarray:
        """Создание LUT для синуса"""
        lut_size = 1024
        lut = np.zeros(lut_size, dtype=np.int16)
        
        for i in range(lut_size):
            # Угол от 0 до 2π
            angle = 2 * np.pi * i / lut_size
            sin_val = np.sin(angle)
            lut[i] = int(sin_val * self.fixed_point_scale)
        
        return lut
    
    def _create_cos_lut(self) -> np.ndarray:
        """Создание LUT для косинуса"""
        lut_size = 1024
        lut = np.zeros(lut_size, dtype=np.int16)
        
        for i in range(lut_size):
            # Угол от 0 до 2π
            angle = 2 * np.pi * i / lut_size
            cos_val = np.cos(angle)
            lut[i] = int(cos_val * self.fixed_point_scale)
        
        return lut
    
    def _create_atan2_lut(self) -> np.ndarray:
        """Создание LUT для atan2 (упрощенная версия)"""
        lut_size = 256
        lut = np.zeros((lut_size, lut_size), dtype=np.int16)
        
        for i in range(lut_size):
            for j in range(lut_size):
                # Нормализация к [-1, 1]
                y = (i - lut_size // 2) / (lut_size // 2)
                x = (j - lut_size // 2) / (lut_size // 2)
                
                if abs(x) > 1e-10 or abs(y) > 1e-10:
                    atan2_val = np.arctan2(y, x)
                    lut[i, j] = int(atan2_val * self.fixed_point_scale / np.pi)
        
        return lut
    
    def float_to_fixed(self, x: float) -> int:
        """Конвертация float в fixed-point INT16"""
        fixed = int(x * self.fixed_point_scale)
        return max(MIN_INT16, min(MAX_INT16, fixed))
    
    def fixed_to_float(self, x: int) -> float:
        """Конвертация fixed-point INT16 в float"""
        return x / self.fixed_point_scale
    
    def lookup_sin(self, angle: float) -> int:
        """Поиск синуса через LUT"""
        # Нормализация угла к [0, 2π]
        angle = angle % (2 * np.pi)
        
        # Индекс в LUT
        index = int(angle * len(self.sin_lut) / (2 * np.pi))
        index = index % len(self.sin_lut)
        
        return self.sin_lut[index]
    
    def lookup_cos(self, angle: float) -> int:
        """Поиск косинуса через LUT"""
        # Нормализация угла к [0, 2π]
        angle = angle % (2 * np.pi)
        
        # Индекс в LUT
        index = int(angle * len(self.cos_lut) / (2 * np.pi))
        index = index % len(self.cos_lut)
        
        return self.cos_lut[index]
    
    def lookup_atan2(self, y: float, x: float) -> int:
        """Поиск atan2 через LUT"""
        # Нормализация к [-1, 1]
        y_norm = max(-1, min(1, y))
        x_norm = max(-1, min(1, x))
        
        # Индексы в LUT
        y_index = int((y_norm + 1) * (len(self.atan2_lut) - 1) / 2)
        x_index = int((x_norm + 1) * (len(self.atan2_lut[0]) - 1) / 2)
        
        y_index = max(0, min(len(self.atan2_lut) - 1, y_index))
        x_index = max(0, min(len(self.atan2_lut[0]) - 1, x_index))
        
        return self.atan2_lut[y_index, x_index]
    
    def compute_pitch_roll(self, accel_x: float, accel_y: float, accel_z: float) -> Tuple[int, int]:
        """Вычисление pitch и roll в fixed-point"""
        # Нормализация акселерометра
        accel_x_norm = accel_x / ACCEL_SCALE
        accel_y_norm = accel_y / ACCEL_SCALE
        accel_z_norm = accel_z / ACCEL_SCALE
        
        # Вычисление pitch: atan2(-ax, sqrt(ay^2 + az^2))
        ay_sq = accel_y_norm ** 2
        az_sq = accel_z_norm ** 2
        denominator = np.sqrt(ay_sq + az_sq)
        
        if denominator > 1e-10:
            pitch = np.arctan2(-accel_x_norm, denominator)
        else:
            pitch = 0
        
        # Вычисление roll: atan2(ay, az)
        roll = np.arctan2(accel_y_norm, accel_z_norm)
        
        # Конвертация в fixed-point
        pitch_fixed = self.float_to_fixed(pitch)
        roll_fixed = self.float_to_fixed(roll)
        
        return pitch_fixed, roll_fixed
    
    def determine_pose(self, pitch: float, roll: float) -> str:
        """Определение позы с гистерезисом"""
        # Нормализация pitch и roll
        pitch_norm = self.fixed_to_float(pitch) / np.pi
        roll_norm = self.fixed_to_float(roll) / np.pi
        
        # Определение позы на основе pitch
        if pitch_norm > self.pose_hysteresis['supine']['threshold']:
            new_pose = 'supine'
        elif pitch_norm < self.pose_hysteresis['prone']['threshold']:
            new_pose = 'prone'
        elif roll_norm > self.pose_hysteresis['left']['threshold']:
            new_pose = 'left'
        elif roll_norm < self.pose_hysteresis['right']['threshold']:
            new_pose = 'right'
        else:
            # Если в зоне неопределенности, сохраняем текущую позу
            new_pose = self.current_pose
        
        # Применение гистерезиса
        if new_pose != self.current_pose:
            threshold = self.pose_hysteresis[new_pose]['threshold']
            hysteresis = self.pose_hysteresis[new_pose]['hysteresis']
            
            if new_pose in ['supine', 'prone']:
                if abs(pitch_norm - threshold) < hysteresis:
                    new_pose = self.current_pose
            else:  # left, right
                if abs(roll_norm - threshold) < hysteresis:
                    new_pose = self.current_pose
        
        return new_pose
    
    def compute_pose_stability(self) -> float:
        """Вычисление стабильности позы"""
        if len(self.pose_history) < 2:
            return 1.0
        
        # Подсчет смен поз
        pose_changes = 0
        for i in range(1, len(self.pose_history)):
            if self.pose_history[i] != self.pose_history[i-1]:
                pose_changes += 1
        
        # Стабильность = 1 - (количество смен / общее количество измерений)
        stability = 1.0 - (pose_changes / (len(self.pose_history) - 1))
        
        return max(0.0, min(1.0, stability))
    
    def update_pose(self, accel_x: float, accel_y: float, accel_z: float) -> Dict:
        """Обновление оценки позы"""
        # Вычисление pitch и roll
        pitch_fixed, roll_fixed = self.compute_pitch_roll(accel_x, accel_y, accel_z)
        
        # Определение позы
        new_pose = self.determine_pose(pitch_fixed, roll_fixed)
        
        # Обновление истории
        self.pose_history.append(new_pose)
        if len(self.pose_history) > self.max_history_size:
            self.pose_history.pop(0)
        
        # Обновление текущей позы
        self.current_pose = new_pose
        
        # Вычисление стабильности
        self.pose_stability = self.compute_pose_stability()
        
        # Подсчет смен поз
        pose_changes = 0
        for i in range(1, len(self.pose_history)):
            if self.pose_history[i] != self.pose_history[i-1]:
                pose_changes += 1
        
        return {
            'pose_state': new_pose,
            'pose_stability': self.pose_stability,
            'pitch': pitch_fixed,
            'roll': roll_fixed,
            'pose_changes': pose_changes,
            'pose_history': self.pose_history.copy()
        }
    
    def get_pose_stats(self, window_size: int = 10) -> Dict:
        """Получение статистики позы за окно"""
        if len(self.pose_history) == 0:
            return {
                'pose_state_mode': 'supine',
                'pose_stability': 1.0,
                'pitch_mean': 0,
                'pitch_std': 0,
                'roll_mean': 0,
                'roll_std': 0,
                'pose_changes': 0
            }
        
        # Режим позы
        pose_counts = {}
        for pose in self.pose_history:
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
        
        pose_state_mode = max(pose_counts, key=pose_counts.get)
        
        # Статистика pitch и roll (упрощенно)
        pitch_mean = 0
        pitch_std = 0
        roll_mean = 0
        roll_std = 0
        
        return {
            'pose_state_mode': pose_state_mode,
            'pose_stability': self.pose_stability,
            'pitch_mean': self.float_to_fixed(pitch_mean),
            'pitch_std': self.float_to_fixed(pitch_std),
            'roll_mean': self.float_to_fixed(roll_mean),
            'roll_std': self.float_to_fixed(roll_std),
            'pose_changes': len([i for i in range(1, len(self.pose_history)) 
                               if self.pose_history[i] != self.pose_history[i-1]])
        } 