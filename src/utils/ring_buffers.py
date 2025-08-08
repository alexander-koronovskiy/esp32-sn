#!/usr/bin/env python3
"""
Кольцевые буферы и WindowAggregator для ESP32
Статические буферы без динамических аллокаций
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

class RingBuffer:
    """Статический кольцевой буфер для ESP32"""
    
    def __init__(self, size: int, dtype=np.int16):
        self.size = size
        self.dtype = dtype
        self.buffer = np.zeros(size, dtype=dtype)
        self.head = 0
        self.tail = 0
        self.count = 0
        self.is_full = False
    
    def push(self, value) -> None:
        """Добавление значения в буфер"""
        self.buffer[self.head] = value
        self.head = (self.head + 1) % self.size
        
        if self.is_full:
            self.tail = (self.tail + 1) % self.size
        else:
            self.count += 1
            if self.count == self.size:
                self.is_full = True
    
    def pop(self):
        """Извлечение значения из буфера"""
        if self.count == 0:
            return None
        
        value = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.size
        self.count -= 1
        self.is_full = False
        
        return value
    
    def peek(self, index: int = 0):
        """Просмотр значения без извлечения"""
        if index >= self.count:
            return None
        
        return self.buffer[(self.tail + index) % self.size]
    
    def get_all(self) -> np.ndarray:
        """Получение всех значений в правильном порядке"""
        if self.count == 0:
            return np.array([], dtype=self.dtype)
        
        if self.is_full:
            # Буфер полный, начинаем с tail
            result = np.zeros(self.size, dtype=self.dtype)
            for i in range(self.size):
                result[i] = self.buffer[(self.tail + i) % self.size]
            return result
        else:
            # Буфер не полный, начинаем с 0
            result = np.zeros(self.count, dtype=self.dtype)
            for i in range(self.count):
                result[i] = self.buffer[(self.tail + i) % self.size]
            return result
    
    def clear(self) -> None:
        """Очистка буфера"""
        self.head = 0
        self.tail = 0
        self.count = 0
        self.is_full = False

class SnoringRingBuffer:
    """Специализированный кольцевой буфер для данных храпа"""
    
    def __init__(self, size: int = 600):  # 60 секунд при 10 Гц
        self.size = size
        self.timestamps = RingBuffer(size, dtype=np.int32)
        self.labels = RingBuffer(size, dtype=np.int8)
        self.p_snore = RingBuffer(size, dtype=np.int16)
        self.is_snoring = RingBuffer(size, dtype=np.int8)
        self.breath_energy = RingBuffer(size, dtype=np.int32)
        self.snore_energy = RingBuffer(size, dtype=np.int32)
        self.speech_energy = RingBuffer(size, dtype=np.int32)
        self.pose_state = RingBuffer(size, dtype=np.int8)
        self.pitch = RingBuffer(size, dtype=np.int16)
        self.roll = RingBuffer(size, dtype=np.int16)
    
    def push_instant_data(self, data: Dict) -> None:
        """Добавление мгновенных данных"""
        current_time = int(time.time() * 1000)  # миллисекунды
        
        self.timestamps.push(current_time)
        self.labels.push(data.get('label_3c', 0))
        self.p_snore.push(data.get('p_snore', 0))
        self.is_snoring.push(data.get('is_snoring', 0))
        self.breath_energy.push(data.get('breath_energy', 0))
        self.snore_energy.push(data.get('snore_energy', 0))
        self.speech_energy.push(data.get('speech_energy', 0))
        self.pose_state.push(data.get('pose_state', 0))
        self.pitch.push(data.get('pitch', 0))
        self.roll.push(data.get('roll', 0))
    
    def get_window_data(self, window_sec: int = 30) -> Dict:
        """Получение данных за окно времени"""
        window_samples = window_sec * 10  # 10 Гц
        
        # Получение всех данных
        timestamps = self.timestamps.get_all()
        labels = self.labels.get_all()
        p_snore = self.p_snore.get_all()
        is_snoring = self.is_snoring.get_all()
        breath_energy = self.breath_energy.get_all()
        snore_energy = self.snore_energy.get_all()
        speech_energy = self.speech_energy.get_all()
        pose_state = self.pose_state.get_all()
        pitch = self.pitch.get_all()
        roll = self.roll.get_all()
        
        # Ограничение окном
        if len(timestamps) > window_samples:
            start_idx = len(timestamps) - window_samples
            timestamps = timestamps[start_idx:]
            labels = labels[start_idx:]
            p_snore = p_snore[start_idx:]
            is_snoring = is_snoring[start_idx:]
            breath_energy = breath_energy[start_idx:]
            snore_energy = snore_energy[start_idx:]
            speech_energy = speech_energy[start_idx:]
            pose_state = pose_state[start_idx:]
            pitch = pitch[start_idx:]
            roll = roll[start_idx:]
        
        return {
            'timestamps': timestamps,
            'labels': labels,
            'p_snore': p_snore,
            'is_snoring': is_snoring,
            'breath_energy': breath_energy,
            'snore_energy': snore_energy,
            'speech_energy': speech_energy,
            'pose_state': pose_state,
            'pitch': pitch,
            'roll': roll
        }

class WindowAggregator:
    """Агрегатор окон для извлечения признаков прогноза"""
    
    def __init__(self):
        self.ring_buffer = SnoringRingBuffer()
    
    def compute_snore_share(self, labels: np.ndarray) -> float:
        """Вычисление доли храпа в окне"""
        if len(labels) == 0:
            return 0.0
        
        # Подсчет храпа (Light_Snoring + Heavy_Snoring)
        snore_count = np.sum(labels > 0)
        return snore_count / len(labels)
    
    def compute_episode_length(self, labels: np.ndarray) -> float:
        """Вычисление длины эпизода храпа"""
        if len(labels) == 0:
            return 0.0
        
        # Поиск непрерывных сегментов храпа
        episode_lengths = []
        current_length = 0
        
        for label in labels:
            if label > 0:  # Храп
                current_length += 1
            else:  # Нет храпа
                if current_length > 0:
                    episode_lengths.append(current_length)
                    current_length = 0
        
        # Добавляем последний эпизод, если он есть
        if current_length > 0:
            episode_lengths.append(current_length)
        
        if episode_lengths:
            return max(episode_lengths) / 10.0  # Конвертация в секунды
        else:
            return 0.0
    
    def compute_snore_transitions(self, labels: np.ndarray) -> int:
        """Подсчет переходов No_Snoring ↔ Snoring"""
        if len(labels) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(labels)):
            prev_snoring = labels[i-1] > 0
            curr_snoring = labels[i] > 0
            
            if prev_snoring != curr_snoring:
                transitions += 1
        
        return transitions
    
    def compute_energy_stats(self, energy_data: np.ndarray) -> Dict:
        """Вычисление статистик энергии"""
        if len(energy_data) == 0:
            return {'mean': 0, 'max': 0, 'std': 0}
        
        return {
            'mean': float(np.mean(energy_data)),
            'max': float(np.max(energy_data)),
            'std': float(np.std(energy_data))
        }
    
    def compute_pose_stats(self, pose_data: np.ndarray, pitch_data: np.ndarray, roll_data: np.ndarray) -> Dict:
        """Вычисление статистик позы"""
        if len(pose_data) == 0:
            return {
                'pose_state_mode': 0,
                'pose_stability': 1.0,
                'pitch_mean': 0,
                'pitch_std': 0,
                'roll_mean': 0,
                'roll_std': 0,
                'pose_changes': 0
            }
        
        # Режим позы
        unique_poses, counts = np.unique(pose_data, return_counts=True)
        pose_state_mode = unique_poses[np.argmax(counts)]
        
        # Статистики pitch и roll
        pitch_mean = float(np.mean(pitch_data))
        pitch_std = float(np.std(pitch_data))
        roll_mean = float(np.mean(roll_data))
        roll_std = float(np.std(roll_data))
        
        # Подсчет смен поз
        pose_changes = 0
        for i in range(1, len(pose_data)):
            if pose_data[i] != pose_data[i-1]:
                pose_changes += 1
        
        # Стабильность позы
        pose_stability = 1.0 - (pose_changes / max(1, len(pose_data) - 1))
        
        return {
            'pose_state_mode': int(pose_state_mode),
            'pose_stability': pose_stability,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'roll_mean': roll_mean,
            'roll_std': roll_std,
            'pose_changes': pose_changes
        }
    
    def compute_accel_stats(self, pitch_data: np.ndarray, roll_data: np.ndarray) -> Dict:
        """Вычисление статистик акселерометра"""
        if len(pitch_data) == 0:
            return {
                'accel_active_ratio': 0.0,
                'jerk_events': 0
            }
        
        # Активность акселерометра (изменения pitch/roll)
        pitch_diff = np.abs(np.diff(pitch_data))
        roll_diff = np.abs(np.diff(roll_data))
        
        # Порог активности
        activity_threshold = 0.1
        active_samples = np.sum((pitch_diff > activity_threshold) | (roll_diff > activity_threshold))
        accel_active_ratio = active_samples / max(1, len(pitch_diff))
        
        # Jerk события (резкие изменения)
        jerk_threshold = 0.5
        jerk_events = np.sum((pitch_diff > jerk_threshold) | (roll_diff > jerk_threshold))
        
        return {
            'accel_active_ratio': float(accel_active_ratio),
            'jerk_events': int(jerk_events)
        }
    
    def aggregate_window_features(self, window_sec: int = 30) -> Dict:
        """Агрегация признаков для окна"""
        # Получение данных из кольцевого буфера
        window_data = self.ring_buffer.get_window_data(window_sec)
        
        if len(window_data['labels']) == 0:
            return self._empty_features()
        
        # Вычисление признаков
        snore_share = self.compute_snore_share(window_data['labels'])
        episode_length = self.compute_episode_length(window_data['labels'])
        snore_transitions = self.compute_snore_transitions(window_data['labels'])
        
        # Статистики энергии
        snore_energy_stats = self.compute_energy_stats(window_data['snore_energy'])
        breath_energy_stats = self.compute_energy_stats(window_data['breath_energy'])
        
        # Статистики позы
        pose_stats = self.compute_pose_stats(
            window_data['pose_state'],
            window_data['pitch'],
            window_data['roll']
        )
        
        # Статистики акселерометра
        accel_stats = self.compute_accel_stats(
            window_data['pitch'],
            window_data['roll']
        )
        
        # Временные признаки
        temporal_features = self._compute_temporal_features()
        
        return {
            'window_sec': window_sec,
            'snore_share': snore_share,
            'episode_len_sec': episode_length,
            'snore_transitions': snore_transitions,
            'snore_energy_mean': snore_energy_stats['mean'],
            'snore_energy_max': snore_energy_stats['max'],
            'breath_energy_mean': breath_energy_stats['mean'],
            'snore_breath_ratio': snore_energy_stats['mean'] / max(1e-10, breath_energy_stats['mean']),
            'pose_state_mode': pose_stats['pose_state_mode'],
            'pose_stability': pose_stats['pose_stability'],
            'pitch_mean': pose_stats['pitch_mean'],
            'pitch_std': pose_stats['pitch_std'],
            'roll_mean': pose_stats['roll_mean'],
            'roll_std': pose_stats['roll_std'],
            'pose_changes': pose_stats['pose_changes'],
            'accel_active_ratio': accel_stats['accel_active_ratio'],
            'jerk_events': accel_stats['jerk_events'],
            **temporal_features
        }
    
    def _compute_temporal_features(self) -> Dict:
        """Вычисление временных признаков"""
        current_time = time.time()
        hour = (current_time // 3600) % 24
        
        return {
            'hour_of_day': hour,
            'sleep_duration': 0.0,  # Требует дополнительной логики
            'sleep_stage': 0  # Требует дополнительной логики
        }
    
    def _empty_features(self) -> Dict:
        """Пустые признаки при отсутствии данных"""
        return {
            'window_sec': 0,
            'snore_share': 0.0,
            'episode_len_sec': 0.0,
            'snore_transitions': 0,
            'snore_energy_mean': 0.0,
            'snore_energy_max': 0.0,
            'breath_energy_mean': 0.0,
            'snore_breath_ratio': 0.0,
            'pose_state_mode': 0,
            'pose_stability': 1.0,
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'roll_mean': 0.0,
            'roll_std': 0.0,
            'pose_changes': 0,
            'accel_active_ratio': 0.0,
            'jerk_events': 0,
            'hour_of_day': 0,
            'sleep_duration': 0.0,
            'sleep_stage': 0
        }
    
    def push_instant_data(self, data: Dict) -> None:
        """Добавление мгновенных данных в буфер"""
        self.ring_buffer.push_instant_data(data) 