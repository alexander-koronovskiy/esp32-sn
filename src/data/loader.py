"""
Модуль для загрузки и обработки данных ЭЭГ.
"""

import numpy as np
import pandas as pd
import mne
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import os

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class EEGDataLoader(LoggerMixin):
    """Класс для загрузки и обработки данных ЭЭГ."""
    
    def __init__(self, config: Config):
        """
        Инициализация загрузчика данных.
        
        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self.sampling_rate = config.get("preprocessing.sampling_rate", 256)
        self.window_size = config.get("preprocessing.window_size", 30)
        self.overlap = config.get("preprocessing.overlap", 0.5)
        
    def load_eeg_data(self, file_path: str) -> mne.io.Raw:
        """
        Загружает данные ЭЭГ из файла.
        
        Args:
            file_path: Путь к файлу с данными ЭЭГ
            
        Returns:
            Объект Raw с данными ЭЭГ
        """
        self.logger.info(f"Загрузка данных ЭЭГ из {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Определяем тип файла по расширению
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.edf', '.bdf']:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_ext == '.fif':
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        elif file_ext == '.set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
        
        self.logger.info(f"Данные загружены: {raw.info}")
        return raw
    
    def preprocess_raw_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Предобработка сырых данных ЭЭГ.
        
        Args:
            raw: Сырые данные ЭЭГ
            
        Returns:
            Предобработанные данные
        """
        self.logger.info("Начало предобработки данных")
        
        # Фильтрация по частотам
        raw.filter(l_freq=0.5, h_freq=100.0, verbose=False)
        
        # Удаление артефактов (простой подход)
        raw.notch_filter(freqs=50, verbose=False)  # Удаление сетевого шума
        
        # Нормализация
        if self.config.get("preprocessing.normalize", True):
            data = raw.get_data()
            data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            raw._data = data
        
        self.logger.info("Предобработка завершена")
        return raw
    
    def segment_data(self, raw: mne.io.Raw) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментирует данные на окна.
        
        Args:
            raw: Данные ЭЭГ
            
        Returns:
            Кортеж (сегменты, временные метки)
        """
        self.logger.info("Сегментация данных на окна")
        
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # Размер окна в сэмплах
        window_samples = int(self.window_size * sfreq)
        # Шаг окна в сэмплах
        step_samples = int(window_samples * (1 - self.overlap))
        
        segments = []
        timestamps = []
        
        for start in range(0, data.shape[1] - window_samples + 1, step_samples):
            segment = data[:, start:start + window_samples]
            segments.append(segment)
            timestamps.append(start / sfreq)
        
        segments = np.array(segments)
        timestamps = np.array(timestamps)
        
        self.logger.info(f"Создано {len(segments)} сегментов")
        return segments, timestamps
    
    def load_annotations(self, file_path: str) -> pd.DataFrame:
        """
        Загружает аннотации стадий сна.
        
        Args:
            file_path: Путь к файлу с аннотациями
            
        Returns:
            DataFrame с аннотациями
        """
        self.logger.info(f"Загрузка аннотаций из {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл с аннотациями не найден: {file_path}")
        
        # Поддерживаемые форматы аннотаций
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            annotations = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            annotations = pd.read_excel(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла аннотаций: {file_ext}")
        
        self.logger.info(f"Загружено {len(annotations)} аннотаций")
        return annotations
    
    def align_annotations_with_segments(
        self, 
        annotations: pd.DataFrame, 
        timestamps: np.ndarray,
        time_column: str = 'time',
        stage_column: str = 'stage'
    ) -> np.ndarray:
        """
        Выравнивает аннотации с сегментами данных.
        
        Args:
            annotations: DataFrame с аннотациями
            timestamps: Временные метки сегментов
            time_column: Название колонки с временем
            stage_column: Название колонки со стадией
            
        Returns:
            Массив меток для сегментов
        """
        self.logger.info("Выравнивание аннотаций с сегментами")
        
        # Преобразуем стадии сна в числовые метки
        stage_mapping = {
            'Wake': 0,
            'N1': 1,
            'N2': 2,
            'N3': 3,
            'REM': 4
        }
        
        # Создаем массив меток
        labels = np.zeros(len(timestamps), dtype=int)
        
        for i, timestamp in enumerate(timestamps):
            # Находим аннотацию для данного времени
            segment_start = timestamp
            segment_end = timestamp + self.window_size
            
            # Ищем аннотацию, которая покрывает большую часть сегмента
            mask = (
                (annotations[time_column] <= segment_end) & 
                (annotations[time_column] + annotations.get('duration', self.window_size) >= segment_start)
            )
            
            if mask.any():
                # Берем аннотацию с наибольшим перекрытием
                overlaps = []
                for idx in annotations[mask].index:
                    ann_start = annotations.loc[idx, time_column]
                    ann_end = ann_start + annotations.loc[idx].get('duration', self.window_size)
                    
                    overlap_start = max(segment_start, ann_start)
                    overlap_end = min(segment_end, ann_end)
                    overlap = max(0, overlap_end - overlap_start)
                    overlaps.append((overlap, idx))
                
                if overlaps:
                    best_overlap_idx = max(overlaps, key=lambda x: x[0])[1]
                    stage = annotations.loc[best_overlap_idx, stage_column]
                    labels[i] = stage_mapping.get(stage, 0)
        
        self.logger.info(f"Выровнено {len(labels)} меток")
        return labels
    
    def load_and_preprocess(
        self, 
        eeg_file: str, 
        annotations_file: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Полный цикл загрузки и предобработки данных.
        
        Args:
            eeg_file: Путь к файлу с данными ЭЭГ
            annotations_file: Путь к файлу с аннотациями (опционально)
            
        Returns:
            Кортеж (сегменты, метки)
        """
        # Загружаем данные ЭЭГ
        raw = self.load_eeg_data(eeg_file)
        
        # Предобрабатываем данные
        raw = self.preprocess_raw_data(raw)
        
        # Сегментируем данные
        segments, timestamps = self.segment_data(raw)
        
        # Загружаем и выравниваем аннотации, если предоставлены
        labels = None
        if annotations_file:
            annotations = self.load_annotations(annotations_file)
            labels = self.align_annotations_with_segments(annotations, timestamps)
        
        return segments, labels
    
    def save_processed_data(
        self, 
        segments: np.ndarray, 
        labels: Optional[np.ndarray], 
        output_path: str
    ):
        """
        Сохраняет обработанные данные.
        
        Args:
            segments: Сегменты данных
            labels: Метки (опционально)
            output_path: Путь для сохранения
        """
        self.logger.info(f"Сохранение обработанных данных в {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем сегменты
        np.save(f"{output_path}_segments.npy", segments)
        
        # Сохраняем метки, если есть
        if labels is not None:
            np.save(f"{output_path}_labels.npy", labels)
        
        self.logger.info("Данные сохранены")
    
    def load_processed_data(self, base_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Загружает обработанные данные.
        
        Args:
            base_path: Базовый путь к файлам
            
        Returns:
            Кортеж (сегменты, метки)
        """
        self.logger.info(f"Загрузка обработанных данных из {base_path}")
        
        segments = np.load(f"{base_path}_segments.npy")
        
        labels = None
        labels_path = f"{base_path}_labels.npy"
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
        
        self.logger.info(f"Загружено {len(segments)} сегментов")
        return segments, labels 