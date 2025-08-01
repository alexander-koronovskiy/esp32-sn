"""
Тесты для модуля загрузки данных.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch

from src.utils.config import Config
from src.data.loader import EEGDataLoader


class TestEEGDataLoader:
    """Тесты для класса EEGDataLoader."""
    
    @pytest.fixture
    def config(self):
        """Создает тестовую конфигурацию."""
        config_dict = {
            'preprocessing': {
                'sampling_rate': 256,
                'window_size': 30,
                'overlap': 0.5,
                'normalize': True
            }
        }
        config = Mock(spec=Config)
        config.get.side_effect = lambda key, default=None: {
            'preprocessing.sampling_rate': 256,
            'preprocessing.window_size': 30,
            'preprocessing.overlap': 0.5,
            'preprocessing.normalize': True
        }.get(key, default)
        return config
    
    @pytest.fixture
    def data_loader(self, config):
        """Создает экземпляр загрузчика данных."""
        return EEGDataLoader(config)
    
    def test_init(self, data_loader, config):
        """Тест инициализации загрузчика данных."""
        assert data_loader.config == config
        assert data_loader.sampling_rate == 256
        assert data_loader.window_size == 30
        assert data_loader.overlap == 0.5
    
    def test_load_annotations_csv(self, data_loader):
        """Тест загрузки аннотаций из CSV файла."""
        # Создаем временный CSV файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,duration,stage\n")
            f.write("0,30,Wake\n")
            f.write("30,30,N1\n")
            f.write("60,30,N2\n")
            temp_file = f.name
        
        try:
            annotations = data_loader.load_annotations(temp_file)
            assert len(annotations) == 3
            assert list(annotations.columns) == ['time', 'duration', 'stage']
            assert annotations.iloc[0]['stage'] == 'Wake'
        finally:
            os.unlink(temp_file)
    
    def test_load_annotations_invalid_format(self, data_loader):
        """Тест загрузки аннотаций с неподдерживаемым форматом."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test data\n")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Неподдерживаемый формат файла аннотаций"):
                data_loader.load_annotations(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_align_annotations_with_segments(self, data_loader):
        """Тест выравнивания аннотаций с сегментами."""
        # Создаем тестовые аннотации
        annotations = pd.DataFrame({
            'time': [0, 30, 60, 90],
            'duration': [30, 30, 30, 30],
            'stage': ['Wake', 'N1', 'N2', 'N3']
        })
        
        # Создаем временные метки сегментов
        timestamps = np.array([0, 15, 30, 45, 60, 75, 90, 105])
        
        # Выравниваем аннотации
        labels = data_loader.align_annotations_with_segments(annotations, timestamps)
        
        # Проверяем результаты
        assert len(labels) == len(timestamps)
        assert labels[0] == 0  # Wake
        assert labels[2] == 1  # N1
        assert labels[4] == 2  # N2
        assert labels[6] == 3  # N3
    
    def test_save_and_load_processed_data(self, data_loader):
        """Тест сохранения и загрузки обработанных данных."""
        # Создаем тестовые данные
        segments = np.random.randn(10, 2, 7680)  # 10 сегментов, 2 канала, 30 сек * 256 Гц
        labels = np.random.randint(0, 5, 10)
        
        # Сохраняем данные
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            data_loader.save_processed_data(segments, labels, temp_path)
            
            # Проверяем, что файлы созданы
            assert os.path.exists(f"{temp_path}_segments.npy")
            assert os.path.exists(f"{temp_path}_labels.npy")
            
            # Загружаем данные
            loaded_segments, loaded_labels = data_loader.load_processed_data(temp_path)
            
            # Проверяем, что данные загружены корректно
            np.testing.assert_array_equal(segments, loaded_segments)
            np.testing.assert_array_equal(labels, loaded_labels)
            
        finally:
            # Удаляем временные файлы
            for suffix in ['_segments.npy', '_labels.npy']:
                if os.path.exists(f"{temp_path}{suffix}"):
                    os.unlink(f"{temp_path}{suffix}")
    
    @patch('mne.io.read_raw_edf')
    def test_load_eeg_data_edf(self, mock_read_raw, data_loader):
        """Тест загрузки данных ЭЭГ в формате EDF."""
        # Мокаем объект Raw
        mock_raw = Mock()
        mock_raw.info = {'sfreq': 256}
        mock_read_raw.return_value = mock_raw
        
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
            temp_file = f.name
        
        try:
            raw = data_loader.load_eeg_data(temp_file)
            assert raw == mock_raw
            mock_read_raw.assert_called_once_with(temp_file, preload=True, verbose=False)
        finally:
            os.unlink(temp_file)
    
    def test_load_eeg_data_invalid_format(self, data_loader):
        """Тест загрузки данных ЭЭГ с неподдерживаемым форматом."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Неподдерживаемый формат файла"):
                data_loader.load_eeg_data(temp_file)
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__]) 