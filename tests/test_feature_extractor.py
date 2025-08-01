"""
Тесты для модуля извлечения признаков.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.utils.config import Config
from src.features.extractor import FeatureExtractor


class TestFeatureExtractor:
    """Тесты для класса FeatureExtractor."""
    
    @pytest.fixture
    def config(self):
        """Создает тестовую конфигурацию."""
        config = Mock(spec=Config)
        config.get.side_effect = lambda key, default=None: {
            'preprocessing.sampling_rate': 256,
            'features.time_domain': True,
            'features.frequency_domain': True,
            'features.statistical': True,
            'features.spectral_power': True,
            'features.entropy': True,
            'features.correlation': True
        }.get(key, default)
        
        # Мокаем метод get_features_params
        config.get_features_params.return_value = {
            'time_domain': True,
            'frequency_domain': True,
            'statistical': True,
            'spectral_power': True,
            'entropy': True,
            'correlation': True
        }
        
        return config
    
    @pytest.fixture
    def feature_extractor(self, config):
        """Создает экземпляр экстрактора признаков."""
        return FeatureExtractor(config)
    
    def test_init(self, feature_extractor, config):
        """Тест инициализации экстрактора признаков."""
        assert feature_extractor.config == config
        assert feature_extractor.sampling_rate == 256
        assert len(feature_extractor.filter_bands) == 5
    
    def test_extract_time_domain_features(self, feature_extractor):
        """Тест извлечения признаков во временной области."""
        # Создаем тестовый сегмент
        segment = np.random.randn(2, 7680)  # 2 канала, 30 сек * 256 Гц
        
        features = feature_extractor.extract_time_domain_features(segment)
        
        # Проверяем, что признаки извлечены
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Проверяем наличие базовых признаков для каждого канала
        for ch_idx in range(2):
            ch_name = f"ch_{ch_idx}"
            assert f"{ch_name}_mean" in features
            assert f"{ch_name}_std" in features
            assert f"{ch_name}_var" in features
            assert f"{ch_name}_skewness" in features
            assert f"{ch_name}_kurtosis" in features
    
    def test_extract_frequency_domain_features(self, feature_extractor):
        """Тест извлечения признаков в частотной области."""
        # Создаем тестовый сегмент
        segment = np.random.randn(2, 7680)  # 2 канала, 30 сек * 256 Гц
        
        features = feature_extractor.extract_frequency_domain_features(segment)
        
        # Проверяем, что признаки извлечены
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Проверяем наличие спектральных признаков для каждого канала
        for ch_idx in range(2):
            ch_name = f"ch_{ch_idx}"
            assert f"{ch_name}_total_power" in features
            assert f"{ch_name}_dominant_freq" in features
            assert f"{ch_name}_spectral_entropy" in features
    
    def test_extract_statistical_features(self, feature_extractor):
        """Тест извлечения статистических признаков."""
        # Создаем тестовый сегмент
        segment = np.random.randn(2, 7680)  # 2 канала, 30 сек * 256 Гц
        
        features = feature_extractor.extract_statistical_features(segment)
        
        # Проверяем, что признаки извлечены
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Проверяем наличие статистических признаков
        assert 'mean_across_channels' in features
        assert 'std_across_channels' in features
        assert 'var_across_channels' in features
        assert 'signal_complexity' in features
    
    def test_extract_spectral_power_features(self, feature_extractor):
        """Тест извлечения признаков спектральной мощности."""
        # Создаем тестовый сегмент
        segment = np.random.randn(2, 7680)  # 2 канала, 30 сек * 256 Гц
        
        features = feature_extractor.extract_spectral_power_features(segment)
        
        # Проверяем, что признаки извлечены
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Проверяем наличие признаков мощности для каждого канала и диапазона
        for ch_idx in range(2):
            ch_name = f"ch_{ch_idx}"
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                assert f"{ch_name}_{band}_relative_power" in features
                assert f"{ch_name}_{band}_absolute_power" in features
                assert f"{ch_name}_{band}_log_power" in features
    
    def test_extract_entropy_features(self, feature_extractor):
        """Тест извлечения признаков энтропии."""
        # Создаем тестовый сегмент
        segment = np.random.randn(2, 7680)  # 2 канала, 30 сек * 256 Гц
        
        features = feature_extractor.extract_entropy_features(segment)
        
        # Проверяем, что признаки извлечены
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Проверяем наличие признаков энтропии для каждого канала
        for ch_idx in range(2):
            ch_name = f"ch_{ch_idx}"
            assert f"{ch_name}_shannon_entropy" in features
            assert f"{ch_name}_renyi_entropy" in features
            assert f"{ch_name}_tsallis_entropy" in features
    
    def test_extract_correlation_features(self, feature_extractor):
        """Тест извлечения признаков корреляции."""
        # Создаем тестовый сегмент с несколькими каналами
        segment = np.random.randn(3, 7680)  # 3 канала, 30 сек * 256 Гц
        
        features = feature_extractor.extract_correlation_features(segment)
        
        # Проверяем, что признаки извлечены
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Проверяем наличие признаков корреляции
        assert 'mean_correlation' in features
        assert 'std_correlation' in features
        assert 'max_correlation' in features
        assert 'min_correlation' in features
    
    def test_extract_correlation_features_single_channel(self, feature_extractor):
        """Тест извлечения признаков корреляции для одного канала."""
        # Создаем тестовый сегмент с одним каналом
        segment = np.random.randn(1, 7680)  # 1 канал, 30 сек * 256 Гц
        
        features = feature_extractor.extract_correlation_features(segment)
        
        # Проверяем, что признаки не извлечены (недостаточно каналов)
        assert isinstance(features, dict)
        assert len(features) == 0
    
    def test_extract_all_features(self, feature_extractor):
        """Тест извлечения всех признаков."""
        # Создаем тестовый сегмент
        segment = np.random.randn(2, 7680)  # 2 канала, 30 сек * 256 Гц
        
        features = feature_extractor.extract_all_features(segment)
        
        # Проверяем, что признаки извлечены
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Проверяем наличие признаков из разных категорий
        assert any('_mean' in key for key in features.keys())  # Временные признаки
        assert any('_power' in key for key in features.keys())  # Частотные признаки
        assert 'mean_across_channels' in features  # Статистические признаки
    
    def test_extract_features_from_segments(self, feature_extractor):
        """Тест извлечения признаков из всех сегментов."""
        # Создаем тестовые сегменты
        segments = np.random.randn(5, 2, 7680)  # 5 сегментов, 2 канала, 30 сек * 256 Гц
        
        features_df = feature_extractor.extract_features_from_segments(segments)
        
        # Проверяем, что DataFrame создан
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 5
        assert 'segment_id' in features_df.columns
        
        # Проверяем, что признаки извлечены
        assert features_df.shape[1] > 1  # Больше одной колонки (segment_id + признаки)
    
    def test_reduce_dimensionality(self, feature_extractor):
        """Тест уменьшения размерности признаков."""
        # Создаем тестовый DataFrame с признаками
        features_df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'segment_id': range(100)
        })
        
        # Уменьшаем размерность
        reduced_df = feature_extractor.reduce_dimensionality(features_df, n_components=2)
        
        # Проверяем, что размерность уменьшена
        assert isinstance(reduced_df, pd.DataFrame)
        assert len(reduced_df) == 100
        assert 'segment_id' in reduced_df.columns
        assert 'pca_0' in reduced_df.columns
        assert 'pca_1' in reduced_df.columns


if __name__ == "__main__":
    pytest.main([__file__]) 