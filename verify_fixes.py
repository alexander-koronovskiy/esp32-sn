#!/usr/bin/env python3
"""
Скрипт для проверки исправлений.
"""

import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_feature_extractor_fixes():
    """Тестирует исправления в экстракторе признаков."""
    print("Тестирование исправлений в экстракторе признаков...")
    
    from src.features.extractor import FeatureExtractor
    
    # Создаем мок конфигурации
    config = Mock()
    config.get.side_effect = lambda key, default=None: {
        'preprocessing.sampling_rate': 256,
        'features.time_domain': True,
        'features.frequency_domain': True,
        'features.statistical': True,
        'features.spectral_power': True,
        'features.entropy': True,
        'features.correlation': True
    }.get(key, default)
    
    config.get_features_params.return_value = {
        'time_domain': True,
        'frequency_domain': True,
        'statistical': True,
        'spectral_power': True,
        'entropy': True,
        'correlation': True
    }
    
    # Создаем экстрактор признаков
    feature_extractor = FeatureExtractor(config)
    
    # Тестируем с различными типами данных
    test_cases = [
        ("Нормальные данные", np.random.randn(2, 7680)),
        ("Нулевые данные", np.zeros((2, 7680))),
        ("Постоянные данные", np.ones((2, 7680)) * 5),
        ("Один канал", np.random.randn(1, 7680)),
    ]
    
    for name, segment in test_cases:
        print(f"  Тестирование: {name}")
        try:
            # Тестируем все методы извлечения признаков
            features = feature_extractor.extract_all_features(segment)
            assert isinstance(features, dict)
            assert len(features) > 0
            print(f"    ✓ {name} - успешно")
        except Exception as e:
            print(f"    ✗ {name} - ошибка: {str(e)}")
            return False
    
    return True

def test_demo_fix():
    """Тестирует исправление демо."""
    print("Тестирование исправления демо...")
    
    try:
        from demo import create_synthetic_data
        
        # Создаем данные с увеличенными длительностями стадий
        eeg_data, labels = create_synthetic_data(duration_minutes=15, sampling_rate=256, n_channels=2)
        
        # Проверяем, что данные созданы
        assert eeg_data.shape[0] == 2  # 2 канала
        assert eeg_data.shape[1] > 0   # Есть данные
        assert len(labels) == eeg_data.shape[1]  # Метки соответствуют данным
        
        # Проверяем, что есть разные стадии
        unique_labels = np.unique(labels)
        assert len(unique_labels) > 1  # Есть разные стадии
        
        print("  ✓ Синтетические данные созданы корректно")
        
        # Проверяем, что каждая стадия имеет достаточно сегментов
        from src.data.loader import EEGDataLoader
        from src.utils.config import Config
        
        # Создаем мок конфигурации
        config = Mock(spec=Config)
        config.get.side_effect = lambda key, default=None: {
            'preprocessing.sampling_rate': 256,
            'preprocessing.window_size': 30,
            'preprocessing.overlap': 0.5,
            'preprocessing.normalize': True
        }.get(key, default)
        
        data_loader = EEGDataLoader(config)
        
        # Создаем объект Raw для сегментации
        import mne
        info = mne.create_info(ch_names=['EEG1', 'EEG2'], sfreq=256, ch_types=['eeg'] * 2)
        raw = mne.io.RawArray(eeg_data, info)
        
        # Сегментируем данные
        segments, timestamps = data_loader.segment_data(raw)
        
        # Создаем метки для сегментов
        segment_labels = []
        for i, timestamp in enumerate(timestamps):
            start_sample = int(timestamp * 256)
            end_sample = start_sample + int(30 * 256)  # 30 секунд
            segment_label_samples = labels[start_sample:end_sample]
            if len(segment_label_samples) > 0:
                segment_label = np.bincount(segment_label_samples).argmax()
            else:
                segment_label = 0
            segment_labels.append(segment_label)
        
        segment_labels = np.array(segment_labels)
        
        # Проверяем, что каждый класс имеет достаточно сегментов для стратификации
        unique_segment_labels, counts = np.unique(segment_labels, return_counts=True)
        min_segments_per_class = min(counts)
        
        print(f"  Количество сегментов по классам: {dict(zip(unique_segment_labels, counts))}")
        print(f"  Минимальное количество сегментов на класс: {min_segments_per_class}")
        
        if min_segments_per_class >= 2:
            print("  ✓ Достаточно сегментов для стратифицированного разделения")
            return True
        else:
            print("  ✗ Недостаточно сегментов для стратифицированного разделения")
            return False
            
    except Exception as e:
        print(f"  ✗ Ошибка в тестировании демо: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция проверки."""
    print("=" * 60)
    print("ПРОВЕРКА ИСПРАВЛЕНИЙ")
    print("=" * 60)
    
    success = True
    
    # Тестируем исправления в экстракторе признаков
    if not test_feature_extractor_fixes():
        success = False
    
    # Тестируем исправление демо
    if not test_demo_fix():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("ВСЕ ИСПРАВЛЕНИЯ ПРОВЕРЕНЫ УСПЕШНО!")
        print("Теперь тесты должны проходить.")
    else:
        print("ЕСТЬ ПРОБЛЕМЫ С ИСПРАВЛЕНИЯМИ!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main() 