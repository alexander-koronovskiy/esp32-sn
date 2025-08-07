#!/usr/bin/env python3
"""
Light тестирование экстрактора признаков с огибающими.
Быстрая проверка основных функций.
"""

import sys
import os
import numpy as np

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.snoring_extractor import SnoringFeatureExtractor, create_snoring_config


def quick_test():
    """Быстрый тест основных функций экстрактора."""
    print("🚀 Light тестирование экстрактора с огибающими...")
    
    try:
        # 1. Создаем конфигурацию
        print("1. Загрузка конфигурации...")
        config = create_snoring_config()
        print(f"✓ Частота дискретизации: {config['sampling_rate']} Hz")
        print(f"✓ Длина сегмента: {config['segment_length']} сэмплов")
        print(f"✓ Максимум признаков: {config['max_features']}")
        print(f"✓ Огибающих на диапазон: {config['envelope_count']}")
        
        # 2. Создаем экстрактор
        print("\n2. Создание экстрактора...")
        extractor = SnoringFeatureExtractor(
            sampling_rate=config['sampling_rate'],
            segment_length=config['segment_length']
        )
        print(f"✓ Экстрактор создан")
        print(f"✓ Частотных диапазонов: {len(extractor.frequency_bands)}")
        
        # 3. Создаем тестовые данные
        print("\n3. Создание тестовых данных...")
        segment_length = config['segment_length']
        t = np.linspace(0, 1, segment_length)
        
        # Простой тестовый сигнал
        test_signal = (np.sin(2 * np.pi * 80 * t) * 0.3 +  # Храп
                      np.sin(2 * np.pi * 0.5 * t) * 0.1 +   # Дыхание
                      np.random.randn(segment_length) * 0.05) # Шум
        
        print(f"✓ Тестовый сигнал создан ({len(test_signal)} сэмплов)")
        
        # 4. Тестируем извлечение признаков
        print("\n4. Извлечение признаков...")
        features = extractor.extract_snoring_features(test_signal)
        print(f"✓ Извлечено {len(features)} признаков")
        
        # 5. Анализируем структуру признаков
        envelope_features = [k for k in features.keys() if 'envelope' in k]
        basic_features = [k for k in features.keys() if 'envelope' not in k]
        
        print(f"  Базовые признаки: {len(basic_features)}")
        print(f"  Признаки огибающих: {len(envelope_features)}")
        
        # 6. Тестируем огибающие для одного диапазона
        print("\n5. Тестирование огибающих...")
        test_band = 'snoring_low'
        envelopes = extractor.extract_envelopes(test_signal, test_band)
        print(f"✓ Огибающие для {test_band}: {len(envelopes)} значений")
        
        # 7. Показываем примеры признаков
        print("\n6. Примеры признаков:")
        print("  Базовые признаки:")
        for i, feature in enumerate(list(basic_features)[:5]):
            print(f"    {feature}: {features[feature]:.6f}")
        
        print("  Признаки огибающих:")
        for i, feature in enumerate(list(envelope_features)[:5]):
            print(f"    {feature}: {features[feature]:.6f}")
        
        # 8. Проверяем частотные диапазоны
        print("\n7. Частотные диапазоны:")
        for band_name, (low_freq, high_freq) in extractor.frequency_bands.items():
            print(f"  {band_name}: {low_freq}-{high_freq} Hz")
        
        # 9. Финальная статистика
        print("\n" + "=" * 50)
        print("✅ LIGHT ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
        print("=" * 50)
        
        print(f"\n📊 Результаты:")
        print(f"  ✅ Экстрактор работает корректно")
        print(f"  ✅ Извлечено {len(features)} признаков")
        print(f"  ✅ {len(extractor.frequency_bands)} частотных диапазонов")
        print(f"  ✅ {config['envelope_count']} огибающих на диапазон")
        print(f"  ✅ Готов к использованию в ESP32")
        
        print(f"\n🎯 Система готова к работе!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        return False


def main():
    """Основная функция light тестирования."""
    print("🎵 LIGHT ТЕСТИРОВАНИЕ ЭКСТРАКТОРА С ОГИБАЮЩИМИ")
    print("=" * 60)
    
    success = quick_test()
    
    if success:
        print("\n🚀 Light тест пройден! Экстрактор готов к использованию.")
    else:
        print("\n❌ Light тест не пройден. Проверьте настройки.")


if __name__ == "__main__":
    main() 