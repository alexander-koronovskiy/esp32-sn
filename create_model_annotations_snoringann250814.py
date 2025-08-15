#!/usr/bin/env python3
"""
Скрипт для создания разметки с префиксом model_ для данных из папки SnoringAnn250814
Использует обученную модель по ТЗ: max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Импорты из нашего проекта
from src.features.feature_extractor import SnoringFeatureExtractor
from src.utils.data_loader import SnoringDataLoader


class SnoringAnn250814ModelAnnotator:
    """Класс для создания разметки с префиксом model_ для SnoringAnn250814"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results = {}
        
    def load_model(self):
        """Загружает модель по ТЗ"""
        print("🔄 Загружаю модель по ТЗ...")
        
        # Загружаем модель
        model_path = self.models_dir / "improved_39_features_tz_compliant" / "snoring_classifier_tz_compliant.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Модель по ТЗ загружена")
        else:
            raise FileNotFoundError(f"Модель по ТЗ не найдена: {model_path}")
        
        # Загружаем скалер
        scaler_path = self.models_dir / "improved_39_features_tz_compliant" / "scaler_tz_compliant.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Скалер по ТЗ загружен")
        else:
            raise FileNotFoundError(f"Скалер по ТЗ не найден: {scaler_path}")
        
        # Загружаем названия признаков
        feature_names_path = self.models_dir / "improved_39_features_tz_compliant" / "feature_names_tz_compliant.txt"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = []
                for line in f.readlines():
                    if ':' in line:
                        self.feature_names.append(line.split(':', 1)[1].strip())
            print(f"✅ Загружено названий признаков: {len(self.feature_names)}")
        else:
            raise FileNotFoundError(f"Файл названий признаков не найден: {feature_names_path}")
        
        return True
    
    def find_snoringann250814_folders(self):
        """Находит все подпапки в SnoringAnn250814"""
        snoringann_dir = Path("SnoringAnn250814")
        if not snoringann_dir.exists():
            raise FileNotFoundError("Папка SnoringAnn250814 не найдена")
        
        folders = []
        for item in snoringann_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                for subitem in item.iterdir():
                    if subitem.is_dir() and not subitem.name.startswith('.'):
                        folders.append(subitem)
        
        print(f"📁 Найдено папок в SnoringAnn250814: {len(folders)}")
        return folders
    
    def process_single_folder(self, folder_path):
        """Обрабатывает одну папку и создает разметку с префиксом model_"""
        folder_name = folder_path.name
        print(f"\n🔍 Обрабатываю папку: {folder_name}")
        
        try:
            # Ищем оригинальный файл аннотаций
            ann_files = [f for f in folder_path.glob("*_ann.txt") if not f.name.startswith('model_')]
            if not ann_files:
                print(f"⚠️ Файл аннотаций не найден в {folder_name}")
                return None
            
            ann_file = ann_files[0]
            print(f"📄 Файл аннотаций: {ann_file.name}")
            
            # Ищем CSV файлы
            csv_files = list(folder_path.glob("*.csv"))
            csv_files = [f for f in csv_files if not f.name.startswith('settings')]
            csv_files.sort()
            
            print(f"📊 Найдено CSV файлов: {len(csv_files)}")
            
            # Загружаем данные и извлекаем признаки
            data_loader = SnoringDataLoader(str(folder_path))
            data_loader.parse_annotations(ann_file)
            
            extractor = SnoringFeatureExtractor()
            
            all_features = []
            all_window_times = []
            
            # Получаем базовую дату из папки
            base_date = data_loader.get_base_date_from_folder()
            print(f"📅 Используем базовую дату: {base_date}")
            
            for csv_file in csv_files:
                try:
                    features, window_times = extractor.extract_features_from_csv(csv_file, base_date)
                    if features.shape[0] > 0:
                        all_features.append(features)
                        all_window_times.extend(window_times)
                except Exception as e:
                    print(f"⚠️ Ошибка при обработке {csv_file.name}: {e}")
                    continue
            
            if not all_features:
                print(f"⚠️ Не удалось извлечь признаки из {folder_name}")
                return None
            
            # Объединяем все признаки
            X = np.vstack(all_features)
            print(f"📊 Извлечено признаков: {X.shape}")
            
            # Нормализуем признаки
            X_scaled = self.scaler.transform(X)
            
            # Делаем предсказания моделью
            predictions = self.model.predict(X_scaled)
            
            print(f"🎯 Сделано предсказаний: {len(predictions)}")
            print(f"📊 Распределение предсказаний: {np.bincount(predictions)}")
            
            # Создаем файл с предсказаниями модели
            self.create_model_annotations(folder_path, ann_file, all_window_times, predictions)
            
            # Возвращаем результаты
            return {
                'folder': folder_name,
                'total_windows': len(predictions),
                'predictions_distribution': np.bincount(predictions).tolist(),
                'snoring_periods': self.count_snoring_periods(all_window_times, predictions)
            }
            
        except Exception as e:
            print(f"❌ Ошибка при обработке {folder_name}: {e}")
            return None
    
    def count_snoring_periods(self, window_times, predictions):
        """Подсчитывает количество периодов с храпом (8 секунд каждый)"""
        snoring_periods = 0
        
        for window_time, pred in zip(window_times, predictions):
            if pred == 1:  # Snoring
                snoring_periods += 1
        
        return snoring_periods
    
    def create_model_annotations(self, folder_path, original_ann_file, window_times, predictions):
        """Создает файл аннотаций с предсказаниями модели"""
        try:
            # Получаем базовое имя файла аннотаций
            ann_name = original_ann_file.stem  # без расширения
            
            # Создаем с префиксом model_
            model_ann_name = f"model_{ann_name}.txt"
            model_ann_path = folder_path / model_ann_name
            
            print(f"📝 Создаю файл разметки модели: {model_ann_name}")
            
            # Каждое предсказание = период 8 секунд (скользящее окно 8с с шагом 1с)
            snoring_periods = []
            
            for window_time, pred in zip(window_times, predictions):
                if pred == 1:  # Snoring
                    # Каждое окно дает предсказание на период 8 секунд
                    start_time = window_time
                    end_time = window_time + timedelta(seconds=8)  # 8 секунд!
                    snoring_periods.append((start_time, end_time))
            
            # Записываем в файл
            with open(model_ann_path, 'w') as f:
                for start_time, end_time in snoring_periods:
                    # Форматируем время как в оригинале
                    start_str = start_time.strftime("%H:%M:%S.%f")[:-3]  # до миллисекунд
                    end_str = end_time.strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"W,{start_str},{end_str}\n")
            
            print(f"✅ Файл разметки модели создан: {model_ann_name}")
            print(f"📊 Найдено периодов с храпом: {len(snoring_periods)}")
            
        except Exception as e:
            print(f"❌ Ошибка при создании файла разметки: {e}")
    
    def run_annotation(self):
        """Запускает создание разметки для всех папок"""
        print("🚀 Начинаю создание разметки с префиксом model_ для SnoringAnn250814...")
        
        # Загружаем модель
        self.load_model()
        
        # Находим папки для обработки
        folders = self.find_snoringann250814_folders()
        
        # Обрабатываем каждую папку
        for folder in folders:
            result = self.process_single_folder(folder)
            if result:
                self.results[folder.name] = result
        
        # Создаем общий отчет
        self.create_summary_report()
        
        print(f"\n🎉 Создание разметки завершено! Обработано папок: {len(self.results)}")
    
    def create_summary_report(self):
        """Создает общий отчет по созданию разметки"""
        if not self.results:
            print("⚠️ Нет результатов для отчета")
            return
        
        report_path = Path("results") / "snoringann250814_model_annotation_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        print(f"\n📊 Создаю общий отчет: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О СОЗДАНИИ РАЗМЕТКИ С ПРЕФИКСОМ MODEL_ ДЛЯ SNORINGANN250814\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего обработано папок: {len(self.results)}\n")
            f.write(f"Модель: max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced\n\n")
            
            # Общая статистика
            total_windows = sum(r['total_windows'] for r in self.results.values())
            total_snoring_periods = sum(r['snoring_periods'] for r in self.results.values())
            
            f.write("ОБЩАЯ СТАТИСТИКА:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Общее количество окон: {total_windows}\n")
            f.write(f"Общее количество периодов с храпом: {total_snoring_periods}\n\n")
            
            # Детальные результаты по папкам
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ПАПКАМ:\n")
            f.write("-" * 50 + "\n")
            
            for folder_name, result in self.results.items():
                f.write(f"\nПапка: {folder_name}\n")
                f.write(f"  Окон: {result['total_windows']}\n")
                f.write(f"  Периодов с храпом: {result['snoring_periods']}\n")
                f.write(f"  Распределение предсказаний: {result['predictions_distribution']}\n")
            
            # Сохраняем результаты в JSON
            json_path = Path("results") / "snoringann250814_model_annotation_results.json"
            with open(json_path, 'w', encoding='utf-8') as json_f:
                json.dump(self.results, json_f, indent=2, default=str)
            
            f.write(f"\n\nДетальные результаты сохранены в: {json_path}\n")
        
        print(f"✅ Общий отчет создан: {report_path}")


def main():
    """Основная функция"""
    try:
        annotator = SnoringAnn250814ModelAnnotator()
        annotator.run_annotation()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 