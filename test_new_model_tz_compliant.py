#!/usr/bin/env python3
"""
Скрипт для тестирования новой модели по ТЗ с генерацией файлов model_*_ann.txt
Использует обученную модель: max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Импорты из нашего проекта
from src.features.feature_extractor import SnoringFeatureExtractor
from src.utils.data_loader import SnoringDataLoader


class TZCompliantModelTester:
    """Класс для тестирования новой модели по ТЗ"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results = {}
        
    def load_new_model(self):
        """Загружает новую модель по ТЗ"""
        print("🔄 Загружаю новую модель по ТЗ...")
        
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
        
        # Проверяем гиперпараметры модели
        print(f"📋 Гиперпараметры модели:")
        print(f"  max_depth: {self.model.max_depth}")
        print(f"  min_samples_leaf: {self.model.min_samples_leaf}")
        print(f"  criterion: {self.model.criterion}")
        print(f"  class_weight: {self.model.class_weight}")
        
        return True
    
    def find_snoring_data_folders(self):
        """Находит все подпапки в snoring_data"""
        snoring_data_dir = Path("snoring_data")
        if not snoring_data_dir.exists():
            raise FileNotFoundError("Папка snoring_data не найдена")
        
        folders = []
        for item in snoring_data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                for subitem in item.iterdir():
                    if subitem.is_dir() and not subitem.name.startswith('.'):
                        folders.append(subitem)
        
        print(f"📁 Найдено папок для тестирования: {len(folders)}")
        return folders
    
    def test_single_folder(self, folder_path):
        """Тестирует новую модель на одной папке"""
        folder_name = folder_path.name
        print(f"\n🔍 Тестирую папку: {folder_name}")
        
        try:
            # Ищем оригинальный файл аннотаций (без префикса model_)
            ann_files = [f for f in folder_path.glob("*_ann.txt") if not f.name.startswith('model_')]
            if not ann_files:
                print(f"⚠️ Оригинальный файл аннотаций не найден в {folder_name}")
                return None
            
            ann_file = ann_files[0]
            print(f"📄 Файл аннотаций (оригинал): {ann_file.name}")
            
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
            
            # Делаем предсказания новой моделью
            predictions = self.model.predict(X_scaled)
            prediction_proba = self.model.predict_proba(X_scaled)
            
            print(f"🎯 Сделано предсказаний: {len(predictions)}")
            print(f"📊 Распределение предсказаний: {np.bincount(predictions)}")
            
            # Получаем оригинальные метки
            labels = data_loader.get_labels_for_windows(all_window_times)
            if labels is None:
                print(f"⚠️ Не удалось получить метки для {folder_name}")
                return None
            
            labels = np.array(labels)
            print(f"🏷️ Распределение меток: {np.bincount(labels)}")
            
            # Вычисляем метрики
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            print(f"📈 Точность: {accuracy:.4f}")
            print(f"📈 Точность (precision): {precision:.4f}")
            print(f"📈 Полнота (recall): {recall:.4f}")
            print(f"📈 F1-score: {f1:.4f}")
            
            # Создаем файл с предсказаниями новой модели
            self.create_model_annotations(folder_path, ann_file, all_window_times, predictions, "new_tz_compliant")
            
            # Возвращаем результаты
            return {
                'folder': folder_name,
                'total_windows': len(predictions),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions_distribution': np.bincount(predictions).tolist(),
                'labels_distribution': np.bincount(labels).tolist()
            }
            
        except Exception as e:
            print(f"❌ Ошибка при тестировании {folder_name}: {e}")
            return None
    
    def create_model_annotations(self, folder_path, original_ann_file, window_times, predictions, model_type):
        """Создает файл аннотаций с предсказаниями новой модели"""
        try:
            # Получаем базовое имя файла аннотаций
            ann_name = original_ann_file.stem  # без расширения
            
            if model_type == "new_tz_compliant":
                model_ann_name = f"model_tz_compliant_{ann_name}.txt"
            else:
                model_ann_name = f"model_{ann_name}.txt"
                
            model_ann_path = folder_path / model_ann_name
            
            print(f"📝 Создаю файл предсказаний новой модели: {model_ann_name}")
            
            # Группируем предсказания по периодам
            snoring_periods = []
            current_start = None
            
            for i, (window_time, pred) in enumerate(zip(window_times, predictions)):
                if pred == 1:  # Snoring
                    if current_start is None:
                        current_start = window_time
                else:  # No snoring
                    if current_start is not None:
                        # Завершаем период храпа
                        end_time = window_times[i-1] + timedelta(seconds=8)  # +8 секунд (длина окна)
                        snoring_periods.append((current_start, end_time))
                        current_start = None
            
            # Если последний период храпа не завершен
            if current_start is not None:
                end_time = window_times[-1] + timedelta(seconds=8)
                snoring_periods.append((current_start, end_time))
            
            # Записываем в файл
            with open(model_ann_path, 'w') as f:
                for start_time, end_time in snoring_periods:
                    # Форматируем время как в оригинале
                    start_str = start_time.strftime("%H:%M:%S.%f")[:-3]  # до миллисекунд
                    end_str = end_time.strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"W,{start_str},{end_str}\n")
            
            print(f"✅ Файл предсказаний новой модели создан: {model_ann_name}")
            print(f"📊 Найдено периодов храпа: {len(snoring_periods)}")
            
        except Exception as e:
            print(f"❌ Ошибка при создании файла предсказаний: {e}")
    
    def run_tests(self):
        """Запускает тестирование новой модели на всех папках"""
        print("🚀 Начинаю тестирование новой модели по ТЗ на всех данных...")
        
        # Загружаем новую модель
        self.load_new_model()
        
        # Находим папки для тестирования
        folders = self.find_snoring_data_folders()
        
        # Тестируем каждую папку
        for folder in folders:
            result = self.test_single_folder(folder)
            if result:
                self.results[folder.name] = result
        
        # Создаем общий отчет
        self.create_summary_report()
        
        print(f"\n🎉 Тестирование новой модели по ТЗ завершено! Обработано папок: {len(self.results)}")
    
    def create_summary_report(self):
        """Создает общий отчет по тестированию новой модели"""
        if not self.results:
            print("⚠️ Нет результатов для отчета")
            return
        
        report_path = Path("results") / "tz_compliant_model_testing_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        print(f"\n📊 Создаю общий отчет: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О ТЕСТИРОВАНИИ НОВОЙ МОДЕЛИ ПО ТЗ\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего протестировано папок: {len(self.results)}\n")
            f.write(f"Модель: max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced\n\n")
            
            # Общая статистика
            total_windows = sum(r['total_windows'] for r in self.results.values())
            avg_accuracy = np.mean([r['accuracy'] for r in self.results.values()])
            avg_precision = np.mean([r['precision'] for r in self.results.values()])
            avg_recall = np.mean([r['recall'] for r in self.results.values()])
            avg_f1 = np.mean([r['f1_score'] for r in self.results.values()])
            
            f.write("ОБЩАЯ СТАТИСТИКА НОВОЙ МОДЕЛИ ПО ТЗ:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Общее количество окон: {total_windows}\n")
            f.write(f"Средняя точность: {avg_accuracy:.4f}\n")
            f.write(f"Средняя точность (precision): {avg_precision:.4f}\n")
            f.write(f"Средняя полнота (recall): {avg_recall:.4f}\n")
            f.write(f"Средний F1-score: {avg_f1:.4f}\n\n")
            
            # Детальные результаты по папкам
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ПАПКАМ:\n")
            f.write("-" * 50 + "\n")
            
            for folder_name, result in self.results.items():
                f.write(f"\nПапка: {folder_name}\n")
                f.write(f"  Окон: {result['total_windows']}\n")
                f.write(f"  Точность: {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  F1-score: {result['f1_score']:.4f}\n")
                f.write(f"  Распределение предсказаний: {result['predictions_distribution']}\n")
                f.write(f"  Распределение меток: {result['labels_distribution']}\n")
            
            # Сохраняем результаты в JSON
            json_path = Path("results") / "tz_compliant_model_testing_results.json"
            with open(json_path, 'w', encoding='utf-8') as json_f:
                json.dump(self.results, json_f, indent=2, default=str)
            
            f.write(f"\n\nДетальные результаты сохранены в: {json_path}\n")
        
        print(f"✅ Общий отчет создан: {report_path}")


def main():
    """Основная функция"""
    try:
        tester = TZCompliantModelTester()
        tester.run_tests()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 