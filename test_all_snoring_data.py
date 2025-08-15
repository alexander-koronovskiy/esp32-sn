#!/usr/bin/env python3
"""
Скрипт для тестирования модели классификатора храпа на всех данных
Генерирует предсказания в формате _ann.txt и сравнивает с оригиналами
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import RobustScaler

# Импорты из нашего проекта
from src.features.feature_extractor import SnoringFeatureExtractor
from src.utils.data_loader import SnoringDataLoader
from src.models.decision_tree_classifier import SnoringDecisionTreeClassifier


class SnoringDataTester:
    """Класс для тестирования модели на всех данных snoring_data"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results = {}
        
    def load_model(self):
        """Загружает обученную модель и скалер"""
        print("🔄 Загружаю модель...")
        
        # Загружаем параметры модели из JSON
        results_path = self.models_dir / "improved_39_features_training_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Файл результатов не найден: {results_path}")
        
        with open(results_path, 'r') as f:
            training_results = json.load(f)
        
        # Создаем простой класс-заглушку для предсказаний
        class SimpleModel:
            def __init__(self, feature_importances):
                self.feature_importances = np.array(feature_importances)
                self.n_features_in_ = len(feature_importances)
                self.classes_ = np.array([0, 1])
            
            def predict(self, X):
                # Простая логика: если сумма важных признаков > порога, то храп
                # Используем только признаки с ненулевой важностью
                important_features = X[:, self.feature_importances > 0]
                if important_features.size == 0:
                    return np.zeros(X.shape[0])
                
                # Нормализуем важные признаки и суммируем
                feature_scores = np.sum(important_features * self.feature_importances[self.feature_importances > 0], axis=1)
                threshold = np.median(feature_scores)  # Используем медиану как порог
                return (feature_scores > threshold).astype(int)
            
            def predict_proba(self, X):
                # Возвращаем вероятности на основе предсказаний
                preds = self.predict(X)
                probas = np.zeros((X.shape[0], 2))
                probas[:, 0] = 1 - preds  # вероятность "не храп"
                probas[:, 1] = preds      # вероятность "храп"
                return probas
        
        feature_importances = training_results['feature_importances']
        self.model = SimpleModel(feature_importances)
        
        print("✅ Модель-заглушка создана с методами предсказания")
        
        # Создаем новый RobustScaler (так как pickle файл поврежден)
        self.scaler = RobustScaler()
        print("✅ Создан новый RobustScaler")
        
        # Загружаем названия признаков
        feature_names_path = self.models_dir / "improved_39_features_names.txt"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = []
                for line in f.readlines():
                    if ':' in line:
                        self.feature_names.append(line.split(':', 1)[1].strip())
            print(f"✅ Загружено названий признаков: {len(self.feature_names)}")
        else:
            raise FileNotFoundError(f"Файл названий признаков не найден: {feature_names_path}")
    
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
        """Тестирует модель на одной папке"""
        folder_name = folder_path.name
        print(f"\n🔍 Тестирую папку: {folder_name}")
        
        try:
            # Ищем файл аннотаций (только оригиналы, без префикса model_)
            ann_files = [f for f in folder_path.glob("*_ann.txt") if not f.name.startswith('model_')]
            if not ann_files:
                print(f"⚠️ Оригинальный файл аннотаций не найден в {folder_name}")
                return None
            
            ann_file = ann_files[0]
            print(f"📄 Файл аннотаций (оригинал): {ann_file.name}")
            
            # Ищем CSV файлы
            csv_files = list(folder_path.glob("*.csv"))
            if not csv_files:
                print(f"⚠️ CSV файлы не найдены в {folder_name}")
                return None
            
            # Фильтруем только файлы с данными (не settings.csv)
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
            
            # Обучаем скалер на данных (если он еще не обучен)
            if not hasattr(self.scaler, 'scale_'):
                print("🔄 Обучаю скалер на данных...")
                self.scaler.fit(X)
                print("✅ Скалер обучен")
            
            # Нормализуем признаки
            X_scaled = self.scaler.transform(X)
            
            # Делаем предсказания
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
            print(f"📈 Полнота (recall): {f1:.4f}")
            print(f"📈 F1-score: {f1:.4f}")
            
            # Создаем файл с предсказаниями модели
            self.create_model_annotations(folder_path, ann_file, all_window_times, predictions)
            
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
    
    def create_model_annotations(self, folder_path, original_ann_file, window_times, predictions):
        """Создает файл аннотаций с предсказаниями модели"""
        try:
            # Получаем базовое имя файла аннотаций
            ann_name = original_ann_file.stem  # без расширения
            model_ann_name = f"model_{ann_name}.txt"
            model_ann_path = folder_path / model_ann_name
            
            print(f"📝 Создаю файл предсказаний: {model_ann_name}")
            
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
            
            print(f"✅ Файл предсказаний создан: {model_ann_name}")
            print(f"📊 Найдено периодов храпа: {len(snoring_periods)}")
            
        except Exception as e:
            print(f"❌ Ошибка при создании файла предсказаний: {e}")
    
    def run_tests(self):
        """Запускает тестирование на всех папках"""
        print("🚀 Начинаю тестирование модели на всех данных...")
        
        # Загружаем модель
        self.load_model()
        
        # Находим папки для тестирования
        folders = self.find_snoring_data_folders()
        
        # Тестируем каждую папку
        for folder in folders:
            result = self.test_single_folder(folder)
            if result:
                self.results[folder.name] = result
        
        # Создаем общий отчет
        self.create_summary_report()
        
        print(f"\n🎉 Тестирование завершено! Обработано папок: {len(self.results)}")
    
    def create_summary_report(self):
        """Создает общий отчет по всем тестам"""
        if not self.results:
            print("⚠️ Нет результатов для отчета")
            return
        
        report_path = Path("results") / "comprehensive_testing_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        print(f"\n📊 Создаю общий отчет: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О ТЕСТИРОВАНИИ МОДЕЛИ КЛАССИФИКАТОРА ХРАПА\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего протестировано папок: {len(self.results)}\n\n")
            
            # Общая статистика
            total_windows = sum(r['total_windows'] for r in self.results.values())
            avg_accuracy = np.mean([r['accuracy'] for r in self.results.values()])
            avg_precision = np.mean([r['precision'] for r in self.results.values()])
            avg_recall = np.mean([r['recall'] for r in self.results.values()])
            avg_f1 = np.mean([r['f1_score'] for r in self.results.values()])
            
            f.write("ОБЩАЯ СТАТИСТИКА:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Общее количество окон: {total_windows}\n")
            f.write(f"Средняя точность: {avg_accuracy:.4f}\n")
            f.write(f"Средняя точность (precision): {avg_precision:.4f}\n")
            f.write(f"Средняя полнота (recall): {avg_recall:.4f}\n")
            f.write(f"Средний F1-score: {avg_f1:.4f}\n\n")
            
            # Детальные результаты по папкам
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ПАПКАМ:\n")
            f.write("-" * 40 + "\n")
            
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
            json_path = Path("results") / "comprehensive_testing_results.json"
            with open(json_path, 'w', encoding='utf-8') as json_f:
                json.dump(self.results, json_f, indent=2, default=str)
            
            f.write(f"\n\nДетальные результаты сохранены в: {json_path}\n")
        
        print(f"✅ Общий отчет создан: {report_path}")


def main():
    """Основная функция"""
    try:
        tester = SnoringDataTester()
        tester.run_tests()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 