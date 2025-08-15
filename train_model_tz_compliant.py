#!/usr/bin/env python3
"""
Скрипт для обучения модели классификатора храпа по ТЗ
Гиперпараметры: max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Импорты из нашего проекта
from src.features.feature_extractor import SnoringFeatureExtractor
from src.utils.data_loader import SnoringDataLoader


class TZCompliantModelTrainer:
    """Класс для обучения модели по ТЗ"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = None
        self.results = {}
        
    def load_all_data(self):
        """Загружает все данные из snoring_data"""
        print("🔄 Загружаю все данные...")
        
        snoring_data_dir = Path("snoring_data")
        if not snoring_data_dir.exists():
            raise FileNotFoundError("Папка snoring_data не найдена")
        
        all_features = []
        all_labels = []
        all_window_times = []
        
        # Находим все папки
        folders = []
        for item in snoring_data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                for subitem in item.iterdir():
                    if subitem.is_dir() and not subitem.name.startswith('.'):
                        folders.append(subitem)
        
        print(f"📁 Найдено папок: {len(folders)}")
        
        # Обрабатываем каждую папку
        for folder in folders:
            print(f"\n🔍 Обрабатываю папку: {folder.name}")
            
            try:
                # Ищем оригинальный файл аннотаций (без префикса model_)
                ann_files = [f for f in folder.glob("*_ann.txt") if not f.name.startswith('model_')]
                if not ann_files:
                    print(f"⚠️ Оригинальный файл аннотаций не найден в {folder.name}")
                    continue
                
                ann_file = ann_files[0]
                print(f"📄 Файл аннотаций: {ann_file.name}")
                
                # Ищем CSV файлы
                csv_files = list(folder.glob("*.csv"))
                csv_files = [f for f in csv_files if not f.name.startswith('settings')]
                csv_files.sort()
                
                print(f"📊 Найдено CSV файлов: {len(csv_files)}")
                
                # Загружаем данные
                data_loader = SnoringDataLoader(str(folder))
                data_loader.parse_annotations(ann_file)
                
                print(f"📊 Загружено аннотаций: {len(data_loader.annotations)}")
                
                # Извлекаем признаки
                extractor = SnoringFeatureExtractor()
                base_date = data_loader.get_base_date_from_folder()
                
                folder_features = []
                folder_window_times = []
                
                for csv_file in csv_files:
                    try:
                        features, window_times = extractor.extract_features_from_csv(csv_file, base_date)
                        if features.shape[0] > 0:
                            folder_features.append(features)
                            folder_window_times.extend(window_times)
                    except Exception as e:
                        print(f"⚠️ Ошибка при обработке {csv_file.name}: {e}")
                        continue
                
                if not folder_features:
                    print(f"⚠️ Не удалось извлечь признаки из {folder.name}")
                    continue
                
                # Объединяем признаки
                X_folder = np.vstack(folder_features)
                print(f"📊 Извлечено признаков: {X_folder.shape}")
                
                # Получаем метки
                labels = data_loader.get_labels_for_windows(folder_window_times)
                if labels is None:
                    print(f"⚠️ Не удалось получить метки для {folder.name}")
                    continue
                
                labels = np.array(labels)
                print(f"🏷️ Распределение меток: {np.bincount(labels)}")
                
                # Добавляем к общим данным
                all_features.append(X_folder)
                all_labels.extend(labels)
                all_window_times.extend(folder_window_times)
                
                print(f"✅ Папка {folder.name} обработана успешно")
                
            except Exception as e:
                print(f"❌ Ошибка при обработке {folder.name}: {e}")
                continue
        
        if not all_features:
            raise ValueError("Не удалось загрузить данные ни из одной папки")
        
        # Объединяем все данные
        self.X = np.vstack(all_features)
        self.y = np.array(all_labels)
        self.window_times = all_window_times
        
        print(f"\n📊 ИТОГО загружено:")
        print(f"  Признаков: {self.X.shape}")
        print(f"  Меток: {len(self.y)}")
        print(f"  Распределение меток: {np.bincount(self.y)}")
        
        # Получаем названия признаков
        self.feature_names = extractor.get_feature_names()
        print(f"  Названий признаков: {len(self.feature_names)}")
        
        return True
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Разбивает данные на train/validation/test"""
        print(f"\n🔄 Разбиваю данные: train/validation/test")
        
        # Сначала разделяем на train+val и test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Затем разделяем train+val на train и validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Разбивка данных:")
        print(f"  Train: {X_train.shape[0]} образцов")
        print(f"  Validation: {X_val.shape[0]} образцов")
        print(f"  Test: {X_test.shape[0]} образцов")
        
        # Сохраняем разбивку
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        return True
    
    def normalize_features(self):
        """Нормализует признаки"""
        print(f"\n🔄 Нормализую признаки...")
        
        # Обучаем скалер на train данных
        self.scaler.fit(self.X_train)
        
        # Нормализуем все наборы
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Признаки нормализованы")
        return True
    
    def train_model(self):
        """Обучает модель по ТЗ"""
        print(f"\n🚀 Обучаю модель по ТЗ...")
        
        # Гиперпараметры по ТЗ
        tz_params = {
            'max_depth': 5,
            'min_samples_leaf': 10,
            'criterion': 'gini',
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        print(f"📋 Гиперпараметры по ТЗ:")
        for param, value in tz_params.items():
            print(f"  {param}: {value}")
        
        # Создаем модель
        self.model = DecisionTreeClassifier(**tz_params)
        
        # Обучаем на train данных
        print(f"\n🔄 Обучение модели...")
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print(f"✅ Модель обучена!")
        print(f"📊 Структура дерева:")
        print(f"  Количество узлов: {self.model.tree_.node_count}")
        print(f"  Глубина: {self.model.get_depth()}")
        print(f"  Количество листьев: {self.model.get_n_leaves()}")
        
        return True
    
    def evaluate_model(self):
        """Оценивает модель"""
        print(f"\n📊 Оценка модели...")
        
        # Предсказания на всех наборах
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_val_pred = self.model.predict(self.X_val_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Метрики для каждого набора
        results = {}
        
        for name, y_true, y_pred in [('train', self.y_train, y_train_pred), 
                                    ('validation', self.y_val, y_val_pred),
                                    ('test', self.y_test, y_test_pred)]:
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions_distribution': np.bincount(y_pred).tolist(),
                'labels_distribution': np.bincount(y_true).tolist()
            }
            
            print(f"\n📈 {name.capitalize()} набор:")
            print(f"  Точность: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  Распределение предсказаний: {np.bincount(y_pred)}")
            print(f"  Распределение меток: {np.bincount(y_true)}")
        
        self.results = results
        return True
    
    def analyze_feature_importance(self):
        """Анализирует важность признаков"""
        print(f"\n🔍 Анализ важности признаков...")
        
        feature_importances = self.model.feature_importances_
        
        # Создаем DataFrame для анализа
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importances
        })
        
        # Сортируем по важности
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Добавляем категории признаков
        def categorize_feature(feature_name):
            if feature_name.startswith('b100_'):
                return 'Audio (b100)'
            elif feature_name.startswith('b400_'):
                return 'Audio (b400)'
            elif feature_name.startswith('b1000_'):
                return 'Audio (b1000)'
            elif feature_name.startswith('env_'):
                return 'Audio (env)'
            elif feature_name.startswith('accel_'):
                return 'Accelerometer'
            elif feature_name.startswith('b400_b100') or feature_name.startswith('b400_b1000'):
                return 'Mixed Features'
            else:
                return 'Other'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        # Выводим топ-10 признаков
        print(f"\n🏆 Топ-10 важных признаков:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f} ({row['category']})")
        
        # Сохраняем результаты
        self.feature_importances = importance_df
        return True
    
    def save_model(self):
        """Сохраняет модель и результаты"""
        print(f"\n💾 Сохраняю модель и результаты...")
        
        # Создаем папку для новой модели
        model_dir = self.models_dir / "improved_39_features_tz_compliant"
        model_dir.mkdir(exist_ok=True)
        
        # Сохраняем модель
        import pickle
        model_path = model_dir / "snoring_classifier_tz_compliant.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Сохраняем скалер
        scaler_path = model_dir / "scaler_tz_compliant.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Сохраняем названия признаков
        feature_names_path = model_dir / "feature_names_tz_compliant.txt"
        with open(feature_names_path, 'w') as f:
            for i, name in enumerate(self.feature_names):
                f.write(f"{i}: {name}\n")
        
        # Сохраняем результаты обучения
        results_path = model_dir / "training_results_tz_compliant.json"
        training_results = {
            'hyperparameters': {
                'max_depth': 5,
                'min_samples_leaf': 10,
                'criterion': 'gini',
                'class_weight': 'balanced'
            },
            'data_shapes': {
                'train': self.X_train.shape,
                'validation': self.X_val.shape,
                'test': self.X_test.shape
            },
            'results': self.results,
            'feature_importances': self.feature_importances['importance'].tolist(),
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Сохраняем feature importances в CSV
        csv_path = model_dir / "feature_importances_tz_compliant.csv"
        self.feature_importances.to_csv(csv_path, index=False)
        
        print(f"✅ Модель сохранена в: {model_dir}")
        return True
    
    def create_visualizations(self):
        """Создает визуализации"""
        print(f"\n🎨 Создаю визуализации...")
        
        model_dir = self.models_dir / "improved_39_features_tz_compliant"
        model_dir.mkdir(exist_ok=True)
        
        # 1. Feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importances.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances - TZ Compliant Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        importance_plot_path = model_dir / "feature_importances_tz_compliant.png"
        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrix для test набора
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test_scaled))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Snoring', 'Snoring'],
                   yticklabels=['No Snoring', 'Snoring'])
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_plot_path = model_dir / "confusion_matrix_tz_compliant.png"
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Визуализации сохранены в: {model_dir}")
        return True
    
    def run_training(self):
        """Запускает полный процесс обучения"""
        print("🚀 Начинаю обучение модели по ТЗ...")
        
        try:
            # 1. Загружаем данные
            self.load_all_data()
            
            # 2. Разбиваем на наборы
            self.split_data()
            
            # 3. Нормализуем признаки
            self.normalize_features()
            
            # 4. Обучаем модель
            self.train_model()
            
            # 5. Оцениваем модель
            self.evaluate_model()
            
            # 6. Анализируем важность признаков
            self.analyze_feature_importance()
            
            # 7. Сохраняем модель
            self.save_model()
            
            # 8. Создаем визуализации
            self.create_visualizations()
            
            print(f"\n🎉 Обучение завершено успешно!")
            print(f"📁 Модель сохранена в: models/improved_39_features_tz_compliant/")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Основная функция"""
    trainer = TZCompliantModelTrainer()
    success = trainer.run_training()
    
    if success:
        print(f"\n✅ Модель по ТЗ успешно обучена!")
        print(f"🎯 Гиперпараметры: max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced")
    else:
        print(f"\n❌ Ошибка при обучении модели")


if __name__ == "__main__":
    main() 