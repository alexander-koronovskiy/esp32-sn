#!/usr/bin/env python3
"""
Скрипт для загрузки и обработки реальных данных храпа
Адаптирован для работы с 2-классовой разметкой
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import glob
from pathlib import Path

def load_csv_data(csv_file: str) -> Tuple[np.ndarray, List[str]]:
    """
    Загрузка CSV данных и создание меток
    
    Args:
        csv_file: Путь к CSV файлу
        
    Returns:
        data: Массив данных
        labels: Список меток (W для бодрствования, пустая строка для храпа)
    """
    print(f"Загружаю файл: {csv_file}")
    
    # Загрузка CSV
    df = pd.read_csv(csv_file, sep=';')
    
    # Проверка структуры
    print(f"Количество строк: {len(df)}")
    print(f"Количество колонок: {len(df.columns)}")
    print(f"Колонки: {list(df.columns)}")
    
    # Конвертация в numpy массив с правильными типами
    data = df.values.astype(np.float64)
    
    # Создание меток: все строки без метки (храп)
    # В реальности нужно использовать аннотации из ann.txt
    labels = [''] * len(data)  # Пустая строка = Snoring
    
    return data, labels

def load_annotations(ann_file: str) -> List[Tuple[str, float, float]]:
    """
    Загрузка аннотаций из ann.txt
    
    Args:
        ann_file: Путь к файлу аннотаций
        
    Returns:
        annotations: Список (метка, время_начала, время_окончания)
    """
    annotations = []
    
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        label = parts[0]  # W или пустая строка
                        start_time = parts[1]  # время начала
                        end_time = parts[2]    # время окончания
                        annotations.append((label, start_time, end_time))
    
    return annotations

def create_labels_from_annotations(data_length: int, annotations: List[Tuple[str, float, float]], 
                                 sample_rate: float = 1.0) -> List[str]:
    """
    Создание меток на основе аннотаций
    
    Args:
        data_length: Количество сэмплов в данных
        annotations: Список аннотаций
        sample_rate: Частота дискретизации (сэмплов в секунду)
        
    Returns:
        labels: Список меток для каждого сэмпла
    """
    labels = [''] * data_length  # По умолчанию все сэмплы = Snoring
    
    for label, start_time_str, end_time_str in annotations:
        # Парсинг времени (формат: HH:MM:SS.mmm)
        start_seconds = parse_time_to_seconds(start_time_str)
        end_seconds = parse_time_to_seconds(end_time_str)
        
        # Конвертация в индексы сэмплов
        start_sample = int(start_seconds * sample_rate)
        end_sample = int(end_seconds * sample_rate)
        
        # Установка меток
        for i in range(start_sample, min(end_sample, data_length)):
            if 0 <= i < data_length:
                labels[i] = label  # W для бодрствования
    
    return labels

def parse_time_to_seconds(time_str: str) -> float:
    """
    Парсинг времени в секунды
    
    Args:
        time_str: Строка времени в формате HH:MM:SS.mmm
        
    Returns:
        seconds: Время в секундах
    """
    try:
        # Разбор времени
        time_parts = time_str.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        
        # Разбор секунд с миллисекундами
        seconds_part = time_parts[2]
        if '.' in seconds_part:
            seconds, milliseconds = seconds_part.split('.')
            seconds = int(seconds)
            milliseconds = int(milliseconds) / 1000.0
        else:
            seconds = int(seconds_part)
            milliseconds = 0.0
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
        return total_seconds
        
    except Exception as e:
        print(f"Ошибка парсинга времени '{time_str}': {e}")
        return 0.0

def load_snoring_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[List[str]], List[str]]:
    """
    Загрузка всего датасета храпа
    
    Args:
        data_dir: Папка с данными
        
    Returns:
        all_data: Список массивов данных
        all_labels: Список списков меток
        file_names: Список имен файлов
    """
    all_data = []
    all_labels = []
    file_names = []
    
    # Поиск всех CSV файлов
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    
    print(f"Найдено {len(csv_files)} CSV файлов")
    
    for csv_file in csv_files:
        try:
            # Пропускаем settings.csv
            if 'settings.csv' in csv_file:
                continue
                
            # Загрузка данных
            data, labels = load_csv_data(csv_file)
            
            # Поиск соответствующего файла аннотаций
            csv_path = Path(csv_file)
            ann_file = csv_path.parent / f"{csv_path.stem.split('_')[0]}_ann.txt"
            
            if ann_file.exists():
                # Загрузка аннотаций
                annotations = load_annotations(str(ann_file))
                print(f"Загружено {len(annotations)} аннотаций из {ann_file}")
                
                # Создание меток на основе аннотаций
                # Предполагаем, что каждый CSV файл содержит 1 секунду данных
                sample_rate = 1.0  # 1 сэмпл в секунду
                labels = create_labels_from_annotations(len(data), annotations, sample_rate)
            else:
                print(f"Файл аннотаций не найден: {ann_file}")
                # Используем метки по умолчанию (все = Snoring)
            
            # Добавление в общий список
            all_data.append(data)
            all_labels.append(labels)
            file_names.append(csv_file)
            
        except Exception as e:
            print(f"Ошибка загрузки файла {csv_file}: {e}")
            continue
    
    return all_data, all_labels, file_names

def prepare_training_data(all_data: List[np.ndarray], all_labels: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготовка данных для обучения
    
    Args:
        all_data: Список массивов данных
        all_labels: Список списков меток
        
    Returns:
        X: Матрица признаков
        y: Вектор меток
    """
    X_list = []
    y_list = []
    
    for data, labels in zip(all_data, all_labels):
        # Каждая строка CSV = один сэмпл
        for i, (row, label) in enumerate(zip(data, labels)):
            # Проверяем данные на корректность
            if len(row) >= 25 and not np.any(np.isnan(row)):
                X_list.append(row)
                y_list.append(label)
    
    # Конвертация в numpy массивы
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Подготовлено данных: {X.shape}")
    
    # Подсчет меток
    label_counts = {}
    for label in y:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Распределение меток: {label_counts}")
    
    return X, y

def main():
    """Основная функция"""
    print("🚀 Загрузка реальных данных храпа")
    
    # Путь к данным
    data_dir = "snoring_data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Папка {data_dir} не найдена")
        return
    
    # Загрузка датасета
    all_data, all_labels, file_names = load_snoring_dataset(data_dir)
    
    if not all_data:
        print("❌ Не удалось загрузить данные")
        return
    
    # Подготовка данных для обучения
    X, y = prepare_training_data(all_data, all_labels)
    
    if len(X) == 0:
        print("❌ Недостаточно данных для обучения")
        return
    
    print(f"✅ Успешно загружено {len(X)} сэмплов")
    print(f"📊 Размер признаков: {X.shape[1]}")
    print(f"🏷️  Метки: {set(y)}")
    
    # Сохранение подготовленных данных
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "real_data_features.npy"), X)
    np.save(os.path.join(output_dir, "real_data_labels.npy"), y)
    
    print(f"💾 Данные сохранены в {output_dir}")
    
    # Пример использования с SnoringClassifier2Class
    try:
        from src.models.snoring_classifier_2class import SnoringClassifier2Class
        
        print("\n🧪 Тестирование SnoringClassifier2Class")
        
        # Создание классификатора
        classifier = SnoringClassifier2Class(model_type='random_forest')
        
        # Обучение модели
        print("📚 Обучение модели...")
        training_result = classifier.train(X, y, data_type='csv')
        
        print(f"✅ Модель обучена!")
        print(f"📊 Точность: {training_result['accuracy']:.3f}")
        print(f"🔢 Количество признаков: {training_result['feature_count']}")
        print(f"📈 Распределение классов: {training_result['class_distribution']}")
        
        # Тестирование предсказания
        print("\n🔮 Тестирование предсказания...")
        test_sample = X[0:1]  # Первый сэмпл
        prediction = classifier.predict(test_sample, data_type='csv')
        
        print(f"📋 Результат предсказания:")
        print(f"   Класс: {prediction['class']}")
        print(f"   ID класса: {prediction['class_id']}")
        print(f"   Вероятности: {prediction['probabilities']}")
        print(f"   Вероятность храпа: {prediction['p_snore']:.3f}")
        print(f"   Есть храп: {prediction['is_snoring']}")
        print(f"   Уверенность: {prediction['confidence']:.3f}")
        
        # Сохранение модели
        model_path = os.path.join(output_dir, "snoring_classifier_2class.pkl")
        classifier.save_model(model_path)
        print(f"💾 Модель сохранена в {model_path}")
        
    except ImportError as e:
        print(f"⚠️  Не удалось импортировать SnoringClassifier2Class: {e}")
        print("📝 Создайте файл src/models/snoring_classifier_2class.py")

if __name__ == "__main__":
    main() 