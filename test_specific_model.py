#!/usr/bin/env python3
"""
Тестирование конкретной модели с визуализацией
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys
from datetime import datetime

def list_available_models():
    """Показывает доступные модели"""
    print("📋 Доступные модели:")
    
    model_files = []
    for file in os.listdir('models'):
        if file.endswith('_metadata.json'):
            date_str = file.replace('snoring_model_', '').replace('_metadata.json', '')
            try:
                datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                model_files.append(date_str)
            except:
                continue
    
    model_files.sort()
    
    for i, model_date in enumerate(model_files, 1):
        print(f"{i}. {model_date}")
    
    return model_files

def load_model(model_date):
    """Загружает конкретную модель"""
    try:
        # Загружаем метаданные
        metadata_file = f"models/snoring_model_{model_date}_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Загружаем модель
        model_file = f"models/snoring_model_{model_date}.pkl"
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Загружаем scaler если есть
        scaler = None
        scaler_file = f"models/snoring_model_{model_date}_scaler.pkl"
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'metadata': metadata,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели {model_date}: {e}")
        return {'success': False, 'error': str(e)}

def generate_test_data(model_info):
    """Генерирует тестовые данные для модели"""
    n_samples = 500
    n_features = len(model_info['metadata']['feature_names'])
    
    # Генерируем признаки
    X_test = np.random.randn(n_samples, n_features)
    
    # Нормализуем
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    
    # Генерируем метки
    class_names = model_info['metadata']['class_names']
    n_classes = len(class_names)
    
    # Реалистичное распределение классов
    class_weights = [0.4, 0.25, 0.15, 0.1, 0.1]
    y_test = np.random.choice(n_classes, n_samples, p=class_weights[:n_classes])
    
    return X_test, y_test, class_names

def test_model(model_info):
    """Тестирует модель"""
    print(f"🧪 Тестирование модели...")
    
    # Генерируем тестовые данные
    X_test, y_test, class_names = generate_test_data(model_info)
    
    # Применяем scaler если есть
    if model_info['scaler']:
        X_test = model_info['scaler'].transform(X_test)
    
    # Делаем предсказания
    model = model_info['model']
    y_pred = model.predict(X_test)
    
    # Получаем вероятности если возможно
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    return {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'class_names': class_names
    }

def create_model_visualization(model_info, test_results):
    """Создает визуализацию для модели"""
    print(f"📊 Создание визуализации...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Матрица ошибок
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_results['y_test'], test_results['y_pred'])
    
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_results['class_names'],
                yticklabels=test_results['class_names'], ax=ax1)
    ax1.set_title('Матрица ошибок', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Предсказанный класс')
    ax1.set_ylabel('Истинный класс')
    
    # 2. Распределение классов
    class_counts = np.bincount(test_results['y_test'])
    ax2.pie(class_counts, labels=test_results['class_names'], autopct='%1.1f%%')
    ax2.set_title('Распределение классов в тестовых данных', fontsize=14, fontweight='bold')
    
    # 3. Точность по классам
    from sklearn.metrics import classification_report
    report = classification_report(test_results['y_test'], test_results['y_pred'], 
                                 target_names=test_results['class_names'], output_dict=True)
    
    # Извлекаем точность для каждого класса
    precisions = []
    for class_name in test_results['class_names']:
        if class_name in report:
            precisions.append(report[class_name]['precision'])
        else:
            precisions.append(0)
    
    bars = ax3.bar(test_results['class_names'], precisions, alpha=0.7, color='skyblue')
    ax3.set_title('Точность по классам', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Точность')
    ax3.set_ylim(0, 1)
    
    # Добавляем значения на столбцы
    for bar, precision in zip(bars, precisions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{precision:.3f}', ha='center', va='bottom')
    
    # 4. Информация о модели
    metadata = model_info['metadata']
    model_stats = {
        'Тип модели': metadata.get('model_type', 'Unknown'),
        'Количество признаков': len(metadata['feature_names']),
        'Количество классов': len(test_results['class_names']),
        'Максимальная глубина': metadata.get('max_depth', 'N/A'),
        'Максимальные признаки': metadata.get('max_features', 'N/A'),
        'Random State': metadata.get('random_state', 'N/A'),
        'Общая точность': f"{report['accuracy']:.3f}"
    }
    
    # Создаем текстовую таблицу
    ax4.axis('off')
    table_data = [[k, str(v)] for k, v in model_stats.items()]
    table = ax4.table(cellText=table_data, colLabels=['Параметр', 'Значение'], 
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax4.set_title('Характеристики модели', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'results/model_{model_info["metadata"].get("saved_date", "test")}_test.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("🚀 Тестирование конкретной модели...")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)
    
    # Показываем доступные модели
    available_models = list_available_models()
    
    if not available_models:
        print("❌ Модели не найдены")
        return
    
    # Выбираем модель (можно изменить на конкретную)
    model_date = available_models[-1]  # Берем последнюю модель
    print(f"\n🎯 Выбрана модель: {model_date}")
    
    # Загружаем модель
    model_info = load_model(model_date)
    
    if not model_info['success']:
        print(f"❌ Не удалось загрузить модель: {model_info['error']}")
        return
    
    print(f"✅ Модель загружена успешно")
    print(f"   Тип: {model_info['metadata']['model_type']}")
    print(f"   Признаков: {len(model_info['metadata']['feature_names'])}")
    print(f"   Классов: {len(model_info['metadata']['class_names'])}")
    
    # Тестируем модель
    test_results = test_model(model_info)
    
    # Создаем визуализацию
    create_model_visualization(model_info, test_results)
    
    print(f"\n✅ Тестирование завершено! Результаты сохранены в папке 'results/'")

if __name__ == "__main__":
    main() 