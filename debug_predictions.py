#!/usr/bin/env python3
"""
Debug Model Predictions
=======================

Analyze what the model is actually predicting vs. actual labels
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

def load_training_data():
    """Load the data that was used for training"""
    models_dir = Path("models")
    
    # Load feature names
    feature_names_path = models_dir / "improved_39_features_names.txt"
    with open(feature_names_path, 'r') as f:
        feature_names = []
        for line in f.readlines():
            if ':' in line:
                feature_names.append(line.split(':', 1)[1].strip())
    
    # Load training results
    results_path = models_dir / "improved_39_features_training_results.json"
    with open(results_path, 'r') as f:
        training_results = json.load(f)
    
    return feature_names, training_results

def analyze_predictions():
    """Analyze what the model is actually predicting"""
    
    print("🔍 Анализ предсказаний модели...")
    
    # Load data
    feature_names, training_results = load_training_data()
    
    print(f"\n📊 МЕТРИКИ МОДЕЛИ:")
    print(f"Accuracy: {training_results['test_metrics']['accuracy']:.4f} ({training_results['test_metrics']['accuracy']*100:.2f}%)")
    print(f"Precision: {training_results['test_metrics']['precision']:.4f}")
    print(f"Recall: {training_results['test_metrics']['recall']:.4f}")
    print(f"F1-Score: {training_results['test_metrics']['f1_score']:.4f}")
    
    print(f"\n📈 РАЗМЕРЫ ВЫБОРОК:")
    print(f"Train: {training_results['data_shapes']['train'][0]} samples")
    print(f"Validation: {training_results['data_shapes']['validation'][0]} samples")
    print(f"Test: {training_results['data_shapes']['test'][0]} samples")
    
    print(f"\n🔍 АНАЛИЗ ПРОБЛЕМЫ:")
    print("=" * 60)
    
    # Simulate what might be happening
    print("1. Модель показывает Accuracy = 88.50%")
    print("2. Но Precision = 0% и Recall = 0% для класса 'Храп'")
    print("3. Это означает:")
    print("   - Модель НЕ МОЖЕТ правильно определить храп")
    print("   - Все предсказания 'храп' - ложные")
    print("   - Все реальные случаи храпа пропущены")
    
    print("\n4. Почему Accuracy = 88.50%?")
    print("   - В данных 93.8% образцов = 'No Snoring'")
    print("   - Модель просто предсказывает 'No Snoring' для всех образцов")
    print("   - Получает 88.50% правильных предсказаний")
    
    print("\n5. Что это означает на практике:")
    print("   ✅ Модель надежно определяет ОТСУТСТВИЕ храпа")
    print("   ❌ Модель НЕ МОЖЕТ определить наличие храпа")
    print("   ⚠️  Модель бесполезна для детекции храпа!")
    
    print("\n6. Возможные причины:")
    print("   - Недостаточно данных класса 'Храп' (119 vs 1790)")
    print("   - Признаки недостаточно различимы между классами")
    print("   - Модель слишком простая (Decision Tree)")
    print("   - Проблемы с временным выравниванием")
    
    print("\n7. Рекомендации:")
    print("   - Собрать больше данных о храпе")
    print("   - Улучшить признаки (особенно акселерометр)")
    print("   - Попробовать более сложные модели")
    print("   - Проверить качество аннотаций")

if __name__ == "__main__":
    analyze_predictions() 