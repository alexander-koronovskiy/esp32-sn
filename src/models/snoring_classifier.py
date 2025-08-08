"""
Специализированный классификатор для детекции храпа.
Поддерживает детекцию в реальном времени и предсказание будущего храпа.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import json
from datetime import datetime

# Импорт LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class SnoringClassifier:
    """Классификатор для детекции храпа."""
    
    def __init__(self, model_type: str = 'lightgbm', max_depth: int = 6, 
                 n_estimators: int = 150, learning_rate: float = 0.05, 
                 random_state: int = 42):
        """
        Инициализация классификатора храпа.
        
        Args:
            model_type: Тип модели ('lightgbm', 'random_forest', 'svm', 'logistic', 'decision_tree')
            max_depth: Максимальная глубина дерева
            n_estimators: Количество деревьев (для LightGBM и Random Forest)
            learning_rate: Скорость обучения (для LightGBM)
            random_state: Случайное состояние
        """
        self.model_type = model_type
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_names = []
        self.class_names = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']  # 3-классная классификация
        
    def create_model(self) -> Any:
        """
        Создает модель в зависимости от типа.
        
        Returns:
            Модель классификатора
        """
        if self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM не установлен. Установите: pip install lightgbm")
            return LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                reg_lambda=1.0,  # L2 регуляризация
                scale_pos_weight=1.0,  # Компенсация дисбаланса
                random_state=self.random_state,
                verbose=-1
            )
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif self.model_type == 'svm':
            return SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                random_state=self.random_state,
                probability=True
            )
        
        elif self.model_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                multi_class='ovr'
            )
        
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Обучает модель детекции храпа.
        
        Args:
            X: Признаки для обучения
            y: Метки классов
            feature_names: Названия признаков
            
        Returns:
            Словарь с метриками обучения
        """
        # Сохраняем названия признаков
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Создаем модель
        self.model = self.create_model()
        
        # Обучение
        self.model.fit(X, y)
        
        # Оценка производительности
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        return {
            'accuracy': float(accuracy),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'model_type': self.model_type
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Предсказанные метки классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности классов.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Матрица вероятностей классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Для моделей без predict_proba возвращаем dummy вероятности
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), len(self.class_names)))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
    
    def predict_snoring_future(self, current_features: np.ndarray, 
                              history_features: Optional[np.ndarray] = None,
                              time_horizon: int = 30) -> Dict[str, float]:
        """
        Предсказание храпа в будущем (через time_horizon секунд).
        
        Args:
            current_features: Текущие признаки
            history_features: Исторические признаки (опционально)
            time_horizon: Горизонт предсказания в секундах
            
        Returns:
            Словарь с вероятностями будущего храпа
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        # Объединяем текущие и исторические признаки
        if history_features is not None:
            combined_features = np.concatenate([current_features, history_features.flatten()])
        else:
            combined_features = current_features
        
        # Делаем предсказание
        proba = self.predict_proba(combined_features.reshape(1, -1))[0]
        
        # Анализируем вероятность будущего храпа
        snoring_proba = proba[1]  # Вероятность храпа
        no_snoring_proba = proba[0]  # Вероятность отсутствия храпа
        
        # Оценка риска храпа
        risk_level = "low"
        if snoring_proba > 0.7:
            risk_level = "high"
        elif snoring_proba > 0.4:
            risk_level = "medium"
        
        return {
            'snoring_probability': float(snoring_proba),
            'no_snoring_probability': float(no_snoring_proba),
            'risk_level': risk_level,
            'time_horizon_seconds': time_horizon,
            'prediction_confidence': float(max(proba)),
            'class_probabilities': dict(zip(self.class_names, proba.tolist()))
        }
    
    def detect_snoring_window(self, audio_segment: np.ndarray, 
                            feature_extractor: Any) -> Dict[str, Any]:
        """
        Детекция храпа в текущем окне.
        
        Args:
            audio_segment: Аудио сегмент
            feature_extractor: Экстрактор признаков
            
        Returns:
            Словарь с результатами детекции
        """
        # Извлекаем признаки
        features = feature_extractor.extract_snoring_features(audio_segment)
        feature_vector = np.array(list(features.values()))
        
        # Делаем предсказание
        prediction = self.predict(feature_vector.reshape(1, -1))[0]
        proba = self.predict_proba(feature_vector.reshape(1, -1))[0]
        
        # Анализируем результат
        is_snoring = prediction == 1  # Бинарная классификация
        snoring_intensity = "heavy" if proba[1] > 0.7 else "light" if proba[1] > 0.4 else "none"
        
        return {
            'is_snoring': bool(is_snoring),
            'snoring_intensity': snoring_intensity,
            'predicted_class': self.class_names[prediction],
            'confidence': float(max(proba)),
            'class_probabilities': dict(zip(self.class_names, proba.tolist())),
            'features': features
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Оценивает производительность модели.
        
        Args:
            X: Признаки для оценки
            y: Истинные метки
            
        Returns:
            Словарь с метриками оценки
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        # Предсказания
        y_pred = self.model.predict(X)
        y_proba = self.predict_proba(X)
        
        # Метрики
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=self.class_names, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Специфичные для храпа метрики
        snoring_detection_accuracy = self._calculate_snoring_detection_accuracy(y, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'snoring_detection_accuracy': float(snoring_detection_accuracy),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'precision': float(report['weighted avg']['precision']),
            'recall': float(report['weighted avg']['recall']),
            'f1_score': float(report['weighted avg']['f1-score'])
        }
    
    def _calculate_snoring_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисляет точность детекции храпа.
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Точность детекции храпа
        """
        # Подсчитываем правильные предсказания храпа
        snoring_correct = np.sum((y_true == 1) & (y_pred == 1))
        snoring_total = np.sum(y_true == 1)
        
        if snoring_total == 0:
            return 0.0
        
        return snoring_correct / snoring_total
    
    def save_model(self, file_path: str):
        """
        Сохраняет модель.
        
        Args:
            file_path: Путь для сохранения
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Сохраняем модель
        joblib.dump(self.model, file_path)
        
        # Сохраняем метаданные
        metadata = {
            'model_type': self.model_type,
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = file_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, file_path: str):
        """
        Загружает модель.
        
        Args:
            file_path: Путь к модели
        """
        # Загружаем модель
        self.model = joblib.load(file_path)
        
        # Загружаем метаданные
        metadata_path = file_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_type = metadata.get('model_type', self.model_type)
                self.max_depth = metadata.get('max_depth', self.max_depth)
                self.n_estimators = metadata.get('n_estimators', self.n_estimators)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.feature_names = metadata.get('feature_names', self.feature_names)
                self.class_names = metadata.get('class_names', self.class_names)


def create_snoring_classifier(config: Dict) -> SnoringClassifier:
    """
    Создает экземпляр SnoringClassifier с заданной конфигурацией.
    
    Args:
        config: Конфигурация модели
        
    Returns:
        Экземпляр SnoringClassifier
    """
    return SnoringClassifier(
        model_type=config.get('model_type', 'lightgbm'),
        max_depth=config.get('max_depth', 6),
        n_estimators=config.get('n_estimators', 150),
        learning_rate=config.get('learning_rate', 0.05),
        random_state=config.get('random_state', 42)
    ) 