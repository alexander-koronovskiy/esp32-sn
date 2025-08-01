"""
Модуль с классификаторами для стадий сна.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class SleepStageClassifier(LoggerMixin):
    """Базовый класс для классификаторов стадий сна."""
    
    def __init__(self, config: Config):
        """
        Инициализация классификатора.
        
        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self.model = None
        self.model_type = config.get("model.type", "random_forest")
        self.model_params = config.get_model_params()
        
    def create_model(self) -> Any:
        """
        Создает модель в зависимости от типа.
        
        Returns:
            Модель классификатора
        """
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", 10),
                min_samples_split=self.model_params.get("min_samples_split", 2),
                min_samples_leaf=self.model_params.get("min_samples_leaf", 1),
                random_state=self.config.get("model.random_state", 42),
                n_jobs=-1
            )
        
        elif self.model_type == "svm":
            return SVC(
                C=self.model_params.get("C", 1.0),
                kernel=self.model_params.get("kernel", "rbf"),
                gamma=self.model_params.get("gamma", "scale"),
                random_state=self.config.get("model.random_state", 42),
                probability=True
            )
        
        elif self.model_type == "neural_network":
            return MLPClassifier(
                hidden_layer_sizes=tuple(self.model_params.get("hidden_layer_sizes", [100, 50])),
                activation=self.model_params.get("activation", "relu"),
                solver=self.model_params.get("solver", "adam"),
                alpha=self.model_params.get("alpha", 0.0001),
                max_iter=self.model_params.get("max_iter", 1000),
                random_state=self.config.get("model.random_state", 42)
            )
        
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.config.get("model.random_state", 42),
                max_iter=1000
            )
        
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(
                random_state=self.config.get("model.random_state", 42)
            )
        
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                learning_rate=self.model_params.get("learning_rate", 0.1),
                max_depth=self.model_params.get("max_depth", 3),
                random_state=self.config.get("model.random_state", 42)
            )
        
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Обучает модель.
        
        Args:
            X: Признаки для обучения
            y: Метки классов
            
        Returns:
            Словарь с метриками обучения
        """
        self.logger.info(f"Обучение модели типа: {self.model_type}")
        
        # Создаем модель
        self.model = self.create_model()
        
        # Обучаем модель
        self.model.fit(X, y)
        
        # Оцениваем на обучающих данных
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        metrics = {
            'train_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.logger.info(f"Точность на обучающих данных: {accuracy:.4f}")
        self.logger.info(f"Кросс-валидация: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Предсказанные метки классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности классов.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Матрица вероятностей классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Оценивает модель на тестовых данных.
        
        Args:
            X: Признаки для тестирования
            y: Истинные метки классов
            
        Returns:
            Словарь с метриками оценки
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        self.logger.info("Оценка модели на тестовых данных")
        
        # Предсказания
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Метрики
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Дополнительные метрики
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
        
        self.logger.info(f"Точность на тестовых данных: {accuracy:.4f}")
        
        return metrics
    
    def save_model(self, file_path: str):
        """
        Сохраняет модель в файл.
        
        Args:
            file_path: Путь для сохранения модели
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        self.logger.info(f"Сохранение модели в {file_path}")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path)
        
        self.logger.info("Модель сохранена")
    
    def load_model(self, file_path: str):
        """
        Загружает модель из файла.
        
        Args:
            file_path: Путь к файлу модели
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл модели не найден: {file_path}")
        
        self.logger.info(f"Загрузка модели из {file_path}")
        
        self.model = joblib.load(file_path)
        
        self.logger.info("Модель загружена")


class EnsembleClassifier(SleepStageClassifier):
    """Ансамблевый классификатор."""
    
    def __init__(self, config: Config):
        """
        Инициализация ансамблевого классификатора.
        
        Args:
            config: Конфигурация проекта
        """
        super().__init__(config)
        self.models = {}
        self.model_weights = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """
        Добавляет модель в ансамбль.
        
        Args:
            name: Имя модели
            model: Модель классификатора
            weight: Вес модели в ансамбле
        """
        self.models[name] = model
        self.model_weights[name] = weight
        self.logger.info(f"Добавлена модель {name} с весом {weight}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Обучает все модели в ансамбле.
        
        Args:
            X: Признаки для обучения
            y: Метки классов
            
        Returns:
            Словарь с метриками обучения
        """
        self.logger.info("Обучение ансамблевого классификатора")
        
        metrics = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Обучение модели {name}")
            model.fit(X, y)
            
            # Оценка на обучающих данных
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            metrics[f"{name}_train_accuracy"] = accuracy
            
            self.logger.info(f"Точность модели {name}: {accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания с помощью ансамбля.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Предсказанные метки классов
        """
        if not self.models:
            raise ValueError("Ансамбль пуст. Добавьте модели с помощью add_model().")
        
        # Получаем предсказания всех моделей
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Взвешенное голосование
        ensemble_pred = np.zeros(X.shape[0], dtype=int)
        total_weight = sum(self.model_weights.values())
        
        for name, pred in predictions.items():
            weight = self.model_weights[name] / total_weight
            ensemble_pred += weight * pred
        
        # Округляем до ближайшего класса
        return np.round(ensemble_pred).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности классов для ансамбля.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Матрица вероятностей классов
        """
        if not self.models:
            raise ValueError("Ансамбль пуст. Добавьте модели с помощью add_model().")
        
        # Получаем вероятности всех моделей
        probabilities = {}
        for name, model in self.models.items():
            probabilities[name] = model.predict_proba(X)
        
        # Взвешенное усреднение вероятностей
        ensemble_proba = np.zeros_like(list(probabilities.values())[0])
        total_weight = sum(self.model_weights.values())
        
        for name, proba in probabilities.items():
            weight = self.model_weights[name] / total_weight
            ensemble_proba += weight * proba
        
        return ensemble_proba


class ModelFactory:
    """Фабрика для создания моделей."""
    
    @staticmethod
    def create_classifier(config: Config) -> SleepStageClassifier:
        """
        Создает классификатор на основе конфигурации.
        
        Args:
            config: Конфигурация проекта
            
        Returns:
            Классификатор
        """
        model_type = config.get("model.type", "random_forest")
        
        if model_type == "ensemble":
            return EnsembleClassifier(config)
        else:
            return SleepStageClassifier(config)
    
    @staticmethod
    def create_ensemble_classifier(config: Config) -> EnsembleClassifier:
        """
        Создает ансамблевый классификатор.
        
        Args:
            config: Конфигурация проекта
            
        Returns:
            Ансамблевый классификатор
        """
        return EnsembleClassifier(config) 