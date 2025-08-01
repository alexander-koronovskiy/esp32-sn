"""
Модуль для работы с конфигурацией проекта.
"""

import yaml
import os
from typing import Dict, Any


class Config:
    """Класс для управления конфигурацией проекта."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Инициализация конфигурации.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Конфигурационный файл не найден: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получает значение из конфигурации по ключу.
        
        Args:
            key: Ключ в формате 'section.subsection.parameter'
            default: Значение по умолчанию
            
        Returns:
            Значение параметра
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_path(self, data_type: str) -> str:
        """
        Получает путь к данным определенного типа.
        
        Args:
            data_type: Тип данных (train, test, validation, raw)
            
        Returns:
            Путь к данным
        """
        return self.get(f"data.{data_type}_path")
    
    def get_model_params(self, model_type: str = None) -> Dict[str, Any]:
        """
        Получает параметры модели.
        
        Args:
            model_type: Тип модели (если None, берется из конфигурации)
            
        Returns:
            Параметры модели
        """
        if model_type is None:
            model_type = self.get("model.type")
        
        return self.get(f"model.{model_type}", {})
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """
        Получает параметры предобработки данных.
        
        Returns:
            Параметры предобработки
        """
        return self.get("preprocessing", {})
    
    def get_features_params(self) -> Dict[str, Any]:
        """
        Получает параметры извлечения признаков.
        
        Returns:
            Параметры извлечения признаков
        """
        return self.get("features", {})
    
    def get_training_params(self) -> Dict[str, Any]:
        """
        Получает параметры обучения.
        
        Returns:
            Параметры обучения
        """
        return self.get("training", {})
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """
        Получает параметры оценки модели.
        
        Returns:
            Параметры оценки
        """
        return self.get("evaluation", {})
    
    def get_output_paths(self) -> Dict[str, str]:
        """
        Получает пути для сохранения результатов.
        
        Returns:
            Словарь с путями
        """
        return self.get("output", {})
    
    def save_config(self, output_path: str):
        """
        Сохраняет текущую конфигурацию в файл.
        
        Args:
            output_path: Путь для сохранения
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Удобная функция для загрузки конфигурации.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Объект конфигурации
    """
    return Config(config_path) 