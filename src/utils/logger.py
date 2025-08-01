"""
Модуль для настройки логирования.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "sleep_stage_classifier",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Настраивает и возвращает логгер.
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу для логирования (опционально)
        log_format: Формат сообщений логов
        
    Returns:
        Настроенный логгер
    """
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Создаем форматтер
    formatter = logging.Formatter(log_format)
    
    # Создаем обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Создаем обработчик для файла, если указан
    if log_file:
        # Создаем директорию для логов, если её нет
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "sleep_stage_classifier") -> logging.Logger:
    """
    Получает существующий логгер или создает новый.
    
    Args:
        name: Имя логгера
        
    Returns:
        Логгер
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Миксин для добавления логирования в классы."""
    
    @property
    def logger(self) -> logging.Logger:
        """Возвращает логгер для класса."""
        return get_logger(self.__class__.__name__) 