# Sleep Stage Classifier

Проект для автоматической классификации стадий сна на основе данных ЭЭГ.

## Описание

Этот проект использует машинное обучение для автоматической классификации стадий сна:
- Бодрствование (Wake)
- Легкий сон (N1, N2)
- Глубокий сон (N3)
- Быстрый сон (REM)

## Структура проекта

```
sleep-stage-classifier/
├── data/                   # Данные для обучения и тестирования
├── models/                 # Сохраненные модели
├── src/                    # Исходный код
│   ├── data/              # Обработка данных
│   ├── features/          # Извлечение признаков
│   ├── models/            # Модели машинного обучения
│   └── utils/             # Утилиты
├── notebooks/             # Jupyter notebooks
├── tests/                 # Тесты
├── requirements.txt        # Зависимости
└── config.yaml           # Конфигурация
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd sleep-stage-classifier
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Обучение модели
```bash
python src/train.py --config config.yaml
```

### Предсказание
```bash
python src/predict.py --model models/best_model.pkl --data data/test_data.csv
```

### Оценка модели
```bash
python src/evaluate.py --model models/best_model.pkl --data data/test_data.csv
```

## Технологии

- Python 3.8+
- Scikit-learn
- NumPy
- Pandas
- Matplotlib/Seaborn
- PyTorch (опционально)
- MNE (для работы с ЭЭГ)

## Лицензия

MIT License 