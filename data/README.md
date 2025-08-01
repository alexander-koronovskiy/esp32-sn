# Данные для классификации стадий сна

Эта папка содержит данные для обучения и тестирования модели классификации стадий сна.

## Структура папки

```
data/
├── raw/                    # Сырые данные ЭЭГ
│   ├── subject_001.edf    # Данные ЭЭГ в формате EDF
│   ├── subject_002.edf
│   └── ...
├── processed/              # Обработанные данные
│   ├── segments/          # Сегментированные данные
│   └── features/          # Извлеченные признаки
├── annotations/           # Аннотации стадий сна
│   ├── subject_001.csv   # Аннотации в формате CSV
│   ├── subject_002.csv
│   └── ...
└── README.md             # Этот файл
```

## Форматы данных

### Данные ЭЭГ
- **Форматы**: EDF, BDF, FIF, SET
- **Частота дискретизации**: 256 Гц (по умолчанию)
- **Каналы**: Минимум 1 канал ЭЭГ

### Аннотации стадий сна
- **Формат**: CSV или Excel
- **Колонки**:
  - `time`: Время начала сегмента (в секундах)
  - `duration`: Продолжительность сегмента (в секундах)
  - `stage`: Стадия сна (Wake, N1, N2, N3, REM)

### Пример аннотаций (CSV)
```csv
time,duration,stage
0,30,Wake
30,30,N1
60,30,N2
90,30,N2
120,30,N3
150,30,REM
```

## Подготовка данных

1. Поместите файлы ЭЭГ в папку `raw/`
2. Создайте аннотации и поместите их в папку `annotations/`
3. Убедитесь, что имена файлов аннотаций соответствуют именам файлов ЭЭГ

## Использование

Для обучения модели используйте:
```bash
python src/train.py --data data/raw/subject_001.edf --annotations data/annotations/subject_001.csv
```

Для предсказания:
```bash
python src/predict.py --model models/best_model.pkl --data data/raw/subject_002.edf
```

Для оценки:
```bash
python src/evaluate.py --model models/best_model.pkl --data data/raw/subject_002.edf --annotations data/annotations/subject_002.csv
``` 