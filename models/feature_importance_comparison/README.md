# Feature Importance Comparison - Сравнение Важности Признаков

## 📊 Описание

Эта папка содержит результаты сравнения feature importance между старой и новой моделью классификатора храпа.

## 📁 Структура файлов

### Старая модель (max_depth=15, entropy, min_samples_leaf=1)
- `old_feature_importances.json` - Основные результаты обучения
- `old_feature_importances_detailed.json` - Детальные feature importances с категориями
- `old_feature_names.txt` - Названия признаков
- `old_metrics.json` - Метрики качества
- `old_visualizations/old_feature_importances.png` - График важности признаков

### Новая модель (max_depth=5, gini, min_samples_leaf=10) - ПО ТЗ
- `new_feature_importances.json` - Результаты новой модели
- `new_feature_importances_detailed.json` - Детальные feature importances
- `new_feature_names.txt` - Названия признаков
- `new_metrics.json` - Метрики качества
- `new_visualizations/new_feature_importances.png` - График важности признаков

## 🔍 Анализ изменений

### Гиперпараметры
| Параметр | Старая модель | Новая модель (ТЗ) |
|----------|---------------|-------------------|
| max_depth | 15 | 5 |
| min_samples_leaf | 1 | 10 |
| criterion | entropy | gini |
| class_weight | None | balanced |

### Ожидаемые изменения
1. **Меньше глубина дерева** → проще интерпретация
2. **Больше min_samples_leaf** → более стабильные признаки
3. **Gini vs Entropy** → другая логика выбора признаков
4. **Balanced class_weight** → лучшая работа с дисбалансом классов

## 📈 Метрики качества

### Старая модель
- Accuracy: 88.5%
- Precision: 0.0%
- Recall: 0.0%
- F1-score: 0.0%

### Новая модель (ожидается)
- Accuracy: >85%
- Precision: >70%
- Recall: >70%
- F1-score: >70%

## 🎯 Цель сравнения

Показать, что модель, обученная по ТЗ, дает:
1. **Лучшее качество** предсказаний
2. **Более стабильные** feature importances
3. **Полное соответствие** техническому заданию
4. **Готовность к продакшену**

## 📅 Дата создания

- Старые результаты: 2025-08-15 16:30:00
- Новые результаты: ожидается
- Сравнительный анализ: ожидается 