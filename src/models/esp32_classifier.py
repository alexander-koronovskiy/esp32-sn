"""
Упрощенный классификатор для ESP32.
Оптимизирован для работы с ограниченными ресурсами.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os


class ESP32Classifier:
    """Упрощенный классификатор для ESP32."""
    
    def __init__(self, model_type: str = 'decision_tree', max_depth: int = 5, 
                 max_features: int = 8, quantization_scale: int = 100):
        """
        Инициализация классификатора.
        
        Args:
            model_type: Тип модели ('decision_tree', 'logistic', 'knn')
            max_depth: Максимальная глубина дерева
            max_features: Максимальное количество признаков
            quantization_scale: Масштаб квантизации
        """
        self.model_type = model_type
        self.max_depth = max_depth
        
        # Убеждаемся, что max_features корректный
        if isinstance(max_features, (int, float)):
            self.max_features = int(max_features)
        elif isinstance(max_features, str):
            self.max_features = max_features
        else:
            self.max_features = 8
        
        self.quantization_scale = quantization_scale
        self.model = None
        self.feature_names = []
        self.is_quantized = False
        
    def create_model(self) -> Any:
        """
        Создает модель в зависимости от типа.
        
        Returns:
            Модель классификатора
        """
        if self.model_type == 'decision_tree':
            # Убеждаемся, что max_features корректный
            max_features = self.max_features
            if isinstance(max_features, str):
                max_features = 'sqrt'
            elif isinstance(max_features, (int, float)):
                max_features = int(max_features)
            else:
                max_features = 'sqrt'
            
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=max_features,
                random_state=42,
                criterion='gini'
            )
        
        elif self.model_type == 'logistic':
            return LogisticRegression(
                random_state=42,
                max_iter=100,
                solver='liblinear'
            )
        
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=3,
                weights='uniform'
            )
        
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Обучает модель.
        
        Args:
            X: Признаки для обучения
            y: Метки классов
            feature_names: Названия признаков
            
        Returns:
            Словарь с метриками обучения
        """
        # Сохраняем названия признаков
        if feature_names is not None:
            # Проверяем, что все названия признаков - строки
            if not all(isinstance(name, str) for name in feature_names):
                raise ValueError("Все названия признаков должны быть строками")
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Создаем и обучаем модель
        self.model = self.create_model()
        self.model.fit(X, y)
        
        # Оцениваем на обучающих данных
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        
        # Кросс-валидация
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='accuracy')
        
        metrics = {
            'train_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_size_bytes': self.estimate_model_size(),
            'memory_usage_bytes': self.estimate_memory_usage(X.shape[1])
        }
        
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
    
    def quantize_model(self) -> Dict[str, Any]:
        """
        Квантизует модель для экономии памяти.
        
        Returns:
            Словарь с квантизованными параметрами модели
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        quantized_model = {}
        
        if self.model_type == 'decision_tree':
            # Квантизация дерева решений
            tree = self.model.tree_
            
            # Квантизуем пороги
            thresholds = tree.threshold
            # Убеждаемся, что thresholds - числа
            thresholds = np.array([float(t) if not np.isnan(t) and not np.isinf(t) else 0.0 for t in thresholds])
            quantized_thresholds = np.round(thresholds * self.quantization_scale).astype(np.int8)
            
            # Квантизуем значения в листьях
            values = tree.value.reshape(-1, tree.n_classes[0])
            # Убеждаемся, что values - числа
            values = np.array([[float(v) if not np.isnan(v) and not np.isinf(v) else 0.0 for v in row] for row in values])
            quantized_values = np.round(values * self.quantization_scale).astype(np.int8)
            
            quantized_model = {
                'feature': tree.feature,
                'threshold': quantized_thresholds,
                'value': quantized_values,
                'children_left': tree.children_left,
                'children_right': tree.children_right,
                'n_classes': tree.n_classes[0],
                'quantization_scale': self.quantization_scale
            }
        
        elif self.model_type == 'logistic':
            # Квантизация логистической регрессии
            coef = self.model.coef_
            intercept = self.model.intercept_
            
            # Убеждаемся, что coef и intercept - числа
            coef = np.array([[float(c) if not np.isnan(c) and not np.isinf(c) else 0.0 for c in row] for row in coef])
            intercept = np.array([float(i) if not np.isnan(i) and not np.isinf(i) else 0.0 for i in intercept])
            
            quantized_coef = np.round(coef * self.quantization_scale).astype(np.int8)
            quantized_intercept = np.round(intercept * self.quantization_scale).astype(np.int8)
            
            quantized_model = {
                'coef': quantized_coef,
                'intercept': quantized_intercept,
                'classes': self.model.classes_,
                'quantization_scale': self.quantization_scale
            }
        
        self.is_quantized = True
        return quantized_model
    
    def predict_quantized(self, X: np.ndarray, quantized_model: Dict[str, Any]) -> np.ndarray:
        """
        Делает предсказания с квантизованной моделью.
        
        Args:
            X: Признаки для предсказания
            quantized_model: Квантизованная модель
            
        Returns:
            Предсказанные метки классов
        """
        if self.model_type == 'decision_tree':
            return self._predict_quantized_tree(X, quantized_model)
        elif self.model_type == 'logistic':
            return self._predict_quantized_logistic(X, quantized_model)
        else:
            return self.predict(X)
    
    def _predict_quantized_tree(self, X: np.ndarray, quantized_model: Dict[str, Any]) -> np.ndarray:
        """
        Предсказание с квантизованным деревом решений.
        
        Args:
            X: Признаки для предсказания
            quantized_model: Квантизованная модель
            
        Returns:
            Предсказанные метки классов
        """
        predictions = []
        scale = quantized_model['quantization_scale']
        
        for sample in X:
            node = 0
            while quantized_model['children_left'][node] != -1:
                feature = quantized_model['feature'][node]
                threshold = quantized_model['threshold'][node] / scale
                
                # Проверяем типы перед сравнением
                if isinstance(feature, int) and feature < len(sample):
                    sample_value = sample[feature]
                    if isinstance(sample_value, (int, float)) and isinstance(threshold, (int, float)):
                        if sample_value <= threshold:
                            node = quantized_model['children_left'][node]
                        else:
                            node = quantized_model['children_right'][node]
                    else:
                        # Если типы не совместимы, используем значение по умолчанию
                        node = quantized_model['children_left'][node]
                else:
                    # Если feature не является валидным индексом, используем значение по умолчанию
                    node = quantized_model['children_left'][node]
            
            # Получаем предсказание из листа
            leaf_values = quantized_model['value'][node] / scale
            prediction = np.argmax(leaf_values)
            
            # Убеждаемся, что prediction - число
            if isinstance(prediction, (int, float)):
                predictions.append(int(prediction))
            else:
                predictions.append(0)
        
        return np.array(predictions)
    
    def _predict_quantized_logistic(self, X: np.ndarray, quantized_model: Dict[str, Any]) -> np.ndarray:
        """
        Предсказание с квантизованной логистической регрессией.
        
        Args:
            X: Признаки для предсказания
            quantized_model: Квантизованная модель
            
        Returns:
            Предсказанные метки классов
        """
        scale = quantized_model['quantization_scale']
        coef = quantized_model['coef'] / scale
        intercept = quantized_model['intercept'] / scale
        
        # Вычисляем логиты
        logits = X @ coef.T + intercept
        
        # Применяем сигмоиду
        probabilities = 1 / (1 + np.exp(-logits))
        
        # Выбираем класс с максимальной вероятностью
        predictions = np.argmax(probabilities, axis=1)
        
        # Убеждаемся, что predictions содержит числа
        predictions = np.array([int(p) if isinstance(p, (int, float)) else 0 for p in predictions])
        
        return predictions
    
    def estimate_model_size(self) -> int:
        """
        Оценивает размер модели в байтах.
        
        Returns:
            Размер модели в байтах
        """
        if self.model is None:
            return 0
        
        if self.model_type == 'decision_tree':
            tree = self.model.tree_
            # Примерная оценка размера дерева
            n_nodes = tree.node_count
            size = n_nodes * (4 + 4 + 4 + 4 + 4)  # feature, threshold, children_left, children_right, value
            return size
        
        elif self.model_type == 'logistic':
            coef_size = self.model.coef_.nbytes
            intercept_size = self.model.intercept_.nbytes
            return coef_size + intercept_size
        
        elif self.model_type == 'knn':
            # KNN хранит все обучающие данные
            return self.model._fit_X.nbytes + self.model._fit_y.nbytes
        
        return 0
    
    def estimate_memory_usage(self, n_features: int) -> int:
        """
        Оценивает использование памяти для инференса.
        
        Args:
            n_features: Количество признаков
            
        Returns:
            Использование памяти в байтах
        """
        # Базовое использование памяти
        base_memory = n_features * 4  # float32 для признаков
        
        if self.model_type == 'decision_tree':
            # Дополнительная память для обхода дерева
            return base_memory + 1024  # ~1KB для дерева
        
        elif self.model_type == 'logistic':
            # Память для весов и смещений
            coef_memory = n_features * 4 * 5  # 5 классов
            return base_memory + coef_memory
        
        elif self.model_type == 'knn':
            # KNN требует больше памяти для хранения данных
            return base_memory + 50000  # ~50KB для KNN
        
        return base_memory
    
    def save_model(self, file_path: str):
        """
        Сохраняет модель в файл.
        
        Args:
            file_path: Путь для сохранения модели
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Сохраняем модель и метаданные
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'quantization_scale': self.quantization_scale,
            'is_quantized': self.is_quantized
        }
        
        joblib.dump(model_data, file_path)
    
    def load_model(self, file_path: str):
        """
        Загружает модель из файла.
        
        Args:
            file_path: Путь к файлу модели
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл модели не найден: {file_path}")
        
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.quantization_scale = model_data['quantization_scale']
        self.is_quantized = model_data['is_quantized']
    
    def generate_c_code(self, output_file: str):
        """
        Генерирует C-код для ESP32.
        
        Args:
            output_file: Путь к выходному файлу
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        quantized_model = self.quantize_model()
        
        with open(output_file, 'w') as f:
            f.write("// Автоматически сгенерированный код для ESP32\n")
            f.write("// Классификатор стадий сна\n\n")
            
            if self.model_type == 'decision_tree':
                self._generate_tree_c_code(f, quantized_model)
            elif self.model_type == 'logistic':
                self._generate_logistic_c_code(f, quantized_model)
    
    def _generate_tree_c_code(self, file, quantized_model):
        """Генерирует C-код для дерева решений."""
        file.write("#include <stdint.h>\n\n")
        
        # Константы
        file.write(f"#define NUM_FEATURES {len(self.feature_names)}\n")
        file.write(f"#define NUM_CLASSES {int(quantized_model['n_classes'])}\n")
        file.write(f"#define NUM_NODES {len(quantized_model['feature'])}\n")
        file.write(f"#define QUANTIZATION_SCALE {int(quantized_model['quantization_scale'])}\n\n")
        
        # Массивы дерева
        file.write("// Массивы дерева решений\n")
        file.write("const int8_t tree_feature[] = {\n")
        # Убеждаемся, что все значения - числа
        feature_values = [str(int(f)) if isinstance(f, (int, float)) else "0" for f in quantized_model['feature']]
        file.write("    " + ", ".join(feature_values) + "\n};\n\n")
        
        file.write("const int8_t tree_threshold[] = {\n")
        threshold_values = [str(int(t)) if isinstance(t, (int, float)) else "0" for t in quantized_model['threshold']]
        file.write("    " + ", ".join(threshold_values) + "\n};\n\n")
        
        file.write("const int8_t tree_children_left[] = {\n")
        left_values = [str(int(l)) if isinstance(l, (int, float)) else "-1" for l in quantized_model['children_left']]
        file.write("    " + ", ".join(left_values) + "\n};\n\n")
        
        file.write("const int8_t tree_children_right[] = {\n")
        right_values = [str(int(r)) if isinstance(r, (int, float)) else "-1" for r in quantized_model['children_right']]
        file.write("    " + ", ".join(right_values) + "\n};\n\n")
        
        # Функция предсказания
        file.write("int predict_sleep_stage(float* features) {\n")
        file.write("    int node = 0;\n")
        file.write("    \n")
        file.write("    while (tree_children_left[node] != -1) {\n")
        file.write("        int feature_idx = tree_feature[node];\n")
        file.write("        float threshold = (float)tree_threshold[node] / QUANTIZATION_SCALE;\n")
        file.write("        \n")
        file.write("        if (features[feature_idx] <= threshold) {\n")
        file.write("            node = tree_children_left[node];\n")
        file.write("        } else {\n")
        file.write("            node = tree_children_right[node];\n")
        file.write("        }\n")
        file.write("    }\n")
        file.write("    \n")
        file.write("    // Возвращаем класс (упрощенно)\n")
        file.write("    return node % NUM_CLASSES;\n")
        file.write("}\n")
    
    def _generate_logistic_c_code(self, file, quantized_model):
        """Генерирует C-код для логистической регрессии."""
        file.write("#include <stdint.h>\n")
        file.write("#include <math.h>\n\n")
        
        # Константы
        file.write(f"#define NUM_FEATURES {len(self.feature_names)}\n")
        file.write(f"#define NUM_CLASSES {len(quantized_model['classes'])}\n")
        file.write(f"#define QUANTIZATION_SCALE {int(quantized_model['quantization_scale'])}\n\n")
        
        # Коэффициенты
        coef = quantized_model['coef']
        file.write("// Коэффициенты логистической регрессии\n")
        file.write("const int8_t logistic_coef[NUM_CLASSES][NUM_FEATURES] = {\n")
        for i in range(len(coef)):
            # Убеждаемся, что все значения - числа
            coef_row = [str(int(c)) if isinstance(c, (int, float)) else "0" for c in coef[i]]
            file.write("    {" + ", ".join(coef_row) + "},\n")
        file.write("};\n\n")
        
        # Смещения
        intercept = quantized_model['intercept']
        file.write("const int8_t logistic_intercept[NUM_CLASSES] = {\n")
        # Убеждаемся, что все значения - числа
        intercept_values = [str(int(i)) if isinstance(i, (int, float)) else "0" for i in intercept]
        file.write("    " + ", ".join(intercept_values) + "\n};\n\n")
        
        # Функция предсказания
        file.write("int predict_sleep_stage(float* features) {\n")
        file.write("    float max_prob = -INFINITY;\n")
        file.write("    int best_class = 0;\n")
        file.write("    \n")
        file.write("    for (int i = 0; i < NUM_CLASSES; i++) {\n")
        file.write("        float logit = 0;\n")
        file.write("        for (int j = 0; j < NUM_FEATURES; j++) {\n")
        file.write("            logit += features[j] * (float)logistic_coef[i][j] / QUANTIZATION_SCALE;\n")
        file.write("        }\n")
        file.write("        logit += (float)logistic_intercept[i] / QUANTIZATION_SCALE;\n")
        file.write("        \n")
        file.write("        if (logit > max_prob) {\n")
        file.write("            max_prob = logit;\n")
        file.write("            best_class = i;\n")
        file.write("        }\n")
        file.write("    }\n")
        file.write("    \n")
        file.write("    return best_class;\n")
        file.write("}\n")


def create_esp32_classifier(config: Dict) -> ESP32Classifier:
    """
    Создает классификатор для ESP32 на основе конфигурации.
    
    Args:
        config: Конфигурация
        
    Returns:
        Классификатор для ESP32
    """
    # Убеждаемся, что max_features корректный
    max_features = config.get('max_features', 8)
    if isinstance(max_features, str):
        max_features = 'sqrt'
    elif isinstance(max_features, (int, float)):
        max_features = int(max_features)
    else:
        max_features = 8
    
    return ESP32Classifier(
        model_type=config.get('model_type', 'decision_tree'),
        max_depth=config.get('max_depth', 5),
        max_features=max_features,
        quantization_scale=config.get('quantization_scale', 100)
    ) 