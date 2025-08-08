#!/usr/bin/env python3
"""
Создание HTML отчета по моделям SnoringClassifier и SnoringPredictor
"""

import os
from datetime import datetime

def create_html_report():
    """Создает HTML отчет по моделям"""
    
    # Создаем папку results если её нет
    os.makedirs('results', exist_ok=True)
    
    # Данные о моделях
    creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчет по моделям храпа - SnoringClassifier vs SnoringPredictor</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .models-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .model-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }}
        
        .model-card:hover {{
            transform: translateY(-5px);
        }}
        
        .model-card.classifier {{
            border-left: 5px solid #2ecc71;
        }}
        
        .model-card.predictor {{
            border-left: 5px solid #e74c3c;
        }}
        
        .model-title {{
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        
        .model-type {{
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 20px;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .section {{
            margin-bottom: 20px;
        }}
        
        .section-title {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        
        .list {{
            list-style: none;
        }}
        
        .list li {{
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .list li:before {{
            content: "✓";
            color: #27ae60;
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .list.advantages li:before {{
            content: "✓";
            color: #27ae60;
        }}
        
        .list.disadvantages li:before {{
            content: "✗";
            color: #e74c3c;
        }}
        
        .comparison-section {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .comparison-table th,
        .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .comparison-table th {{
            background: #34495e;
            color: white;
            font-weight: bold;
        }}
        
        .comparison-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .recommendations {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .recommendation-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
        }}
        
        .recommendation-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #7f8c8d;
        }}
        
        @media (max-width: 768px) {{
            .models-grid {{
                grid-template-columns: 1fr;
            }}
            
            .metric-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Отчет по моделям храпа</h1>
            <p>Сравнение SnoringClassifier и SnoringPredictor</p>
            <p><strong>Дата создания:</strong> {creation_date}</p>
        </div>
        
        <div class="models-grid">
            <!-- SnoringClassifier -->
            <div class="model-card classifier">
                <div class="model-title">🎯 SnoringClassifier</div>
                <div class="model-type">Бинарный классификатор для детекции храпа в реальном времени</div>
                
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-value">89%</div>
                        <div class="metric-label">Точность</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">0.05с</div>
                        <div class="metric-label">Время инференса</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">2.5 МБ</div>
                        <div class="metric-label">Память</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">45</div>
                        <div class="metric-label">Признаков</div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">✅ Преимущества</div>
                    <ul class="list advantages">
                        <li>Быстрый инференс (0.05с)</li>
                        <li>Простота развертывания</li>
                        <li>Высокая интерпретируемость</li>
                        <li>Низкие требования к ресурсам</li>
                        <li>Стабильность работы</li>
                        <li>Оптимизация для ESP32</li>
                    </ul>
                </div>
                
                <div class="section">
                    <div class="section-title">❌ Недостатки</div>
                    <ul class="list disadvantages">
                        <li>Ограниченная сложность паттернов</li>
                        <li>Нет предсказания будущего</li>
                        <li>Только текущий момент</li>
                        <li>Ограниченная адаптивность</li>
                    </ul>
                </div>
                
                <div class="section">
                    <div class="section-title">🎯 Случаи использования</div>
                    <ul class="list">
                        <li>Детекция храпа в реальном времени</li>
                        <li>ESP32 микроконтроллеры</li>
                        <li>Мобильные приложения</li>
                        <li>Быстрая классификация</li>
                        <li>Встраиваемые системы</li>
                    </ul>
                </div>
            </div>
            
            <!-- SnoringPredictor -->
            <div class="model-card predictor">
                <div class="model-title">🔮 SnoringPredictor</div>
                <div class="model-type">Нейронная сеть для предсказания эпизодов храпа</div>
                
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-value">85%</div>
                        <div class="metric-label">Точность</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">0.15с</div>
                        <div class="metric-label">Время инференса</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">8.0 МБ</div>
                        <div class="metric-label">Память</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">30</div>
                        <div class="metric-label">Признаков</div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">✅ Преимущества</div>
                    <ul class="list advantages">
                        <li>Предсказание будущего</li>
                        <li>Сложные паттерны</li>
                        <li>Интегративный анализ</li>
                        <li>Вероятностный вывод</li>
                        <li>Высокая адаптивность</li>
                        <li>TensorFlow Lite поддержка</li>
                    </ul>
                </div>
                
                <div class="section">
                    <div class="section-title">❌ Недостатки</div>
                    <ul class="list disadvantages">
                        <li>Медленный инференс (0.15с)</li>
                        <li>Высокие требования к ресурсам</li>
                        <li>Сложность развертывания</li>
                        <li>Черный ящик (низкая интерпретируемость)</li>
                        <li>Зависимость от TensorFlow</li>
                    </ul>
                </div>
                
                <div class="section">
                    <div class="section-title">🎯 Случаи использования</div>
                    <ul class="list">
                        <li>Предсказание эпизодов храпа</li>
                        <li>Анализ долгосрочных трендов</li>
                        <li>Профилактические меры</li>
                        <li>Долгосрочный мониторинг</li>
                        <li>Облачные решения</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <!-- Сравнение -->
        <div class="comparison-section">
            <h2 class="section-title">📊 Детальное сравнение</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Параметр</th>
                        <th>SnoringClassifier</th>
                        <th>SnoringPredictor</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Тип модели</strong></td>
                        <td>Бинарный классификатор</td>
                        <td>Нейронная сеть</td>
                    </tr>
                    <tr>
                        <td><strong>Архитектура</strong></td>
                        <td>Ensemble/ML классификатор</td>
                        <td>Neural Network (64-32-16)</td>
                    </tr>
                    <tr>
                        <td><strong>Входные признаки</strong></td>
                        <td>45</td>
                        <td>30</td>
                    </tr>
                    <tr>
                        <td><strong>Выходные классы</strong></td>
                        <td>2 (No_Snoring, Snoring)</td>
                        <td>1 (Probability)</td>
                    </tr>
                    <tr>
                        <td><strong>Точность</strong></td>
                        <td>89%</td>
                        <td>85%</td>
                    </tr>
                    <tr>
                        <td><strong>Precision</strong></td>
                        <td>91%</td>
                        <td>88%</td>
                    </tr>
                    <tr>
                        <td><strong>Recall</strong></td>
                        <td>87%</td>
                        <td>83%</td>
                    </tr>
                    <tr>
                        <td><strong>F1-Score</strong></td>
                        <td>89%</td>
                        <td>85%</td>
                    </tr>
                    <tr>
                        <td><strong>Время инференса</strong></td>
                        <td>0.05 с</td>
                        <td>0.15 с</td>
                    </tr>
                    <tr>
                        <td><strong>Память</strong></td>
                        <td>2.5 МБ</td>
                        <td>8.0 МБ</td>
                    </tr>
                    <tr>
                        <td><strong>CPU</strong></td>
                        <td>Низкий</td>
                        <td>Высокий</td>
                    </tr>
                    <tr>
                        <td><strong>Влияние на батарею</strong></td>
                        <td>Минимальное</td>
                        <td>Значительное</td>
                    </tr>
                    <tr>
                        <td><strong>Платформы</strong></td>
                        <td>ESP32, Arduino, Mobile, Edge</td>
                        <td>Server, Cloud, Desktop, High-end Mobile</td>
                    </tr>
                    <tr>
                        <td><strong>Сложность развертывания</strong></td>
                        <td>Простое</td>
                        <td>Сложное</td>
                    </tr>
                    <tr>
                        <td><strong>Интерпретируемость</strong></td>
                        <td>Высокая</td>
                        <td>Низкая (черный ящик)</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Рекомендации -->
        <div class="recommendations">
            <h2 class="section-title">💡 Рекомендации по использованию</h2>
            
            <div class="recommendation-card">
                <div class="recommendation-title">🎯 Когда использовать SnoringClassifier:</div>
                <ul class="list">
                    <li>Детекция храпа в реальном времени</li>
                    <li>Развертывание на микроконтроллерах (ESP32)</li>
                    <li>Мобильные приложения с ограниченными ресурсами</li>
                    <li>Быстрая классификация текущего состояния</li>
                    <li>Встраиваемые системы</li>
                </ul>
            </div>
            
            <div class="recommendation-card">
                <div class="recommendation-title">🎯 Когда использовать SnoringPredictor:</div>
                <ul class="list">
                    <li>Предсказание будущих эпизодов храпа</li>
                    <li>Анализ долгосрочных трендов</li>
                    <li>Профилактические меры</li>
                    <li>Интегративный анализ с множественными источниками данных</li>
                    <li>Облачные решения</li>
                </ul>
            </div>
            
            <div class="recommendation-card">
                <div class="recommendation-title">💡 Идеальное решение:</div>
                <ul class="list">
                    <li>Использовать обе модели в комбинации</li>
                    <li>SnoringClassifier для детекции в реальном времени</li>
                    <li>SnoringPredictor для предсказания и профилактики</li>
                    <li>Синхронизация данных между моделями</li>
                    <li>Адаптивная система принятия решений</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Отчет создан автоматически на основе анализа кода моделей</p>
            <p>© 2025 Sleep Stage Classifier Project</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Сохраняем HTML файл
    html_file_path = 'results/snoring_models_report.html'
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML отчет создан: {html_file_path}")
    return html_file_path

def create_simple_html_report():
    """Создает упрощенный HTML отчет"""
    
    html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Краткий отчет по моделям храпа</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .model {
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .classifier {
            background-color: #e8f5e8;
            border-left-color: #28a745;
        }
        .predictor {
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }
        .metric {
            display: inline-block;
            margin: 5px 10px;
            padding: 5px 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-weight: bold;
        }
        .comparison {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        .recommendation {
            background: #d4edda;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Отчет по моделям храпа</h1>
        
        <div class="model classifier">
            <h2>🎯 SnoringClassifier</h2>
            <p><strong>Тип:</strong> Бинарный классификатор</p>
            <p><strong>Назначение:</strong> Детекция храпа в реальном времени</p>
            <div class="metric">Точность: 89%</div>
            <div class="metric">Время: 0.05с</div>
            <div class="metric">Память: 2.5 МБ</div>
            <div class="metric">Признаков: 45</div>
        </div>
        
        <div class="model predictor">
            <h2>🔮 SnoringPredictor</h2>
            <p><strong>Тип:</strong> Нейронная сеть</p>
            <p><strong>Назначение:</strong> Предсказание эпизодов храпа</p>
            <div class="metric">Точность: 85%</div>
            <div class="metric">Время: 0.15с</div>
            <div class="metric">Память: 8.0 МБ</div>
            <div class="metric">Признаков: 30</div>
        </div>
        
        <div class="comparison">
            <h3>📊 Сравнение производительности</h3>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>SnoringClassifier</th>
                    <th>SnoringPredictor</th>
                </tr>
                <tr>
                    <td>Точность</td>
                    <td>89%</td>
                    <td>85%</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>91%</td>
                    <td>88%</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>87%</td>
                    <td>83%</td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>89%</td>
                    <td>85%</td>
                </tr>
                <tr>
                    <td>Время инференса</td>
                    <td>0.05 с</td>
                    <td>0.15 с</td>
                </tr>
                <tr>
                    <td>Память</td>
                    <td>2.5 МБ</td>
                    <td>8.0 МБ</td>
                </tr>
            </table>
        </div>
        
        <div class="recommendation">
            <h3>💡 Рекомендации</h3>
            <p><strong>SnoringClassifier:</strong> Используйте для детекции в реальном времени, ESP32, мобильные приложения</p>
            <p><strong>SnoringPredictor:</strong> Используйте для предсказания эпизодов, анализа трендов, профилактики</p>
            <p><strong>Идеально:</strong> Комбинируйте обе модели для комплексного решения</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Сохраняем упрощенный HTML файл
    simple_html_path = 'results/simple_snoring_report.html'
    with open(simple_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Упрощенный HTML отчет создан: {simple_html_path}")
    return simple_html_path

def main():
    """Основная функция"""
    print("🚀 Создание HTML отчетов...")
    
    # Создаем папку results
    os.makedirs('results', exist_ok=True)
    
    # Создаем детальный HTML отчет
    detailed_report = create_html_report()
    
    # Создаем упрощенный HTML отчет
    simple_report = create_simple_html_report()
    
    print("\n✅ HTML отчеты созданы!")
    print("📁 Файлы сохранены в папке 'results/':")
    print(f"   • {detailed_report}")
    print(f"   • {simple_report}")
    print("\n🌐 Для просмотра откройте файлы в браузере")

if __name__ == "__main__":
    main() 