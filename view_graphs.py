#!/usr/bin/env python3
"""
Просмотр всех графиков и визуализаций в проекте
"""

import os
import glob
import subprocess
import platform

def list_all_graphs():
    """Показывает все доступные графики"""
    print("📊 ДОСТУПНЫЕ ГРАФИКИ И ВИЗУАЛИЗАЦИИ")
    print("=" * 60)
    
    # Ищем все PNG файлы в папке results
    png_files = glob.glob("results/*.png")
    
    if png_files:
        print("\n🖼️  Готовые PNG графики:")
        for i, file in enumerate(png_files, 1):
            size = os.path.getsize(file) / 1024  # KB
            filename = os.path.basename(file)
            print(f"{i}. {filename} ({size:.1f} KB)")
    else:
        print("❌ PNG графики не найдены")
    
    # Ищем все HTML файлы
    html_files = glob.glob("results/*.html")
    
    if html_files:
        print("\n🌐 HTML отчеты:")
        for i, file in enumerate(html_files, 1):
            size = os.path.getsize(file) / 1024  # KB
            filename = os.path.basename(file)
            print(f"{i}. {filename} ({size:.1f} KB)")
    else:
        print("❌ HTML отчеты не найдены")
    
    # Ищем скрипты визуализации
    viz_scripts = [
        "simple_visualization.py",
        "visualize_results.py", 
        "create_snoring_visualizations.py",
        "model_visualization.py",
        "visualize_results_from_file.py"
    ]
    
    print("\n🎨 Скрипты для создания графиков:")
    for i, script in enumerate(viz_scripts, 1):
        if os.path.exists(script):
            print(f"{i}. {script} - создает новые графики")
        else:
            print(f"{i}. {script} - НЕ НАЙДЕН")
    
    return png_files, html_files

def open_file(file_path):
    """Открывает файл в зависимости от ОС"""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path])
        elif platform.system() == "Windows":
            subprocess.run(["start", file_path], shell=True)
        else:  # Linux
            subprocess.run(["xdg-open", file_path])
        print(f"✅ Открыт: {file_path}")
    except Exception as e:
        print(f"❌ Ошибка открытия {file_path}: {e}")

def create_graphs_summary():
    """Создает сводку по графикам"""
    print("\n📋 СВОДКА ПО ГРАФИКАМ")
    print("=" * 60)
    
    graphs_info = {
        "results/prediction_results.png": {
            "description": "Результаты предсказаний моделей",
            "type": "Анализ производительности",
            "size": "212 KB"
        },
        "results/snoring_analysis.png": {
            "description": "Детальный анализ храпа",
            "type": "Временные ряды и статистика",
            "size": "2.0 MB"
        },
        "results/prediction_simulation.png": {
            "description": "Симуляция работы предиктора",
            "type": "Симуляция моделей",
            "size": "452 KB"
        },
        "results/snoring_visualization.png": {
            "description": "Визуализация паттернов храпа",
            "type": "Анализ паттернов",
            "size": "595 KB"
        },
        "results/snoring_models_report.html": {
            "description": "Детальный HTML отчет по моделям",
            "type": "Интерактивный отчет",
            "size": "18 KB"
        },
        "results/simple_snoring_report.html": {
            "description": "Упрощенный HTML отчет",
            "type": "Краткий отчет",
            "size": "5 KB"
        }
    }
    
    for filename, info in graphs_info.items():
        if os.path.exists(filename):
            print(f"\n📊 {os.path.basename(filename)}")
            print(f"   Описание: {info['description']}")
            print(f"   Тип: {info['type']}")
            print(f"   Размер: {info['size']}")

def main():
    """Основная функция"""
    print("🚀 Просмотр графиков и визуализаций")
    print("=" * 60)
    
    # Показываем все доступные графики
    png_files, html_files = list_all_graphs()
    
    # Создаем сводку
    create_graphs_summary()
    
    print("\n💡 КАК ПОСМОТРЕТЬ ГРАФИКИ:")
    print("=" * 60)
    
    if png_files:
        print("\n🖼️  Для просмотра PNG графиков:")
        for file in png_files:
            filename = os.path.basename(file)
            print(f"   • open results/{filename}")
    
    if html_files:
        print("\n🌐 Для просмотра HTML отчетов:")
        for file in html_files:
            filename = os.path.basename(file)
            print(f"   • open results/{filename}")
    
    print("\n🎨 Для создания новых графиков:")
    print("   • python3 simple_visualization.py")
    print("   • python3 visualize_results.py")
    print("   • python3 create_snoring_visualizations.py")
    print("   • python3 model_visualization.py")
    print("   • python3 visualize_results_from_file.py")
    
    print("\n📁 Все графики сохраняются в:")
    print("   • Папка results/ (PNG и HTML)")
    
    # Предлагаем открыть графики
    if png_files or html_files:
        print("\n🎯 Хотите открыть графики? (y/n): ", end="")
        try:
            response = input().lower()
            if response == 'y':
                print("\n📂 Открываю графики...")
                
                # Открываем PNG файлы
                for file in png_files:
                    open_file(file)
                
                # Открываем HTML файлы
                for file in html_files:
                    open_file(file)
                
                print("\n✅ Все графики открыты!")
        except KeyboardInterrupt:
            print("\n❌ Отменено пользователем")

if __name__ == "__main__":
    main() 