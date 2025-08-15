#!/usr/bin/env python3
"""
Скрипт для визуализации feature importance новой модели по ТЗ
Создает детальные графики важности признаков с группировкой по категориям
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Настройка стиля для красивых графиков
plt.style.use('default')
sns.set_palette("husl")

# Настройка для корректного отображения русских символов
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 12)


class TZCompliantFeatureImportanceVisualizer:
    """Класс для визуализации feature importance новой модели по ТЗ"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.feature_importances = None
        self.feature_names = None
        
    def load_data(self):
        """Загружает данные о важности признаков"""
        print("🔄 Загружаю данные о важности признаков...")
        
        # Загружаем CSV с важностью признаков
        csv_path = self.models_dir / "improved_39_features_tz_compliant" / "feature_importances_tz_compliant.csv"
        if csv_path.exists():
            self.feature_importances = pd.read_csv(csv_path)
            print(f"✅ Загружены данные о важности {len(self.feature_importances)} признаков")
        else:
            raise FileNotFoundError(f"Файл важности признаков не найден: {csv_path}")
        
        # Загружаем названия признаков
        names_path = self.models_dir / "improved_39_features_tz_compliant" / "feature_names_tz_compliant.txt"
        if names_path.exists():
            self.feature_names = {}
            with open(names_path, 'r') as f:
                for line in f.readlines():
                    if ':' in line:
                        idx, name = line.strip().split(':', 1)
                        self.feature_names[int(idx)] = name.strip()
            print(f"✅ Загружено {len(self.feature_names)} названий признаков")
        else:
            raise FileNotFoundError(f"Файл названий признаков не найден: {names_path}")
        
        # Названия признаков уже есть в колонке 'feature'
        self.feature_importances['feature_name'] = self.feature_importances['feature']
        
        return True
    
    def create_feature_importance_plot(self):
        """Создает основной график важности признаков"""
        print("📊 Создаю график важности признаков...")
        
        # Сортируем по важности
        df_sorted = self.feature_importances.sort_values('importance', ascending=True)
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Цвета для категорий
        colors = {
            'Audio (env)': '#FF6B6B',
            'Audio (b100)': '#4ECDC4', 
            'Audio (b400)': '#45B7D1',
            'Audio (b1000)': '#96CEB4',
            'Accelerometer': '#FFEAA7'
        }
        
        # Создаем горизонтальные бары
        bars = ax.barh(range(len(df_sorted)), df_sorted['importance'], 
                      color=[colors.get(cat, '#CCCCCC') for cat in df_sorted['category']])
        
        # Настройка осей
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['feature_name'], fontsize=9)
        ax.set_xlabel('Важность признака', fontsize=12, fontweight='bold')
        ax.set_title('Важность признаков для новой модели по ТЗ\n(max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Добавляем значения на бары
        for i, (bar, importance) in enumerate(zip(bars, df_sorted['importance'])):
            if importance > 0.01:  # Показываем только значимые значения
                ax.text(importance + 0.005, i, f'{importance:.3f}', 
                       va='center', fontsize=8, fontweight='bold')
        
        # Добавляем легенду
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=cat) 
                          for cat, color in colors.items()]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Сетка
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = Path("results_visualization") / "tz_compliant_feature_importance_detailed.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ График сохранен: {output_path}")
        
        return fig
    
    def create_category_importance_plot(self):
        """Создает график важности по категориям"""
        print("📊 Создаю график важности по категориям...")
        
        # Группируем по категориям
        category_importance = self.feature_importances.groupby('category')['importance'].sum().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Цвета для категорий
        colors = {
            'Audio (env)': '#FF6B6B',
            'Audio (b100)': '#4ECDC4', 
            'Audio (b400)': '#45B7D1',
            'Audio (b1000)': '#96CEB4',
            'Accelerometer': '#FFEAA7'
        }
        
        # Создаем горизонтальные бары
        bars = ax.barh(range(len(category_importance)), category_importance.values,
                      color=[colors.get(cat, '#CCCCCC') for cat in category_importance.index])
        
        # Настройка осей
        ax.set_yticks(range(len(category_importance)))
        ax.set_yticklabels(category_importance.index, fontsize=11, fontweight='bold')
        ax.set_xlabel('Общая важность категории', fontsize=12, fontweight='bold')
        ax.set_title('Важность категорий признаков для новой модели по ТЗ', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Добавляем значения на бары
        for i, (bar, importance) in enumerate(zip(bars, category_importance.values)):
            ax.text(importance + 0.01, i, f'{importance:.3f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        # Сетка
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = Path("results_visualization") / "tz_compliant_category_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ График категорий сохранен: {output_path}")
        
        return fig
    
    def create_top_features_plot(self):
        """Создает график топ-15 самых важных признаков"""
        print("📊 Создаю график топ-15 признаков...")
        
        # Выбираем топ-15 признаков
        top_features = self.feature_importances.nlargest(15, 'importance')
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Цвета для категорий
        colors = {
            'Audio (env)': '#FF6B6B',
            'Audio (b100)': '#4ECDC4', 
            'Audio (b400)': '#45B7D1',
            'Audio (b1000)': '#96CEB4',
            'Accelerometer': '#FFEAA7'
        }
        
        # Создаем горизонтальные бары
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=[colors.get(cat, '#CCCCCC') for cat in top_features['category']])
        
        # Настройка осей
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature_name'], fontsize=10)
        ax.set_xlabel('Важность признака', fontsize=12, fontweight='bold')
        ax.set_title('Топ-15 самых важных признаков новой модели по ТЗ', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Добавляем значения на бары
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax.text(importance + 0.005, i, f'{importance:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        # Добавляем легенду
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=cat) 
                          for cat, color in colors.items()]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Сетка
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = Path("results_visualization") / "tz_compliant_top15_features.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ График топ-15 сохранен: {output_path}")
        
        return fig
    
    def create_importance_distribution_plot(self):
        """Создает график распределения важности признаков"""
        print("📊 Создаю график распределения важности...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Гистограмма важности
        ax1.hist(self.feature_importances['importance'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Важность признака', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Количество признаков', fontsize=11, fontweight='bold')
        ax1.set_title('Распределение важности признаков', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Box plot по категориям
        categories = self.feature_importances['category'].unique()
        importance_by_category = [self.feature_importances[self.feature_importances['category'] == cat]['importance'].values 
                                for cat in categories]
        
        box_plot = ax2.boxplot(importance_by_category, labels=categories, patch_artist=True)
        
        # Цвета для box plot
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for patch, color in zip(box_plot['boxes'], colors[:len(categories)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Категория признаков', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Важность признака', fontsize=11, fontweight='bold')
        ax2.set_title('Распределение важности по категориям', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = Path("results_visualization") / "tz_compliant_importance_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ График распределения сохранен: {output_path}")
        
        return fig
    
    def create_summary_report(self):
        """Создает текстовый отчет о важности признаков"""
        print("📝 Создаю текстовый отчет...")
        
        # Сортируем по важности
        df_sorted = self.feature_importances.sort_values('importance', ascending=False)
        
        # Статистика по категориям
        category_stats = self.feature_importances.groupby('category').agg({
            'importance': ['sum', 'mean', 'count']
        }).round(4)
        
        # Топ-10 признаков
        top_10 = df_sorted.head(10)
        
        # Создаем отчет
        report_path = Path("results") / "tz_compliant_feature_importance_analysis.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ НОВОЙ МОДЕЛИ ПО ТЗ\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Модель: max_depth=5, min_samples_leaf=10, criterion=gini, class_weight=balanced\n")
            f.write(f"Всего признаков: {len(self.feature_importances)}\n\n")
            
            f.write("СТАТИСТИКА ПО КАТЕГОРИЯМ:\n")
            f.write("-" * 50 + "\n")
            for category in category_stats.index:
                total_imp = category_stats.loc[category, ('importance', 'sum')]
                mean_imp = category_stats.loc[category, ('importance', 'mean')]
                count = category_stats.loc[category, ('importance', 'count')]
                f.write(f"{category}:\n")
                f.write(f"  Общая важность: {total_imp:.4f}\n")
                f.write(f"  Средняя важность: {mean_imp:.4f}\n")
                f.write(f"  Количество признаков: {count}\n\n")
            
            f.write("ТОП-10 САМЫХ ВАЖНЫХ ПРИЗНАКОВ:\n")
            f.write("-" * 50 + "\n")
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:2d}. {row['feature_name']}\n")
                f.write(f"    Категория: {row['category']}\n")
                f.write(f"    Важность: {row['importance']:.4f}\n\n")
            
            f.write("ПРИЗНАКИ С НУЛЕВОЙ ВАЖНОСТЬЮ:\n")
            f.write("-" * 50 + "\n")
            zero_importance = self.feature_importances[self.feature_importances['importance'] == 0]
            for _, row in zero_importance.iterrows():
                f.write(f"  {row['feature_name']} ({row['category']})\n")
        
        print(f"✅ Текстовый отчет создан: {report_path}")
        
        return report_path
    
    def run_visualization(self):
        """Запускает полную визуализацию"""
        print("🚀 Начинаю визуализацию feature importance для новой модели по ТЗ...")
        
        # Загружаем данные
        self.load_data()
        
        # Создаем все графики
        self.create_feature_importance_plot()
        self.create_category_importance_plot()
        self.create_top_features_plot()
        self.create_importance_distribution_plot()
        
        # Создаем текстовый отчет
        self.create_summary_report()
        
        print("\n🎉 Визуализация feature importance завершена!")
        print("📁 Все файлы сохранены в папке results_visualization/")
        print("📝 Текстовый отчет сохранен в results/tz_compliant_feature_importance_analysis.txt")


def main():
    """Основная функция"""
    try:
        visualizer = TZCompliantFeatureImportanceVisualizer()
        visualizer.run_visualization()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 