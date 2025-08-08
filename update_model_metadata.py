#!/usr/bin/env python3
"""
Скрипт для обновления метаданных существующих моделей
Убирает классы Snoring_Start и Snoring_End
"""

import os
import json
import glob
from datetime import datetime

def update_model_metadata():
    """Обновляет метаданные существующих моделей"""
    print("🔄 ОБНОВЛЕНИЕ МЕТАДАННЫХ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Находим все файлы метаданных
    metadata_files = glob.glob("models/snoring_model_*_metadata.json")
    
    print(f"📊 Найдено файлов метаданных: {len(metadata_files)}")
    
    updated_count = 0
    skipped_count = 0
    
    for metadata_file in metadata_files:
        try:
            # Читаем текущие метаданные
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Проверяем, нужно ли обновление
            class_names = metadata.get('class_names', [])
            has_old_classes = 'Snoring_Start' in class_names or 'Snoring_End' in class_names
            
            if has_old_classes:
                print(f"🔄 Обновление: {os.path.basename(metadata_file)}")
                
                # Обновляем классы
                metadata['class_names'] = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
                
                # Добавляем информацию об обновлении
                metadata['updated_at'] = datetime.now().isoformat()
                metadata['update_note'] = 'Removed Snoring_Start and Snoring_End classes'
                
                # Сохраняем обновленные метаданные
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                updated_count += 1
                print(f"   ✅ Обновлено: {', '.join(metadata['class_names'])}")
                
            else:
                print(f"⏭️  Пропуск: {os.path.basename(metadata_file)} (уже обновлено)")
                skipped_count += 1
                
        except Exception as e:
            print(f"❌ Ошибка обновления {metadata_file}: {e}")
    
    print(f"\n📋 РЕЗУЛЬТАТЫ ОБНОВЛЕНИЯ:")
    print(f"   Обновлено файлов: {updated_count}")
    print(f"   Пропущено файлов: {skipped_count}")
    print(f"   Всего обработано: {len(metadata_files)}")

def create_backup():
    """Создает резервные копии метаданных"""
    print("\n💾 СОЗДАНИЕ РЕЗЕРВНЫХ КОПИЙ")
    print("=" * 60)
    
    metadata_files = glob.glob("models/snoring_model_*_metadata.json")
    
    for metadata_file in metadata_files:
        try:
            # Создаем имя резервной копии
            backup_file = metadata_file.replace('.json', '_backup.json')
            
            # Копируем файл
            with open(metadata_file, 'r') as f:
                content = f.read()
            
            with open(backup_file, 'w') as f:
                f.write(content)
            
            print(f"💾 Резервная копия: {os.path.basename(backup_file)}")
            
        except Exception as e:
            print(f"❌ Ошибка создания резервной копии {metadata_file}: {e}")

def verify_updates():
    """Проверяет результаты обновления"""
    print("\n🔍 ПРОВЕРКА ОБНОВЛЕНИЙ")
    print("=" * 60)
    
    metadata_files = glob.glob("models/snoring_model_*_metadata.json")
    
    correct_count = 0
    incorrect_count = 0
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            class_names = metadata.get('class_names', [])
            expected_classes = ['No_Snoring', 'Light_Snoring', 'Heavy_Snoring']
            
            if class_names == expected_classes:
                print(f"✅ {os.path.basename(metadata_file)} - корректно")
                correct_count += 1
            else:
                print(f"❌ {os.path.basename(metadata_file)} - некорректно")
                print(f"   Ожидалось: {expected_classes}")
                print(f"   Получено: {class_names}")
                incorrect_count += 1
                
        except Exception as e:
            print(f"❌ Ошибка проверки {metadata_file}: {e}")
            incorrect_count += 1
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
    print(f"   Корректных файлов: {correct_count}")
    print(f"   Некорректных файлов: {incorrect_count}")
    print(f"   Всего проверено: {len(metadata_files)}")

def main():
    """Основная функция"""
    print("🚀 ОБНОВЛЕНИЕ МЕТАДАННЫХ МОДЕЛЕЙ ХРАПА")
    print("=" * 80)
    
    # 1. Создаем резервные копии
    create_backup()
    
    # 2. Обновляем метаданные
    update_model_metadata()
    
    # 3. Проверяем результаты
    verify_updates()
    
    print(f"\n✅ Обновление метаданных завершено!")

if __name__ == "__main__":
    main() 