import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel
import time
import os

def load_base_model(model_name="Den4ikAI/rubert_large_squad_2", num_labels=13):
    """Загрузка базовой модели без адаптера"""
    print("Загрузка базовой модели...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

def load_model_with_adapter(model_name="Den4ikAI/rubert_large_squad_2", adapter_path="./models/lora_adapter", num_labels=13):
    """Загрузка модели с LoRA адаптером"""
    print("Загрузка модели с LoRA адаптером...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    # Проверяем существование адаптера
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"Адаптер не найден по пути: {adapter_path}")
        model = base_model
    return model, tokenizer

def run_inference(model, tokenizer, text, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Запуск инференса и замер времени"""
    model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Замер времени
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()
    
    return outputs, end_time - start_time

def compare_models(test_texts):
    """Сравнение моделей с адаптером и без"""
    print("=== Сравнение моделей с LoRA адаптером и без ===\n")
    
    # Загрузка моделей
    try:
        base_model, base_tokenizer = load_base_model()
        adapter_model, adapter_tokenizer = load_model_with_adapter()
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")
        return
    
    # Параметры моделей
    base_params = sum(p.numel() for p in base_model.parameters())
    adapter_params = sum(p.numel() for p in adapter_model.parameters())
    
    print(f"Параметры базовой модели: {base_params:,}")
    if hasattr(adapter_model, 'peft_config'):
        trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
        print(f"Общие параметры с адаптером: {adapter_params:,}")
        print(f"Обучаемые параметры: {trainable_params:,} ({(trainable_params/base_params)*100:.4f}%)")
    print()
    
    # Тестирование на примерах
    for i, text in enumerate(test_texts):
        print(f"Тест {i+1}: {text[:50]}...")
        
        # Базовая модель
        try:
            base_outputs, base_time = run_inference(base_model, base_tokenizer, text)
            print(f"  Базовая модель время: {base_time:.4f} сек")
        except Exception as e:
            print(f"  Ошибка базовой модели: {e}")
            continue
        
        # Модель с адаптером
        try:
            adapter_outputs, adapter_time = run_inference(adapter_model, adapter_tokenizer, text)
            print(f"  Модель с LoRA время: {adapter_time:.4f} сек")
            
            # Разница во времени
            time_diff = ((base_time - adapter_time) / base_time) * 100
            print(f"  Разница во времени: {time_diff:.2f}% {'быстрее' if time_diff > 0 else 'медленнее'}")
            
            # Сравнение результатов
            base_logits = base_outputs.logits.cpu()
            adapter_logits = adapter_outputs.logits.cpu()
            
            # Средняя разница в предсказаниях
            diff = torch.mean(torch.abs(base_logits - adapter_logits)).item()
            print(f"  Средняя разница в логитах: {diff:.6f}")
        except Exception as e:
            print(f"  Ошибка модели с адаптером: {e}")
        
        print()

def main():
    # Примеры медицинских текстов для тестирования
    test_texts = [
        "Пациент жалуется на головную боль и повышение температуры тела до 38.5°C",
        "Диагноз: острый бронхит. Назначено: Амоксициллин 500 мг 3 раза в день",
        "Артериальное давление 140/90 мм рт. ст., частота сердечных сокращений 85 ударов в минуту",
        "Результаты анализов: глюкоза 5.6 ммоль/л, холестерин 4.2 ммоль/л",
        "После операции пациенту назначена физиотерапия и диета №5"
    ]
    
    compare_models(test_texts)

if __name__ == "__main__":
    main()