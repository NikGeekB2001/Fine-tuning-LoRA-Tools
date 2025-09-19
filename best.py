import os
import json
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Загрузка переменных окружения
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

class ModelTester:
    def __init__(self, model_name="Den4ikAI/rubert_large_squad_2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Используется устройство: {self.device}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN)
        
        # Загружаем маппинг меток
        self.load_label_mappings()
        
    def load_label_mappings(self):
        """Загружаем маппинг меток из датасета"""
        # Карта меток для расшифровки (BIO-формат для Named Entity Recognition)
        self.label_map = {
            "0": "O",          # Вне сущности (Outside) - токен не относится к медицинской сущности
            "1": "B-DISEASE",  # Начало болезни (Beginning of Disease) - первый токен в названии заболевания или диагноза
            "2": "I-DISEASE",  # Продолжение болезни (Inside Disease) - последующие токены в названии заболевания или диагноза
            "3": "B-SYMPTOM",  # Начало симптома (Beginning of Symptom) - первый токен в названии симптома или жалобы пациента
            "4": "I-SYMPTOM",  # Продолжение симптома (Inside Symptom) - последующие токены в названии симптома или жалобы пациента
            "5": "B-DRUG",     # Начало лекарства (Beginning of Drug) - первый токен в названии лекарственного препарата или медикамента
            "6": "I-DRUG",     # Продолжение лекарства (Inside Drug) - последующие токены в названии лекарственного препарата или медикамента
            "7": "B-ANATOMY",  # Начало анатомии (Beginning of Anatomy) - первый токен в названии органа, ткани или части тела
            "8": "I-ANATOMY",  # Продолжение анатомии (Inside Anatomy) - последующие токены в названии органа, ткани или части тела
            "9": "B-PROCEDURE", # Начало процедуры (Beginning of Procedure) - первый токен в названии медицинской процедуры или исследования
            "10": "I-PROCEDURE", # Продолжение процедуры (Inside Procedure) - последующие токены в названии медицинской процедуры или исследования
            "11": "B-FINDING", # Начало результата (Beginning of Finding) - первый токен в названии результата обследования или медицинского наблюдения
            "12": "I-FINDING"  # Продолжение результата (Inside Finding) - последующие токены в названии результата обследования или медицинского наблюдения
        }

        try:
            dataset = load_dataset("Rexhaif/ru-med-ner", token=HF_API_TOKEN)
            train_dataset = dataset["train"]

            if hasattr(train_dataset.features["ner_tags"], 'feature') and hasattr(train_dataset.features["ner_tags"].feature, 'names'):
                label_names = train_dataset.features["ner_tags"].feature.names
            else:
                all_labels = set()
                for example in train_dataset:
                    all_labels.update(example["ner_tags"])
                label_names = sorted(list(all_labels))

            self.id2label = {i: label for i, label in enumerate(label_names)}
            self.label2id = {label: i for i, label in enumerate(label_names)}
            self.num_labels = len(label_names)

            print(f"📋 Загружены метки: {list(self.id2label.values())}")

        except Exception as e:
            print(f"❌ Ошибка при загрузке датасета: {e}")
            # Fallback к сохранённым меткам
            with open("./models/lora_adapter/id2label.json", "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.id2label = {int(k): self.label_map[str(v)] for k, v in loaded.items()}
            self.label2id = {v: k for k, v in self.id2label.items()}
            self.num_labels = len(self.id2label)
    
    def load_base_model(self):
        """Загружаем базовую модель без LoRA"""
        print("🔄 Загрузка базовой модели...")
        base_model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            token=HF_API_TOKEN,
            ignore_mismatched_sizes=True
        )
        base_model.to(self.device)
        return base_model
    
    def load_lora_model(self, adapter_path="./models/lora_adapter"):
        """Загружаем модель с LoRA адаптером"""
        print("🔄 Загрузка LoRA модели...")
        
        # Сначала загружаем базовую модель
        base_model = self.load_base_model()
        
        # Настраиваем LoRA конфигурацию
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="TOKEN_CLS",
        )
        
        # Применяем LoRA к базовой модели
        lora_model = get_peft_model(base_model, lora_config)
        
        # Загружаем веса адаптера
        try:
            lora_model.load_adapter(adapter_path, "default")
            print("✅ LoRA адаптер успешно загружен")
        except Exception as e:
            print(f"❌ Ошибка загрузки LoRA адаптера: {e}")
            return None
            
        lora_model.to(self.device)
        return lora_model
    
    def predict_single_text(self, model, text):
        """Предсказание для одного текста"""
        # Токенизация
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512,
            is_split_into_words=False
        ).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Декодирование токенов и меток
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[pred.item()] for pred in predictions[0]]
        # Преобразуем числовые метки в строковые
        predicted_labels = [self.label_map[str(label)] for label in predicted_labels]
        
        # Фильтруем специальные токены
        filtered_results = []
        for token, label in zip(tokens, predicted_labels):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                filtered_results.append((token, label))
                
        return filtered_results
    
    def predict_batch(self, model, texts):
        """Пакетное предсказание"""
        all_predictions = []
        
        for text in tqdm(texts, desc="Предсказания"):
            pred = self.predict_single_text(model, text)
            all_predictions.append(pred)
            
        return all_predictions
    
    def evaluate_on_dataset(self, model, test_size=100):
        """Оценка на части датасета"""
        print(f"📊 Загрузка тестовых данных (размер: {test_size})...")
        
        try:
            dataset = load_dataset("Rexhaif/ru-med-ner", token=HF_API_TOKEN)
            
            # Берём часть данных для теста
            if "test" in dataset:
                test_data = dataset["test"].select(range(min(test_size, len(dataset["test"]))))
            else:
                # Если нет тестовой выборки, берём из валидационной или обучающей
                if "validation" in dataset:
                    test_data = dataset["validation"].select(range(min(test_size, len(dataset["validation"]))))
                else:
                    test_data = dataset["train"].select(range(-test_size, len(dataset["train"])))  # Последние примеры
            
            true_predictions = []
            true_labels = []
            
            print("🔄 Обработка примеров...")
            for example in tqdm(test_data):
                tokens = example["tokens"]
                true_tags = example["ner_tags"]
                
                # Создаём текст из токенов
                text = " ".join(tokens)
                
                # Получаем предсказания
                predictions = self.predict_single_text(model, text)
                
                # Извлекаем только метки (без токенов)
                pred_labels = [label for _, label in predictions]
                true_label_names = [self.id2label[tag] for tag in true_tags]
                # Преобразуем числовые метки в строковые
                true_label_names = [self.label_map[str(label)] for label in true_label_names]

                # Выравниваем длины (может отличаться из-за подтокенизации)
                min_len = min(len(pred_labels), len(true_label_names))
                if min_len > 0:
                    true_predictions.append(pred_labels[:min_len])
                    true_labels.append(true_label_names[:min_len])
            
            # Вычисляем метрики
            metrics = {
                "accuracy": accuracy_score(true_labels, true_predictions),
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }
            
            return metrics, true_labels, true_predictions
            
        except Exception as e:
            print(f"❌ Ошибка при оценке на датасете: {e}")
            return None, None, None
    
    def create_test_examples(self):
        """Создаём собственные тестовые примеры с разметкой"""
        test_examples = [
            {
                "text": "Пациент жалуется на боль в груди и одышку",
                "expected": [
                    ("Пациент", "O"),
                    ("жалуется", "O"),
                    ("на", "O"),
                    ("боль", "B-SYMPTOM"),
                    ("в", "O"),
                    ("груди", "B-ANATOMY"),
                    ("и", "O"),
                    ("одышку", "B-SYMPTOM")
                ]
            },
            {
                "text": "Диагноз: инфаркт миокарда",
                "expected": [
                    ("Диагноз", "O"),
                    (":", "O"),
                    ("инфаркт", "B-DISEASE"),
                    ("миокарда", "I-DISEASE")
                ]
            },
            {
                "text": "Назначили аспирин от головной боли",
                "expected": [
                    ("Назначили", "O"),
                    ("аспирин", "B-DRUG"),
                    ("от", "O"),
                    ("головной", "B-SYMPTOM"),
                    ("боли", "I-SYMPTOM")
                ]
            },
            {
                "text": "Сделали рентген лёгких",
                "expected": [
                    ("Сделали", "O"),
                    ("рентген", "B-PROCEDURE"),
                    ("лёгких", "B-ANATOMY")
                ]
            },
            {
                "text": "Обнаружена опухоль в печени размером 3 см",
                "expected": [
                    ("Обнаружена", "O"),
                    ("опухоль", "B-FINDING"),
                    ("в", "O"),
                    ("печени", "B-ANATOMY"),
                    ("размером", "O"),
                    ("3", "O"),
                    ("см", "O")
                ]
            }
        ]
        return test_examples
    
    def evaluate_on_custom_examples(self, model, test_examples):
        """Оценка на кастомных примерах"""
        print("🧪 Тестирование на кастомных примерах...")
        
        all_true_labels = []
        all_predictions = []
        
        for example in test_examples:
            text = example["text"]
            expected = example["expected"]
            
            # Получаем предсказания
            predictions = self.predict_single_text(model, text)
            
            print(f"\n📝 Текст: {text}")
            print("👁️ Ожидаемое vs 🤖 Предсказанное:")
            
            # Сравниваем токен к токену
            for i, (token, pred_label) in enumerate(predictions):
                if i < len(expected):
                    expected_token, expected_label = expected[i]
                    match = "✅" if pred_label == expected_label else "❌"
                    print(f"  {token} -> {pred_label} (ожидалось: {expected_label}) {match}")
                else:
                    print(f"  {token} -> {pred_label} (лишний токен)")
            
            # Собираем метки для общих метрик
            pred_labels = [str(label) if isinstance(label, int) else label for _, label in predictions]
            true_labels = [str(label) if isinstance(label, int) else label for _, label in expected]
            
            # Выравниваем длины
            min_len = min(len(pred_labels), len(true_labels))
            if min_len > 0:
                all_predictions.append(pred_labels[:min_len])
                all_true_labels.append(true_labels[:min_len])
        
        # Вычисляем общие метрики
        if all_true_labels and all_predictions:
            metrics = {
                "accuracy": accuracy_score(all_true_labels, all_predictions),
                "precision": precision_score(all_true_labels, all_predictions),
                "recall": recall_score(all_true_labels, all_predictions),
                "f1": f1_score(all_true_labels, all_predictions),
            }
            return metrics
        else:
            return {}
    
    def compare_models(self, test_size=50):
        """Полное сравнение базовой модели и LoRA модели"""
        print("🔄 Начинаем полное сравнение моделей...\n")
        
        # Загружаем модели
        base_model = self.load_base_model()
        lora_model = self.load_lora_model()
        
        if lora_model is None:
            print("❌ Не удалось загрузить LoRA модель")
            return
        
        # Создаём тестовые примеры
        test_examples = self.create_test_examples()
        
        print("=" * 60)
        print("🧪 ТЕСТ 1: Кастомные примеры")
        print("=" * 60)
        
        print("\n🤖 Базовая модель:")
        base_custom_metrics = self.evaluate_on_custom_examples(base_model, test_examples)
        
        print("\n🚀 LoRA модель:")
        lora_custom_metrics = self.evaluate_on_custom_examples(lora_model, test_examples)
        
        print("\n📊 Сравнение метрик на кастомных примерах:")
        print(f"{'Метрика':<12} {'Базовая':<10} {'LoRA':<10} {'Улучшение':<12}")
        print("-" * 50)
        for metric in ["accuracy", "precision", "recall", "f1"]:
            base_val = base_custom_metrics.get(metric, 0)
            lora_val = lora_custom_metrics.get(metric, 0)
            improvement = lora_val - base_val
            improvement_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
            print(f"{metric:<12} {base_val:<10.3f} {lora_val:<10.3f} {improvement_str:<12}")
        
        print("\n" + "=" * 60)
        print("🧪 ТЕСТ 2: Реальный датасет")
        print("=" * 60)
        
        print("\n🤖 Оценка базовой модели на датасете...")
        base_metrics, _, _ = self.evaluate_on_dataset(base_model, test_size)
        
        print("\n🚀 Оценка LoRA модели на датасете...")
        lora_metrics, true_labels, lora_predictions = self.evaluate_on_dataset(lora_model, test_size)
        
        if base_metrics and lora_metrics:
            print("\n📊 Сравнение метрик на реальном датасете:")
            print(f"{'Метрика':<12} {'Базовая':<10} {'LoRA':<10} {'Улучшение':<12}")
            print("-" * 50)
            for metric in ["accuracy", "precision", "recall", "f1"]:
                base_val = base_metrics[metric]
                lora_val = lora_metrics[metric]
                improvement = lora_val - base_val
                improvement_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
                print(f"{metric:<12} {base_val:<10.3f} {lora_val:<10.3f} {improvement_str:<12}")
        
        # Детальный отчёт
        if true_labels and lora_predictions:
            print("\n📋 Детальный отчёт по классам:")
            report = classification_report(true_labels, lora_predictions)
            print(report)
        
        print("\n🎉 Сравнение завершено!")
        
        return {
            "custom_metrics": {"base": base_custom_metrics, "lora": lora_custom_metrics},
            "dataset_metrics": {"base": base_metrics, "lora": lora_metrics}
        }

def main():
    """Главная функция для запуска тестирования"""
    print("🚀 Запуск системы тестирования LoRA модели")
    print("=" * 60)
    
    # Создаём тестер
    tester = ModelTester()
    
    # Запускаем полное сравнение
    results = tester.compare_models(test_size=100)
    
    print("\n✅ Все тесты завершены!")

if __name__ == "__main__":
    main()
