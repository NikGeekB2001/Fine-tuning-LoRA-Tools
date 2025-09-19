import os
import json
import torch
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка переменных окружения
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Функция для токенизации и выравнивания меток
def tokenize_and_align_labels(examples, tokenizer, id2label):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
    )

    labels = []
    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(ner_tags[word_idx])
            else:
                label_ids.append(ner_tags[word_idx] if True else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Функция для вычисления метрик
def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

def train_lora():
    # Загрузка датасета
    dataset = load_dataset("Rexhaif/ru-med-ner", token=HF_API_TOKEN)
    train_dataset = dataset["train"]

    # Извлечение меток
    print("Features:", train_dataset.features)
    if hasattr(train_dataset.features["ner_tags"], 'feature') and hasattr(train_dataset.features["ner_tags"].feature, 'names'):
        label_names = train_dataset.features["ner_tags"].feature.names
    else:
        # Если не ClassLabel, получить уникальные метки из данных
        all_labels = set()
        for example in train_dataset:
            all_labels.update(example["ner_tags"])
        label_names = sorted(list(all_labels))
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    num_labels = len(label_names)

    print(f"Обнаруженные метки NER: {label_names}")
    print(f"Количество меток: {num_labels}")

    # Инициализация токенайзера и модели
    model_name = "с"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        token=HF_API_TOKEN,
    )

    # Перемещаем модель на GPU
    model.to(device)

    # Настройка LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Токенизация данных
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, id2label),
        batched=True,
    )

    # Настройка параметров обучения
    training_args = TrainingArguments(
        output_dir="./models/lora_adapter",
        eval_strategy="no",  # Отключаем оценку для простоты
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=False,
        report_to="none",
    )

    # Инициализация DataCollator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    # Обучение
    print("Начинаем обучение LoRA-адаптера на GPU...")
    trainer.train()
    print("Обучение завершено.")

    # Сохранение модели
    model.save_pretrained("./models/lora_adapter")
    tokenizer.save_pretrained("./models/lora_adapter")

    # Сохранение id2label
    with open("./models/lora_adapter/id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=4)

    print("LoRA-адаптер, токенизатор и id2label сохранены в './models/lora_adapter'.")

if __name__ == "__main__":
    train_lora()
