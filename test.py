import os
import torch
import time  # Добавлен недостающий импорт
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model

from peft import LoraConfig, get_peft_model, TaskType

# Загрузка переменных окружения
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Тест импорта и загрузки модели
try:
    model_name = "Den4ikAI/rubert_large_squad_2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_API_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=10,  # Пример
        id2label={i: f"label_{i}" for i in range(10)},
        label2id={f"label_{i}": i for i in range(10)},
        use_auth_token=HF_API_TOKEN,
    )
    print("Модель и токенизатор загружены успешно.")

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
    print("LoRA настроена успешно.")

except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")

# Создаем большие тензоры
x = torch.randn(10000, 10000).cuda()
y = torch.randn(10000, 10000).cuda()

# Измеряем время операции
start = time.time()
z = torch.mm(x, y)
torch.cuda.synchronize()  # Ждем завершения GPU операции
end = time.time()

print(f"Matrix multiplication took {end-start:.2f} seconds on GPU")
# Проверка GPU
print(f"Используется устройство: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Загрузка модели и токенизатора
model_name = "Den4ikAI/rubert_large_squad_2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=None)  # Используем token вместо use_auth_token
model = AutoModelForTokenClassification.from_pretrained(model_name, token=None, num_labels=3)

# Настройка LoRA
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, peft_config)
print("Модель и токенизатор загружены успешно.")

# Информация о параметрах
model.print_trainable_parameters()

print("LoRA настроена успешно.")

# Тест производительности GPU (если доступен)
if torch.cuda.is_available():
    # Создаем большие тензоры
    x = torch.randn(10000, 10000).cuda()
    y = torch.randn(10000, 10000).cuda()

    # Измеряем время операции
    start = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()  # Ждем завершения GPU операции
    end = time.time()

    print(f"Matrix multiplication took {end-start:.2f} seconds on GPU")
else:
    print("GPU не доступен, тест производительности пропущен")