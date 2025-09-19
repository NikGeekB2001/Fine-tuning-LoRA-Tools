# Fine-tuning RuBERT Large SQuAD 2 с LoRA 

Этот проект демонстрирует обучение модели Den4ikAI/rubert_large_squad_2 с использованием LoRA (Low-Rank Adaptation) для задачи на медицинских данных. Обучение проводится на GPU для повышения производительности.

## Структура проекта

```
..
├── .env                              # Файл для хранения токена Hugging Face
├── scripts/                          # Скрипты для обучения
│   └── train_lora.py                # Скрипт для обучения LoRA
├── 1.py                              # Python скрипт
├── analysis_utils.py                 # Утилиты для анализа
├── requirements.txt                  # Зависимости проекта
├── test_adapter_comparison.py        # Тест сравнения адаптеров
├── test_error_handling.py            # Тест обработки ошибок
├── Test_GPU.py                       # Тест GPU
├── test.py                           # Основной тестовый файл
├── test0.py                          # Дополнительный тестовый файл
├── README.md                         # Описание проекта
└── распределение_истинных_меток.png  # График распределения меток
```

## Установка

1. Клонируйте репозиторий или создайте структуру проекта.

2. Установите PyTorch с поддержкой CUDA (для GPU):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

3. Создайте файл `.env` в корне проекта и добавьте ваш Hugging Face токен:
   ```
   HF_API_TOKEN=your_hugging_face_token_here
   ```

4. Установите остальные зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Запуск обучения на GPU

1. Настройте accelerate:
   ```bash
   accelerate config
   ```
   Ответьте на вопросы, выбрав:
   - This machine
   - Количество GPU (обычно 1)
   - Использование FP16/BF16 (yes)

2. Запустите обучение:
   ```bash
   accelerate launch scripts/train_lora.py
   ```

## Объяснение процесса обучения на GPU

- `accelerate launch`: Автоматически настраивает обучение для использования доступных GPU.
- `model.to(device)`: Перемещает модель на GPU.
- `per_device_train_batch_size`: Определяет количество примеров, обрабатываемых за один раз на каждом GPU. Чем больше батч, тем быстрее обучение, но требуется больше VRAM.
- LoRA: Эффективно обучает только небольшую часть параметров, что позволяет обучать большие модели даже на GPU с ограниченной VRAM.




