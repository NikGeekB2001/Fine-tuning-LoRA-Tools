import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def test_without_token():
    try:
        model_name = "Den4ikAI/rubert_large_squad_2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=None)
        model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=None)
        print("Модель загружена без токена (публичная модель).")
    except Exception as e:
        print(f"Ошибка без токена: {e}")

def test_without_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("GPU не доступен, используется CPU.")
    else:
        print("GPU доступен.")

if __name__ == "__main__":
    test_without_token()
    test_without_gpu()
