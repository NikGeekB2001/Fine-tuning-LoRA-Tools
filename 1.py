import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Простой тест тензора на GPU
    x = torch.randn(3, 3).cuda()
    print(f"Tensor on GPU: {x.device}")
else:
    print("No CUDA device")

# source C:/Users/kolin/Fine-tuning-rag-lora/venv/Scripts/activate

