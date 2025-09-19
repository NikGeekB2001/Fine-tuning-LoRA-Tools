import torch
import time

# Создаем большие тензоры
x = torch.randn(10000, 10000).cuda()
y = torch.randn(10000, 10000).cuda()

# Измеряем время операции
start = time.time()
z = torch.mm(x, y)
torch.cuda.synchronize()  # Ждем завершения GPU операции
end = time.time()

print(f"Matrix multiplication took {end-start:.2f} seconds on GPU")