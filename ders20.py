import torch

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# -1 buyuklugu diger boyutlardan cikarilir
print(x.size(), y.size(), z.size())