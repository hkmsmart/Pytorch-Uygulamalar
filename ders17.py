import torch

x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)
# new_* yontemler boyutlari alir
print("Double ")
print(x)

x = torch.randn_like(x, dtype=torch.float)
print("Float ")
print(x)

print("size ")
print(x.size())