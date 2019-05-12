"""       """
import torch
from torch.nn import Linear
import numpy as np

#random sayi
girdi = torch.rand(1)
print("girdi",girdi)

#1 girdi ,1 cikti
Linear11 = Linear(in_features=1,out_features=1)

print("agirlik w:",Linear11.weight)
print("Y ekseni b:",Linear11.bias)


print("m*x + b, m*girdi + b, w*girdi + b")

print(Linear11.weight*girdi+Linear11.bias)

print("------")
#1 e 5 metrix
Lin1 = Linear(in_features=1,out_features=5,bias=True)
print("Lin1")
print(Lin1.weight)

Lin2 = Linear(in_features=5,out_features=1)
print("Lin2")
print(Lin2.weight)

print("------")

print(Lin2(Lin1(girdi)))

