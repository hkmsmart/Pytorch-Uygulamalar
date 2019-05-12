import torch

a = torch.ones(5)
print('torch')
print(a)

b = a.numpy()
print("numpy")
print(b)

#Numpy dizisinin deger olarak nasil degistigini gorun.
a.add_(1)
print("add")
print(a)
print(b)