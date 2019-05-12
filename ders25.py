import torch

x = torch.ones(2, 2, requires_grad=True)
print("X")
print(x)

y = x + 2
print("Y")
print(y)

#y Bir islemin sonucunda yaratildi, bu nedenle bir grad_fn.
print("grad_fn")
print(y.grad_fn)

#Tarihinde daha fazla islem yapin y
z = y * y * 3
out = z.mean()
print("Z")
print(z, out)