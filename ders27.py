import torch
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print("Y")
print(y)
#Simdi bu durumda yartik bir skaler degildir. torch.autograd tam Jacobian tam olarak hesaplayamadi,
#ancak sadece vektor-Jacobian urununu istiyorsak, vektoru backward arguman olarak iletin:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print("x.grad")
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)
#.requires_grad=Truekod blogunu icine alarak otomatik autograd'in
#Tensors'taki gecmisini takip etmesini durdurabilirsiniz . with torch.no_grad():
with torch.no_grad():
    print("x ** 2).requires_grad)")
    print((x ** 2).requires_grad)