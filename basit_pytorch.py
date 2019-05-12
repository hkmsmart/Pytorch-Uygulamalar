import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Temel autograd ornek 1               (Line 25 to 39)
# 2. Temel autograd ornek 2               (Line 46 to 83)
# 3. Numpy'den veri yukleniyor               (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline ozel veri kumesi icin    (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Kaydet ve modeli yukle                  (Line 183 to 189)


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Tensors olusturma.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Hesaplamali bir grafik olusturun.
y = w * x + b    # y = 2 * x + 3

# Hesaplama gradyanlari.
y.backward()

# Degradleri yazdirin.
print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1


# ================================================================== #
#                    2. Temel autograd ornek 2                       #
# ================================================================== #

# sekil tensoru olusturun (10, 3) ve (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Tamamen balli bir katman olusturun.
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Kayip fonksiyonu ve iyilestirici olusturun.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Dogrudan gecis.
pred = linear(x)

# Hesaplama kaybi
loss = criterion(pred, y)
print('loss: ', loss.item())

# Geriye dogru gecis
loss.backward()

# Degradeleri yazdirin.
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

# 1 adimli degrade
optimizer.step()

# Degrade inisini dusuk seviyede de gerceklestirebilirsiniz.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# 1 adimli degrade inisinden sonra kaybi yazdirin.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Numpy'den veri yukleniyor                   #
# ================================================================== #

# Bir numpy dizi olusturun.
x = np.array([[1, 2], [3, 4]])

# Numpy dizisini bir mesale tensorune donusturun.
y = torch.from_numpy(x)

# Torc tensorunu bir numpy dizisine donusturun.
z = y.numpy()


# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

# CIFAR-10 veri setini indirin ve kurun.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# Bir veri cifti al (diskten veri oku).
image, label = train_dataset[0]
print (image.size())
print (label)

# Veri yukleyici (bu, kuyruklari ve iplikleri cok basit bir sekilde saglar).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# Yineleme basladiginda, sira ve is parcacigi dosyalardan veri yuklemeye baslar.
data_iter = iter(train_loader)

# Mini toplu goruntuler ve etiketler.
images, labels = data_iter.next()

# Veri yukleyicinin gercek kullanimi asagidaki gibidir.
for images, labels in train_loader:
    # Egitim kodu buraya yazilmalidir.
    pass


# ================================================================== #
#                5. Kaydet ve modeli yukle                           #
# ================================================================== #

# Ozel veri kumenizi asagidaki gibi olusturmalisiniz.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Dosya yollarini veya dosya adlarinin bir listesini baslatin.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Dosyadan bir veri okuyun (orn. Numpy.fromfile, PIL.Image.open).
        # 2. Verileri onceden isleyin (ornegin torchvision.Transform).
        # 3. Bir veri cifti dondurun (ornegin, resim ve etiket).
        pass
    def __len__(self):
        # 0'i veri kumenizin toplam boyutuna degistirmelisiniz.
        return 0

# Onceden olusturulmus veri yukleyiciyi kullanabilirsiniz.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Onceden hazirlanmis ResNet-18'i indirin ve yukleyin.
resnet = torchvision.models.resnet18(pretrained=True)

# Modelin yalnizca ust katmanina ince ayar yapmak istiyorsaniz, asagidaki gibi ayarlayin.
for param in resnet.parameters():
    param.requires_grad = False

# Ince ayar icin ust katmani degistirin.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Dogrudan gecis.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Kaydet ve modeli yukle                     #
# ================================================================== #

# Tum modeli kaydedin ve yukleyin.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Yalnizca model parametrelerini kaydedin ve yukleyin (onerilir).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))