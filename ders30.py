import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
import torchvision

from ders29 import trainloader, classes


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# rastgele egitim goruntuleri al
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resimleri goster
imshow(torchvision.utils.make_grid(images))
# etiketleri yazdir
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
