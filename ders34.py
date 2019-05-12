import torchvision
from matplotlib.pyplot import imshow

from ders30 import imshow
from ders29 import classes,testloader
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
