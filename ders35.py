import torch

from ders29 import testloader
from ders31 import net

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('10000 test goruntusunde agin dogrulugu %d %%' % (
    100 * correct / total))