import torch.optim as optim
from torch import nn

from ders31 import net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
