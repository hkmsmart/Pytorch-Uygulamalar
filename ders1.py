# numpy paketini np olarak ekledik. 

import numpy as np


a = np.arange(15).reshape(3,5)

a.shape
#(3,5) olarak sonuc almaniz gerekiyor.

a.ndim
#2

a.dtype.name
#'int64'

a.itemsize
#8

a.size
#15 3 ve 5 in carpimi

type(a)
#&lt;type 'numpy.ndarray'&gt;

b=np.array([6,7,8])
b
#array([6, 7, 8])

type(b)
print(a)
print("---")
print(b)
#&lt;type 'numpy.ndarray'&gt;
