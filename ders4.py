import numpy as np
from numpy import pi

# sadece pi sayisini almis olduk.

np.linspace(0,2,9)
#  0 ile 2 arasinda bu sayilarda dahil olmak uzere 9 tane sayi olusturduk.

np.array([0. , 0.25, 0.5 , 0.75, 1. , 1.25, 1.5 , 1.75, 2.])

x=np.linspace(0,2*pi,100)
# 0 ilke 2*pi arasinda 100 tane sayi oluiturduk

x.size

#100

f = np.sin(x)
print(f)
#x degerlerine gore sinus degerlerinin bulundugu bir dizi yarattik.
