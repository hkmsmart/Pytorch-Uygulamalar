import numpy as np

a = np.random.random((2, 3))

# array([[ 0.72172506, 0.93579579, 0.46396908],
#       [ 0.15173845, 0.38972205, 0.76270919]])

print("--a.sum()--")
print(a.sum())
# dizi icerisindeki elemanlarin toplamini bulduk
# 3.4256596156639314

print("--a.min()--")
print(a.min())
# en kucuk degeri bulduk
# 0.151738451440807

print("--a.max()--")
print(a.max())
# en buyuk degeri bulduk
# 0.93579578504321492

b = np.arange(15).reshape(3, 5)
# 3*5 matris yarattik
print("--b--")
print(b)

# array([[ 0, 1, 2, 3, 4],
#              [ 5, 6, 7, 8, 9],
#              [10, 11, 12, 13, 14]])