import numpy as np

x = np.arange(10) ** 2
print("--x--")
print(x)

# array([ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81])

x[2:6]
# 2.eleman dahil 6. eleman haric  dahil et.

# array([ 4, 9, 16, 25])

y = x[2:5]
# y dizisine 2.den 5.ye kadar olan elemanlar atandi. Aslinda gercekten dizi yaratmadik.
print("--y--")
print(y)

# array([ 4, 9, 16])

# Eger biz x' in 2 'den 5' e kadar olan elemanlarini degistirsek y' nin elemanlarininda degistigini goruruz.

#x[2:3] = [12, 13, 14]


# array([12,13,14])

# !!! Eger gercekten bir "y" dizisi yaratmak istiyorsaniz su islemi yapmalisiniz.

y = list(x[2:5])
print("--y = list(x[2:5])--")
print(y)

#array([12, 13, 14])

# simdi tekrardan X[2:5] ' i baska degerlere atayalim ve "y" dizisini tekrardan ekrana bastiralim.

x[2:5] = [4, 9, 16]
print("--x[2:5] = [4, 9, 16]--")
print(x)

# array([12,13,14])