import numpy as np

baseball = [[180, 78.4],

            [215, 102.7],

            [210, 98.5],

            [188, 75.2]]

# 2D dizi olusturma

np_baseball = np.array(baseball)
print("-- #2D dizi olusturma--")
print(np_baseball)

# 1.satiri yazdirma islemi
print("-- # 1.satiri yazdirma islemi--")
print(np_baseball[0, :])

#  2. satirin butun sutunlarini bu diziye kaydedelim

np_kilo = np_baseball[:, 1]
print("---2. satirin butun sutunlarini bu diziye kaydedelim-")
print(np_kilo)

# 3.oyuncunun boyunu yazdirma
print("-- # 3.oyuncunun boyunu yazdirma--")
print(np_baseball[2, 0])
