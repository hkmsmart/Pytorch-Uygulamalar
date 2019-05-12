import numpy as np

# Random deger atamak icin randint(ilk_deger, ikinci_deger, size) methodunu cagiriyoruz.

boy = np.random.randint(175, 200, 10)

kilo = np.random.randint(75, 100, 10)

bmi = kilo / boy ** 2
# herbir oyuncunun vucut kitle indeksini bmi dizisine kaydettik

print(bmi)
# ve ekrana yazdirdik