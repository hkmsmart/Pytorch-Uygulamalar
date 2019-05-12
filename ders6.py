import numpy as np

A = np.array( [[1,1],[0,1]] )
B = np.array( [[2,0],[3,4]] )

A * B
 #standart 2 matrisin carpimi

 #np.array([[2, 0], [0, 4]])

A.dot(B)
# matrix product
print("--np.dot(A, B)--")
print(np.dot(A, B))

#np.array([[5, 4],[3, 4]])

a = np.ones((3, 4), dtype=float)

a = a * 3

print("--a = a * 3-")
print(a)
#    array ([[3.,3.,3.,3.],
#    [3.,3.,3.,3.],
#    [3.,3.,3.,3.]])

d = np.random.random((3,4))
# 0 ile 1 arasinda uniform olarak agitilmis 3 * 4  bir dizi elde ettik.

d = d + a
print("--b = b + a-")
print(d)
# b ile a yi toplayip tekrar b ye yazdik

# array([[ 3.098576 , 3.05622794, 3.9892649 , 3.43405857],
#         3.42536648, 3.02038322, 3.15685286, 3.7162542 ],
#         [ 3.76245798, 3.2472704 , 3.0372553 , 3.71383375]])