import numpy as np
a=np.array([[1.0,2.0], [3.0,4.0]])
print(a)
#[[ 1.  2.]
# [ 3.  4.]]

import numpy as np
a=np.array([[1.0,2.0], [3.0,4.0]])
print(a)
#[[ 1.  2.]
# [ 3.  4.]]

a.transpose()
print("--Matris Devrik   a.transpose()--")
print(a.transpose())
# Transpose alma
#array([[ 1., 3.],
#       [ 2., 4.]])

np.linalg.inv(a)
# Tersini alma
#array([[-2. , 1. ],
#              [ 1.5, -0.5]])
u=np.eye(2)
# unit 2x2 matrix; "eye" represents "I"
print("--matris tersi  np.eye(2)  --")
print(u)
#array([[ 1., 0.],
#              [ 0., 1.]])


j=np.array([[0.0,-1.0], [1.0,0.0]])
# Dot product
print("--Dot Product (Aritmetik islemler)  np.dot (j, j) --")
print(np.dot (j, j) )

# matrix product

#array([[-1., 0.],
#              [ 0., -1.]])
print("--Linear Trace (izleme)  np.trace(u) --")
print(np.trace(u))
 # trace
#2.0