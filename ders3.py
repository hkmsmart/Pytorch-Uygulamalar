import numpy as np
a = np.zeros( (3,4) )

# array([[ 0., 0., 0., 0.],

#        [ 0., 0., 0., 0.],

#        [ 0., 0., 0., 0.]])
print("--np.zeros--")
print(a)


b = np.ones( (2,3,4), dtype=np.int16 )

#array([[[ 1, 1, 1, 1],

#        [ 1, 1, 1, 1],

#        [ 1, 1, 1, 1]],

#       [[ 1, 1, 1, 1],

#        [ 1, 1, 1, 1],

#        [ 1, 1, 1, 1]]], dtype=int16)

print("--np.ones--")
print(b)


c = np.empty( (2,3) )

#array([[ 3.73603959e-262, 6.02658058e-154, 6.55490914e-260],
print("--np.empty--")
print(c)
