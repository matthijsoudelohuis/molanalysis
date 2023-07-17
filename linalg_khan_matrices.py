
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### Distributive property of matrices: 

m = 3
n = 4
k = 5

A = np.round(np.random.rand(k,m)*5)

B = np.round(np.random.rand(m,n)*5)

C = np.round(np.random.rand(m,n)*5)

print(A @ (B + C))
#same thing as:
print(A @ B + A @ C)


################# Inverse of matrices #################

# The matrix that undoes what the original matrix did
# F-1(F(x)) = I
# F(F-1(Y)) = I
# So applying the function and the inverse of the function 
# gets the same value back, which is the same as leaving it untouched (Identity matrix)

# So if there is something in the matrix where information is lost or can not be retrieved
# then there is no inverse of the matrix

m = 4
n = 4

A = np.round(np.random.rand(m,n)*5)
A_inv = np.linalg.inv(A)
print(A_inv)

### What if a matrix is not square:

m = 4
n = 2

A = np.round(np.random.rand(m,n)*5)

A_inv = np.linalg.inv(A)

# This does not work, why not? Because this matrix takes 
# in 4-dimensional vectors and returns 2 dimensional vectors.
# This operation can be inverted, because the original 2d vectors 
# will not give the same result in 4D

#And the same for broad matrices: 
m = 2
n = 4

A = np.round(np.random.rand(m,n)*5)

A_inv = np.linalg.inv(A)

#How about a matrix with linearly dependent columns:

A = np.array([[1,2,0,-1],
             [2,4,9,-5],
             [3,6,-2,5],
             [4,8,-4,8]])

print(np.linalg.matrix_rank(A))

A_inv = np.linalg.inv(A)
# This gives an error: singular matrix. This means that 
# there is a nullspace that is not recovered upon the inverse

### The inverse of a diagonal matrix: 
A = np.diag([5,2,1,0.2,0.1])

A_inv = np.linalg.inv(A)

print(A @ A_inv) # a matrix times its inverse should give the identity matrix
             
##### 

