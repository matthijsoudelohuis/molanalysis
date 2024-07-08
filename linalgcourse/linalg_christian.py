# Probing Christian with linalg questions

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from scipy import linalg

m = 5 
n = 8 

X = np.round(np.random.randn(m,n)*5)

# Why are the eigenvalues of A.T @ A and A @ A.T the same?

print(X.T @ X)
eigvals1, eigvecs1 = np.linalg.eig(X.T @ X)
print(X @ X.T)
eigvals2, eigvecs2 = np.linalg.eig(X @ X.T)

# The intuition behind this is that in the end the way the matrix is, depends
# on both the rows and the columns, how the elements are distributed. 
# If you tranpose the matrix, the rows and columns get flipped, but their relation
# is still the same. 
# For a matrix A that is mxn, eigendecomposition finds axes for which 
# inputs (inputs are linear combinations of the columns of A, m x 1) 
# merely get stretched. The eigenvalues indicate the amount of stretching. 
# If you transpose the matrix eigendecomposition finds the inputs (nx1), 
# where the inputs are again combination of columns (but rows of original A)
# along which the input is stretched. The fact that the eigenvalues of A.T and A
# are the same is similar to the fact that the rank(A) = rank(A.T)
# Another way of seeing this is that they share the same characteristic polynomial. 
# The eigenvectors are different however (combinations of rows, versus combinations
# of columns of A)

U,S,V = np.linalg.svd(X) # U is mxm, S is mx1, V is nxn

#If you want to project the data along the first principal dimension:

Xp = U[:,:1].T @ X

#if you want to reconstruct the data using the first two dimensions of X:
Xhat = U[:,:1] @ U[:,:1].T @ X
plt.imshow(X)
plt.imshow(U[:,:2] @ U[:,:2].T @ X)

# That the eigenvalues/eigenvectors relate to the singular values / vectors 
# makes sense if you look at how they are related: 
# if X = USV.T
# then X.T @ X = V @ S.T @ U.T @ U @ S @ V.T
# then U.t @ U cancel out
# so: X.T @ X = V @ S.T @ S @ V.T
# this means that for diagonal matrices S.T @ S the singular values are squared
# and X.T @ X is of the form P D P-1 (which is eigendecomposition)
# So singular values are the sqrt of eigenvalues of the covariance matrix of X:

U,S,V = np.linalg.svd(X) # U is mxm, S is mx1, V is nxn

print(S)
print(np.sqrt(eigvals2))

## Furthermore the eigenvectors of the covariance matrix X @ X.T are the left singular vectors
# and the eigenvectors of the covariance matrix X.T @ X are the right singular vectors

print(U)
print(eigvecs2)
np.allclose(np.abs(U), np.abs(eigvecs2))
print(V)
print(eigvecs1)
np.allclose(np.abs(V), np.abs(eigvecs1))

## What is whitening a matrix? Whitening a matrix involves an operation such that when computing 
# the covariance of the whitened matrix it returns the identity matrix I.
# So any dependencies between columns and rows are removed and everything is scaled to unit vectors.
# So what is happening: 
# The idea is that for some matrix X, there exists transformation C where:
# Xw = C @ X where cov(Aw) = I (unit diagonal covariance)

# so let's construct some low rank matrix: 

m = 12
n = 20 
r = 3

X = np.random.randn(m,r) @ np.random.randn(r,n) * 10

X = X - X.mean(axis=1)[:,np.newaxis]

plt.imshow(X)

plt.imshow(np.cov(X))

# Now the condition C.T @ C = S-1 (inverse ofthe covariance)
# So basically you want to undo / cancel any covariance that exists in X. 

U, s, Vt = np.linalg.svd(X, full_matrices=False)

# U and Vt are the singular matrices, and s contains the singular values.
# Since the rows of both U and Vt are orthonormal vectors, then U * Vt
# will be white
X_white = np.dot(U, Vt)

plt.imshow(X_white)

plt.imshow(np.cov(X_white))

# So basically you are undoing any stretching of the matrix (you are omitting
#  the singular values). This is pretty interesting. 

# sklearn pca also has an option to whiten your data

from sklearn.decomposition import PCA
pca = PCA(whiten=True)
whitened = pca.fit_transform(X)
plt.imshow(np.cov(whitened))

# These transformations of the data, however, don't preserve the scale. So generate data
# in arbitrary units.

# Even better:
##### There is whitening with PCA, but also with ZCA. let's see the difference: 

# Create data #let's create multivariate Gaussian distributed data
np.random.seed(1)
mu = [0,0]
sigma = [[6,5], [5,6]]
n = 1000
x = np.random.multivariate_normal(mu, sigma, size=n)
print('x.shape:', x.shape, '\n')
 
# Zero center data
xc = x - np.mean(x, axis=0)
print(xc.shape)
xc = xc.T
print('xc.shape:', xc.shape, '\n')
 
# Calculate Covariance matrix
# Note: 'rowvar=True' because each row is considered as a feature
# Note: 'bias=True' to divide the sum of squared variances by 'n' instead of 'n-1'
xcov = np.cov(xc, rowvar=True, bias=True)
print(xcov) #should be close to original simga:
print(sigma) 
 
# Calculate Eigenvalues and Eigenvectors
w, v = linalg.eig(xcov) # .eigh()
# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits
print("Eigenvalues:\n", w.real.round(4), '\n')
print("Eigenvectors:\n", v, '\n')

# Calculate inverse square root of Eigenvalues
# Optional: Add '.1e5' to avoid division errors if needed
# Create a diagonal matrix
diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
diagw = diagw.real.round(4) #convert to real and round off
print("Diagonal matrix for inverse square root of Eigenvalues:\n", diagw, '\n')
 
# Calculate Rotation (optional)
# Note: To see how data can be rotated
xrot = np.dot(v, xc)
 
# Whitening transform using PCA (Principal Component Analysis)
wpca = np.dot(np.dot(diagw, v.T), xc)
# The whitened data is computed by counteracting the eigenvalues of the data
# So projecting the data through the eigenvectors and performing the inverse of the eigenscaling

# this is similar but not the same as: 
# U,S,V = np.linalg.svd(xc,full_matrices=False)
# wpca = np.dot(U,V) #here the 
 
# Whitening transform using ZCA (Zero Component Analysis)
wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
 
 
fig = plt.figure(figsize=(10,3))
 
plt.subplot(1,4,1)
plt.scatter(x[:,0], x[:,1], label='original', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.subplot(1,4,2)
plt.scatter(xc[0,:], xc[1,:], label='centered', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.subplot(1,4,3)
plt.scatter(wpca[0,:], wpca[1,:], label='wpca', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.subplot(1,4,4)
plt.scatter(wzca[0,:], wzca[1,:], label='wzca', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.tight_layout()


##### How does regularization of multiple linear regression involve the identity matrix: 
# Y = XB: 
# and we can find B through OLS like so:
# B = (X.T @ X)-1 @ X.T @ Y
# The loss function is described as || Y - Yhat ||2 (which is the frobenius norm, L2 norm, each element squared)
# If you want to add a regularization term you add it to the end: 
# || Y - XB ||2 + lambda || B ||2
# Solving this equation involves setting the derivative with respect to B to zero. 
# The derivative of matrices to zero is somewhat similar to finding derivatives of scalars. 
# d/dB = || Y - XB ||2 + lambda || B ||2 = 0
# d/dB = deriv Y*2 - 2XBY + X*2B*2 + lambda B*2 
# d/dB = - 2XY + 2BX*2 + lambda 2B
# d/dB = -XY + BX*2 + lambda B (divide by 2)
# let's say that Y = m x T, X = m x p and B is p x T, then let's rearrange to get good dims (sloppy)
# d/dB = - X.T @ Y + B (X.T @ X + lambda @ I) (take the inverse of expression right) and move B over to the right:
# B = (X.T @ X + lambda I)-1 @ X.T @ Y

# This gives you an easy solution to OLS with L2 regularization. L1 regularization on the other hand
# does not allow you to compute the derivative so easily, much more work. That's why ridge reg is preffered for 
# multiple linear regression. 



### Symmetric versus non-symmetric orthogonalization
# indeed if Gram-Schmidt orthogonalization is applied to two or more vectors, then the order matters. 
# Basically for every new vector, the part this is in the direction of the previous ones is
# projected out. There is a way, called Lowdin symmetric orthogonalization to have each vector be 
# affected by a similar amount to achieve orthogonality between all vectros. 


m = 5
S = np.random.randn(m,m)

S = S.T @ S
print("Overlap matrix")
print(S)

lam_s, l_s = np.linalg.eigh(S)
lam_s = lam_s * np.eye(len(lam_s))
lam_sqrt_inv = np.sqrt(np.linalg.inv(lam_s))
symm_orthog = np.dot(l_s, np.dot(lam_sqrt_inv, l_s.T))

print("Symmetric orthogonalization matrix")
print(symm_orthog)

# Covariances and variances in matrix form: 
# X and Y are matrices,e.g. two area data


m = 15
n = 25
T = 200

X = np.random.randn(T,m)
Y = np.random.randn(T,n)


# np.cov(X,Y) = X.T @ Y
# np.cov(X) = X.T @ X
# np.cov(Y) = Y.T @ Y

# so the OLS regression we saw earlier: B = (X.T @ X)-1 @ X.T @ Y
# is basically saying take the covariance between X and Y and 
# divide by variance of X (inverse of X.T X)
# So regression is removing the effect of variance within X
# and is therefore not symmetric with respect to treating X and Y
# CCA and PLS are symmetric: 
# CCA: cov(X,Y) / cov(X)*cov(Y) you normalize by both variance in X and Y
# so you get out a correlation value, not a covariance value
# PLS: cov(X,Y) not scaled by the variance in each of the areas