
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.linalg as LA

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


# The inverse of a diagonal matrix is just 1/matrix
# Every diagonal element is a scaling factor of that axis.
# So undoing the scaling means 1/scalar
A_inv = np.diag(1/np.diag(A))

print(A @ A_inv) # a matrix times its inverse should give the identity matrix


##### Eignevalues and signluar values


A = np.array([[1,2,0,-1],
             [2,-1,9,-2],
             [3,6,-2,5],
             [4,8,-4,8]])

eigenvalues,eigenvectors = LA.eig(A)

# Eigenvectors describe orthogonal axes along which the matrix is stretched. 
# The amount of stretching along each axis is the eigenvalue. 

# The matrix can thus be described as converting from the standard unit vectors
# to the eigenbasis, scaling by the eigenvalues and converting back.
# Converting back and to is unintuitive, because the inverse describes how to go TO
# the eigenbasis.

print(A)
print(np.round(eigenvectors @ np.diag(eigenvalues) @ LA.inv(eigenvectors)))

# Eigenvectors describe along which dimensions the input is merely scaled.

print(eigenvectors[:,0]) #The eigenvector as input
print(A @ eigenvectors[:,0]) #what does A do to the eigenvector: simply scale
print(eigenvalues[0] * eigenvectors[:,0]) #thus eigenvector scaling can 
                                        # be described as scaling by eigenvalue 0

# Because any vector that is a multiple of an eigenvector is also an 
# eigenvector, the eigenvectors are unit normalized: 
LA.norm(eigenvectors,axis=0) #the norms are one
# However, the eigenvectors don't have to be orthogonal to each other:
np.dot(eigenvectors[:,0],eigenvectors[:,1]) #column wise
np.dot(eigenvectors[0,:],eigenvectors[1,:]) #row wise

# That is becuse it can be stretched along non-orthogonal axes.
# However, if eigendecomposition is performed on a symmetric matrix, 
# then the eigenvectors are orthogonal to each other.

eigenvalues,eigenvectors = LA.eig(A.T @ A)
np.dot(eigenvectors[:,0],eigenvectors[:,1]) #column wise
np.dot(eigenvectors[0,:],eigenvectors[1,:]) #row wise

################# QR decomposition #################
# To get an orthonormal basis, perform QR decomposition:
q,r = LA.qr(A) #get orthonormal axis
LA.norm(q,axis=0) #the norms are one
np.dot(q[:,0],q[:,1]) #column wise

# How do eigenvalues and singular values relate to each other?
# Eigenvalues and eigenvectors can only be computed for square matrices. 
# So let's take a non-square matrix:
A = np.array([[1,2,0], #same matrix A as before
             [2,-1,9],
             [3,6,-2],
             [4,8,-4]])

eigenvalues,eigenvectors = LA.eig(A) #error because must be square

#So how are the dimensions of a non-square matrix computed that 
# describe how the matrix is stretched? This is done by applying 
# kind of eigendecomposition on the two covariance matrices of A:
# A.T @ A and A @ A.T

# Let's say A is ordered by samples x features
# The first covariance matrix describes the covariance between the 
# covariance between features: (for cov should be normalized by sqrtN)
covA1 = A.T @ A
print(covA1)
#It is a 3x3 matrix because we have 3 features

# The second covariance matrix describes the covariance 
# between the samples and is 4 by 4:
covA2 = A @ A.T
print(covA2)

evalA1,evecA1 = LA.eig(covA1) #
evalA2,evecA2 = LA.eig(covA2) #

# Because this is a symmetric matrix the eigenvalues are guaranteed, 
# furthermore, they will be orthogonal to each other. 
# SO: the transpose of the eigenvectors times eigenvectors is identity matrix:
print(np.round(evecA1.T @ evecA1,2))

# The eigenvalues of A1 and A2 have both 3 nonzero elements. Why is this?
# Well the covariance of the rows gives a 4 x 4 matrix, but of rank 3. 
# There is one nullspace dimension in which nothing gets stretched.
# In fact the eigenvalues of covA1 and covA2 are the same.
# Why is this? I don't know. 
# The number of eigenvalues is therefore lower or equal than min(m,n)

# The eigenvectors of covA1 and covA2 thus both describe which vectors
# will only be stretched in their covariance matrix. 
# Importantly they can be thought of as the input that in some low rank 
# space of the matrix is exactly aligned. Then it gets scaled and mapped onto 
# the output eigenvectors of the matrix.
# So the SVD is basically using eigendecomposition on both sides of the matrix:
# Left and right eigendecomposition. The scaling is then not the eigenvalues, 
# but the singular values which are the sqrt of the eigenvalues..

U,S,Vh = LA.svd(A)

#The order of the eigen/singular values doesn't necessarily 
# have to be the same:
print(evalA1)
print(S**2)

# So the right singular vectors are the eigenvectors of covA1:
print(evecA1)
print(Vh.T)

#Two things can be different however. The sign of the vectors can be different.
# and the ordering of the eigenvectors can be different.
# The first is because the combination of positive and negative singular 
# vector pairs is arbitrary: the XXX problem. There is no unique solution.
# The ordering is because the eigenvalues don't come out sorted. 
# If you sort the output of both algorithms than you would get the same. 

# Similarly, the left singular vectors are the eigenvectors of covA2:
print(evecA2)
print(U)

###  Actually we did not compute the covariance with the transpose operations. 
# For that we need to subtract the column mean and divide by number of features
# as so: 
Asub = A - np.mean(A,axis=0)
covA2   = ( Asub.T @ Asub ) / (np.shape(Asub)[0]-1)
cov2    = np.cov(A,rowvar=False)
print(covA2)
print(cov2)
Asub = (A.T - np.mean(A,axis=1)).T
covA1 = ( Asub @ Asub.T ) / (np.shape(Asub)[1]-1)
cov1= np.cov(A,rowvar=True)
print(covA1)
print(cov1)

evalA1,evecA1 = LA.eig(covA1) 
evalA2,evecA2 = LA.eig(covA2) #
# Now the eigenvalues don't correspond, because the matrix was 
# normalized differently to compute the covariance (column/row subtr)

##### PCA ####
from sklearn.decomposition import PCA

# Perform eigendecomposition on the covariance matrix:
evals,evecs = LA.eig(np.cov(A,rowvar=False))
srt = np.argsort(evals)[::-1] # indices of eigenvalues from largest to smallest
evals = evals[srt] # sorted eigenvalues
evecs = evecs[:, srt] # sorted eigenvectors

print(evals)

#Perform 'PCA':
pca = PCA()
pca.fit(A)
print(pca.explained_variance_) #These are the eigenvalues

#The eigenvectors:
print(pca.components_.T)
print(evecs)
#These might not be the same because of 

# The variance and cumulative variance explained by each PC
total_var = evals.sum()
var_exp = [(v / total_var)*100 for v in evals]
cumulative_var = [np.sum(var_exp[0:i]) for i in range(1, len(var_exp)+1)]

pca.singular_values_
pca.components_

# The number of components can be estimated using several methods: 

A = np.random.randn(100,10)

pca = PCA(n_components=0.9)
pca.fit(A)
print(pca.n_components_)
pca = PCA(n_components=0.95)
pca.fit(A)
print(pca.n_components_)

pca.explained_variance_ratio_


B = np.random.randn(100,2)
C = np.random.randn(2,10)
A = B @ C

pca = PCA(n_components='mle')
pca.fit(A)
print(pca.n_components_)

########### Ordinary Least Squares ###########

A = np.array([[2,-1],
              [1,2],
              [1,1]])

b = np.array([[2,1,4]]).T

#Solve for x: Ax = b

# here we are using the fact that b when going through A.T 
# is forced through the column space of A and when A*x does
#  the same, it is the projection of b onto the subspace of A, 
# which is the closest you can get to a solution.

# OLS:
# A.T @ A @ x = A.T @ b
# Right hand side is simple matrix vector product
# Left hand side, have to get A.T @ A ('covariance matrix') and 
# multiply both sides with inverse of this:
# A.T @ A always has an inverse
x = LA.inv(A.T @ A) @ A.T @ b
print(x)
print(b)
print(A @ x)

# Numpy linalg has the following function to find the 
# least square solution:
x,res,rank,s = LA.lstsq(A,b,rcond=None)
print(x)

### Another example:
# Fitting a line through 4 points in 2D
# Very simple regression, but now using matrix algebra to solve: 
# Four points in 2D: (-1,0), (0,1), (1,2) and (2,1)

x = np.array([-1, 0, 1, 2])
y = np.array([0, 1, 2, 1])

# By examining the coefficients, we see that the line should have a
# gradient of roughly 1 and cut the y-axis at, more or less, -1.

# We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
# and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:

A = np.vstack([x, np.ones(len(x))]).T

m = LA.inv(A.T @ A) @ A.T @ y
print(m)

m = np.linalg.lstsq(A, y, rcond=None)[0]

_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, m[0]*x + m[1], 'r', label='Fitted line')
_ = plt.legend()

# an alternative way of defining linear regression:

# Finding beta in a different way:
beta1 = x.reshape(-1,1).T @ y / LA.norm(x)**2
# and with an intercept:
xsub = x - np.mean(x)
ysub = y - np.mean(y)
beta1 = xsub.reshape(-1,1).T @ ysub / LA.norm(xsub)**2
beta0 = np.mean(y) - beta1*np.mean(x)

_ = plt.plot(x, beta1*x + beta0 , 'b', label='Reg line')


# Importantly, when defined this way we see that X @ Y is the inner product
# of X and Y (dot product) divided by the variance of x. So how much do 
# you go up in Y for going up one unit in X?
# This is the same as the correlation (covariance divided by variance 
# of each and this can be rewritten as:
# B1 = cov(x,y) /  var(x) = cor(x,y) / sqrt(var(y) / var(x))

# This holds true for univariate linear regression. In the case of 
# multivariate regression, this is only true if the predictors are orthogonal.

# This means that you could get the coefficients in an iterative procedure,
# you orthogonalize the predictors in each step when finding the coefficients. 
# I.e. you regress each time on the residuals. You subtract whichever pred
# dimension you found and then regress again. 
# Claim: this is exactly the same as you find when applying multivariate 
# linear regression using the matrix products above.
# Why? Well, if we have orthogonal predictors, then finding the coefficient by 
# linear regression and removing it, leaves the rest of the regression steps 
# unaffected. So the next step again finds any variability in Y related to one 
# orhtogonal dimension in X.

### How is this all related to projection geometrically? #####

# Now project the datapoints onto the fitted line:
# We are formulating the regression line as a subspace in 2D
# The projection matrix is simply the coefficient matrix m
#note that we add 1 to the input space for the intercept.
_ = plt.plot(x, A @ m, 'o',c='r', label='Projected data', markersize=10)

# Projection matrices: 
# If L is some subspace in Rn, then the span of L is the liear combination of 
# vectors v1 to vk, each in Rn. If V contains K linearly indpendent column
# vectors v than make up its column space, col(V) is the span of L.

# Any inputs in Rn can be mapped onto the subspace L. This is a linear 
# transformation from Rn to Rn. 
# This function has a projection map that is Rn by Rn. Furthermore, it is 
# symmetric (P.T = P) and idempotent P**2 = P)
# For any inputs within this subspace Px = x, and for any inputs orthogonal
# Px = 0

#Importantly, the linear regression fit yhat is exactly the projecion of y 
# onto the subspace of X. 
# if yhat = X * B = X @ [(X.T @ X)-1 @ X.T @ y] where brackets indicate P projmat
# So we have defined the projection map

# This gives some interesting insights actually. If any vector y that lies within 
# within the subspace is unaffected by the regression (i.e. is perfectly predicte).
# However, any vector that is orthogonal is fully unpredicted. 
# The orthogonal complement of the subspace and the subspace itself fully span Rn,
# i.e. Pl + Pl-comp = Identity matrix. 
# That means that for each y you can get the residual by projecting with the 
# orthogonal complement of the subspace L.
# y-yhat is orthogonal to each column of X! 
# projection is: A @ (A.T @ A)-1 @ A.T @ x



###### So what about reduced rank regression #####################
# Here let's start with multivariate regression:

n = 20
p = 5

X = np.random.randn(n,p)
B = np.random.randn(p,1)
Y = X @ B + np.random.randn(n,1)*0.1

Bhat = np.linalg.lstsq(X, Y, rcond=None)[0]

print(B)
print(Bhat)

# Now let's introduce a multivariate output Y 
# (e.g. some neural population that we are trying to predict)

n = 20
p = 5
m = 20

X = np.random.randn(n,p)
B = np.random.randn(p,m)
Y = X @ B + np.random.randn(n,m)*0.1

Bhat = np.linalg.lstsq(X, Y, rcond=None)[0]

# Bhat = LA.inv(X.T @ X) @ X.T @ Y #or by matrix multiplication

fig,axes = plt.subplots(1,2,figsize=(8,16))
axes[0].imshow(B)
axes[1].imshow(Bhat)

# Now B is a matrix of coefficients of p by m which give how Y is
# predicted as a series of linear combinations of X and B. 
# Each column of B describes as a linear combination of predictors that 
# predicts the first column m1 in Y. Each row of B can be thought of 
# as how the first predictor influences the different Y variables. 

# Multivariate linear regression finds p coefficients 
# that predict how X is related onto Y

# Now let's look at reduced rank regression:

# Instead of matrix B, we are going to fit the outer product of two 
# matrices W and L, which are p x r and r x m
# W describes how each of the ranks r maps onto the different predictors
# and L describes how the ranks map onto the Y variables

import scipy as sp
from scipy import linalg
from copy import copy

# W = np.random.randn(p,r)
# L = np.random.randn(r,m)
# Bhat = W @ L

# So let's define normal multiple linear regression 
# with the option of ridge regression regularizing parameter labmda:
def LM(Y, X, lam=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    Y_hat = X @ B_hat
    return B_hat


# Let's take our same example data again, but a little bit higher dim:
n = 200
p = 50
m = 200

X = np.random.randn(n,p)
B = np.random.randn(p,m)
Y = X @ B + np.random.randn(n,m)*0.1

#We can also calculate some error metric:
def Rss(Y, Yhat, normed=True):
    """ evaluate (normalized) model error """
    e = Yhat - Y
    Rss = np.trace(e.T @ e)
    if normed:
        Rss /= Y.shape[0]
    return Rss

rss = Rss(Y, Yhat, normed=True)

print(np.linalg.norm(Y-Yhat, 'fro'))

e = Yhat - Y
autocov = e.T @ e
print(np.trace(autocov))
np.sum(autocov[np.eye(20)==1])

def low_rank_approx(A, r, mode='left'):
    """ calculate low rank approximation of matrix A and 
    decomposition into L and W """
    # decomposing and low rank approximation of A
    U, s, Vh = linalg.svd(A)
    S = linalg.diagsvd(s,U.shape[0],s.shape[0])

    # L and W
    if mode is 'left':
        L = U[:,:r] @ S[:r,:r]
        W = Vh[:r,:]
    if mode is 'right':
        L = U[:,:r] 
        W = S[:r,:r] @ Vh[:r,:]
    
    return L, W

def RRR(Y, X, B_hat, r, mode='left'):
    """ reduced rank regression by low rank approx of B_hat """
    L, W = low_rank_approx(B_hat,r, mode=mode)
    B_hat_lr = L @ W
    Y_hat_lr = X @ B_hat_lr
    return B_hat_lr


def xval_rank(Y, X, lam, ranks, K=5):
    # K = 5 # k-fold xval
    # ranks = list(range(2,7)) # ranks to check

    ix = np.arange(Y.shape[0])
    sp.random.shuffle(ix)
    ix_splits = chunk(ix,K)

    Rsss_lm = np.zeros(K) # to calculate the distribution
    Rsss_rrr = np.zeros((K,len(ranks))) # to evaluate against

    # k-fold
    for k in tqdm(range(K),desc='xval rank'):
        # get train/test indices
        l = list(range(K))
        l.remove(k)
        train_ix = np.concatenate([ix_splits[i] for i in l])
        test_ix = ix_splits[k]

        # LM error
        B_hat = LM(Y[train_ix], X[train_ix], lam=lam)
        Y_hat_test = X[test_ix] @ B_hat
        Rsss_lm[k] = Rss(Y[test_ix], Y_hat_test)

        # RRR error for all ranks
        for i,r in enumerate(tqdm(ranks)):
            B_hat_lr = RRR(Y[train_ix], X[train_ix], B_hat, r)
            Y_hat_lr_test = X[test_ix] @ B_hat_lr
            Rsss_rrr[k,i] = Rss(Y[test_ix], Y_hat_lr_test)

    return Rsss_lm, Rsss_rrr



# we could get the bhat matrix by direct multiple linear regression:
Bhat_lm = np.linalg.lstsq(X, Y, rcond=None)[0]
Yhat_lm = X @ Bhat

# LM model run with ridge regression (regularization):
Bhat_rr = LM(Y, X, lam=0.01)
Yhat_rr = X @ Bhat_rr

#Now we can apply reduced rank regression:
r = 3
# RRR
Bhat_rrr = RRR(Y, X, Bhat_rr, r=r)
Yhat_lr = X @ Bhat_rrr

fig,axes = plt.subplots(4,1,figsize=(10,8))
axes[0].imshow(B)
axes[1].imshow(Bhat_lm)
axes[2].imshow(Bhat_rr)
axes[3].imshow(Bhat_rrr)

# Based on the fit parameters we can reconstruct Y:


L, W = low_rank_approx(Bhat_lr,r)
l = copy(L[1:]) # wo intercept

print(np.shape(L))
print(np.shape(W))


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X               = scipy.randn(num, dimX)
    W               = scipy.dot(scipy.randn(dimX, rrank), scipy.randn(rrank, dimY))
    Y               = scipy.dot(X, W) + scipy.randn(num, dimY) * noise
    return X, Y
