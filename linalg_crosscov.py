
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


m = 5 #5 neurons area 1
n = 8 #8 neurons area 2
p = 20 #20 timepoints

X = np.round(np.random.rand(m,p)*5)

Y = np.round(np.random.rand(n,p)*5)

covA = np.cov(X)
covB = np.cov(Y)

crosscov = np.cov(X,Y) #calculate the cross-covariance
np.shape(crosscov)
#This computes the full covariance matrix including the covariance matrix of the matrices
#X and Y themselves. It appends matrices and then np.cov
plt.figure()
plt.imshow(crosscov)


# If X is rank m x p, then np.cov(X, rowvar=False) is equal
# to S after

cX = X - X.mean(axis=0)[np.newaxis,:]
S = np.dot(cX.T, cX)/(n-1.)


# If we also have Y rank n x p, then the upper p x p
# quadrant of

np.cov(X, y=Y, rowvar=True)

# is equal to S after

XY = np.vstack((X,Y))
cXY = XY - XY.mean(axis=1)[:,np.newaxis]
S = np.dot(cXY, cXY.T)/(p-1.)

# Thus we can see that the total covariance is composed
# of four parts:

covXX = S[:m,:m] # upper left
covXY = S[:m,m:] # upper right
covYX = S[m:,:m] # lower left
covYY = S[m:,m:] # lower right

# Often we just want the upper-right p x p quadrant. Thus
# we can save 75 % of the cpu time by not computing the rest.

