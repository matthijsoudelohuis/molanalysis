"""
@author: Matthijs oude Lohuis
Champalimaud 2023

"""

import scipy as sp
import numpy as np
from scipy import linalg
from tqdm import tqdm
from scipy.optimize import minimize
from utils.psth import construct_behav_matrix_ts_F
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA as ICA

def EV(Y,Y_hat):
    e = Y - Y_hat
    ev = 1 - np.trace(e.T @ e) / np.trace(Y.T @ Y) #fraction of variance explained
    return ev

def LM(Y, X, lam=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    Y_hat = X @ B_hat
    return B_hat
    
def Rss(Y, Y_hat, normed=True):
    """ evaluate (normalized) model error """
    e = Y_hat - Y
    Rss = np.trace(e.T @ e)
    if normed:
        Rss /= Y.shape[0]
    return Rss

def low_rank_approx(A, r, mode='left'):
    """ calculate low rank approximation of matrix A and 
    decomposition into L and W """
    # decomposing and low rank approximation of A
    U, s, Vh = linalg.svd(A)
    S = linalg.diagsvd(s,U.shape[0],s.shape[0])

    # L and W
    if mode == 'left':
        L = U[:,:r] @ S[:r,:r]
        W = Vh[:r,:]
    if mode == 'right':
        L = U[:,:r] 
        W = S[:r,:r] @ Vh[:r,:]
    
    return L, W

def RRR(Y, X, B_hat, r, mode='left'):
    """ reduced rank regression by low rank approx of B_hat """
    L, W = low_rank_approx(B_hat,r, mode=mode)
    B_hat_lr = L @ W
    Y_hat_lr = X @ B_hat_lr
    return B_hat_lr


def chunk(A, n_chunks=10):
    """ split A into n chunks, ignore overhaning samples """
    chunk_size = sp.floor(A.shape[0] / n_chunks)
    drop = int(A.shape[0] % chunk_size)
    if drop != 0:
        A = A[:-drop]
    A_chunks = sp.split(A, n_chunks)
    return A_chunks

def pca_rank_est(A, th=0.99):
    """ estimate rank by explained variance on PCA """
    pca = PCA(n_components=A.shape[1])
    pca.fit(A)
    var_exp = sp.cumsum(pca.explained_variance_ratio_) < th
    return 1 + np.sum(var_exp)

def ica_orth(A, r=None):
    if r is None:
        r = pca_rank_est(A)

    I = ICA(n_components=r).fit(A.T)
    P = I.transform(A.T)
    K = A @ P
    return K

def xval_ridge_reg_lambda(Y, X, K=5):
    
    def obj_fun(lam, Y_train, X_train, Y_test, X_test):
        B_hat = LM(Y_train, X_train, lam=lam)
        Y_hat_test = X_test @ B_hat
        return Rss(Y_test, Y_hat_test)

    ix = sp.arange(X.shape[0])
    np.random.shuffle(ix)
    # sp.random.shuffle(ix)
    ix_chunks = chunk(ix, K)

    lambdas = []
    for i in tqdm(range(K),desc="xval lambda"):
        l = list(range(K))
        l.remove(i)

        ix_train = sp.concatenate([ix_chunks[j] for j in l])
        ix_test = ix_chunks[i]
        
        x0 = np.array([1])
        res = minimize(obj_fun, x0, args=(Y[ix_train], X[ix_train],
                       Y[ix_test], X[ix_test]), bounds=[[0,np.inf]],
                       options=dict(maxiter=100, disp=True))
        lambdas.append(res.x)

    return sp.average(lambdas)

def xval_rank(Y, X, lam, ranks, K=5):
    # K = 5 # k-fold xval
    # ranks = list(range(2,7)) # ranks to check

    ix = np.arange(Y.shape[0])
    np.random.shuffle(ix)
    ix_splits = np.array_split(ix,K)

    Rsss_lm = np.zeros(K) # to calculate the distribution
    Rsss_rrr = np.zeros((K,len(ranks))) # to evaluate against
    EV_rrr = np.zeros((K,len(ranks)))

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
            EV_rrr[k,i] = EV(Y[test_ix], Y_hat_lr_test)
    return Rsss_lm, Rsss_rrr,EV_rrr

def regress_out_behavior_modulation(ses,X=None,Y=None,nvideoPCs = 30,rank=2,lam=0.8):
    if X is None:
        X,Xlabels = construct_behav_matrix_ts_F(ses,nvideoPCs=nvideoPCs)

    if Y is None:
        Y = ses.calciumdata.to_numpy()
        
    assert X.shape[0] == Y.shape[0],'number of samples of calcium activity and interpolated behavior data do not match'

    ## LM model run
    B_hat = LM(Y, X, lam=lam)

    B_hat_rr = RRR(Y, X, B_hat, r=rank, mode='left')
    Y_hat_rr = X @ B_hat_rr

    Y_out = Y - Y_hat_rr
    # print("EV of behavioral modulation: %1.4f" % EV(Y,Y_hat_rr))

    return Y_out