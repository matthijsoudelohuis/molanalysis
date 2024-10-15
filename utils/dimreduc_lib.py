
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from utils.RRRlib import *

def remove_dim(data,remove_method,remove_rank):

    if remove_method == 'PCA':
        pca = PCA(n_components=remove_rank)
        pca.fit(data.T)
        data_T = pca.transform(data.T)
        data_hat = pca.inverse_transform(data_T).T
    elif remove_method == 'FA':
        fa = FactorAnalysis(n_components=remove_rank, max_iter=1000)
        fa.fit(data.T)
        data_T = fa.transform(data.T)
        data_hat = np.dot(data_T, fa.components_).T

    elif remove_method == 'RRR':
        X = np.vstack((sessions[ises].respmat_runspeed,sessions[ises].respmat_videome))[:,trial_ori==ori].T
        Y = data.T
        ## LM model run
        B_hat = LM(Y, X, lam=10)

        B_hat_rr = RRR(Y, X, B_hat, r=remove_rank, mode='left')
        data_hat = (X @ B_hat_rr).T

    else: raise ValueError('unknown remove_method')

    return data_hat