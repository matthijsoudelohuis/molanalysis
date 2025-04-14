import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA

def get_powerlaw(ss, trange):
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
    
    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha,ypred

def shuff_cvPCA(X, nshuff=5):
    ''' X is 2 x stimuli x neurons '''
    nc = min(1024, X.shape[1])
    ss=np.zeros((nshuff,nc))
    for k in range(nshuff):
        iflip = np.random.rand(X.shape[1]) > 0.5
        X0 = X.copy()
        X0[0,iflip] = X[1,iflip]
        X0[1,iflip] = X[0,iflip]
        ss[k]=cvPCA(X0)
    return ss

def cvPCA(X):
    ''' X is 2 x stimuli x neurons '''
    # do PCA on data
    pca = PCA(n_components=min(1024, X.shape[1])).fit(X[0].T)
    # get the components
    u = pca.components_.T
    # get the singular values
    sv = pca.singular_values_
    
    # project train data onto components
    xproj = X[0].T @ (u / sv)
    # project train data onto components
    cproj0 = X[0] @ xproj
    # project test data onto components
    cproj1 = X[1] @ xproj
    
    # compute the correlation between the two sets of components
    ss = (cproj0 * cproj1).sum(axis=0)
    return ss

