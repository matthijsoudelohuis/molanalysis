"""
@author: Matthijs oude Lohuis
Champalimaud 2023

"""

import scipy as sp
import numpy as np
from scipy import linalg
from scipy.linalg import orth, qr, svd
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.decomposition import PCA, FactorAnalysis,FastICA
from sklearn.model_selection import KFold
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils.psth import construct_behav_matrix_ts_F
from utils.shuffle_lib import my_shuffle


def EV(Y,Y_hat):
    # e = Y - Y_hat
    # ev = 1 - np.trace(e.T @ e) / np.trace(Y.T @ Y) #fraction of variance explained
    ev = 1 - np.nanvar(Y-Y_hat) / np.nanvar(Y)
    return ev

def LM(Y, X, lam=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    # Y_hat = X @ B_hat
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

    return U[:,:r],S[:r,:r],Vh[:r,:]


def RRR(Y, X, B_hat, r):
    """ reduced rank regression by low rank approx of Y_hat """
    
    Y_hat = X @ B_hat

    U,S,V = low_rank_approx(Y_hat,r)

    # Y_hat_rr =  X @ B_hat @ U @ U.T
    Y_hat_rr =  U @ U.T @ X @ B_hat

    return Y_hat_rr,U,S,V

def RRR_cvR2(Y, X, rank,lam=0,kfold=5):
    # Input: 
    # Y is activity in area 2, X is activity in area 1

    # Function:
    # X is of shape K x N (samples by features), Y is of shape K x M
    # K is the number of samples, N is the number of neurons in area 1,
    # M is the number of neurons in area 2

    # multiple linear regression, B_hat is of shape N x M:
    # B_hat               = LM(Y,X, lam=lam) 
    #RRR: do SVD decomp of Y_hat, 
    # U is of shape K x r, S is of shape r x r, V is of shape r x M
    # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r)

    kf      = KFold(n_splits=kfold,shuffle=True)

    R2_cv_folds = np.full((kfold),np.nan)

    X                   = zscore(X,axis=0)  #Z score activity for each neuron across trials/timepoints
    Y                   = zscore(Y,axis=0)

    for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
        
        X_train, X_test     = X[idx_train], X[idx_test]
        Y_train, Y_test     = Y[idx_train], Y[idx_test]

        B_hat_train         = LM(Y_train,X_train, lam=lam)

        Y_hat_train         = X_train @ B_hat_train

        # decomposing and low rank approximation of A
        # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
        
        U, s, V = svds(Y_hat_train,k=rank,which='LM')

        B_rrr               = B_hat_train @ V.T @ V #project beta coeff into low rank subspace

        Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace

        R2_cv_folds[ikf] = EV(Y_test,Y_hat_rr_test)

    return np.nanmean(R2_cv_folds)

def RRR_wrapper(Y, X, nN=None,nM=None,nK=None,lam=0,nranks=25,kfold=5,nmodelfits=5):
    #Reduced rank regression with unknown rank: 
    # Input: 
    # Y is activity in area 2, X is activity in area 1

    # Function:
    # X is of shape K x N (samples by features), Y is of shape K x M
    # K is the number of samples, N is the number of neurons in area 1,
    # M is the number of neurons in area 2

    # multiple linear regression, B_hat is of shape N x M:
    # B_hat               = LM(Y,X, lam=lam) 
    #RRR: do SVD decomp of Y_hat, 
    # U is of shape K x r, S is of shape r x r, V is of shape r x M
    # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r)

    kf      = KFold(n_splits=kfold,shuffle=True)

    # Data format: 
    K,N     = np.shape(X)
    M       = np.shape(Y)[1]

    nN  = nN or min(M,N)
    nM  = nM or nN or min(M,N)
    nK  = nK or K

    assert nM<=M and nN<=N, "number of subsampled neurons must be smaller than M and N"
    assert nK<=K, "number of subsampled timepoints must be smaller than number of timepoints"
    if nK/np.min([nN,nM]) < 5 and lam==0:
        print('Warning: number of samples per feature (e.g. trials per neuron) is less than 5 and no regularization is applied. This may result in overfitting.')

    R2_cv_folds = np.full((nranks,nmodelfits,kfold),np.nan)

    for imf in range(nmodelfits):
        idx_areax_sub           = np.random.choice(N,nN,replace=False)
        idx_areay_sub           = np.random.choice(M,nM,replace=False)

        X_sub                   = X[:,idx_areax_sub]
        Y_sub                   = Y[:,idx_areay_sub]
        
        X_sub                   = zscore(X_sub,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y_sub                   = zscore(Y_sub,axis=0)
    
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X_sub)):
            
            X_train, X_test     = X_sub[idx_train], X_sub[idx_test]
            Y_train, Y_test     = Y_sub[idx_train], Y_sub[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=lam)

            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of A
            # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
            
            U, s, V = svds(Y_hat_train,k=np.min((nranks,nN,nM))-1,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            S = linalg.diagsvd(s,U.shape[0],s.shape[0])

            for r in range(nranks):
                B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace

                Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace

                R2_cv_folds[r,imf,ikf] = EV(Y_test,Y_hat_rr_test)

    repmean,rank = rank_from_R2(R2_cv_folds.reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)

    return repmean,rank,R2_cv_folds

def RRR_decompose(Y, X, B, S, nN=None,nM=None,nK=None,lam=0,nranks=25,kfold=5,nmodelfits=5):
    #Reduced rank regression with unknown rank: 
    # Input: 
    # Y is activity in area 2, X is activity in area 1
    # B is behavioral activity
    # S is stimulus condition

    # Function:
    # X is of shape K x N (samples by features), Y is of shape K x M
    # B is of shape K x O (samples by behavioral features), S is of shape K x 1
    # K is the number of samples, N is the number of neurons in area 1,
    # M is the number of neurons in area 2

    # multiple linear regression, B_hat is of shape N x M:
    # B_hat               = LM(Y,X, lam=lam) 
    #RRR: do SVD decomp of Y_hat, 
    # U is of shape K x r, S is of shape r x r, V is of shape r x M
    # Y_hat_rr,U,S,V     = RRR(Y, X, B_hat, r)
    
    # Compute a stimulus related and behavior related subspace in area 2 and decompose the predicted 
    # activity into stimulus related, behavioral related or unknown

    kf      = KFold(n_splits=kfold,shuffle=True)

    # Data format: 
    K,N     = np.shape(X)
    M       = np.shape(Y)[1]

    nN  = nN or min(M,N)
    nM  = nM or nN or min(M,N)
    nK  = nK or K

    assert nM<=M and nN<=N, "number of subsampled neurons must be smaller than M and N"
    assert nK<=K, "number of subsampled timepoints must be smaller than number of timepoints"
    if nK/np.min([nN,nM]) < 5 and lam==0:
        print('Warning: number of samples per feature (e.g. trials per neuron) is less than 5 and no regularization is applied. This may result in overfitting.')

    R2_cv_folds         = np.full((4,nranks,nmodelfits,kfold),np.nan) # 4 subspaces: full,stim,behav,unknown

    # U_stim,pca          = compute_stim_subspace(Y, S, n_components=5)
    # U_behav             = compute_behavior_subspace_linear(Y, B, n_components=5)

    # U_stim_orth2, U_behav_orth2 = orthogonalize_subspaces(U_stim.T, U_behav.T)
    # U_stim_orth2, U_behav_orth2 = U_stim_orth2.T, U_behav_orth2.T
    # #order of orthogonalization matters so do for both orderings:
    # U_stim_orth1, U_behav_orth1 = orthogonalize_subspaces(U_behav.T,U_stim.T)
    # U_stim_orth1, U_behav_orth1 = U_stim_orth1.T, U_behav_orth1.T

    for imf in range(nmodelfits):
        idx_areax_sub           = np.random.choice(N,nN,replace=False)
        idx_areay_sub           = np.random.choice(M,nM,replace=False)

        X_sub                   = X[:,idx_areax_sub]
        Y_sub                   = Y[:,idx_areay_sub]
        
        X_sub                   = zscore(X_sub,axis=0)  #Z score activity for each neuron across trials/timepoints
        Y_sub                   = zscore(Y_sub,axis=0)
    
        U_stim,pca              = compute_stim_subspace(Y_sub, S, n_components=None)
        U_behav                 = compute_behavior_subspace_linear(Y_sub, B, n_components=int(nM**0.35))

        U_stim_orth2, U_behav_orth2 = orthogonalize_subspaces(U_stim.T, U_behav.T)
        U_stim_orth2, U_behav_orth2 = U_stim_orth2.T, U_behav_orth2.T
        #order of orthogonalization matters so do for both orderings:
        U_stim_orth1, U_behav_orth1 = orthogonalize_subspaces(U_behav.T,U_stim.T)
        U_stim_orth1, U_behav_orth1 = U_stim_orth1.T, U_behav_orth1.T

        for ikf, (idx_train, idx_test) in enumerate(kf.split(X_sub)):
            
            X_train, X_test     = X_sub[idx_train], X_sub[idx_test]
            Y_train, Y_test     = Y_sub[idx_train], Y_sub[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=lam)

            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of A
            # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
            U, s, V = svds(Y_hat_train,k=np.min((nranks,nN,nM))-1,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            # S = linalg.diagsvd(s,U.shape[0],s.shape[0])

            for r in range(nranks):
                B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace

                Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace

                R2_cv_folds[0,r,imf,ikf] = EV(Y_test,Y_hat_rr_test)
# 
                # Y_hat_stim1          = project_onto_subspace(Y_hat_rr_test, U_stim_orth1[:,idx_areay_sub])
                Y_hat_stim1          = project_onto_subspace(Y_hat_rr_test, U_stim_orth1)
                ev1                 = EV(Y_test,Y_hat_stim1)
                Y_hat_stim2          = project_onto_subspace(Y_hat_rr_test, U_stim_orth2)
                ev2                 = EV(Y_test,Y_hat_stim2)
                R2_cv_folds[1,r,imf,ikf] = np.mean([ev1,ev2])

                Y_hat_behav1          = project_onto_subspace(Y_hat_rr_test, U_behav_orth1)
                ev1                 = EV(Y_test,Y_hat_behav1)
                Y_hat_behav2          = project_onto_subspace(Y_hat_rr_test, U_behav_orth2)
                ev2                 = EV(Y_test,Y_hat_behav2)
                R2_cv_folds[2,r,imf,ikf] = np.mean([ev1,ev2])

        # print(f"Fraction of predicted variance that is behavior-related: {ev_behav / ev_total:.3f}")

    R2_cv_folds[3]     = R2_cv_folds[0] - R2_cv_folds[1] - R2_cv_folds[2]

    # repmean = 
    # repmean,rank = rank_from_R2(R2_cv_folds.reshape([nranks,nmodelfits*kfold]),nranks,nmodelfits*kfold)

    return R2_cv_folds


def rank_from_R2(data,nranks,nrepetitions):
    """
    find rank at which performance first exceeds max performance minus std across repetitions

    Parameters
    ----------
    data : array of shape (nranks,nrepetitions)
        data to evaluate
    nranks : int
        number of ranks
    nrepetitions : int
        number of repetitions

    Returns
    -------
    rank : int
        rank at which performance first exceeds max performance minus std across repetitions
    """
    assert(data.shape==(nranks,nrepetitions)), 'input data must be of shape (nranks,nrepetitions)'
    
    #find max performance across ranks in the average across oris,models and folds
    repmean    = np.nanmean(data,axis=1)
    maxperf    = np.nanmax(repmean)

    #find variance across repetitions
    repsem      = np.nanmean(np.nanstd(data,axis=0)) / np.sqrt(nrepetitions)

    rank        = np.where(repmean > (maxperf- repsem))[0][0]
    # rank        = np.where(rank==0,np.nan,rank)

    return repmean[rank],rank

def RRR_depricated(Y, X, B_hat, r, mode='left'):
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

    I = FastICA(n_components=r).fit(A.T)
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

def regress_out_behavior_modulation(ses,X=None,Y=None,nvideoPCs = 30,rank=None,nranks=None,lam=0,perCond=False,kfold = 5):
    
    if X is None:
        idx_T   = np.ones(len(ses.trialdata),dtype=bool)
        X       = np.stack((ses.respmat_videome[idx_T],
                        ses.respmat_runspeed[idx_T],
                        ses.respmat_pupilarea[idx_T],
                        ses.respmat_pupilx[idx_T],
                        ses.respmat_pupily[idx_T]),axis=1)
        X       = np.column_stack((X,ses.respmat_videopc[:nvideoPCs,idx_T].T))
        X       = zscore(X,axis=0,nan_policy='omit')

        si      = SimpleImputer()
        X       = si.fit_transform(X)

        # X,Xlabels = construct_behav_matrix_ts_F(ses,nvideoPCs=nvideoPCs)

    if Y is None:
        # Y = ses.calciumdata.to_numpy()

        Y               = ses.respmat[:,idx_T].T
        Y               = zscore(Y,axis=0,nan_policy='omit')

    assert X.shape[0] == Y.shape[0],'number of samples of calcium activity and interpolated behavior data do not match'

    if rank is None:
        if nranks is None: 
            nranks = X.shape[1]
        
        R2_cv_folds = np.full((nranks,kfold),np.nan)
        kf = KFold(n_splits=kfold,shuffle=True)
        for ikf, (idx_train, idx_test) in enumerate(kf.split(X)):
            X_train, X_test     = X[idx_train], X[idx_test]
            Y_train, Y_test     = Y[idx_train], Y[idx_test]

            B_hat_train         = LM(Y_train,X_train, lam=lam)

            Y_hat_train         = X_train @ B_hat_train

            # decomposing and low rank approximation of A
            # U, s, V = linalg.svd(Y_hat_train, full_matrices=False)
            U, s, V = svds(Y_hat,k=nranks,which='LM')
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]
    
            S = linalg.diagsvd(s,U.shape[0],s.shape[0])
            for r in range(nranks):
                B_rrr               = B_hat_train @ V[:r,:].T @ V[:r,:] #project beta coeff into low rank subspace
                Y_hat_rr_test       = X_test @ B_rrr #project test data onto low rank predictive subspace
                R2_cv_folds[r,ikf] = EV(Y_test,Y_hat_rr_test)

        repmean,rank = rank_from_R2(R2_cv_folds,nranks,kfold)

    if perCond:
        Y_hat_rr = np.zeros(Y.shape)
        conds = np.unique(ses.trialdata['stimCond'])
        for ic in conds:
            idx_c = ses.trialdata['stimCond']==ic

            B_hat           = LM(Y[idx_c,:],X[idx_c,:],lam=lam)

            Y_hat           = X[idx_c,:] @ B_hat

            # decomposing and low rank approximation of Y_hat
            # U, s, V = linalg.svd(Y_hat)
            U, s, V = svds(Y_hat,k=rank)
            U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

            S = linalg.diagsvd(s,U.shape[0],s.shape[0])

            #construct low rank subspace prediction
            Y_hat_rr[idx_c,:]       = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]

        Y_out           = Y - Y_hat_rr #subtract prediction
    else:

        B_hat           = LM(Y,X,lam=lam)

        Y_hat           = X @ B_hat

        # decomposing and low rank approximation of Y_hat
        # U, s, V = linalg.svd(Y_hat,full_matrices=False)
        U, s, V = svds(Y_hat,k=rank)
        U, s, V = U[:, ::-1], s[::-1], V[::-1, :]

        S = linalg.diagsvd(s,U.shape[0],s.shape[0])

        #construct low rank subspace prediction
        Y_hat_rr       = U[:,:rank] @ S[:rank,:rank] @ V[:rank,:]

        Y_out           = Y - Y_hat_rr #subtract prediction

    # print("EV of behavioral modulation: %1.4f" % EV(Y,Y_hat_rr))

    return Y,Y_hat_rr,Y_out,rank,EV(Y,Y_hat_rr)

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

def compute_stim_subspace(Y, stimulus, n_components=None):
    """
    Estimate stimulus-related subspace using PCA on mean responses.
    Y: array (samples x features)
    stimulus: array of stimulus labels (samples,)
    n_components: how many PCs to keep (default: all)
    """
    unique_stim = np.unique(stimulus)
    means = np.array([Y[stimulus == s].mean(axis=0) for s in unique_stim])  # shape: (n_conditions x n_features)

    pca = PCA(n_components=n_components)
    pca.fit(means)
    if n_components is None:
        n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9)
    components = pca.components_[:n_components,:]  # shape: (n_components x n_features)

    return components, pca

def project_onto_subspace(Yhat, subspace_basis):
    """
    Project Yhat onto the behavior-related subspace.
    subspace_basis: (k x neurons)
    """
    Yhat_centered = Yhat - Yhat.mean(axis=0)
    projection = Yhat_centered @ subspace_basis.T @ subspace_basis
    return projection

def compute_behavior_subspace_linear(Y, S, n_components=None):
    """
    Estimates behavior-related subspace using linear regression.
    Projects behavioral data S onto Y to extract subspace.
    
    Y: (samples x neurons) - true neural data
    S: (samples x behavioral features) - e.g. running, pupil, etc.
    """
    model = LinearRegression()
    model.fit(S, Y)  # Predict Y from S
    W = model.coef_  # shape: (neurons x behavioral_features)

    # The span of W.T defines the behavior-related directions in neural space
    # Perform SVD to get orthonormal basis
    U, _, _ = np.linalg.svd(W, full_matrices=False)  # shape: (features x neurons)
    
    if n_components is not None:
        U = U[:, :n_components]

    return U.T  # (n_components x neurons)

def compute_subspace_overlap(U1, U2):
    """
    Compute overlap between two subspaces U1 and U2.
    Each is (k x n_features) with orthonormal rows (subspace bases).
    
    Returns:
    - cosines: singular values = cosines of principal angles
    - mean_cosine: average overlap
    - squared_overlap: sum of squared cosines (subspace alignment metric)
    """
    # Ensure row vectors (basis vectors) are orthonormal
    M = U1 @ U2.T  # shape: (k1 x k2)
    _, s, _ = svd(M)  # s: singular values = cos(theta)

    mean_cosine = np.mean(s)
    squared_overlap = np.sum(s**2)

    return {
        'cosines': s,
        'mean_cosine': mean_cosine,
        'squared_overlap': squared_overlap
    }


def orthogonalize_subspaces(U1, U2):
    """
    Orthogonalize two subspaces U1 and U2 of different dimensionality.
    
    Parameters:
    - U1: np.array of shape (d, k1), where d is the number of features and k1 is the dimensionality of the subspace.
    - U2: np.array of shape (d, k2), where d is the number of features and k2 is the dimensionality of the subspace.
    
    Returns:
    - U1_orth: np.array of shape (d, k1), orthogonalized U1.
    - U2_orth: np.array of shape (d, k2), orthogonalized U2.

    # Example usage
    d = 10  # number of features
    k1 = 3  # dimensionality of subspace U1
    k2 = 4  # dimensionality of subspace U2

    # Generate random orthonormal bases for U1 and U2
    np.random.seed(0)
    U1 = np.random.randn(d, k1)
    U2 = np.random.randn(d, k2)

    # Orthogonalize the subspaces
    U1_orth, U2_orth = orthogonalize_subspaces(U1, U2)

    # Ensure orthogonality
    print("Orthogonality check between U1 and U2:")
    print(np.allclose(U1.T @ U2, np.zeros((k1, k2))))
    print("Orthogonality check between U1_orth and U2_orth:")
    print(np.allclose(U1_orth.T @ U2_orth, np.zeros((k1, k2))))
    """
    
    # Ensure U1 and U2 are orthonormal bases
    U1, _ = qr(U1, mode='economic')
    U2, _ = qr(U2, mode='economic')
    
    # Project U1 onto the orthogonal complement of U2
    P_U2 = U2 @ U2.T
    U1_proj = U1 - P_U2 @ U1
    U1_orth, _ = qr(U1_proj, mode='economic')
    
    # Project U2 onto the orthogonal complement of the modified U1
    P_U1_orth = U1_orth @ U1_orth.T
    U2_proj = U2 - P_U1_orth @ U2
    U2_orth, _ = qr(U2_proj, mode='economic')
    
    return U1_orth, U2_orth

def estimate_dimensionality(X,method='participation_ratio'):
    """
    Estimate the dimensionality of a data set X using a PCA approach.
    
    The dimensionality is estimated by computing the number of principal components
    required to explain a certain proportion of the variance (default 95%).
    
    Parameters
    ----------
    X : array (n_samples, n_features)
        Data to analyze
    
    Returns
    -------
    n_components : int
        Estimated number of components

    # Example usage:
    # X = np.random.rand(100, 50)  # Replace with your actual data
    # dimensionality_estimates = estimate_dimensionality(X)
    # print(dimensionality_estimates)

    """
       
    def pca_variance_explained(X, variance_threshold=0.95):
        pca = PCA()
        pca.fit(X)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        return np.argmax(cumulative_variance >= variance_threshold) + 1
    
    def pca_shuffled_data(X):
        # X_shuffled = my_shuffle(X, random_state=0)
        X_shuffled = my_shuffle(X, method='random')
        pca_original = PCA().fit(X)
        pca_shuffled = PCA().fit(X_shuffled)
        return np.sum(pca_original.explained_variance_ > np.max(pca_shuffled.explained_variance_))
    
    def parallel_analysis_pca(X):
        n_samples, n_features = X.shape
        n_iter = 100
        eigenvalues = np.zeros((n_iter, n_features))
        for i in range(n_iter):
            X_random = my_shuffle(X, method='random')
            pca_random = PCA().fit(X_random)
            eigenvalues[i, :] = pca_random.explained_variance_
        mean_eigenvalues = np.mean(eigenvalues, axis=0)
        pca = PCA().fit(X)
        return np.sum(pca.explained_variance_ > mean_eigenvalues)
    
    def participation_ratio(X):
        pca = PCA().fit(X)
        explained_variance = pca.explained_variance_
        return (np.sum(explained_variance) ** 2) / np.sum(explained_variance ** 2)
    
    if method == 'pca_ev':
        return pca_variance_explained(X)
    elif method == 'pca_shuffle':
        return pca_shuffled_data(X)
    elif method == 'parallel_analysis':
        return parallel_analysis_pca(X)
    elif method == 'participation_ratio':
        return participation_ratio(X)
    elif method == 'FA':
        print('Not yet implemented')
    else:
        raise ValueError('Unknown method')


