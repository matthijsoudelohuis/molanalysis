import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.linalg import orth, qr, svd

from utils.RRRlib import *
from utils.shuffle_lib import my_shuffle


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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    Filters out nans in any column
    """
    notnan = np.logical_and(~np.isnan(v1), ~np.isnan(v2))
    v1 = v1[notnan]
    v2 = v2[notnan]
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.rad2deg(angle_rad)

def angles_between(v):
    """ Returns the angle in degrees between each of the columns in vector array v:
    """
    angles = np.full((v.shape[1],v.shape[1]), np.nan)
    for i in range(v.shape[1]):
        for j in range(i+1,v.shape[1]):
            angles[i,j] = angle_between(v[:,i],v[:,j])
            angles[j,i] = angles[i,j]
    return angles

def var_along_dim(data,weights):
    """
    Compute the variance of the data projected onto the weights.
    
    Parameters
    ----------
    data : array (n_samples, n_features)
        Data to project
    weights : array (n_features)
        Weights for projecting the data into a lower dimensional space
    
    Returns
    -------
    ev : float
        Proportion of variance explained by the projection.
    """
    assert data.shape[1] == weights.shape[0], "data and weights must have the same number of features"
    assert weights.ndim == 1, "weights must be a vector"
    
    weights     = unit_vector(weights) # normalize weights
    var_proj    = np.var(np.dot(data, weights)) # compute variance of projected data
    var_tot     = np.var(data, axis=0).sum() # compute total variance of original data
    ev          = var_proj / var_tot # compute proportion of variance explained 
    return ev


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
