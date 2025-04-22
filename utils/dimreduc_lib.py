
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
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
def estimate_dimensionality(X):
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
    
    results = {
        "pca_95_variance_explained": pca_variance_explained(X),
        "pca_shuffled_data": pca_shuffled_data(X),
        "parallel_analysis_pca": parallel_analysis_pca(X),
        "participation_ratio_above_1": participation_ratio(X)
    }
    
    return results
