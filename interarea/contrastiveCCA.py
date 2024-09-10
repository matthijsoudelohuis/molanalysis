import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

def contrastive_cca(X_target, Y_target, X_background, Y_background, alpha=1.0):
    # Standardize the datasets
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_target = scaler_x.fit_transform(X_target)
    Y_target = scaler_y.fit_transform(Y_target)
    X_background = scaler_x.transform(X_background)
    Y_background = scaler_y.transform(Y_background)
    
    # Compute covariance matrices for target datasets
    C_XX_t = np.cov(X_target, rowvar=False)
    C_YY_t = np.cov(Y_target, rowvar=False)
    C_XY_t = np.cov(X_target.T, Y_target.T, rowvar=True)[:X_target.shape[1], X_target.shape[1]:]
    
    # Compute covariance matrices for background datasets
    C_XX_b = np.cov(X_background, rowvar=False)
    C_YY_b = np.cov(Y_background, rowvar=False)
    C_XY_b = np.cov(X_background.T, Y_background.T, rowvar=True)[:X_background.shape[1], X_background.shape[1]:]
    
    # Compute contrastive covariance matrices
    C_XX_c = C_XX_t - alpha * C_XX_b
    C_YY_c = C_YY_t - alpha * C_YY_b
    C_XY_c = C_XY_t - alpha * C_XY_b
    
    # Eigen decomposition
    def generalized_eigenproblem(A, B):
        eigvals, eigvecs = eigh(A, B)
        return eigvals, eigvecs

    # Solve generalized eigenvalue problems
    eigvals_x, eigvecs_x = generalized_eigenproblem(C_XX_c, C_XX_t)
    eigvals_y, eigvecs_y = generalized_eigenproblem(C_YY_c, C_YY_t)

    # Sort eigenvalues and eigenvectors
    idx_x = np.argsort(eigvals_x)[::-1]
    idx_y = np.argsort(eigvals_y)[::-1]
    eigvecs_x = eigvecs_x[:, idx_x]
    eigvecs_y = eigvecs_y[:, idx_y]

    # Select top canonical vectors
    u = eigvecs_x[:, 0]
    v = eigvecs_y[:, 0]

    # Compute contrastive canonical correlation
    contrastive_corr = np.dot(u.T, np.dot(C_XY_c, v))
    
    return contrastive_corr, u, v

# Generate random data for target and background datasets
np.random.seed(0)
X_target = np.random.randn(100, 10)  # Target dataset X
Y_target = np.random.randn(100, 10)  # Target dataset Y
X_background = np.random.randn(100, 10) * 0.5  # Background dataset X
Y_background = np.random.randn(100, 10) * 0.5  # Background dataset Y

# Perform Contrastive CCA
alpha = 1.0
contrastive_corr, u, v = contrastive_cca(X_target, Y_target, X_background, Y_background, alpha)

# Print the contrastive canonical correlation
print("Contrastive Canonical Correlation:", contrastive_corr)
