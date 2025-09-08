import os
import numpy as np
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
from scipy.sparse.linalg import svds
import pandas as pd
from scipy.stats import zscore
from scipy import stats
from sklearn.metrics import r2_score
from utils.RRRlib import *
from preprocessing.locate_rf import *

def im_log(image):
    # Apply log transformation method 
    c = 255 / np.log(1 + np.max(image)) 
    log_image = c * (np.log(image + 1)) 
    return log_image

def im_sqrt(image):
    # Apply sqrt transformation 
    return image ** 0.5

def im_norm(I,min=0,max=100):
    mn = np.percentile(I,min)
    mx = np.percentile(I,max)
    mx -= mn
    I = ((I - mn)/mx) * 255
    I[I<0] =0
    I[I>255] = 255
    return I

def im_norm8(I,min=0,max=100):
    I = im_norm(I,min=min,max=max)
    # Specify the data type so that float value will be converted to int 
    return I.astype(np.uint8)


def load_natural_images(onlyright=False):

    mat_fname = os.path.join(os.getcwd(),'naturalimages','images_natimg2800_all.mat')
    mat_contents = sio.loadmat(mat_fname)
    natimgdata = mat_contents['imgs']

    if onlyright:
        natimgdata = natimgdata[:,180:,:]
        
    return natimgdata


def lowrank_RF(Y, IMdata, lam=0.05,nranks=25,nsub=3):
    """
    Compute a linear low-rank approximation of the receptive field (RF) from the natural image responses

    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nranks : int, default=25
        The number of ranks to keep in the low-rank approximation
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The low-rank approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """
    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    N                   = np.shape(Y)[1]        # N is the number of neurons

    B_hat               = LM(Y, X, lam=lam)     #fit multiple linear regression (with ridge penalty)

    U, s, V             = svds(B_hat, k=nranks) #truncated singular value decomposition of the coefficients

    B_hat_rrr           = U @ np.diag(s) @ V    #reconstruct the coefficients from low rank

    Y_hat               = X @ B_hat_rrr         #predict the trial to trial response from the low rank coefficients

    cRF                 = np.reshape(B_hat_rrr, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

def lowrank_RF_cv(Y, IMdata, lam=0.05,nranks=25,nsub=3,kfold=2):
    """
    Compute a linear low-rank approximation of the receptive field (RF) from the natural image responses
    Crossvalidated version
    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nranks : int, default=25
        The number of ranks to keep in the low-rank approximation
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The low-rank approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """

    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    K,N                 = np.shape(Y)        # K is the number of images, N is the number of neurons
    Y_hat               = np.full((K,N),np.nan)

    kf                  = KFold(n_splits=kfold, shuffle=True)

    B_hat_rrr_folds = np.full((Ly*Lx,N,kfold),np.nan)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        B_hat               = LM(Y_train, X_train, lam=lam)     #fit multiple linear regression (with ridge penalty)

        U, s, V             = svds(B_hat, k=nranks) #truncated singular value decomposition of the coefficients

        B_hat_rrr           = U @ np.diag(s) @ V    #reconstruct the coefficients from low rank

        Y_hat[test_index,:] = X_test @ B_hat_rrr         #predict the trial to trial response from the low rank coefficients
        
        B_hat_rrr_folds[:,:,i] = B_hat_rrr

    B_hat_rrr   = np.nanmean(B_hat_rrr_folds, axis=2)

    cRF         = np.reshape(B_hat_rrr, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

def linear_RF(Y, IMdata, lam=0.05,nsub=3,kfold=2):
    """
    Compute a linear approximation of the receptive field (RF) from the natural image responses
    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The linear approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """

    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    K,N                 = np.shape(Y)        # K is the number of images, N is the number of neurons
    Y_hat               = np.full((K,N),np.nan)

    kf = KFold(n_splits=kfold, shuffle=True)

    B_hat               = LM(Y, X, lam=lam)     #fit multiple linear regression (with ridge penalty)
    
    Y_hat               = X @ B_hat         #predict the trial to trial response from the low rank coefficients


    # B_hat_folds = np.full((Ly*Lx,N,kfold),np.nan)

    # for i, (train_index, test_index) in enumerate(kf.split(X)):

    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = Y[train_index], Y[test_index]

    #     B_hat               = LM(Y_train, X_train, lam=lam)     #fit multiple linear regression (with ridge penalty)

    #     Y_hat[test_index,:] = X_test @ B_hat         #predict the trial to trial response from the low rank coefficients
        
    #     B_hat_folds[:,:,i] = B_hat

    # B_hat   = np.nanmean(B_hat_folds, axis=2)

    cRF         = np.reshape(B_hat, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

def linear_RF_cv(Y, IMdata, lam=0.05,nsub=3,kfold=2):
    """
    Compute a linear approximation of the receptive field (RF) from the natural image responses
    Crossvalidated version
    Parameters
    ----------
    Y : array with shape (K,N)
        The neural responses to N natural images
    IMdata : array with shape (H, W, K)
        The natural image data of shape H x W x K 
        where H is the height of the image, W is the width of the image, and K is the number of images
    lam : float, default=0.05
        The regularization parameter for the multiple linear regression
    nsub : int, default=3
        The downsampling factor for the natural images

    Returns
    -------
    cRF : array with shape (Ly, Lx, N)
        The linear approximation of the receptive field
    Y_hat : array with shape (K,N)
        The predicted neural responses
    """

    IMdata              = IMdata[::nsub, ::nsub, :]         #subsample the natural images
    Ly,Lx,K             = np.shape(IMdata)                  #get dimensions
    X                   = np.reshape(IMdata, (Ly*Lx, K)).T  #X is now pixels by images matrix
    X                   = X / np.linalg.norm(X, axis=0)     # normalize the pixels in each image
    assert np.shape(Y)[0] == np.shape(X)[0], 'Number of neuronal responses does not match number of images'

    K,N                 = np.shape(Y)        # K is the number of images, N is the number of neurons
    Y_hat               = np.full((K,N),np.nan)

    kf = KFold(n_splits=kfold, shuffle=True)

    B_hat_folds = np.full((Ly*Lx,N,kfold),np.nan)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        B_hat               = LM(Y_train, X_train, lam=lam)     #fit multiple linear regression (with ridge penalty)

        Y_hat[test_index,:] = X_test @ B_hat         #predict the trial to trial response from the low rank coefficients
        
        B_hat_folds[:,:,i] = B_hat

    B_hat   = np.nanmean(B_hat_folds, axis=2)

    cRF         = np.reshape(B_hat, (Ly,Lx, N)) #reshape the low rank coefficients to the image space

    return cRF,Y_hat

#Fit each cRF with a 2D gaussian:
def fit_2dgauss_cRF(cRF, nsub,celldata):
    N = np.shape(cRF)[2]

    celldata[['rf_az_RRR', 'rf_el_RRR', 'rf_sx_RRR', 'rf_sy_RRR', 'rf_r2_RRR']] = np.nan
    
    for n in tqdm(range(N),total=N,desc='Fitting 2D gauss to RF'):	
        rfdata = np.abs(cRF[:, :, n])
        gaussian_sigma = 1
        rfdata = gaussian_filter(rfdata,sigma=[gaussian_sigma,gaussian_sigma])

        try:
            popt,pcov,r2,z_fit = fit_2d_gaussian(rfdata)

            celldata.loc[n,'rf_az_RRR']   = popt[0]*nsub
            celldata.loc[n,'rf_el_RRR']   = popt[1]*nsub
            celldata.loc[n,'rf_sx_RRR']   = popt[2]*nsub
            celldata.loc[n,'rf_sy_RRR']   = popt[3]*nsub
            celldata.loc[n,'rf_r2_RRR']   = r2
        except:
            pass
    return celldata
