import torch
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import r2_score 

import fct_facilities as fac 
from fct_stat_models import * 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def compute_LL(x_train, x_test, Nrepet_train, Nrepet_test, add_model, add_varp_model, mul_model, aff_model, expo_model, gen_model):

    # Calculate log-likelihood for additive_model (uses pre-calculated NLL from initialization)
    ll_add = -add_model.NLL
    ll_add_test = -add_model.NLL_test

    # Calculate log-likelihood for additive_varp_model (uses dynamic NLL calculation)
    ll_add_varp = -add_varp_model.loss_nll(x_train, Nrepet_train)
    ll_add_varp_test = -add_varp_model.loss_nll(x_test, Nrepet_test)

    # Calculate log-likelihood for multiplicative_model
    ll_mul = -mul_model.loss_nll(x_train, Nrepet_train)
    ll_mul_test = -mul_model.loss_nll(x_test, Nrepet_test)
    
    # Calculate log-likelihood for affine_model
    ll_aff = -aff_model.loss_nll(x_train, Nrepet_train)
    ll_aff_test = -aff_model.loss_nll(x_test, Nrepet_test)
    
    # Calculate log-likelihood for exponent_model
    ll_expo = -expo_model.loss_nll(x_train, Nrepet_train)
    ll_expo_test = -expo_model.loss_nll(x_test, Nrepet_test)
    
    # Calculate log-likelihood for generalized_model (uses pre-calculated NLL from initialization)
    ll_gen = -gen_model.NLL
    ll_gen_test = -gen_model.NLL_test

    # Collect all training and test log-likelihoods into lists
    ll_train = [ll_add, ll_add_varp.detach().numpy(), ll_mul.detach().numpy(),
                ll_aff.detach().numpy(), ll_expo.detach().numpy(), ll_gen]
    ll_test = [ll_add_test, ll_add_varp_test.detach().numpy(), ll_mul_test.detach().numpy(),
               ll_aff_test.detach().numpy(), ll_expo_test.detach().numpy(), ll_gen_test]

    return ll_train, ll_test


def calculate_nc(x, N, Nrepet, Nstimuli):

    #calculate noise covariance 
    spks_res = np.zeros(( Nrepet*Nstimuli, N )) 
    for i_stim in range(Nstimuli):
        trial_avg = np.mean(x[:, Nrepet*i_stim: Nrepet*(i_stim+1)],1)
        spks_res[Nrepet*i_stim: Nrepet*(i_stim+1),:] = (x[:,Nrepet*i_stim: Nrepet*(i_stim+1)] - trial_avg[:,np.newaxis]).T

    ncov = np.zeros((N, N, Nstimuli)) 
    
    ## plot nc with axis centered around neurons' whose pref ori match with the input
    for i_stim in range(Nstimuli):
        ncov[:,:, i_stim] = np.cov(spks_res[Nrepet*i_stim: Nrepet*(i_stim + 1), :].T)

    return ncov


def calculate_r2_nc(ncov, ncov_add, N, Nstimuli):
    #ncov from data
    x1 = np.zeros((int(N*(N-1)/2), Nstimuli))
    
    #ncov from model
    x2 = np.zeros((int(N*(N-1)/2), Nstimuli))
    
    for i_stim in range(Nstimuli):
        tmp = ncov[:,:, i_stim]
        x1[:, i_stim] = tmp[np.triu_indices(N, 1)]
        
        tmp = ncov_add[:,:, i_stim]
        x2[:, i_stim] = tmp[np.triu_indices(N, 1)]
        
    r2 = r2_score(x1, x2)

    return r2


def calculate_ncov_from_phi(phi):
    #phi is n x n_compo x Nstimuli 
    ncov = np.einsum('ijk, ljk->ilk', phi, phi)
    return ncov


def calculate_phi_from_model(add_model, add_varp_model, mul_model, aff_model, expo_model,gen_model, Nstimuli):
    phi_add = np.tile(add_model.h_p[:,:, None], (1,1,Nstimuli))
    
    phi_add_varp = np.tile(add_varp_model.h_p[:,:,None].detach().numpy(), (1,1,Nstimuli))
    
    phi_mul_tmp = mul_model.alpha_p[:,:,None]*mul_model.d_p[:,None,:]
    phi_mul = phi_mul_tmp.detach().numpy()
    
    phi_aff_tmp = aff_model.alpha_p[:,:,None]*aff_model.d_p[:,None,:]+aff_model.beta_p[:,:,None]
    phi_aff = phi_aff_tmp.detach().numpy()
    
    phi_gen = gen_model.F_p
    
    phi_expo_tmp = expo_model.alpha_p[:,:,None]*(expo_model.d_p[:,None,:]**expo_model.expo_p)+expo_model.beta_p[:,:,None]
    phi_expo = phi_expo_tmp.detach().numpy()
    
    return phi_add, phi_add_varp, phi_mul, phi_aff, phi_expo, phi_gen
