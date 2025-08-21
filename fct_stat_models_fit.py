import torch
import numpy as np
from sklearn.decomposition import FactorAnalysis

import fct_facilities as fac 
from fct_stat_models import * 

lr_general = 1e-3


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def FitModels(i_fold, i_compo, data, train_ind_all, test_ind_all, Nstimuli, Nrepet_train, Nrepet_test, OSI_thr, resultsdir, semaphore=None):

    print(f'*********** Components {i_compo}, Fold {i_fold}', flush=True)

    N = data.shape[0]

    # --- Reshape training and test data --- #
    x_train = np.copy(data[:, train_ind_all[i_fold], :].reshape(N, -1, order='F'))
    x_test = np.copy(data[:, test_ind_all[i_fold], :].reshape(N, -1, order='F'))


    ### --- Generalized Model --- ###
    print ('*** Generalized Model')

    gen_model = generalized_model(x_train, N, Nstimuli, Nrepet_train, x_test, Nrepet_test, i_compo)
    fac.Store(gen_model, f'gen_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)


    ### --- Additive Model --- ###
    print ('*** Additive Model')

    add_model = additive_model(x_train, N, Nstimuli, Nrepet_train, x_test, Nrepet_test, i_compo)
    fac.Store(add_model, f'add_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)


    ### --- Additive Model w/ Variable Private Variance --- ###
    print ('*** Additive Model w/ pvar')

    add_varp_model = additive_varp_model(
        x_train, N, Nstimuli, Nrepet_train, i_compo,
        h_p_init=add_model.h_p.copy(),
        psi_p_init=gen_model.psi_p.copy()
    )
    add_varp_model.train(lr_general, x_train, Nrepet_train)
    fac.Store(add_varp_model, f'add_varp_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)


    ### --- Exponent Model --- ###
    print ('*** Exponent Model')

    expo_model = exponent_model(
        x_train, N, Nstimuli, Nrepet_train, i_compo,
        expo_p_init=0.5,
        beta_p_init=add_model.h_p.copy(),
        psi_p_init=gen_model.psi_p.copy()
    )
    expo_model.train(lr_general, x_train, Nrepet_train)
    fac.Store(expo_model, f'expo_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)


    ### --- Affine Model --- ###
    print ('*** Affine Model')

    aff_model = affine_model(
        x_train, N, Nstimuli, Nrepet_train, i_compo,
        alpha_p_init=np.zeros((N, i_compo)),
        beta_p_init=add_model.h_p.copy(),
        psi_p_init=gen_model.psi_p.copy()
    )
    aff_model.train(lr_general, x_train, Nrepet_train)
    fac.Store(aff_model, f'aff_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)


    ### --- Multiplicative Model --- ###
    print ('*** Multiplicative Model')

    mul_model = multiplicative_model(
        x_train, N, Nstimuli, Nrepet_train, i_compo,
        alpha_p_init=aff_model.alpha_p.detach().numpy().copy(),
        psi_p_init=aff_model.psi_p.detach().numpy().copy()
    )
    mul_model.train(lr_general, x_train, Nrepet_train)
    fac.Store(mul_model, f'mul_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)

    ###
    
    if semaphore is not None:
        semaphore.release()    
