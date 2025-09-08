#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# from scipy.linalg import norm
# from scipy.stats import vonmises
# from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
# from tqdm import tqdm
import pickle

os.chdir('e:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive
from loaddata.session import Session
from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_respmat
from utils.tuning import compute_tuning_wrapper
from utils.plot_lib import * #get all the fixed color schemes
from utils.gain_lib import * 

from fct_stat_models import * 
from fct_stat_models_utils import * 
import fct_facilities as fac 

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\GainModel\\')

#%%
# Multiplicative: 
# - self.d_p (torch.Tensor): Learned stimulus-specific mean responses. shape (n_neurons, n_stimuli)
# - self.alpha_p (torch.Tensor): Learned global loading matrix. Global loading matrix, shape (n_neurons, n_compo)
# - self.psi_p (torch.Tensor): Learned diagonal of private noise covariance. shape (n_neurons, n_stimuli)

#Affine: 
# Model: x_st = d_s + A_s * z_st + epsilon_st
# where A_s = alpha_p * diag(d_s) + beta_p, with d_s being the stimulus-specific mean,
# alpha_p and beta_p being global loading matrices, z_st are latent variables, and
# epsilon_st is private noise.

#%%%

# add_model = fac.Retrieve(f'add_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)
resultsdir = 'E:\\Python\\GainAnalysis\\Results\\'

# Orientation Selectivity Index (OSI) threshold for neuron filtering
OSI_thr = 0.35

# Number of folds for K-fold cross-validation
Nfolds = 5
# Number of latent components
Ncompo = 3

r2_nc_all = np.zeros((6, Ncompo, Nfolds))

N = 2134

alpha_p = np.empty((N, Ncompo, Nfolds))
beta_p  = np.empty((N, Ncompo, Nfolds))
pop_coupling = np.empty((N, Nfolds))

# psi_p = np.empty((Ncompo, Nfolds))

for i_fold in range(Nfolds):

    print('*** Fold ', i_fold)

    # --- Reshape training and test data --- #
    # x_train = data[:, train_ind_all[i_fold], :].reshape(N, -1, order='F')
    # x_test = data[:, test_ind_all[i_fold], :].reshape(N, -1, order='F')
    
    # for i_compo in 1+np.arange(Ncompo):

    # print('Components ', i_compo)

    # Retrieve previously fitted model instances
    # add_model = fac.Retrieve(f'add_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)
    # add_varp_model = fac.Retrieve(f'add_varp_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)
    # mul_model = fac.Retrieve(f'mul_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)
    # aff_model = fac.Retrieve(f'aff_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)
    # gen_model = fac.Retrieve(f'gen_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)
    # expo_model = fac.Retrieve(f'expo_model_fold{i_fold}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)

    # add_model.x
    # add_model.x_recon

    # mul_model.alpha_p
    # mul_model.psi_p
    # mul_model.x_recon
    # mul_model.x_recon

    aff_model = fac.Retrieve(f'aff_model_fold{i_fold}_compo{Ncompo}_OSI{OSI_thr}.p', resultsdir)
    alpha_p[:,:,i_fold] = aff_model.alpha_p.detach().numpy()
    beta_p[:,:,i_fold] = aff_model.beta_p.detach().numpy()
    # psi_p[:,i_fold] = aff_model.psi_p.detach().numpy()

    resp        = zscore(aff_model.x.T,axis=0)
    poprate     = np.mean(resp, axis=1)

    pop_coupling[:,i_fold] = [np.corrcoef(resp[:,i],poprate)[0,1] for i in range(N)]

    # aff_model = fac.Retrieve(f'aff_model_fold{1}_compo{i_compo}_OSI{OSI_thr}.p', resultsdir)
    # alpha_p_1 = aff_model.alpha_p.detach().numpy()
    # plt.scatter(alpha_p_0, alpha_p_1)
    # # Compute actual noise covariance from test data
    # ncov = calculate_nc(x_test, N, Nrepet_test, Nstimuli)

    # # Compute predicted common noise from models
    # phi = calculate_phi_from_model(add_model, add_varp_model, mul_model, aff_model, expo_model, gen_model, Nstimuli)

    # ncov_from_phi = [np.zeros((N, N, Nstimuli))] * 6

    # for model_i, phi_i in enumerate(phi):
    #     ncov_from_phi[model_i] = calculate_ncov_from_phi(phi_i)

    # # Calculate R2 for noise covariance for each model
    # r2_nc = []

    # for ncov_model in ncov_from_phi:
    #     r2_model = calculate_r2_nc(ncov, ncov_model, N, Nstimuli)
    #     r2_nc.append(r2_model)

    # r2_nc_all[:, i_compo-1, i_fold] = r2_nc

#%% compute correlation coefficient of the parameter estimates for each fold.

alpha_p_corr = np.zeros((Ncompo, Nfolds, Nfolds))
beta_p_corr = np.zeros((Ncompo, Nfolds, Nfolds))

for i_compo in range(1, Ncompo+1):
    for i_fold in range(Nfolds):
        for j_fold in range(Nfolds):
            alpha_p_corr[i_compo-1,i_fold,j_fold] = np.corrcoef(alpha_p[:,i_compo-1,i_fold], alpha_p[:,i_compo-1,j_fold])[0,1]
            beta_p_corr[i_compo-1,i_fold,j_fold] = np.corrcoef(beta_p[:,i_compo-1,i_fold], beta_p[:,i_compo-1,j_fold])[0,1]

fig, ax = plt.subplots(2,Ncompo,figsize=(Ncompo*3,6))
for i in range(Ncompo):
    ax[0,i].imshow(alpha_p_corr[i,:,:],vmin=0,vmax=1,cmap='coolwarm')
    ax[0,i].set_title(f'Alpha compo {i+1}')
    ax[1,i].imshow(beta_p_corr[i,:,:],vmin=0,vmax=1,cmap='coolwarm')
    ax[1,i].set_title(f'Beta compo {i+1}')
plt.tight_layout()
plt.show()
my_savefig(fig,savedir,'Corr_alpha_beta_folds',formats=['png'])

#%% Show the correlation between the population coupling and the multiplicative
# gain parameter and additive gain parameter
fig, axes = plt.subplots(2,Ncompo,figsize=(Ncompo*3,6))

if Ncompo == 1:
    axes = np.expand_dims(axes, axis=1)

alpha_p_avg = np.mean(alpha_p, axis=2)
beta_p_avg = np.mean(beta_p, axis=2)
pop_coupling_avg = np.mean(pop_coupling, axis=1)

from scipy.stats import linregress

for i_compo in range(Ncompo):

    slope, intercept, r_value, p_value, std_err = linregress(pop_coupling_avg, alpha_p_avg[:,i_compo])

    axes[0,i_compo].scatter(pop_coupling_avg, alpha_p_avg[:,i_compo], s=1, color='k')
    axes[0,i_compo].plot(pop_coupling_avg, slope*pop_coupling_avg + intercept, color='r')
    axes[0,i_compo].set_title('Mult compo %d, R2=%.2f' % (i_compo+1, r_value),color='r', fontsize=10)
    # axes[0,i_compo].text(0,0.5,s='Mult. compo %d, R2=%.2f' % (i_compo+1, r_value),color='r',
                        #  fontsize=9,transform=axes[0,i_compo].transAxes)
    slope, intercept, r_value, p_value, std_err = linregress(pop_coupling_avg, beta_p_avg[:,i_compo])

    axes[1,i_compo].scatter(pop_coupling_avg, beta_p_avg[:,i_compo], s=1, color='k')
    axes[1,i_compo].plot(pop_coupling_avg, slope*pop_coupling_avg + intercept, color='r')
    axes[1,i_compo].set_title('Add compo %d, R2=%.2f' % (i_compo+1, r_value),color='r', fontsize=10)
    # axes[1,i_compo].text(0,0.5,s='Add. compo %d, R2=%.2f' % (i_compo+1, r_value),color='r',
                        #  fontsize=9,transform=axes[1,i_compo].transAxes)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=1)
my_savefig(fig,savedir,'Corr_alpha_beta_popcoupling',formats=['png'])

#%% 
fig,axes = plt.subplots(1,1,figsize=(4,3))
ax = axes
i_compo = 0
from sklearn.cluster import KMeans
data = np.column_stack((alpha_p_avg,beta_p_avg))
kmeans = KMeans(n_clusters=1, random_state=0).fit(data)
ax.scatter(alpha_p_avg[:,i_compo],beta_p_avg[:,i_compo],c=kmeans.labels_,s=1,cmap='viridis')
# ax.scatter(alpha_p_avg[:,i_compo],beta_p_avg[:,i_compo],color='k',s=1)
ax.set_xlim(np.percentile(alpha_p_avg[:,i_compo],[0.5,99.5]))
ax.set_ylim(np.percentile(beta_p_avg[:,i_compo],[0.5,99.5]))

sns.despine(fig=fig, top=True, right=True, offset=3)



#%%  
nNeurons        = 1000
nTrials         = 3200

noise_level     = 15
gain_level      = 5
offset_level    = 0

noris           = 16

oris            = np.linspace(0,360,noris+1)[:-1]
locs            = np.random.rand(nNeurons) * np.pi * 2  # circular mean
kappa           = 2  # concentration

tuning_var      = np.random.rand(nNeurons) #how strongly tuned neurons are

ori_trials      = np.random.choice(oris,nTrials)

R = np.empty((nNeurons,nTrials))
for iN in range(nNeurons):
    tuned_resp = vonmises.pdf(np.deg2rad(ori_trials), loc=locs[iN], kappa=kappa)
    R[iN,:] = (tuned_resp / np.max(tuned_resp)) * tuning_var[iN]


# The distribution of gains across trials determines the distribution of points within each column of the cone
# I.e. if gain is rand <0,1> then there are trials with zero gain. If gain is strongly dependent on locomotion
# then there are many trials with large gains in sessions where the mouse is continuously moving.
# However, it seems that gain (1 + weights * trials) should be positive, otherwise tuning response is inverted
