#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.stats import vonmises
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress,binned_statistic
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator
from sklearn.decomposition import PCA

os.chdir('c:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive
from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\TransferFunctions')

#%% Define nonlinearities:


def lin(x):
    return x

def relu(x):
    return np.maximum(0, x)

def softplus(x, beta=1.0):
    return np.log1p(np.exp(beta * x)) / beta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exp(x):
    return np.maximum(0, np.exp(x) - 1)  # Shifted to be zero at x=0

def tanh(x):
    return np.tanh((x))+1  # Shifted to be zero at x=0

def powerlaw(x, p=2):
    return np.maximum(0, x) ** p



#%%
def tuning_input(stim, pref=0.0, width=1.0, gain=1.0):
    return gain * np.exp(-(stim - pref)**2 / (2 * width**2))


#%%
def simulate_responses(
    x,
    nonlinearity,
    noise_std=0.2,
    n_trials=1000
):
    responses = []
    for _ in range(n_trials):
        noisy_x = x + np.random.normal(0, noise_std, size=x.shape)
        y = nonlinearity(noisy_x)
        responses.append(y)
    return np.array(responses)

#%% Show transfer functions for different nonlinearities:
nonlinearities = [lin, relu, lambda x: softplus(x, beta=2), 
                sigmoid, tanh, lambda x: powerlaw(x, p=2), exp]
nonlinearity_names = ['Linear', 'ReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Power-law (p=2)', 'Exp']
nnonlinearities = len(nonlinearities)

operating_range = np.array([[0,1],
                            [-0.5,1],
                            [-3,3],
                            [-5,5],
                            [-2.5,2.5],
                            [-.5,3],
                            [-.5,2]])

fig, axes = plt.subplots(3,3,figsize=(6, 6))
axes = axes.flatten()
x = np.linspace(-10, 10, 100)
x = np.linspace(-5, 5, 100)
# x = np.linspace(-1, 1, 100)

for i, nonlinearity in enumerate(nonlinearities):
    ax = axes[i]
    y = nonlinearity(x)
    ax.plot(x, y)
    ax.set_title(nonlinearity_names[i])
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.grid()
plt.tight_layout()
sns.despine()
# my_savefig(plt.gcf(),savedir,f'{nonlinearity_names[i]}_Nonlinearity_TransferFunction')
# my_savefig(fig,savedir,f'Tranfer_functions_overview')

#%% Plotting:

nonlinearities = [lin, relu, softplus, sigmoid, tanh, lambda x: powerlaw(x, p=2), exp]
nonlinearity_names = ['Linear', 'ReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Power-law (p=2)', 'Exp']
nnonlinearities = len(nonlinearities)

tuninglevels = np.linspace(0.1, 1.0, 5)
ntuninglevels = len(tuninglevels)

fig, axes = plt.subplots(ntuninglevels, nnonlinearities, figsize=(18, 10), 
                         sharex=True, sharey='col')
for i, nonlinearity in enumerate(nonlinearities):
    for j, gain in enumerate(tuninglevels):
        stim = np.linspace(-5, 5, 100)
        input = tuning_input(stim, pref=0.0, width=1.0, gain=gain) * gain

        input /= np.max(input) 
        input *= (operating_range[i,1] -  operating_range[i,0])
        input += operating_range[i,0]

        # responses = simulate_responses(input, lambda x: nonlinearity(x * gain), noise_std=0.2, n_trials=1000)
        responses = simulate_responses(input, lambda x: nonlinearity(x), noise_std=0.2, n_trials=1000)
        mean_response = responses.mean(axis=0)
        std_response = responses.std(axis=0)
        # variance = responses.var(axis=0)
        # fano_factor = variance / (mean_response + 1e-9)

        ax = axes[j, i]
        ax.plot(stim, mean_response, label='Mean Response')
        ax.fill_between(stim, mean_response - std_response, mean_response + std_response, alpha=0.3, label='±1 Std Dev')
        # ax.plot(stim, variance, label='Variance')
        # ax.plot(stim, fano_factor, label='Fano Factor')
        ax.set_title(f'{nonlinearity_names[i]} (Gain={gain:.1f})')
        # ax.legend(
plt.tight_layout()
sns.despine(fig=fig,right=True,top=True)
# my_savefig(fig,savedir,'Nonlinearity_Simulation_PopCouplingModel')

#%% Now for each nonlinearity, compare for neurons with high vs low poppulation rate and population coupling:

#%% Plotting:
ntuninglevels = 5
tuninglevels = np.linspace(0, 1.0, ntuninglevels)
stim = np.linspace(-5, 5, 100)

npopmodulations = 5
# popmodulation = np.linspace(0,1,npopmodulations) #simulate different levels of population modulation
popmodulations = np.linspace(0,0.5,npopmodulations) #simulate different levels of population modulation

fig, axes = plt.subplots(ntuninglevels, npopmodulations, figsize=(18, 10), sharex=True, sharey=True)
for ip, popmod in enumerate(popmodulations):
    for ituning, tuning in enumerate(tuninglevels):
        ax = axes[ituning, ip]
        input = tuning_input(stim, pref=0.0, width=1.0, gain=tuning)
        ax.plot(stim, input, label='Low', color='blue')

        responses = simulate_responses(input, lambda x: nonlinearity(x * tuning), noise_std=0.2, n_trials=1000)

        ax.plot(stim, input + popmod, label='High', color='orange')
        ax.set_title(f'{popmod} (Gain={tuning:.1f})')
plt.tight_layout()
sns.despine(fig=fig,right=True,top=True)
# my_savefig(fig,savedir,'Nonlinearity_Simulation_PopCouplingModel')

#%% Plotting:

nonlinearity_names = ['Linear', 'ReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Power-law (p=2)','Exp']
nonlinearities = [lin, relu, softplus, sigmoid, tanh, lambda x: powerlaw(x, p=2),exp]

nnonlinearities = len(nonlinearities)

ntuninglevels = 5
tuninglevels = np.linspace(0, 1.0, ntuninglevels)
# tuninglevels = np.logspace(-2,0.1, ntuninglevels)
stim = np.linspace(-5, 5, 100)

npopcouplings = 5
# popmodulation = np.linspace(0,1,npopmodulations) #simulate different levels of population modulation
popcouplings = np.linspace(0,1,npopcouplings) #simulate different levels of population modulation
# popmodulations = np.logspace(-2,0,npopmodulations) #simulate different levels of population modulation

npoprates = 5
poprates = np.linspace(0,1,npoprates) #simulate different levels of population modulation
clrs_popcoupling    = sns.color_palette('viridis',npoprates)

for inonlin, nonlinearity in enumerate(nonlinearities):
    fig, axes = plt.subplots(ntuninglevels, npopcouplings, figsize=(10, 8), sharex=True, sharey=True)

    for ipc, popcoupling in enumerate(popcouplings):
        for ituning, tuning in enumerate(tuninglevels):
            ax = axes[ituning, ipc]
            input = tuning_input(stim, pref=0.0, width=1.0, gain=tuning)
            for ipoprate,poprate in enumerate(poprates):
                popmod = popcoupling * poprate
                
                scaledinput = input * tuning + popmod
                # scaledinput /= np.max(scaledinput) 
                scaledinput *= (operating_range[i,1] -  operating_range[i,0])
                scaledinput += operating_range[i,0]

                responses = simulate_responses(scaledinput, nonlinearity, noise_std=0.2, n_trials=1000)
                
                mean_response = responses.mean(axis=0)
                std_response = responses.std(axis=0)

                ax.plot(stim, mean_response, label='Low', color=clrs_popcoupling[ipoprate])
                # ax.fill_between(stim, mean_response - std_response, mean_response + std_response, alpha=0.3,
                                # label='±1 Std Dev', color=clrs_popcoupling[ipoprate])
            plt.suptitle(nonlinearity_names[inonlin])
            # ax.legend(
    plt.tight_layout()
    sns.despine(fig=fig,right=True,top=True)
    my_savefig(fig,savedir,f'Simulation_TuningCurve_HeterogeneousPopCoupling_Model_{nonlinearity_names[inonlin]}')

#%%

######   #####     #       ####### #     #    #     # ####### ######  ####### #       
#     # #     #   # #      #     # ##    #    ##   ## #     # #     # #       #       
#     # #        #   #     #     # # #   #    # # # # #     # #     # #       #       
######  #       #     #    #     # #  #  #    #  #  # #     # #     # #####   #       
#       #       #######    #     # #   # #    #     # #     # #     # #       #       
#       #     # #     #    #     # #    ##    #     # #     # #     # #       #       
#        #####  #     #    ####### #     #    #     # ####### ######  ####### ####### 

#%%  
nNeurons        = 1000
nTrials         = 3200

noris           = 16
tuning_level    = 1
popmodulation_level = 1
noise_std       = 0.1
# noise_std = 0.2

oris            = np.linspace(0,360,noris+1)[:-1]
locs            = np.random.rand(nNeurons) * np.pi * 2  # circular mean
kappa           = 2  # concentration

tuning_var      = np.random.rand(nNeurons) * tuning_level #how strongly tuned neurons are
popcouplings    = np.random.rand((nNeurons))
poprates        = np.random.rand((nTrials)) * popmodulation_level

ori_trials      = np.random.choice(oris,nTrials)

nonlinearity_names = ['Linear', 'ReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Power-law (p=2)','Exp']
nonlinearities      = [lin, relu, softplus, sigmoid, tanh, lambda x: powerlaw(x, p=2),exp]

nnonlinearities = len(nonlinearities)

R = np.empty((nNeurons,nTrials))
for iN in range(nNeurons):
    tuned_resp = vonmises.pdf(np.deg2rad(ori_trials), loc=locs[iN], kappa=kappa)
    R[iN,:] = (tuned_resp / np.max(tuned_resp)) * tuning_var[iN]

inputmat = np.full((nNeurons,nTrials),np.nan)
for iN in range(nNeurons):
    inputmat[iN,:] = R[iN,:] + poprates * popcouplings[iN]

for inonlin, nonlinearity in enumerate(nonlinearities):
    #Scale the input to cover the operating range of this transfer function:
    scaledinputmat = copy.deepcopy(inputmat)
    scaledinputmat /= np.max(scaledinputmat) 
    scaledinputmat *= (operating_range[inonlin,1] -  operating_range[inonlin,0])
    scaledinputmat += operating_range[inonlin,0]

    respmat = np.full((nNeurons,nTrials),np.nan)
    for iN in range(nNeurons):
        respmat[iN,:] = simulate_responses(scaledinputmat[iN,:], nonlinearity, noise_std=noise_std *  (operating_range[inonlin,1] -  operating_range[inonlin,0]), n_trials=1)

    pca         = PCA(n_components=15) #construct PCA object with specified number of components
    Xp          = pca.fit_transform(respmat.T).T #fit pca to response matrix (n_samples by n_features)
    #dimensionality is now reduced from N by K to ncomp by K
    
    oris        = np.sort(np.unique(ori_trials))

    ori_ind     = [np.argwhere(np.array(ori_trials) == iori)[:, 0] for iori in oris]

    shade_alpha = 0.2
    lines_alpha = 0.8
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)), (2, 1))
    
    # projections = [(0, 1), (1, 2), (0, 2)]
    projections = [(0, 1), (1, 2)]
    fig, axes = plt.subplots(1, len(projections), figsize=[
                            len(projections)*3, 3], 
                            sharey=False, sharex=False)
    for ax, proj in zip(axes, projections):
        # plot orientation separately with diff colors
        for t, t_type in enumerate(oris):
            # get all data points for this ori along first PC or projection pairs
            x = Xp[proj[0], ori_ind[t]]
            y = Xp[proj[1], ori_ind[t]]  # and the second
            # each trial is one dot
            ax.scatter(x, y, color=pal[t], s=10, alpha=0.8)
            # ax.scatter(x, y, color=pal[t], s=ses.respmat_videome[ori_ind[t]], alpha=0.8)     #each trial is one dot
            ax.set_xlabel('PC {}'.format(proj[0]+1))  # give labels to axes
            ax.set_ylabel('PC {}'.format(proj[1]+1))

        sns.despine(fig=fig, top=True, right=True)

    # Put a legend to the right of the current axis
    # ax.legend(oris, title='Orientation', frameon=False, fontsize=6, title_fontsize=8,
    #         loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(nonlinearity_names[inonlin])
    plt.tight_layout()
    # my_savefig(fig,savedir,f'PCA_Simulation_HeterogeneousPopCoupling_Model_{nonlinearity_names[inonlin]}')


#%%  
ncomps = 15
fig, axes = plt.subplots(1, 1, figsize=[4,4])
ax = axes
for inonlin, nonlinearity in enumerate(nonlinearities):
    #Scale the input to cover the operating range of this transfer function:
    scaledinputmat = copy.deepcopy(inputmat)
    scaledinputmat /= np.max(scaledinputmat) 
    scaledinputmat *= (operating_range[inonlin,1] -  operating_range[inonlin,0])
    scaledinputmat += operating_range[inonlin,0]

    respmat = np.full((nNeurons,nTrials),np.nan)
    for iN in range(nNeurons):
        respmat[iN,:] = simulate_responses(scaledinputmat[iN,:], nonlinearity, noise_std=noise_std *  (operating_range[inonlin,1] -  operating_range[inonlin,0]), n_trials=1)

    pca         = PCA(n_components=ncomps) #construct PCA object with specified number of components
    pca         = pca.fit(respmat.T) #fit pca to response matrix (n_samples by n_features)

    #dimensionality is now reduced from N by K to ncomp by K
    ax.plot(pca.explained_variance_ratio_,label=nonlinearity_names[inonlin])
ax.set_xticks(range(ncomps))
ax.grid()
ax.legend(nonlinearity_names,frameon=False)
ax.set_xlabel('PCA component')
ax.set_ylabel('Variance')
plt.tight_layout()
sns.despine(fig=fig,top=True)
my_savefig(fig,savedir,f'PCA_EV_TransferFunctions')

#%% 
