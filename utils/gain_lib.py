
#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score

os.chdir('e:\\Python\\molanalysis')

from scipy.stats import vonmises
from loaddata.session import Session
from utils.psth import compute_respmat
from utils.plotting_style import * #get all the fixed color schemes




def plot_respmat(orientations, datasets, labels):
    fig,axes = plt.subplots(1, len(datasets),figsize=(3*len(datasets),4))
    if len(datasets) == 1:
        axes = [axes]
    for d,data in enumerate(datasets):
        ax = axes[d]
        # ax.imshow(data,aspect='auto',vmin=0.1,vmax=0.5,cmap='magma')
        ax.imshow(data,aspect='auto',vmin=np.percentile(datasets[0],20),vmax=np.percentile(datasets[0],90),cmap='magma')
        ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
        ax.set_xticks([0,np.shape(data)[1]],labels=[0,np.shape(data)[1]],fontsize=7)
        ax.set_title(labels[d])
        ax.set_xlabel('Trial',fontsize=9)
        ax.set_ylabel('Neuron',fontsize=9)
        ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    return fig

def plot_tuned_response(orientations, datasets, labels):
    fig,axes = plt.subplots(1, len(datasets),figsize=(3*len(datasets),4))
    if len(datasets) == 1:
        axes = [axes]
    u_oris = np.unique(orientations)
    for d,data in enumerate(datasets):
        ax = axes[d]
        sm = np.array([np.mean(data[:, orientations == i], axis=1) for i in u_oris])
        if d == 0:
            idx = np.argsort(np.argmax(sm,axis=0))
        sm = sm[:,idx]
        ax.imshow(sm.T,aspect='auto',vmin=np.percentile(datasets[0],20),vmax=np.percentile(datasets[0],90),cmap='magma')
        ax.set_xticks(np.arange(len(u_oris)),labels=u_oris,fontsize=7)
        ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
        ax.set_title(labels[d])
        ax.set_xlabel('Orientation',fontsize=9)
        ax.set_ylabel('Neuron',fontsize=9)
        ax.tick_params(axis='x', labelrotation=45)

    fig.tight_layout()

    return fig

def tuned_resp_model(data, stimuli):
    nstim = len(np.unique(stimuli))
    assert nstim == 9 or nstim == 16, 'There should be 9 or 16 unique stimuli, not %d' %nstim
    
    data_hat = np.zeros_like(data)

    for i,stim in enumerate(np.unique(stimuli)):
        data_hat[:,stimuli==stim] = np.mean(data[:, stimuli==stim], axis=1,keepdims=True)

    return data_hat

def pop_rate_gain_model(data, stimuli):
    poprate             = np.nanmean(data,axis=0)
    gain_weights        = np.array([np.corrcoef(poprate,data[n,:])[0,1] for n in range(data.shape[0])])
    gain_trials         = poprate - np.nanmean(data,axis=None)

    ustim,istimeses,stims  = np.unique(stimuli,return_index=True,return_inverse=True)
    nstim = len(ustim)
    assert nstim == 9 or nstim == 16, 'There should be 9 or 16 unique stimuli, not %d' %nstim
    
    # Calculate mean response per stimulus
    sm = np.array([np.mean(data[:,stims == i,], axis=1) for i in range(nstim)])

    if np.mean(poprate) < 1: 
        mfs         = np.arange(10,30,2)
    else:
        mfs         = np.arange(0,0.3,0.025)

    r2data      = []
    for imf,mf in enumerate(mfs):
        data_hat = np.empty_like(data)
        for i in range(nstim):
            data_hat[:,stims == i] = sm[i,:][:,np.newaxis] * (1 + np.outer(gain_weights * mf,gain_trials[stims == i]))

        r2data.append(r2_score(data,data_hat))
    
    mf = mfs[np.argmax(r2data)]
    data_hat = np.empty_like(data)
    for i in range(nstim):
        data_hat[:,stims == i] = sm[i,:][:,np.newaxis] * (1 + np.outer(gain_weights * mf,gain_trials[stims == i]))

    return data_hat