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
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator
from sklearn.decomposition import PCA

os.chdir('c:\\Python\\molanalysis')

from loaddata.get_data_folder import get_local_drive
from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import *
from utils.nonlin_lib import *

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\TransferFunctions')

#%%
cm = 1/2.54  # centimeters in inches

# #%% Define nonlinearities:

# def lin(x):
#     return x

# def relu(x):
#     return np.maximum(0, x)

# def softplus(x, beta=1.0):
#     return np.log1p(np.exp(beta * x)) / beta

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def exp(x):
#     return np.maximum(0, np.exp(x) - 1)  # Shifted to be zero at x=0

# def tanh(x):
#     return np.tanh((x))+1  # Shifted to be zero at x=0

# def powerlaw(x, p=2):
#     return np.maximum(0, x) ** p


# #%% Show transfer functions for different nonlinearities:
# nonlinearities = [lin, relu, lambda x: softplus(x, beta=2), 
#                 sigmoid, tanh, lambda x: powerlaw(x, p=2), exp]
# nonlinearity_names = ['Linear', 'ReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Power-law (p=2)', 'Exp']
# nnonlinearities = len(nonlinearities)

# operating_range = np.array([[0,1],
#                             [-0.5,1],
#                             [-3,3],
#                             [-5,5],
#                             [-2.5,2.5],
#                             [-.5,3],
#                             [-.5,2]])

# fig, axes = plt.subplots(3,3,figsize=(6, 6))
# axes = axes.flatten()
# # x = np.linspace(-10, 10, 100)
# x = np.linspace(-5, 5, 100)
# # x = np.linspace(-1, 1, 100)

# for i, nonlinearity in enumerate(nonlinearities):
#     ax = axes[i]
#     y = nonlinearity(x)
#     ax.plot(x, y)
#     ax.set_title(nonlinearity_names[i])
#     ax.set_xlabel('Input')
#     ax.set_ylabel('Output')
#     ax.grid()
# plt.tight_layout()
# sns.despine()
# my_savefig(plt.gcf(),savedir,f'{nonlinearity_names[i]}_Nonlinearity_TransferFunction')
# my_savefig(fig,savedir,f'Tranfer_functions_overview')



#%% 

session_list        = np.array([['LPE11086_2024_01_05']])
session_list        = np.array([['LPE12223_2024_06_10']])
# session_list        = np.array([['LPE12223_2024_06_10','LPE11086_2024_01_05','LPE10919_2023_11_06']])

sessions,nSessions  = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

[sessions[ises].meanresp_orig,sessions[ises].respmat_res] = mean_resp_gr(sessions[ises])

#%% Compute signal and noise correlation matrix:
[N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
oris                            = np.sort(sessions[ises].trialdata['Orientation'].unique())
# trialfilter                     = np.ones(K,bool)
resp_meanori,respmat_res        = mean_resp_gr(sessions[ises],trialfilter=None)
prefori                         = oris[np.argmax(resp_meanori,axis=1)]

sessions[ises].delta_pref       = np.abs(np.mod(np.subtract.outer(prefori, prefori),180))

# Compute signal correlations on all trials: 
# sessions[ises].sig_corr         = np.corrcoef(resp_meanori)

#Compute signal correlation on separate halfs of trials:
trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=~trialfilter)
sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_meanori1, resp_meanori2)[:N, N:] +
                                    np.corrcoef(resp_meanori2, resp_meanori1)[:N, N:])
# Compute noise correlations from residuals:
sessions[ises].NC_alltrials       = np.corrcoef(respmat_res)

# Compute per stimulus, then average:
trial_ori   = sessions[ises].trialdata['Orientation']
noise_corr = np.empty((N,N,len(oris)))  
sessions[ises].NC_perstim = np.empty((N,N,len(oris)))  
for i,ori in enumerate(oris):
    sessions[ises].NC_perstim[:,:,i] = np.corrcoef(respmat_res[:,trial_ori==ori])
    np.fill_diagonal(sessions[ises].NC_perstim[:, :, i],np.nan)

sessions[ises].NC_avgperstim = np.nanmean(sessions[ises].NC_perstim,axis=2)

np.fill_diagonal(sessions[ises].sig_corr,np.nan)
np.fill_diagonal(sessions[ises].delta_pref,np.nan)
np.fill_diagonal(sessions[ises].NC_alltrials,np.nan)
np.fill_diagonal(sessions[ises].NC_avgperstim,np.nan)

#%% Compute the product of the two population coupling:

sessions[ises].coupling_product = np.outer(sessions[ises].celldata['pop_coupling'],
                                    sessions[ises].celldata['pop_coupling'])

fig,axes = plt.subplots(1,1,figsize=(4,4))
ax = axes
sns.histplot(sessions[ises].coupling_product.flatten(),bins=50,ax=ax)
ax.set_xlabel('Product of population coupling')
ax.set_ylabel('Count')

#%% Compute the product of the response per orientation:
sessions[ises].response_product = np.empty((N,N,len(oris)))  
for i,ori in enumerate(oris):
    temp = minmax_scale(sessions[ises].meanresp_orig[:,i])

    # sessions[ises].response_product[:,:,i] = np.outer(temp,temp)
    sessions[ises].response_product[:,:,i] = temp[:,np.newaxis] + temp[np.newaxis,:]
    np.fill_diagonal(sessions[ises].response_product[:, :, i],np.nan)

# fig,axes = plt.subplots(1,1,figsize=(4,4))
# ax = axes
# sns.histplot(sessions[ises].response_product.flatten(),bins=50,ax=ax)
# ax.set_xlabel('Product of mean tuned response')
# ax.set_ylabel('Count')

#%% Plot the scatter between average noise correlations and:
# 1) signal correlation
# 2) product of population coupling
# 3) product of 1 and 2

from utils.corr_lib import filter_sharednan
markersize = 2
markeralpha = 0.1
# idx_N = np.random.choice(N,150,replace=False)
# idx_N = np.random.choice(np.where(sessions[ises].celldata['gOSI']>0.4)[0],150,replace=False)

# NC_data = sessions[ises].NC_alltrials[np.ix_(idx_N,idx_N)]
NC_data = sessions[ises].NC_avgperstim[np.ix_(idx_N,idx_N)]
ydata = NC_data.flatten()

nsubplots = 3
fig,axes = plt.subplots(1,nsubplots,figsize=(nsubplots*3.5*cm,4*cm),sharey=True)
ax = axes[0]
xdata = sessions[ises].sig_corr[np.ix_(idx_N,idx_N)].flatten()
xdata,ydata = filter_sharednan(xdata,ydata)

# sns.scatterplot(x=xdata.flatten(),y=NC_data.flatten(),ax=ax,s=5,alpha=0.2)
sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata,ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
ax.set_xlabel('Signal Corr')
ax.set_ylabel('Noise Correlation')

ax = axes[1]
xdata = sessions[ises].coupling_product[np.ix_(idx_N,idx_N)]
ydata = NC_data.flatten()
xdata,ydata = filter_sharednan(xdata.flatten(),ydata)
sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata.flatten(),ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
ax.set_xlabel('PC product')
# ax.set_ylabel('Noise Correlation')

ax = axes[2]
ydata = NC_data.flatten()
temp1 = sessions[ises].sig_corr[np.ix_(idx_N,idx_N)].flatten()
temp2 = sessions[ises].coupling_product[np.ix_(idx_N,idx_N)].flatten()
temp1 = minmax_scale(temp1,feature_range=(0,1))
# temp1 = zscore(temp1,nan_policy='omit')
temp1 = temp1**0.5
# temp1 = temp1**2
# temp1 = np.log(temp1)

# xdata = temp1 * temp2
temp1 = minmax_scale(temp1)

xdata = temp1 * temp2
# xdata = minmax_scale(temp1) * minmax_scale(temp2)
xdata,ydata = filter_sharednan(xdata,ydata)
# xdata = sessions[ises].sig_corr[np.ix_(idx_N,idx_N)] * sessions[ises].coupling_product[np.ix_(idx_N,idx_N)]
# xdata = minmax_scale(sessions[ises].sig_corr[np.ix_(idx_N,idx_N)]) * sessions[ises].coupling_product[np.ix_(idx_N,idx_N)]

sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata,ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
ax.set_xlabel('PC * signal corr')
# ax.set_ylabel('Noise Correlation')

sns.despine(fig=fig, top=True, right=True, offset=2,trim=False)
plt.tight_layout()

#%% 

#%% Plot the scatter between average noise correlations and:
# 1) signal correlation
# 2) product of population coupling
# 3) product of 1 and 2

from utils.corr_lib import filter_sharednan
markersize = 1
markeralpha = 0.1
# idx_N = np.random.choice(N,150,replace=False)
idx_N = np.random.choice(np.where(sessions[ises].celldata['gOSI']>0.4)[0],
                         50,replace=False)

# NC_data = sessions[ises].NC_alltrials[np.ix_(idx_N,idx_N)]
# NC_data = sessions[ises].NC_avgperstim[np.ix_(idx_N,idx_N)]
NC_data = sessions[ises].NC_perstim[np.ix_(idx_N,idx_N,np.arange(len(oris)))].flatten()


nsubplots = 5
fig,axes = plt.subplots(1,nsubplots,figsize=(nsubplots*3.5*cm,4*cm),sharey=True)
ax = axes[0]
ydata = NC_data.flatten()
xdata_sigcorr = np.repeat(sessions[ises].sig_corr[np.ix_(idx_N,idx_N)][:,:,np.newaxis],
                    len(oris),axis=2).flatten()
xdata,ydata = filter_sharednan(xdata_sigcorr,ydata)

# sns.scatterplot(x=xdata.flatten(),y=NC_data.flatten(),ax=ax,s=5,alpha=0.2)
sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata,ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
ax.set_xlabel('Signal Corr')
ax.set_ylabel('Noise Correlation')

ax = axes[1]
ydata = NC_data.flatten()
xdata_coupling = np.repeat(sessions[ises].coupling_product[np.ix_(idx_N,idx_N)][:,:,np.newaxis],
                    len(oris),axis=2).flatten()
xdata,ydata = filter_sharednan(xdata_coupling,ydata)
sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata.flatten(),ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
ax.set_xlabel('PC product')
# ax.set_ylabel('Noise Correlation')

ax = axes[2]
ydata = NC_data.flatten()
temp2 = xdata_coupling
temp1 = minmax_scale(xdata_sigcorr,feature_range=(0,1))
# temp1 = temp1**0.5
# xdata = temp1 * temp2
temp1 = minmax_scale(temp1)

xdata = temp1 * temp2
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata,ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
ax.set_xlabel('PC * signal corr')
# ax.set_ylabel('Noise Correlation')

ax = axes[3]
ydata = NC_data.flatten()
xdata = sessions[ises].response_product[np.ix_(idx_N,idx_N,np.arange(len(oris)))].flatten()

# xdata = temp1 * temp2
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata,ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
# ax.set_xlabel('PC * signal corr')
ax.set_xlabel('Resp product')

ax = axes[4]
ydata = NC_data.flatten()
temp1 = sessions[ises].coupling_product[np.ix_(idx_N,idx_N)]
temp2 = sessions[ises].response_product[np.ix_(idx_N,idx_N,np.arange(len(oris)))]
# temp2 = minmax_scale(temp2,feature_range=(0,1),axis=1)
temp2 -= np.nanmin(temp2,axis=2,keepdims=True)
temp2 /= np.nanmax(temp2,axis=2,keepdims=True)
xdata = temp1[:,:,np.newaxis] * temp2
xdata= xdata.flatten()
xdata,ydata = filter_sharednan(xdata,ydata)
sns.regplot(x=xdata,y=ydata,ax=ax,scatter_kws={'alpha': markeralpha, 's': markersize},line_kws={'color': 'red'})
ax.text(0.5, 0.95,'R2= %1.2f' % np.corrcoef(xdata,ydata)[0,1]**2,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='red')
ax.set_xlabel('PC * resp product')

sns.despine(fig=fig, top=True, right=True, offset=2,trim=False)
plt.tight_layout()


#%% Plot correlation between two neurons 
def plot_noise_pair(ses,sourcecell,targetcell):
    oris = np.arange(0,360,22.5)
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

    nsubplots = 6
    fig,axes = plt.subplots(1,nsubplots,figsize=(nsubplots*2.5,2.5))
    # fig,axes = plt.subplots(2,2,figsize=(6,6))

    for iN,N in enumerate([sourcecell,targetcell]):
        ax = axes[iN]

        ax.plot(oris,ses.meanresp_orig[N],c='k',linewidth=2)

        for iori,ori in enumerate(oris):
            idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
            ax.scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[N,idx_ori],
                            color=pal[iori],s=5,alpha=0.4)
            # ax.scatter(ses.trialdata['Orientation'][idx_ori],ses.respmat[targetcell,idx_ori],
            #                 color='red',s=5,alpha=0.2)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('Ori')
        ax.set_ylabel('Deconvolved activity')
        # ax.set_title('Tuning Curves')
        ax.set_ylim([0,my_ceil(np.nanpercentile(ses.respmat[N,:],99),-1)])
        ax.set_yticks([0,ax.get_ylim()[1]])
        ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)
    ax = axes[2]
    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        ax.scatter(ses.respmat[sourcecell,idx_ori],ses.respmat[targetcell,idx_ori],
                        c=pal[iori],s=5,alpha=0.2)
    ax_nticks(ax,3)
    ax.set_xlabel('Neuron 1')
    ax.set_ylabel('Neuron 2')
    ax.set_title('Activity')
    # ax.text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])

    ax = axes[3]
    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        ax.scatter(ses.respmat_res[sourcecell,idx_ori],ses.respmat_res[targetcell,idx_ori],
                        c=pal[iori],s=5,alpha=0.2)

    ax_nticks(ax,3)
    ax.set_xlabel('Residual Neuron 1')
    ax.set_ylabel('Residual Neuron 2')
    ax.set_title('Residual activity')
    ax.text(0.5,0.6,'r= %1.2f' % ses.noise_corr[sourcecell,targetcell],transform=ax.transAxes,ha='center',va='center',fontsize=8,color='k')
    # ax.text(250,250,r'NC = %1.2f' % ses.noise_corr[sourcecell,targetcell])
    
    NC = np.empty(len(oris))
    for iori,ori in enumerate(oris):
        idx_ori = np.where(ses.trialdata['Orientation']==ori)[0]
        NC[iori] = np.corrcoef(ses.respmat_res[sourcecell,idx_ori],ses.respmat_res[targetcell,idx_ori])[0,1]
    ax = axes[4]
    ax.plot(oris,NC,c='k',linewidth=2)
    ax.set_xticks(oris[::2],oris[::2].astype(int),rotation=45)

    # ax_nticks(ax,5)
    ax.set_xlabel('Stimulus Orientation')
    ax.set_ylabel('Noise Correlation')
    ax.set_ylim([-0.2,1])
    ax.axhline(0, color='k', lw=0.5, ls=':')
    # ax.set_title('Residual activity')
    # ax.text(0.1,0.8,'r= %1.2f' % ses.noise_corr[sourcecell,targetcell],transform=ax.transAxes,ha='center',va='center',fontsize=10,color='k')
    # ax.text(250,250,r'NC = %1.2f' % s

    ax = axes[5]
    # resp_product = sessions[ises].meanresp_orig[sourcecell,:] * sessions[ises].meanresp_orig[targetcell,:]
    resp_product = minmax_scale(sessions[ises].meanresp_orig[sourcecell,:]) * minmax_scale(sessions[ises].meanresp_orig[targetcell,:])
    ax.scatter(resp_product,NC)
            #    c=pal[ses.trialdata['
            # urves')
    ax.set_ylabel('Noise Correlation')
    # ax.set_title('NC vs Tuning Product')

    sns.despine(fig=fig, top=True, right=True, offset=2,trim=False)
    plt.tight_layout()
    return fig

#%% Find a neuron pair that is strongly tuned, has similar tuning pref and has negative correlation
ises = 0
idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],prctile)
N = len(sessions[ises].celldata)
signal_filter = np.full((N,N),False)
signal_filter[np.ix_(idx,idx)] = True
idx = np.all((sessions[ises].noise_corr > 0.5,sessions[ises].delta_pref == 0,signal_filter),axis=0)
sourcecells,targetcells = np.where(idx)
random_cell = np.random.choice(len(sourcecells))
sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
# my_savefig(fig, os.path.join(savedir,'NoiseCorrelations'), 'NC_example_isotuning_%s_cell%d_%d' % (sessions[ises].session_id,sourcecell,targetcell), formats = ['png']) 

#%% Find a neuron pair that is strongly tuned, has opposite tuning pref and has negative correlation
ises = 0
prctile = 90

idx = sessions[ises].celldata['tuning_var']>np.percentile(sessions[ises].celldata['tuning_var'],prctile)
N = len(sessions[ises].celldata)
signal_filter = np.full((N,N),False)
signal_filter[np.ix_(idx,idx)] = True
idx = np.all((sessions[ises].noise_corr < -0.05,sessions[ises].delta_pref == 90,signal_filter),axis=0)
sourcecells,targetcells = np.where(idx)
random_cell = np.random.choice(len(sourcecells))
sourcecell,targetcell = sourcecells[random_cell],targetcells[random_cell]

fig = plot_noise_pair(sessions[ises],sourcecell,targetcell)
# my_savefig(fig, os.path.join(savedir,'NoiseCorrelations'), 'NC_example_orthotuning_%s_cell%d_%d' % (sessions[ises].session_id,sourcecell,targetcell), formats = ['png']) 

#%% Prediction is the noise correlation is strongest for stimuli that are both preferred
# Show that for a given pair of neurons the product of their response is correlated with
# the noise correlation:






#%%
oris = np.arange(0,360,22.5)


idx_N   = np.all((sessions[ises].noise_corr < -0.05,sessions[ises].delta_pref == 90,signal_filter),axis=0)
idx_N   = np.where((np.sum(sessions[ises].respmat>0,axis=0) / len(sessions[ises].trialdata)) > 0.9)[0]

idx_N = np.random.choice(idx_N,2,replace=False)

fig,axes = plt.subplots(2,2,figsize=(6,6))

for iN,N in enumerate(idx_N):
    for iori,ori in enumerate(oris[:2]):
        ax = axes[iori,iN]
        idx_ori = np.where(sessions[ises].trialdata['Orientation']==ori)[0]
        data = sessions[ises].respmat[N,idx_ori][:,np.newaxis]
        data = data[np.random.choice(np.shape(data)[0],size=50,replace=False)]
        data = np.sort(data,axis=0)

        ax.imshow(data,aspect=0.1,origin='lower',vmin=0,vmax=np.percentile(data,99),cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # ax.set_aspect(0.1)



#%% Pick several examples:

idxs = 




#%% Show nonlinearities at p0 initialization
x = np.linspace(-0.5, 1.25, 300)
clrs_nl  = sns.color_palette('tab10', nNL)

fig, axes = plt.subplots(1, nNL, figsize=(nNL * 2.2, 3), sharey=True)
for i, (name, nl_func, n_shape, p0_shape, _) in enumerate(NL_CONFIGS):
    ax = axes[i]
    y  = nl_func(x, *p0_shape) if n_shape else nl_func(x)
    ax.plot(x, y, color=clrs_nl[i], lw=2)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('u  (θ_k + γ·P + b)')
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    if i == 0:
        ax.set_ylabel('f(u)')
    if p0_shape:
        ax.text(0.05, 0.95, ', '.join([f'{v}' for v in p0_shape]),
                transform=ax.transAxes, fontsize=7, va='top', color='gray')

sns.despine(trim=True, offset=3)

plt.suptitle('Nonlinearities at p0 initialization', fontsize=10, y=1.02)
plt.tight_layout()
# my_savefig(fig, savedir, 'NL_p0_shapes', formats=['png'])

#%% Pick two example neurons: well-tuned with moderate–high pop coupling
ises     = 0
ses      = sessions[ises]
poprate  = np.nanmean(zscore(ses.respmat, axis=1), axis=0)   # (nTrials,)
# poprate  = np.nanmean(ses.respmat, axis=0)   # (nTrials,)
ustim    = np.unique(ses.trialdata['Orientation'])
stim_ids = np.searchsorted(ustim, ses.trialdata['Orientation'].to_numpy())
nstim    = len(ustim)

idx_good = np.where(
    (ses.celldata['gOSI']           > 0.5) &
    # (ses.celldata['gOSI']           <0.2) &
    (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 70))
    # (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 50)) &
    # (ses.celldata['noise_level']   < 20)
    )[0]

# sigmoidaldiff = ses.celldata['R2Sigmoid'] - ses.celldata['R2Power-law (p=2)']
# idx_good = np.where(sigmoidaldiff > np.nanpercentile(sigmoidaldiff, 80))[0]
# np.random.seed(42)
ex_iN    = np.random.choice(idx_good)
# ex_iN = 0
resp_ex  = ses.respmat[ex_iN, :]
# Normalise responses to [0, 1]
r_min     = resp_ex.min()
r_max     = resp_ex.max()
# r_max     = np.percentile(resp_ex, 99)
resp_ex = (resp_ex - r_min) / max(r_max - r_min, 1e-8)
# resp_ex = zscore(resp_ex)

ex_cid   = ses.celldata['cell_id'].iloc[ex_iN]
print(f'Example: {ex_cid}  OSI={ses.celldata["gOSI"].iloc[ex_iN]:.2f}  '
      f'pop_coupling={ses.celldata["pop_coupling"].iloc[ex_iN]:.2f}')

results_ex = fit_nl_models(resp_ex, stim_ids, poprate, NL_CONFIGS)


#%%
[sessions, theta_arr, nlpar_arr, ses_idx_arr] = fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS)
#  fit_nl_models(sessions, nl_configs=NL_CONFIGS, verbose=False):




