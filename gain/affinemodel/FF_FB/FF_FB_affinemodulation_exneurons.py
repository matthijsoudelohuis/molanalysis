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
from scipy.stats import linregress
import statsmodels.formula.api as smf
from scipy import stats

os.chdir('e:\\Python\\molanalysis')

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching
from utils.plot_lib import * #get all the fixed color schemes
from preprocessing.preprocesslib import assign_layer,assign_layer2
from utils.rf_lib import filter_nearlabeled
from utils.RRRlib import regress_out_behavior_modulation

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain\\Affine_FF_vs_FB'

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
# session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%%  Load data properly:        
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
# t_pre       = -1    #pre s
# t_post      = 1.9     #post s
# binsize     = 0.2
calciumversion = 'deconv'
# vidfields   = np.concatenate((['videoPC_%d'%i for i in range(30)],
#                             ['pupil_area','pupil_ypos','pupil_xpos']),axis=0)
# behavfields = np.array(['runspeed','diffrunspeed'])

t_pre       = -1         #pre s
t_post      = 2.17        #post s
binsize     = 1/5.35

for ises in tqdm(range(nSessions),total=nSessions,desc='Loading data'):
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion)
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'], 
                                 method='nearby')
    
    # [sessions[ises].tensor_vid,t_axis] = compute_tensor(sessions[ises].videodata[vidfields], sessions[ises].videodata['ts'], sessions[ises].trialdata['tOnset'], 
    #                              t_pre, t_post, method='binmean',binsize=binsize)
    # #Subsample behavioral data 10 times before binning:
    # sessions[ises].behaviordata.drop('session_id',axis=1,inplace=True)
    # sessions[ises].behaviordata = sessions[ises].behaviordata.groupby(sessions[ises].behaviordata.index // 10).mean()
    # sessions[ises].behaviordata['diffrunspeed'] = np.diff(sessions[ises].behaviordata['runspeed'],prepend=0)
    # [sessions[ises].tensor_run,t_axis] = compute_tensor(sessions[ises].behaviordata[behavfields], sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'], 
    #                              t_pre, t_post, method='binmean',binsize=binsize)
    
    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'behaviordata')
    delattr(sessions[ises],'videodata')

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)


#%% Compute Tuning Metrics (gOSI, gDSI etc.)
sessions = compute_tuning_wrapper(sessions)

#%%
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=50)

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 
    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 


#%%

#%% Show tuning curve when activityin the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5
alphathr                = 0.001 #threshold for correlation with cross area rate

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Regression output:
nboots                  = 100
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)


for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        #Ratio:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        #Difference:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        # meanpopact          = np.nanmean(respdata[idx_N2,:],axis=0) - np.nanmean(respdata[idx_N1,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        errorresp           = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            errorresp[:,i,0]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            errorresp[:,i,1]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]
        error_resp_split[ialp,:,:,idx_ses] = errorresp[idx_N3]

        regressdata          = np.full((N,3),np.nan)
        regress_sig          = np.full((N,2),0)
        for n in range(N):
            xdata = meanresp[n,:,0]
            ydata = meanresp[n,:,1]
            regressdata[n,:] = linregress(xdata,ydata)[:3]
        params_regress[idx_ses,ialp,:] = regressdata[idx_N3]

        if nboots:
            bootregressdata  = np.full((N,nboots,3),np.nan)
            bootregress_sig  = np.full((N,2),0)
            for iboot in range(nboots):
                meanrespboot            = np.empty([N,len(oris),2])
                for i,ori in enumerate(oris):
                    idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
                    idx_K1              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    idx_K2              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    meanrespboot[:,i,0]     = np.nanmean(respdata[:,idx_K1],axis=1)
                    meanrespboot[:,i,1]     = np.nanmean(respdata[:,idx_K2],axis=1)
                for n in range(N):
                    bootregressdata[n,iboot,:] = linregress(meanrespboot[n,:,0],meanrespboot[n,:,1])[:3]

            bootregress_sig[regressdata[:,0]>np.percentile(bootregressdata[:,:,0],97.5,axis=1),0] = 1
            bootregress_sig[regressdata[:,0]<np.percentile(bootregressdata[:,:,0],2.5,axis=1),0] = -1
            bootregress_sig[regressdata[:,1]>np.percentile(bootregressdata[:,:,1],97.5,axis=1),1] = 1
            bootregress_sig[regressdata[:,1]<np.percentile(bootregressdata[:,:,1],2.5,axis=1),1] = -1

            sig_params_regress[idx_ses,ialp,:] = bootregress_sig[idx_N3]

        #Aligned:
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# # Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
# for iN in tqdm(range(nCells),total=nCells,desc='Fitting gain for each neuron'):
#     for ialp,alp in enumerate(arealabelpairs):
#         xdata = mean_resp_split[ialp,:,0,iN]
#         ydata = mean_resp_split[ialp,:,1,iN]
#         params_regress[iN,ialp,:] = linregress(xdata,ydata)[:3]

#%% Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%% Show some example neurons:

#%% use 
ialp = 0
clrs_arealabelpairs = ['green','purple']
legendlabels        = ['FF','FB']

#%% Get good multiplicatively modulated cells by FF or FB:
#mutliplicative: 
idx_examples = np.all((params_regress[:,ialp,0]>np.nanpercentile(params_regress[:,ialp,0],80),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)
#divisive:
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)

print(celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(celldata['cell_id'][idx_examples])
example_cells     = np.random.choice(celldata['cell_id'][idx_examples],2)

#%% Get good additively modulated cells by FB: 
#additive:
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],80),
                       params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],70),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)
# #subtractive:
# idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],70),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],25),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)
print(celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(celldata['cell_id'][idx_examples])

#%% 
ialp = 1
example_cell      = np.random.choice(celldata['cell_id'][~np.isnan(params_regress[:,ialp,0])])

#%% Plot in two ways:
idx_N = np.where(celldata['cell_id']==example_cell)[0][0]
ustim = np.unique(sessions[ises].trialdata['Orientation'])
x = mean_resp_split[ialp,:,0,idx_N]
y = mean_resp_split[ialp,:,1,idx_N]
xerror = error_resp_split[ialp,:,0,idx_N]
yerror = error_resp_split[ialp,:,1,idx_N]

# clrs_stimuli    = sns.color_palette('viridis',8)
fig,axes = plt.subplots(1,2,figsize=(5,2.5))
ax = axes[0]
ax.scatter(ustim,x,color='k',s=10)
ax.plot(ustim,x,color='k',linestyle='-',linewidth=0.5)
ax.errorbar(ustim,x,yerr=xerror,color='k',ls='None',linewidth=1)

ax.scatter(ustim,y,color=clrs_arealabelpairs[ialp],s=10)
ax.plot(ustim,y,color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=0.5)
ax.errorbar(ustim,y,yerr=yerror,color=clrs_arealabelpairs[ialp],ls='None',linewidth=1)
ax.set_xlabel('Orientation',fontsize=10)
ax.set_ylabel('Response',fontsize=10)

ax = axes[1]
ax.scatter(x,y,color='k',s=10)
ax.errorbar(x,y,xerr=xerror,yerr=yerror,color='k',ls='None',linewidth=1)
b = linregress(x, y)
xp = np.linspace(np.percentile(x,0),np.percentile(x,100)*1.1,100)
ax.plot(xp,b[0]*xp+b[1],color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=2)

ax.plot([0,1],[0,1],color='grey',ls='--',linewidth=1)
ax.set_xlim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
ax.set_ylim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])

ax.set_ylabel('FB (high)',fontsize=10)
ax.set_xlabel('FB (Low)',fontsize=10)
ax_nticks(ax,3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])
# my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])
# my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly','BaselineCorrected'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])
