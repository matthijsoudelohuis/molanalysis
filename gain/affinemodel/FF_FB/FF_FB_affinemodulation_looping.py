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
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

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
calciumversion = 'deconv'
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
                    'V1lab-V1unl-PMlabL2/3',
                    # 'V1lab-V1unl-PMunlL5',
                    # 'V1lab-V1unl-PMlabL5',
                    'PMlab-PMunl-V1unlL2/3',
                    'PMlab-PMunl-V1labL2/3',
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

# #Regression output:
# nboots                  = 100
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
# sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

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

        # if nboots:
        #     bootregressdata  = np.full((N,nboots,3),np.nan)
        #     bootregress_sig  = np.full((N,2),0)
        #     for iboot in range(nboots):
        #         meanrespboot            = np.empty([N,len(oris),2])
        #         for i,ori in enumerate(oris):
        #             idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
        #             idx_K1              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
        #             idx_K2              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
        #             meanrespboot[:,i,0]     = np.nanmean(respdata[:,idx_K1],axis=1)
        #             meanrespboot[:,i,1]     = np.nanmean(respdata[:,idx_K2],axis=1)
        #         for n in range(N):
        #             bootregressdata[n,iboot,:] = linregress(meanrespboot[n,:,0],meanrespboot[n,:,1])[:3]

        #     bootregress_sig[regressdata[:,0]>np.percentile(bootregressdata[:,:,0],97.5,axis=1),0] = 1
        #     bootregress_sig[regressdata[:,0]<np.percentile(bootregressdata[:,:,0],2.5,axis=1),0] = -1
        #     bootregress_sig[regressdata[:,1]>np.percentile(bootregressdata[:,:,1],97.5,axis=1),1] = 1
        #     bootregress_sig[regressdata[:,1]<np.percentile(bootregressdata[:,:,1],2.5,axis=1),1] = -1

        #     sig_params_regress[idx_ses,ialp,:] = bootregress_sig[idx_N3]

        #Aligned:
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # # normalize by peak response
        # tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        # meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

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

# #%% use 
# ialp = 0
# clrs_arealabelpairs = ['green','purple']
# legendlabels        = ['FF','FB']

# #%% Get good multiplicatively modulated cells by FF or FB:
# #mutliplicative: 
# idx_examples = np.all((params_regress[:,ialp,0]>np.nanpercentile(params_regress[:,ialp,0],80),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)
# #divisive:
# idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)

# print(celldata['cell_id'][idx_examples])

# example_cell      = np.random.choice(celldata['cell_id'][idx_examples])
# example_cells     = np.random.choice(celldata['cell_id'][idx_examples],2)

# #%% Get good additively modulated cells by FB: 
# #additive:
# idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],80),
#                        params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],70),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)
# # #subtractive:
# # idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],70),
# #                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],25),
# #                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
# #                        ),axis=0)
# print(celldata['cell_id'][idx_examples])

# example_cell      = np.random.choice(celldata['cell_id'][idx_examples])


#%%
                                            
#       ####   ####  #####  # #    #  ####  
#      #    # #    # #    # # ##   # #    # 
#      #    # #    # #    # # # #  # #      
#      #    # #    # #####  # #  # # #  ### 
#      #    # #    # #      # #   ## #    # 
######  ####   ####  #      # #    #  ####  


#%%
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=40)
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%%
fracdata = np.full((narealabelpairs,2,nSessions),np.nan)

for ises in range(nSessions):
    idx_ses = np.isin(celldata['session_id'],sessions[ises].session_id)

    for ialp,alp in enumerate(arealabelpairs):

        idx_N = np.all((idx_ses,
                        ~np.isnan(corrsig_cells[ialp,:]),
                        celldata['nearby'],
                        rangeresp>0.04,
                        # np.any(params_regress[:,:,2] > 0.5,axis=1),
                        ),axis=0)
        fracdata[ialp,0,ises] = np.sum(corrsig_cells[ialp,idx_N]==1) / np.sum(idx_N)
        fracdata[ialp,1,ises] = np.sum(corrsig_cells[ialp,idx_N]==-1) / np.sum(idx_N)
        # fracdata[ialp,1,ises] = np.sum(corrsig_cells[ialp,idx_ses]==-1) / np.sum(~np.isnan(corrsig_cells[ialp,idx_ses]))

clrs = ['black','red']
axtitles = np.array(['FF: +corr','FF: -corr', 'FB: +corr','FB: -corr'])
fig,axes = plt.subplots(1,4,figsize=(12,3))

ax = axes[0]
sns.barplot(data=fracdata[:2,0,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[:2,0,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[0,0,:],fracdata[1,0,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[:2,0,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[:2])
print(p)

ax = axes[1]
sns.barplot(data=fracdata[:2,1,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[:2,1,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[0,1,:],fracdata[1,1,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[:2,1,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[:2])
print(p)

ax = axes[2]
sns.barplot(data=fracdata[2:,0,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[2:,0,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[2,0,:],fracdata[3,0,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[2:,0,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[2:])
print(p)

ax = axes[3]
sns.barplot(data=fracdata[2:,1,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[2:,1,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[2,1,:],fracdata[3,1,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[2:,1,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[2:])
print(p)

for iax,ax in enumerate(axes):
    ax.set_title(axtitles[iax])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_FracSig_%dsessions' % (nSessions), formats = ['png'])

#%% Bootstrapped comparison of correlations and significant correlations with other area: 

# For each bootstrap, for each labeled cell a random nonlabeled cell that is nearby is sampled. 
# This results in an equal number of nonlabeled cells in a paired way. 
# The distribution of correlations is compared to the loop correlation distribution.
# The fraction of significantly positive and negative as well. 

nboots          = 100
minrangeresp    = 0.04
radius          = 50

loopfrac = np.full((2,2),np.nan) # FF vs FB, +corr vs -corr
loopmean = np.full((2),np.nan) # FF vs FB, +corr vs -corr
loopmean_abs = np.full((2),np.nan) # FF vs FB, +corr vs -corr
bootfrac = np.full((2,2,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean = np.full((2,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean_abs = np.full((2,nboots),np.nan) # FF vs FB, +corr vs -corr

binedges = np.linspace(-1,1,50)
nhistbins = len(binedges)-1
loophist =  np.full((2,nhistbins),np.nan) # FF vs FB, +corr vs -corr
boothist =  np.full((2,nhistbins,nboots),np.nan) # FF vs FB, +corr vs -corr

idx_N = np.all((
                # ~np.isnan(corrsig_cells[ialp,:]),
                # celldata['nearby'],
                rangeresp>minrangeresp,
                ),axis=0)

loopfrac[0,0] = np.sum(corrsig_cells[1,idx_N]==1) / np.sum(~np.isnan(corrsig_cells[1,idx_N]))
loopfrac[0,1] = np.sum(corrsig_cells[1,idx_N]==-1) / np.sum(~np.isnan(corrsig_cells[1,idx_N]))
loopfrac[1,0] = np.sum(corrsig_cells[3,idx_N]==1) / np.sum(~np.isnan(corrsig_cells[3,idx_N]))
loopfrac[1,1] = np.sum(corrsig_cells[3,idx_N]==-1) / np.sum(~np.isnan(corrsig_cells[3,idx_N]))
loopmean[0]     = np.nanmean(corrdata_cells[1,idx_N])
loopmean[1]     = np.nanmean(corrdata_cells[3,idx_N])
loopmean_abs[0]     = np.nanmean(np.abs(corrdata_cells[1,idx_N]))
loopmean_abs[1]     = np.nanmean(np.abs(corrdata_cells[3,idx_N]))

histcounts      = np.histogram(corrdata_cells[1,idx_N],bins=binedges)[0]
loophist[0,:]   = np.cumsum(histcounts)/np.sum(histcounts)
histcounts      = np.histogram(corrdata_cells[3,idx_N],bins=binedges)[0]
loophist[1,:]   = np.cumsum(histcounts)/np.sum(histcounts)

idx_PMlab       = np.where(celldata['arealayerlabel'] == 'PMlabL2/3')[0]
idx_V1lab       = np.where(celldata['arealayerlabel'] == 'V1labL2/3')[0]

for iboot in tqdm(range(nboots),total=nboots,desc='Bootstrapping'):
    idx_PMlab_nearby = np.full(len(idx_PMlab),np.nan)
    for iN,N in enumerate(idx_PMlab):
        #get index of which session this labeled cell comes from:
        ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
        #get index of all cells in this session
        idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
        #get index of labeled cell in this session
        idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
        #get index of all unlabeled cells in this session that are nearby this particular labeled cell
        idx_nearby_ses = np.where(np.all((np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<radius),
                                                 rangeresp[idx_ses]>minrangeresp,
                                                 sessions[ises].celldata['redcell']==0,
                                                 ),axis=0))[0]
        #Convert this index to the index in the whole dataset
        idx_nearby = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
        if len(idx_nearby) > 0: #pick a random one from the selected nearby cells
            idx_PMlab_nearby[iN] = np.random.choice(idx_nearby,1) 
    idx_PMlab_nearby = idx_PMlab_nearby[~np.isnan(idx_PMlab_nearby)].astype(int) #remove nans

    bootfrac[0,0,iboot] = np.sum(corrsig_cells[0,idx_PMlab_nearby]==1) / len(idx_PMlab_nearby) #compute fraction of sig pos for this boot
    bootfrac[0,1,iboot] = np.sum(corrsig_cells[0,idx_PMlab_nearby]==-1) / len(idx_PMlab_nearby)
    
    histcounts = np.histogram(corrdata_cells[0,idx_PMlab_nearby],bins=binedges)[0]
    boothist[0,:,iboot] = np.cumsum(histcounts)/np.sum(histcounts)
    bootmean[0,iboot] = np.nanmean(corrdata_cells[0,idx_PMlab_nearby])
    bootmean_abs[0,iboot] = np.nanmean(np.abs(corrdata_cells[0,idx_PMlab_nearby]))

    idx_V1lab_nearby = np.full(len(idx_V1lab),np.nan)
    for iN,N in enumerate(idx_V1lab):
        ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0]
        idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
        idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
        
        idx_nearby_ses = np.where(np.all((np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<radius),
                                                 rangeresp[idx_ses]>minrangeresp,
                                                 sessions[ises].celldata['redcell']==0,
                                                 ),axis=0))[0]
        idx_nearby = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]

        if len(idx_nearby) > 0:
            idx_V1lab_nearby[iN] = np.random.choice(idx_nearby,1)
    idx_V1lab_nearby = idx_V1lab_nearby[~np.isnan(idx_V1lab_nearby)].astype(int)

    bootfrac[1,0,iboot] = np.sum(corrsig_cells[2,idx_V1lab_nearby]==1) / len(idx_V1lab_nearby)
    bootfrac[1,1,iboot] = np.sum(corrsig_cells[2,idx_V1lab_nearby]==-1) / len(idx_V1lab_nearby)

    histcounts = np.histogram(corrdata_cells[2,idx_V1lab_nearby],bins=binedges)[0]
    boothist[1,:,iboot] = np.cumsum(histcounts)/np.sum(histcounts)
    bootmean[1,iboot] = np.nanmean(corrdata_cells[2,idx_V1lab_nearby])
    bootmean_abs[1,iboot] = np.nanmean(np.abs(corrdata_cells[2,idx_V1lab_nearby]))

#%% 
clrs_arealabelpairs = ['green','purple']
legendlabels = ['FF','FB']
axisbuffer = 0.05
lw = 2
# fig,axes = plt.subplots(2,5,figsize=(12,5),sharex='col')
fig,axes = plt.subplots(2,5,figsize=(10,4))
for ialp in range(2):
    axes[ialp,0].plot(binedges[:-1],loophist[ialp,:],color=clrs_arealabelpairs[ialp])
    
    shaded_error(binedges[:-1],np.nanmean(boothist[ialp,:,:],axis=1),np.nanstd(boothist[ialp,:,:],axis=1),
                    ax=axes[ialp,0],color='grey')
    axes[ialp,0].set_xlim([binedges[np.where(loophist[ialp,:]>0)[0][0]],binedges[np.where(loophist[ialp,:]==1)[0][0]]])
    axes[ialp,0].set_ylim([0,1])
    axes[ialp,0].set_ylabel(legendlabels[ialp],fontsize=15,fontweight='bold',color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        axes[ialp,0].set_title('Corr. coeff.')
        # axes[ialp,0].set_xlabel('Corr. coeff.')
    
    axidx = 1
    axes[ialp,axidx].axvline(loopmean[ialp],color=clrs_arealabelpairs[ialp],linewidth=lw)
    sns.histplot(bootmean[ialp,:],ax=axes[ialp,axidx],bins=np.linspace(-.1,1,500),element='step',stat='probability',color='grey')
    axes[ialp,axidx].set_xlim([loopmean[ialp]-axisbuffer,loopmean[ialp]+axisbuffer])
    if loopmean[ialp]<np.percentile(bootmean[ialp,:],2.5) or loopmean[ialp]>np.percentile(bootmean[ialp,:],97.5):
        axes[ialp,axidx].text(loopmean[ialp],0.1,'*',fontsize=20,color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        # axes[ialp,1].set_xlabel('Mean Corr.')
        axes[ialp,axidx].set_title('Mean Corr.')

    axidx = 2
    axes[ialp,axidx].axvline(loopmean_abs[ialp],color=clrs_arealabelpairs[ialp],linewidth=lw)
    sns.histplot(bootmean_abs[ialp,:],ax=axes[ialp,axidx],bins=np.linspace(-.1,1,500),element='step',stat='probability',color='grey')
    axes[ialp,axidx].set_xlim([loopmean_abs[ialp]-axisbuffer,loopmean_abs[ialp]+axisbuffer])
    if loopmean_abs[ialp]<np.percentile(bootmean_abs[ialp,:],2.5) or loopmean_abs[ialp]>np.percentile(bootmean_abs[ialp,:],97.5):
        axes[ialp,axidx].text(loopmean_abs[ialp],0.1,'*',fontsize=20,color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        # axes[ialp,2].set_xlabel('Mean Abs. Corr.')
        axes[ialp,axidx].set_title('Mean Abs. Corr.')
    
    axidx = 3
    axes[ialp,axidx].axvline(loopfrac[ialp,0],color=clrs_arealabelpairs[ialp],linewidth=lw)
    sns.histplot(bootfrac[ialp,0],ax=axes[ialp,axidx],bins=np.linspace(-.1,1,500),element='step',stat='probability',color='grey')
    axes[ialp,axidx].set_xlim([np.percentile(bootfrac[ialp,0],0)-axisbuffer,np.percentile(bootfrac[ialp,0],100)+axisbuffer])
    if loopfrac[ialp,0]<np.percentile(bootfrac[ialp,0],2.5) or loopfrac[ialp,0]>np.percentile(bootfrac[ialp,0],97.5):
        axes[ialp,axidx].text(loopfrac[ialp,0],0.1,'*',fontsize=20,color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        # axes[ialp,axidx].set_xlabel('Frac. Pos. Corr.')
        axes[ialp,axidx].set_title('Frac. Pos. Corr.')
    
    axidx = 4
    axes[ialp,axidx].axvline(loopfrac[ialp,1],color=clrs_arealabelpairs[ialp],linewidth=lw)
    sns.histplot(bootfrac[ialp,1],ax=axes[ialp,axidx],bins=np.linspace(-.1,1,500),element='step',stat='probability',color='grey')
    axes[ialp,axidx].set_xlim([np.percentile(bootfrac[ialp,1],0)-axisbuffer,np.percentile(bootfrac[ialp,1],100)+axisbuffer])
    if loopfrac[ialp,1]<np.percentile(bootfrac[ialp,1],2.5) or loopfrac[ialp,1]>np.percentile(bootfrac[ialp,1],97.5):
        axes[ialp,axidx].text(loopfrac[ialp,1],0.1,'*',fontsize=20,color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        # axes[ialp,4].set_xlabel('Frac. Neg. Corr.')
        axes[ialp,axidx].set_title('Frac. Neg. Corr.')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_Looped_Correlations_bootstrapped_%dsessions' % (nSessions), formats = ['png'])

#%% 




