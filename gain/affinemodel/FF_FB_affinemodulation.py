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

os.chdir('e:\\Python\\molanalysis')

from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes
from preprocessing.preprocesslib import assign_layer,assign_layer2

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

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

#%%
# sessions = compute_pairwise_anatomical_distance(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions)


#%%


#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=250)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 


#%% 
arealayers = np.array(['V1L2/3','PML2/3','PML5'])
narealayers = len(arealayers)
maxnoiselevel = 20
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
clrs = ['black','red']
fig,axes = plt.subplots(1,narealayers,figsize=(narealayers*3,3),sharey=True,sharex=True)

for ial,al in enumerate(arealayers):
    ax = axes[ial]
    idx_N              = np.all((
                                celldata['noise_level']<maxnoiselevel,
                                celldata['arealayer']==al,
                                celldata['OSI']>0.6,
                                    ),axis=0)

    sns.histplot(data=celldata[idx_N],x='pop_coupling',hue='arealayerlabel',color='green',element="step",stat="density", 
                common_norm=False,alpha=0.2,bins=np.linspace(-0.3,1,100),cumulative=True,ax=ax,palette=clrs,legend=False)
    ax.set_title(al)
sns.despine(fig=fig, top=True, right=True,offset=0)

#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
# tuning curve of PM neurons and vice versa for feedback. Because the population activity in V1 and PM cofluctuates,
#  just taking the level of V1 or PM activity would confound the analysis with local activity levels. 
# So therefore I took the 10% of trials with the labeled cells being more active than unlabeled cells vs the 10% trials 
# with unlabeled cells being more active than labeled cells (e.g. for FF: mean of V1lab - mean of V1unl). This would 
# be a proxy of epochs of particularly high FF activity, vs epochs of low FF activity (while controlling for overall 
# activity levels). Then the population tuning curve of PMunl or PMlab is plotted computed on these trials separately.
# You can see that high FF activity has very small divisive effect, while high FB activity has a clear multiplicative 
# effect. I also checked the effect on individual neurons (fitting affine modulation per neuron) but they mainly reflect 
# the mean. There are also additive effects, but the magnitude of the additive effects does not seem larger for PM cells 
# when FF ratio is high (edited) 


arealabelpairs  = [
                    'V1labL2/3-V1unlL2/3-PMunlL2/3',
                    'V1labL2/3-V1unlL2/3-PMunlL5',
                    'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                    'PMlabL5-PMunlL5-V1unlL2/3',
                    ]

arealabelpairs  = [
                    'V1labL2/3-V1unlL2/3-PMunlL2/3',
                    'V1labL2/3-V1unlL2/3-PMlabL2/3',
                    'V1labL2/3-V1unlL2/3-PMunlL5',
                    'V1labL2/3-V1unlL2/3-PMlabL5',
                    'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                    'PMlabL2/3-PMunlL2/3-V1labL2/3',
                    'PMlabL5-PMunlL5-V1unlL2/3',
                    'PMlabL5-PMunlL5-V1labL2/3',
                    ]

# arealabelpairs  = [
                    # 'V1unl-PMunl-PMunl',
                    # 'V1unl-PMunl-PMlab',
                    # 'V1lab-PMunl-PMunl',
                    # 'V1lab-PMunl-PMlab',
                    # 'PMunl-V1unl-V1unl',
                    # 'PMunl-V1unl-V1lab',
                    # 'PMlab-V1unl-V1unl',
                    # 'PMlab-V1unl-V1lab',
                    # ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

nboots                  = 100
perc                    = 20
minnneurons             = 15
maxnoiselevel           = 20
mineventrate            = 0
# mineventrate            = np.nanpercentile(celldata['event_rate'],10)

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
corrdata                = np.full((narealabelpairs,nSessions),np.nan)
corrdata_boot           = np.full((narealabelpairs,nSessions,nboots),np.nan)
corrdata_confounds      = np.full((narealabelpairs,nSessions,3),np.nan)

from utils.rf_lib import filter_nearlabeled

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing correlations between rates and affine modulation'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respmat                = zscore(sessions[ises].respmat, axis=1)
    # poprate             = np.nanmean(data,axis=0)

    idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0]
        idx_N2              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1]
        idx_N3              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2]

        idx_N1              = np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['depth']>depthcutoff,
                                      sessions[ises].celldata['OSI']>0.6,
                                    #   sessions[ises].celldata['gOSI']>0.5,
                                    # idx_nearby,
                                    sessions[ises].celldata['event_rate']>np.nanpercentile(sessions[ises].celldata['event_rate'],mineventrate),

                                      ),axis=0)
        idx_N2              = np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        # sessions[ises].celldata['gOSI']>0.5,
                                          sessions[ises].celldata['OSI']>0.6,
                                        # sessions[ises].celldata['depth']>depthcutoff,
                                    sessions[ises].celldata['event_rate']>np.nanpercentile(sessions[ises].celldata['event_rate'],mineventrate),
                                    # idx_nearby,
                                      ),axis=0)

        if np.sum(idx_N1) < minnneurons or np.sum(idx_N2) < minnneurons or np.sum(idx_N3) < minnneurons:
            continue

        idx_N3              = np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['gOSI']>np.nanpercentile(sessions[ises].celldata['gOSI'],50),
                                    #   sessions[ises].celldata['OSI']>0.5,
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    sessions[ises].celldata['event_rate']>np.nanpercentile(sessions[ises].celldata['event_rate'],mineventrate),
                                      ),axis=0)
        
        meanpopact          = np.nanmean(respmat[idx_N1,:],axis=0) - np.nanmean(respmat[idx_N2,:],axis=0)

        corrdata_confounds[ialp,ises,0]      = np.corrcoef(meanpopact,sessions[ises].respmat_runspeed)[0,1]
        corrdata_confounds[ialp,ises,1]      = np.corrcoef(meanpopact,sessions[ises].respmat_videome)[0,1]

        corrdata[ialp,ises]      = np.corrcoef(meanpopact,np.nanmean(respmat[idx_N3,:],axis=0))[0,1]

        sampleNneurons = min(np.sum(idx_N1),np.sum(idx_N2),np.sum(idx_N3))

        for iboot in range(nboots):
            bootidx_N1          = np.random.choice(np.where(idx_N1)[0],sampleNneurons,replace=True)
            bootidx_N2          = np.random.choice(np.where(idx_N2)[0],sampleNneurons,replace=True)
            bootidx_N3          = np.random.choice(np.where(idx_N3)[0],sampleNneurons,replace=True)
            corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respmat[bootidx_N1,:],axis=0) - np.nanmean(respmat[bootidx_N2,:],axis=0),
                                              np.nanmean(respmat[bootidx_N3,:],axis=0))[0,1]
        
        idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
        idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        for i,ori in enumerate(oris):
            meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)],axis=1)
            meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K2)],axis=1)
            
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response during still trials
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,0] = meanresp_pref[:,:,0] / tempmax
        # meanresp_pref[:,:,1] = meanresp_pref[:,:,1] / tempmax

        # meanresp_pref
        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]


#%%
plotdata = np.nanmean(corrdata_boot,axis=2)
# plotdata = corrdata

fig,ax = plt.subplots(1,1,figsize=(2.5,3))
df = pd.DataFrame({'correlation': plotdata.flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

# my_savefig(fig,savedir,'FF_FB_poprate_Corr_GR_%s_%dsessions' % (layerlabel,nSessions))

#%% 

fig,axes = plt.subplots(1,2,figsize=(6.5,3))
ax=axes[0]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,0].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')

ax=axes[1]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,1].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')

sns.despine(fig=fig, top=True, right=True,offset=3)
axes[0].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)
axes[1].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

# sns.barplot(data=corrdata_confounds[:,:,0],palette='pastel')
# corrdata_confounds

#%% 
data_gainregress_mean = np.full((narealabelpairs,3),np.nan)
# clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    xdata = np.nanmean(mean_resp_split[ialp,:,0,:],axis=1)
    ydata = np.nanmean(mean_resp_split[ialp,:,1,:],axis=1)
    b = linregress(xdata,ydata)
    data_gainregress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = data_gainregress_mean[ialp,0]*xvals + data_gainregress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=0.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.6,s=25)
    ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
    ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.text(0.1,0.8,'Slope: %1.2f\nOffest: %1.2f'%(data_gainregress_mean[ialp,0],data_gainregress_mean[ialp,1]),
            transform=ax.transAxes,fontsize=8)
    # ax.legend(frameon=False,loc='lower right')
    ax.set_xlabel('%s-%s low (Norm. Response)'%(alp.split('-')[0],alp.split('-')[1]))
    ax.set_ylabel('%s-%s high (Norm. Response)'%(alp.split('-')[0],alp.split('-')[1]))
ax.set_xlim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
ax.set_ylim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
my_savefig(fig,savedir,'FF_FB_affinemodulation_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
N = len(celldata)
data_gainregress = np.full((N,narealabelpairs,3),np.nan)
for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
    for ialp,alp in enumerate(arealabelpairs):
        xdata = mean_resp_split[ialp,:,0,iN]
        ydata = mean_resp_split[ialp,:,1,iN]
        b = linregress(xdata,ydata)
        data_gainregress[iN,ialp,:] = b[:3]

#%% 
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
for ialp,alp in enumerate(arealabelpairs):
    for iparam in range(2):
        ax = axes[iparam,ialp]
        # idx_N = data_gainregress[:,ialp,2] > 0.3
        idx_N =  celldata['OSI']>0.5
        # idx_N =  celldata['gOSI']>0.5

        sns.histplot(data=data_gainregress[idx_N,ialp,iparam],color=clrs_arealabelpairs[ialp],
                     ax=ax,stat='probability',bins=np.arange(-1,3,0.1))
        ax.axvline(0,color='grey',ls='--',linewidth=1)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
        ax.plot(np.nanmean(data_gainregress[idx_N,ialp,iparam]),0.25,markersize=10,
                color=clrs_arealabelpairs[ialp],marker='v')
        ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
        if iparam == 0:
            ax.set_xlabel('Slope')
        else:
            ax.set_xlabel('Offset')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,2,figsize=(6,3))
for iparam in range(2):
    ax = axes[iparam]
    # idx_N = data_gainregress[:,:,2] > 0.5
    idx_N =  celldata['OSI']>0.5
    if iparam==1:
        sns.barplot(data=np.abs(data_gainregress[idx_N,:,iparam]),palette=clrs_arealabelpairs,ax=ax)
    else: 
        sns.barplot(data=data_gainregress[idx_N,:,iparam],palette=clrs_arealabelpairs,ax=ax)
    ax.set_xticklabels(arealabelpairs)
    if iparam == 0:
        ax.set_title('Multiplicative')
    else:
        ax.set_title('Additive')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_affinemodulation_barplot_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
for ialp,alp in enumerate(arealabelpairs):
    for iparam in range(2):
        ax = axes[iparam,ialp]
        # idx_N = data_gainregress[:,ialp,2] > 0.3
        idx_N =  celldata['OSI']>0.5
        idx_N = np.all((celldata['OSI']>0.5,
                        ~np.isnan(data_gainregress[:,ialp,iparam])),axis=0)
        
        sns.regplot(x=celldata['pop_coupling'][idx_N],y=data_gainregress[idx_N,ialp,iparam],color=clrs_arealabelpairs[ialp],
                    ax=ax,scatter=True,marker='o',
                    scatter_kws={'alpha':0.5, 's':20, 'edgecolors':'white'},
                    line_kws={'color':clrs_arealabelpairs[ialp], 'ls':'-', 'linewidth':3})
        b = linregress(celldata['pop_coupling'][idx_N],data_gainregress[idx_N,ialp,iparam])

        ax.set_xlim(np.nanpercentile(celldata['pop_coupling'][idx_N],[1,99]))
        ax.set_ylim(np.nanpercentile(data_gainregress[idx_N,ialp,iparam],[1,99]))
        ax.text(0.1,0.8,'R2: %1.2f'%(b[2]),transform=ax.transAxes)
        ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
        ax.set_xlabel('Pop. Coupling')
        if iparam == 0:
            ax.set_ylabel('Slope')
        else:
            ax.set_ylabel('Offset')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_affinemodulation_popcoupling_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])












#%% Check whether the affine modulation modulation depends on the activity of the other area
perc                = 10

arealabelpairs      = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    ]

# arealabelpairs  = [
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab',
#                     # 'PMunl-PMlab'
#                     ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respmat                = zscore(sessions[ises].respmat, axis=1)
    # poprate             = np.nanmean(data,axis=0)
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = sessions[ises].celldata['arealabel'] == alp.split('-')[0]
        idx_N2              = sessions[ises].celldata['arealabel'] == alp.split('-')[1]

        meanpopact          = np.nanmean(respmat[idx_N1,:],axis=0)
        idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
        idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)

        # compute meanresp
        meanresp            = np.empty([N,len(oris),2])
        for i,ori in enumerate(oris):
            meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)],axis=1)
            meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K2)],axis=1)
        
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response during still trials
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        # meanresp_pref
        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N2])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N2]
    
#%% 
data_gainregress_mean = np.full((narealabelpairs,3),np.nan)
clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    xdata = np.nanmean(mean_resp_split[ialp,:,0,:],axis=1)
    ydata = np.nanmean(mean_resp_split[ialp,:,1,:],axis=1)
    b = linregress(xdata,ydata)
    data_gainregress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = data_gainregress_mean[ialp,0]*xvals + data_gainregress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=0.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.6,s=25)
    ax.plot([0,3],[0,3],'grey',ls='--',linewidth=1)
    ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
ax.legend(frameon=False,loc='lower right')
ax.set_xlabel('Low (Norm. Response)')
ax.set_ylabel('High (Norm. Response)')
ax.set_xlim([0,3.5])
ax.set_ylim([0,3.5])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

# temp = mean_resp_split[ialp,i,:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)]
# print(alp,ori,np.nanmean(temp),np.nanstd(temp),np.nanmedian(temp),np.nanpercentile(temp,25),np.nanpercentile(temp,75))



# DEPRECTAED: 





# #%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
# # tuning curve of PM neurons and vice versa for feedback. Because the population activity in V1 and PM cofluctuates,
# #  just taking the level of V1 or PM activity would confound the analysis with local activity levels. 
# # So therefore I took the 10% of trials with the labeled cells being more active than unlabeled cells vs the 10% trials 
# # with unlabeled cells being more active than labeled cells (e.g. for FF: mean of V1lab - mean of V1unl). This would 
# # be a proxy of epochs of particularly high FF activity, vs epochs of low FF activity (while controlling for overall 
# # activity levels). Then the population tuning curve of PMunl or PMlab is plotted computed on these trials separately.
# # You can see that high FF activity has very small divisive effect, while high FB activity has a clear multiplicative 
# # effect. I also checked the effect on individual neurons (fitting affine modulation per neuron) but they mainly reflect 
# # the mean. There are also additive effects, but the magnitude of the additive effects does not seem larger for PM cells 
# # when FF ratio is high (edited) 


# arealabelpairs  = [
#                     'V1lab-V1unl-PMunl',
#                     # 'V1lab-V1unl-PMlab',
#                     'PMlab-PMunl-V1unl',
#                     # 'PMlab-PMunl-V1lab',
#                     ]

# # arealabelpairs  = [
#                     # 'V1unl-PMunl-PMunl',
#                     # 'V1unl-PMunl-PMlab',
#                     # 'V1lab-PMunl-PMunl',
#                     # 'V1lab-PMunl-PMlab',
#                     # 'PMunl-V1unl-V1unl',
#                     # 'PMunl-V1unl-V1lab',
#                     # 'PMlab-V1unl-V1unl',
#                     # 'PMlab-V1unl-V1lab',
#                     # ]


# # arealabelpairs  = [
#                     # 'V1unl-PMunl-PMunl',
#                     # 'V1lab-PMlab-PMlab',
#                     # 'V1lab-PMlab-PMunl',
#                     # 'V1lab-PMlab-PMlab',
#                     # 'PMunl-V1unl-V1unl',
#                     # 'PMunl-V1lab-V1lab',
#                     # 'PMlab-V1lab-V1unl',
#                     # 'PMlab-V1lab-V1lab',
#                     # ]

# narealabelpairs         = len(arealabelpairs)

# celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

# nOris                   = 16
# nCells                  = len(celldata)
# oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

# nboots                  = 100
# perc                    = 20
# minnneurons             = 15
# maxnoiselevel           = 20
# mineventrate            = 10
# # mineventrate            = np.nanpercentile(celldata['event_rate'],10)

# mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
# corrdata                = np.full((narealabelpairs,nSessions),np.nan)
# corrdata_boot           = np.full((narealabelpairs,nSessions,nboots),np.nan)

# # layerlabel = 'L5'
# layerlabel = 'L23'

# from utils.rf_lib import filter_nearlabeled

# for ises in tqdm(range(nSessions),total=nSessions,desc='Computing correlations between rates and affine modulation'):
#     [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

#     respmat                = zscore(sessions[ises].respmat, axis=1)
#     # poprate             = np.nanmean(data,axis=0)

#     idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

#     for ialp,alp in enumerate(arealabelpairs):
#         idx_N1              = sessions[ises].celldata['arealabel'] == alp.split('-')[0]
#         idx_N2              = sessions[ises].celldata['arealabel'] == alp.split('-')[1]
#         idx_N3              = sessions[ises].celldata['arealabel'] == alp.split('-')[2]

#         idx_N1              = np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
#                                       sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                     #   sessions[ises].celldata['tuning_var']>0.025,
#                                     #   sessions[ises].celldata['depth']>depthcutoff,
#                                     #   sessions[ises].celldata['gOSI']>0.5,
#                                     # idx_nearby,
#                                     sessions[ises].celldata['event_rate']>np.nanpercentile(sessions[ises].celldata['event_rate'],mineventrate),

#                                       ),axis=0)
#         idx_N2              = np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
#                                       sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                         # sessions[ises].celldata['gOSI']>0.5,
#                                         # sessions[ises].celldata['depth']>depthcutoff,
#                                     sessions[ises].celldata['event_rate']>np.nanpercentile(sessions[ises].celldata['event_rate'],mineventrate),
#                                     # idx_nearby,
#                                       ),axis=0)

#         if np.sum(idx_N1) < minnneurons or np.sum(idx_N2) < minnneurons or np.sum(idx_N3) < minnneurons:
#             continue

#         idx_N3              = np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[2],
#                                     #   sessions[ises].celldata['tuning_var']>0.025,
#                                     #   sessions[ises].celldata['gOSI']>np.nanpercentile(sessions[ises].celldata['gOSI'],50),
#                                     #   sessions[ises].celldata['OSI']>0.5,
#                                       sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                     sessions[ises].celldata['event_rate']>np.nanpercentile(sessions[ises].celldata['event_rate'],mineventrate),
#                                       ),axis=0)
        
#         meanpopact          = np.nanmean(respmat[idx_N1,:],axis=0) - np.nanmean(respmat[idx_N2,:],axis=0)

#         corrdata[ialp,ises]      = np.corrcoef(meanpopact,np.nanmean(respmat[idx_N3,:],axis=0))[0,1]
#         sampleNneurons = min(np.sum(idx_N1),np.sum(idx_N2),np.sum(idx_N3))

#         for iboot in range(nboots):
#             bootidx_N1          = np.random.choice(np.where(idx_N1)[0],sampleNneurons,replace=True)
#             bootidx_N2          = np.random.choice(np.where(idx_N2)[0],sampleNneurons,replace=True)
#             bootidx_N3          = np.random.choice(np.where(idx_N3)[0],sampleNneurons,replace=True)
#             corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respmat[bootidx_N1,:],axis=0) - np.nanmean(respmat[bootidx_N2,:],axis=0),
#                                               np.nanmean(respmat[bootidx_N3,:],axis=0))[0,1]
        
#         idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
#         idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)

#         # compute meanresp for trials with low and high difference in lab-unl activation
#         meanresp            = np.empty([N,len(oris),2])
#         for i,ori in enumerate(oris):
#             meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)],axis=1)
#             meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K2)],axis=1)
            
#         # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
#         prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

#         meanresp_pref          = meanresp.copy()
#         for n in range(N):
#             meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
#             meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

#         # normalize by peak response during still trials
#         tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
#         meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
#         meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)
#         # meanresp_pref[:,:,0] = meanresp_pref[:,:,0] / tempmax
#         # meanresp_pref[:,:,1] = meanresp_pref[:,:,1] / tempmax

#         # meanresp_pref
#         # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
#         idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
#         mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

# #%%
# plotdata = np.nanmean(corrdata_boot,axis=2)
# # plotdata = corrdata

# fig,ax = plt.subplots(1,1,figsize=(2.5,3))
# df = pd.DataFrame({'correlation': plotdata.flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
# sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 95),palette='pastel',ax=ax)
# sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
# ax.axhline(0,linestyle='--',color='k')
# sns.despine(fig=fig, top=True, right=True,offset=3)
# ax.set_xticklabels(arealabelpairs,rotation=45,fontsize=8)

# my_savefig(fig,savedir,'FF_FB_poprate_Corr_GR_%s_%dsessions' % (layerlabel,nSessions))

# #%% 
# data_gainregress_mean = np.full((narealabelpairs,3),np.nan)
# # clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
# fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

# for ialp,alp in enumerate(arealabelpairs):
#     ax = axes[ialp]
#     xdata = np.nanmean(mean_resp_split[ialp,:,0,:],axis=1)
#     ydata = np.nanmean(mean_resp_split[ialp,:,1,:],axis=1)
#     b = linregress(xdata,ydata)
#     data_gainregress_mean[ialp,:] = b[:3]
#     xvals = np.arange(0,3,0.1)
#     yvals = data_gainregress_mean[ialp,0]*xvals + data_gainregress_mean[ialp,1]
#     ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=0.3)
#     ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.6,s=25)
#     ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
#     ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
#     ax.text(0.1,0.8,'Slope: %1.2f\nOffest: %1.2f'%(data_gainregress_mean[ialp,0],data_gainregress_mean[ialp,1]),
#             transform=ax.transAxes,fontsize=8)
#     # ax.legend(frameon=False,loc='lower right')
#     ax.set_xlabel('%s-%s low (Norm. Response)'%(alp.split('-')[0],alp.split('-')[1]))
#     ax.set_ylabel('%s-%s high (Norm. Response)'%(alp.split('-')[0],alp.split('-')[1]))
# ax.set_xlim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_ylim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

# #%% Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
# N = len(celldata)
# data_gainregress = np.full((N,narealabelpairs,3),np.nan)
# for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
#     for ialp,alp in enumerate(arealabelpairs):
#         xdata = mean_resp_split[ialp,:,0,iN]
#         ydata = mean_resp_split[ialp,:,1,iN]
#         b = linregress(xdata,ydata)
#         data_gainregress[iN,ialp,:] = b[:3]

# #%% 
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
# fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
# for ialp,alp in enumerate(arealabelpairs):
#     for iparam in range(2):
#         ax = axes[iparam,ialp]
#         # idx_N = data_gainregress[:,ialp,2] > 0.3
#         idx_N =  celldata['OSI']>0.5
#         # idx_N =  celldata['gOSI']>0.5

#         sns.histplot(data=data_gainregress[idx_N,ialp,iparam],color=clrs_arealabelpairs[ialp],
#                      ax=ax,stat='probability',bins=np.arange(-1,3,0.1))
#         ax.axvline(0,color='grey',ls='--',linewidth=1)
#         ax.axvline(1,color='grey',ls='--',linewidth=1)
#         ax.plot(np.nanmean(data_gainregress[idx_N,ialp,iparam]),0.25,markersize=10,
#                 color=clrs_arealabelpairs[ialp],marker='v')
#         ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
#         if iparam == 0:
#             ax.set_xlabel('Slope')
#         else:
#             ax.set_xlabel('Offset')

# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

# #%% 
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
# fig,axes = plt.subplots(1,2,figsize=(6,3))
# for iparam in range(2):
#     ax = axes[iparam]
#     # idx_N = data_gainregress[:,:,2] > 0.5
#     idx_N =  celldata['OSI']>0.5
#     if iparam==1:
#         sns.barplot(data=np.abs(data_gainregress[idx_N,:,iparam]),palette=clrs_arealabelpairs,ax=ax)
#     else: 
#         sns.barplot(data=data_gainregress[idx_N,:,iparam],palette=clrs_arealabelpairs,ax=ax)
#     ax.set_xticklabels(arealabelpairs)
#     if iparam == 0:
#         ax.set_title('Multiplicative')
#     else:
#         ax.set_title('Additive')
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_barplot_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

# #%% 
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
# fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
# for ialp,alp in enumerate(arealabelpairs):
#     for iparam in range(2):
#         ax = axes[iparam,ialp]
#         # idx_N = data_gainregress[:,ialp,2] > 0.3
#         idx_N =  celldata['OSI']>0.5
#         idx_N = np.all((celldata['OSI']>0.5,
#                         ~np.isnan(data_gainregress[:,ialp,iparam])),axis=0)
        
#         sns.regplot(x=celldata['pop_coupling'][idx_N],y=data_gainregress[idx_N,ialp,iparam],color=clrs_arealabelpairs[ialp],
#                     ax=ax,scatter=True,marker='o',
#                     scatter_kws={'alpha':0.5, 's':20, 'edgecolors':'white'},
#                     line_kws={'color':clrs_arealabelpairs[ialp], 'ls':'-', 'linewidth':3})
#         b = linregress(celldata['pop_coupling'][idx_N],data_gainregress[idx_N,ialp,iparam])

#         ax.set_xlim(np.nanpercentile(celldata['pop_coupling'][idx_N],[1,99]))
#         ax.set_ylim(np.nanpercentile(data_gainregress[idx_N,ialp,iparam],[1,99]))
#         ax.text(0.1,0.8,'R2: %1.2f'%(b[2]),transform=ax.transAxes)
#         ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
#         ax.set_xlabel('Pop. Coupling')
#         if iparam == 0:
#             ax.set_ylabel('Slope')
#         else:
#             ax.set_ylabel('Offset')

# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_popcoupling_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])





