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
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=50)


#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
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
                                celldata['nearby'],
                                    ),axis=0)

    sns.histplot(data=celldata[idx_N],x='pop_coupling',hue='arealayerlabel',color='green',element="step",stat="density", 
                common_norm=False,fill=False,bins=np.linspace(-0.3,1,1000),cumulative=True,ax=ax,palette=clrs,legend=False)
    ax.set_title(al)
sns.despine(fig=fig, top=True, right=True,offset=0)


#%% FOr every session remove behavior related variability:
rank_behavout = 3
# maxnoiselevel = 20

#%%
for ises in range(nSessions):

    # Convert response_matrix and orientations_vector to numpy arrays
    # response_matrix         = np.array(response_matrix)
    conditions_vector       = np.array(sessions[ises].trialdata['stimCond'])
    conditions              = np.sort(np.unique(conditions_vector))
    C                       = len(conditions)

    resp_mean       = sessions[ises].respmat.copy()
    # resp_res        = sessions[ises].respmat.copy()

    for iC,cond in enumerate(conditions):
        tempmean                            = np.nanmean(sessions[ises].respmat[:,conditions_vector==cond],axis=1)
        # resp_mean[:,iC]                     = tempmean
        resp_mean[:,conditions_vector==cond] = tempmean[:,np.newaxis]

    Y               = sessions[ises].respmat.copy()
    Y               = Y - resp_mean
    [Y_orig,Y_hat,Y_out,rank,ev] = regress_out_behavior_modulation(sessions[ises],X=None,Y=Y.T,
                                nvideoPCs = 30,rank=rank_behavout,lam=0,perCond=True,kfold = 5)
    # print(ev)
    sessions[ises].respmat_behavout = Y_out.T + resp_mean

# plt.imshow(sessions[ises].respmat,cmap='RdBu_r',vmin=0,vmax=100)
# plt.imshow(sessions[ises].respmat_behavout,cmap='RdBu_r',vmin=0,vmax=100)


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

arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    # 'V1lab-V1unl-PMlabL2/3',
                    # 'V1lab-V1unl-PMunlL5',
                    # 'V1lab-V1unl-PMlabL5',
                    'PMlab-PMunl-V1unlL2/3',
                    # 'PMlab-PMunl-V1labL2/3',
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
minnneurons             = 10
maxnoiselevel           = 20

alphathr                = 0.001
# maxnoiselevel           = 100
# mineventrate            = 0

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
corrdata                = np.full((narealabelpairs,nSessions),np.nan)
corrdata_boot           = np.full((narealabelpairs,nSessions,nboots),np.nan)
corrdata_confounds      = np.full((narealabelpairs,nSessions,3),np.nan)
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

# valuematching           = 'pop_coupling'
valuematching           = None
nmatchbins              = 5

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    # respdata            = zscore(sessions[ises].respmat, axis=1)
    # respdata            = zscore(sessions[ises].respmat_behavout, axis=1)

    respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    # idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

    for ialp,alp in enumerate(arealabelpairs):
        # idx_N1              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0]
        # idx_N2              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1]
        # idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2]

        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['OSI']>0.5,
                                    #   sessions[ises].celldata['gOSI']>0.5,
                                    # idx_nearby,

                                      ),axis=0))[0]
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        # sessions[ises].celldata['gOSI']>0.5,
                                        #   sessions[ises].celldata['OSI']>0.5,
                                    # idx_nearby,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['gOSI']>np.nanpercentile(sessions[ises].celldata['gOSI'],50),
                                    #   sessions[ises].celldata['OSI']>0.5,
                                    #   sessions[ises].celldata['gOSI']>0.5,
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        # if len(idx_N1) < minnneurons or len(idx_N2) < minnneurons or len(idx_N3) < minnneurons:
            # continue

        # if len(idx_N1) < minnneurons:
        #     continue

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue

        if valuematching is not None:
            #Get value to match from celldata for V1 matching
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_N1,idx_N2))
            group       = np.concatenate((np.zeros(len(idx_N1)),np.ones(len(idx_N2))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_N1      = np.intersect1d(idx_N1,idx_sub) #recover subset from idx_joint
            idx_N2      = np.intersect1d(idx_N2,idx_sub)
        
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        # meanpopact          = np.nanmean(zscore(respdata[idx_N1,:],axis=1),axis=0)

        corrdata_confounds[ialp,ises,0]      = np.corrcoef(meanpopact,sessions[ises].respmat_runspeed)[0,1]
        corrdata_confounds[ialp,ises,1]      = np.corrcoef(meanpopact,sessions[ises].respmat_videome)[0,1]
        corrdata_confounds[ialp,ises,2]      = np.corrcoef(meanpopact,sessions[ises].respmat_pupilarea)[0,1]

        corrdata[ialp,ises]      = np.corrcoef(meanpopact,np.nanmean(respdata[idx_N3,:],axis=0))[0,1]

        # sampleNneurons = min(np.sum(idx_N1),np.sum(idx_N2),np.sum(idx_N3))
        sampleNneurons = min(len(idx_N1),len(idx_N2),len(idx_N3))

        for iboot in range(nboots):
            bootidx_N1          = np.random.choice(idx_N1,sampleNneurons,replace=True)
            bootidx_N2          = np.random.choice(idx_N2,sampleNneurons,replace=True)
            bootidx_N3          = np.random.choice(idx_N3,sampleNneurons,replace=True)
            # corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respdata[bootidx_N1,:],axis=0) - np.nanmean(respdata[bootidx_N2,:],axis=0),
                                            #   np.nanmean(respdata[bootidx_N3,:],axis=0))[0,1]
            corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respdata[bootidx_N1,:],axis=0),
                                              np.nanmean(respdata[bootidx_N3,:],axis=0))[0,1]
        
        # idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
        # idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)
        # # compute meanresp for trials with low and high difference in lab-unl activation
        # meanresp            = np.empty([N,len(oris),2])
        # ori_ses             = sessions[ises].trialdata['Orientation']
        # oris                = np.unique(ori_ses)
        # for i,ori in enumerate(oris):
        #     meanresp[:,i,0] = np.nanmean(respdata[:,np.logical_and(ori_ses==ori,idx_K1)],axis=1)
        #     meanresp[:,i,1] = np.nanmean(respdata[:,np.logical_and(ori_ses==ori,idx_K2)],axis=1)
        
        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            idx_T               = ori_ses == ori
            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0] = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1] = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response during still trials
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        # meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,0] = meanresp_pref[:,:,0] / tempmax
        # meanresp_pref[:,:,1] = meanresp_pref[:,:,1] / tempmax

        # meanresp_pref
        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
N = len(celldata)
data_gainregress = np.full((N,narealabelpairs,3),np.nan)
for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
    for ialp,alp in enumerate(arealabelpairs):
        xdata = mean_resp_split[ialp,:,0,iN]
        ydata = mean_resp_split[ialp,:,1,iN]
        data_gainregress[iN,ialp,:] = linregress(xdata,ydata)[:3]

#%%
# idx_sigN = corrsig_cells[0,:]==1
# idx_sigN = corrsig_cells[0,:]==-1
# plt.hist(corrdata_cells[0,idx_sigN].flatten())
# corrsig_cells[ialp,idx_ses]==-1

#%%
plotdata = np.nanmean(corrdata_boot,axis=2)

fig,ax = plt.subplots(1,1,figsize=(2.5,3))
df = pd.DataFrame({'correlation': plotdata.flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

# my_savefig(fig,savedir,'FF_FB_poprate_arealayerpairs_GR_%dsessions' % (nSessions))

#%% 

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax=axes[0]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,0].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Running speed')

ax=axes[1]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,1].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Video ME')

ax=axes[2]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,2].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Pupil size')

for ax in axes:
    ax.set_ylim([-0.3,0.3])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
axes[0].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)
axes[1].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)
axes[2].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

my_savefig(fig,savedir,'FF_FB_poprate_confounds_GR_%dsessions' % (nSessions))

#%% 
data_gainregress_mean = np.full((narealabelpairs,3),np.nan)
# clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    # idx_N =  np.array(celldata['gOSI']>0.5)
    # idx_N = data_gainregress[:,ialp,2] > 0.5
    # idx_N = corrsig_cells[ialp,:]==-1

    idx_N =  np.all((
                    # celldata['gOSI']>0.5,
                    # celldata['nearby'],
                    # corrsig_cells[ialp,:]==1,
                    # np.any(mean_resp_split>0.5,axis=(0,1,2)),
                    np.any(data_gainregress[:,:,2] > 0.5,axis=1),
                     ),axis=0)

    xdata = np.nanmean(mean_resp_split[ialp,:,0,idx_N].T,axis=1)
    ydata = np.nanmean(mean_resp_split[ialp,:,1,idx_N].T,axis=1)
    b = linregress(xdata,ydata)
    data_gainregress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = data_gainregress_mean[ialp,0]*xvals + data_gainregress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=1.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.7,s=35)
    ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
    ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.text(0.1,0.8,'Slope: %1.2f\nOffest: %1.2f'%(data_gainregress_mean[ialp,0],data_gainregress_mean[ialp,1]),
            transform=ax.transAxes,fontsize=8)
    # ax.legend(frameon=False,loc='lower right')
    ax.set_xlabel('%s low (events/F0) '%(alp.split('-')[0]))
    ax.set_ylabel('%s high'%(alp.split('-')[0]))
    ax.set_xlim([0,np.nanmax([xdata,ydata])*1.1])
    ax.set_ylim([0,np.nanmax([xdata,ydata])*1.1])
# ax.set_xlim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_ylim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_xlim([np.nanmin(np.nanmean(mean_resp_split,axis=(3)))*1.1,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_ylim([np.nanmin(np.nanmean(mean_resp_split,axis=(3)))*1.1,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_%dsessions' % (nSessions), formats = ['png'])

#%%
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
for ialp,alp in enumerate(arealabelpairs):
    sns.histplot(data_gainregress[:,ialp,2],bins=np.linspace(-1,1.1,25),element='step',stat='probability',
                 color=clrs_arealabelpairs[ialp],fill=False,ax=ax)
ax.set_xlabel('R2')
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
my_savefig(fig,savedir,'AffineModel_R2_%dsessions' % (nSessions), formats = ['png'])

# #%% 
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)

# fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
# for ialp,alp in enumerate(arealabelpairs):
#     for iparam in range(2):
#         ax = axes[iparam,ialp]
#         idx_N = data_gainregress[:,ialp,2] > 0.5
#         # idx_N =  celldata['gOSI']>0.5

#         sns.histplot(data=data_gainregress[idx_N,ialp,iparam],color=clrs_arealabelpairs[ialp],
#                      ax=ax,stat='probability',bins=np.arange(-1,3,0.1))
#         ax.axvline(0,color='grey',ls='--',linewidth=1)
#         ax.axvline(1,color='grey',ls='--',linewidth=1)
#         ax.plot(np.nanmean(data_gainregress[idx_N,ialp,iparam]),0.2,markersize=10,
#                 color=clrs_arealabelpairs[ialp],marker='v')
#         ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
#         if iparam == 0:
#             ax.set_xlabel('Slope')
#         else:
#             ax.set_xlabel('Offset')

# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
clrs_arealabelpairs = ['green','purple']
legendlabels = ['FF','FB']
clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(1,2,figsize=(6,3))
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(-0.5,3,0.1)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.15,0.15,0.01)
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    handles = []
    for ialp,alp in enumerate(arealabelpairs):
        idx_N = data_gainregress[:,ialp,2] > 0.5
        # idx_N =  celldata['gOSI']>0.5

        sns.histplot(data=data_gainregress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],alpha=0.3,fill=True,linewidth=1,
                     ax=ax,stat='probability',bins=bins)
        # ax.axvline(0,color='grey',ls='--',linewidth=1)
        # ax.axvline(1,color='grey',ls='--',linewidth=1)
        handles.append(ax.plot(np.nanmean(data_gainregress[idx_N,ialp,iparam]),0.2,markersize=10,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
        # ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.legend(handles,legendlabels)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
from scipy import stats
clrs_arealabelpairs = ['green','purple']
ticklabels = ['FF','FB']
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,2,figsize=(4.5,3))
for iparam in range(2):
    ax = axes[iparam]
    # idx_N = np.all(data_gainregress[:,:,2] > 0.3,axis=1)
    # idx_N = np.any(data_gainregress[:,:,2] > 0.3,axis=1)
    # idx_N = data_gainregress[:,:,2] > 0.5
    # idx_N =  celldata['OSI']>0.5
    idx_N =  np.all((
                    # celldata['gOSI']>0.5,
                    # celldata['nearby'],
                    # np.any(corrsig_cells==1,axis=0),
                    # np.any(corrsig_cells==-1,axis=0),
                    np.any(data_gainregress[:,:,2] > 0.5,axis=1),
                     ),axis=0)
    sns.barplot(data=data_gainregress[idx_N,:,iparam],palette=clrs_arealabelpairs,
                ax=ax,estimator=np.nanmean,errorbar=('ci', 95))
    if np.shape(data_gainregress)[1]==2:
        h,p = stats.ttest_ind(data_gainregress[idx_N,0,iparam],
                            data_gainregress[idx_N,1,iparam],nan_policy='omit')
        p = p * narealabelpairs
        add_stat_annotation(ax, 0.2, 0.8, np.nanmean(data_gainregress[idx_N,:,iparam],axis=0).max()*1.1, p, h=0)
    elif np.shape(data_gainregress)[1]==4:
        for iidx,idx in enumerate([[0,2],[0,1],[2,3]]):
            h,p = stats.ttest_ind(data_gainregress[idx_N,idx[0],iparam],
                                data_gainregress[idx_N,idx[1],iparam],nan_policy='omit')
            p = p * narealabelpairs
            add_stat_annotation(ax, idx[0], idx[1], np.nanmean(data_gainregress[np.ix_(np.where(idx_N)[0],idx,[iparam])],axis=0).max()*1.2+iidx*0.01, p, h=0.001)
    
        # h,p = stats.ttest_ind(data_gainregress[idx_N,0,iparam],
        #                     data_gainregress[idx_N,2,iparam],nan_policy='omit')
        # add_stat_annotation(ax, 0, 2, np.nanmean(data_gainregress[idx_N,:,iparam],axis=0).max()*1.1, p, h=0.0)
 
    ax.tick_params(labelsize=9,rotation=0)
    ax.set_xticklabels(ticklabels)
    if iparam == 0:
        ax.set_title('Multiplicative')
    else:
        ax.set_title('Additive')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_sigN_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_sigP_barplot_GR%dsessions' % (nSessions), formats = ['png'])

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
                        np.any(data_gainregress[:,:,2] > 0.5,axis=1),
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

#%%






#%% 

clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
for ialp,alp in enumerate(arealabelpairs):
    for iparam in range(2):
        ax = axes[iparam,ialp]
        # idx_N = data_gainregress[:,ialp,2] > 0.3
        idx_N =  celldata['OSI']>0.5
        idx_N = np.all((celldata['gOSI']>0.5,
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
# my_savefig(fig,savedir,'FF_FB_affinemodulation_popcoupling_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])


######  #######  #####  ####### ######  ### #     #  #####  
#     # #       #     # #     # #     #  #  ##    # #     # 
#     # #       #       #     # #     #  #  # #   # #       
#     # #####   #       #     # #     #  #  #  #  # #  #### 
#     # #       #       #     # #     #  #  #   # # #     # 
#     # #       #     # #     # #     #  #  #    ## #     # 
######  #######  #####  ####### ######  ### #     #  #####  

#%% 
from utils.regress_lib import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


#%% Decoding orientation when the FF or FB activity is high or low:

arealabelpairs  = [
                    'V1lab-PMunlL2/3',
                    # 'V1lab-V1unl-PMlabL2/3',
                    # 'V1lab-V1unl-PMunlL5',
                    # 'V1lab-V1unl-PMlabL5',
                    'PMlab-V1unlL2/3',
                    # 'PMlab-PMunl-V1labL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

perc                = 20
minnneurons         = 10
maxnoiselevel       = 20

nmodelfits          = 10
nsampleneurons      = 25
kfold               = 5
# lam               = None
lam                 = 0.05
model_name          = 'SVM'
scoring_type        = 'accuracy_score'

error_cv = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    # respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # data            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # data            = sessions[ises].respmat
    data            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    ori_ses     = sessions[ises].trialdata['Orientation']

    # idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        # idx_N2              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
        #                               sessions[ises].celldata['noise_level']<maxnoiselevel,
        #                               ),axis=0))[0]

        idx_ses             = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][0])
        idx_N2              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                    #   corrsig_cells[ialp,idx_ses]==1,
                                      corrsig_cells[ialp,idx_ses]==-1,
                                    #   corrsig_cells[ialp,idx_ses]==0,
                                    #   np.any(corrsig_cells[:,idx_ses]!=1,axis=0),
                                    #   np.any(corrsig_cells[:,idx_ses]==1,axis=0),
                                      np.any(data_gainregress[idx_ses,:,2] > 0.5,axis=1),
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        if len(idx_N1) < minnneurons or len(idx_N2) < nsampleneurons:
            continue
        
        if lam is None:
            y = ori_ses
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
            X = data.T
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

        # meanpopact          = np.nanmean(zscore(data[idx_N1,:],axis=1),axis=0)
        meanpopact          = np.nanmean(data[idx_N1,:],axis=0)
        
        # meanpopact2          = np.nanmean(data[idx_N2,:],axis=0)
        # plt.scatter(meanpopact,meanpopact2)
        # plt.plot([0,1],[0,1],transform=ax.transAxes)

        # compute index for trials with low and high activity in the other labeled pop
        idx_K1            = []
        idx_K2            = []
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            idx_T               = ori_ses == ori
            idx_K1.append(list(np.where(np.logical_and(idx_T,meanpopact < np.nanpercentile(meanpopact[ori_ses == ori],perc)))[0]))
            idx_K2.append(list(np.where(np.logical_and(idx_T,meanpopact > np.nanpercentile(meanpopact[ori_ses == ori],100-perc)))[0]))
        
        idx_K1 = np.array(idx_K1).flatten()
        idx_K2 = np.array(idx_K2).flatten()

        for imf in range(nmodelfits):
            idx_N_sub = np.random.choice(idx_N2,nsampleneurons,replace=False)
            X = data[np.ix_(idx_N_sub,idx_K1)].T
            y = ori_ses[idx_K1]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            error_cv[ialp,0,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)
            
            X = data[np.ix_(idx_N_sub,idx_K2)].T
            y = ori_ses[idx_K2]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            error_cv[ialp,1,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)
         

#%%
axtitles = np.array(['Feedforward','Feedback']) 
fig,axes = plt.subplots(1,2,figsize=(5,3),sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
    sns.stripplot(data.T,palette=['grey','red'],alpha=0.5,ax=ax,jitter=0.1)
    sns.barplot(data.T,palette=['grey','red'],alpha=0.5,ax=ax)
    sns.lineplot(data,palette=sns.color_palette(['grey'],nSessions),alpha=0.5,ax=ax,legend=False,
                linewidth=2,linestyle='-')

    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(0.5,1/16+0.05,'Chance',fontsize=10,ha='center',va='center')
    ax.set_ylim([0,1])
    ax.set_title(axtitles[ialp],fontsize=12,)
    ax.set_xticks([0,1],labels=['Low','High'])
    if ialp == 0:
        ax.set_ylabel('Decoding Performance')
    else:
        ax.set_ylabel('')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

#Statistics:
testdata    = np.nanmean(error_cv,axis=3) #average over modelfits
df = pd.DataFrame({'perf': testdata.flatten(),
                   'act': np.repeat(np.tile(np.arange(2),2),nSessions),
                   'area': np.repeat(np.arange(2),2*nSessions),
                   'session_id': np.tile(np.arange(nSessions),2*2)
                   })
df.dropna(inplace=True)

model     = smf.mixedlm("perf ~ act * area", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_SigP_%dsessions' % nSessions, formats = ['png'])
# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_SigN_%dsessions' % nSessions, formats = ['png'])

#%% 

#Comments: the interaction effect between FF and FB on decoding 
# is there if in the first part the deconv/F0 is chosen for meanpopact in population 1
# but then in the second half for the decoding you work with respmat original
# This is confounding with the behavioral data again. So need to fix this... 





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





