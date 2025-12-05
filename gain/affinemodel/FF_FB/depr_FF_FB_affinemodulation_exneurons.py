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
sessions = compute_tuning_wrapper(sessions)

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=50)
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

#%% Show example neurons

arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    # 'V1lab-V1unl-PMlabL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    # 'PMlab-PMunl-V1labL2/3',
                    ]


narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

nboots                  = 100
perc                    = 20
minnneurons             = 10
maxnoiselevel = 20

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)


for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    # idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

    for ialp,alp in enumerate(arealabelpairs):
        # idx_N1              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0]
        # idx_N2              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1]
        # idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2]

        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    # idx_nearby,

                                      ),axis=0))[0]
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    # idx_nearby,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue

        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        # meanpopact          = np.nanmean(zscore(respdata[idx_N1,:],axis=1),axis=0)

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

        # meanresp_pref
        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

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

