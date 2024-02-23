# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math, os
try:
    os.chdir('t:\\Python\\molanalysis\\')
except:
    os.chdir('e:\\Python\\molanalysis\\')
os.chdir('c:\\Python\\molanalysis\\')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from utils.tuning import compute_tuning
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import PCA_gratings,PCA_gratings_3D
from utils.plot_lib import shaded_error

savedir = 'C:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\'

##############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE10885','2023_10_19']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')
sessions            = filter_sessions(protocols = ['GR'],load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')
nSessions = len(sessions)

##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# for i in range(nSessions):
#     [sessions[i].tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
#                                  t_pre, t_post, binsize,method='interp_lin')

#Alternative method, much faster:
for ises in range(nSessions):
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

    #hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].behaviordata['runspeed']),(len(sessions[ises].behaviordata['runspeed']),1)))
    sessions[ises].respmat_runspeed = compute_respmat(temp, sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_runspeed = np.squeeze(sessions[ises].respmat_runspeed)

    #hacky way to create dataframe of the video motion with F x 1 with F number of samples:
    temp = pd.DataFrame(np.reshape(np.array(sessions[ises].videodata['motionenergy']),(len(sessions[ises].videodata['motionenergy']),1)))
    sessions[ises].respmat_videome = compute_respmat(temp, sessions[ises].videodata['timestamps'], sessions[ises].trialdata['tOnset'],
                                    t_resp_start=0,t_resp_stop=1,method='mean')
    sessions[ises].respmat_videome = np.squeeze(sessions[ises].respmat_videome)

    delattr(sessions[ises],'calciumdata')
    delattr(sessions[ises],'videodata')
    delattr(sessions[ises],'behaviordata')


############################ Compute tuning metrics: ###################################

for ises in range(nSessions):
    sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='OSI')
    sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='gOSI')
    sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='tuning_var')

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


##### 


sesidx = 5
fig = PCA_gratings_3D(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

fig = PCA_gratings(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

################################ Show response with and without running #################

thr_still   = 0.5 #max running speed for still trials
thr_moving  = 1 #min running speed for moving trials

nOris       = 16
nCells      = len(celldata)
mean_resp_speedsplit = np.empty((nCells,nOris,2))

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    idx_trials_still = sessions[ises].respmat_runspeed<thr_still
    idx_trials_moving = sessions[ises].respmat_runspeed>thr_moving

    # compute meanresp
    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)
    # resp_meanori    = np.empty([N,len(oris)])

    resp_meanori    = np.empty([N,len(oris),2])
    for i,ori in enumerate(oris):
        resp_meanori[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_still)],axis=1)
        resp_meanori[:,i,1] =  np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_trials_moving)],axis=1)
        
    # prefori             = np.argmax(resp_meanori.mean(axis=2),axis=1)
    prefori             = np.argmax(resp_meanori[:,:,0],axis=1)

    resp_meanori_pref       = resp_meanori.copy()
    for n in range(N):
        resp_meanori_pref[n,:,0] = np.roll(resp_meanori[n,:,0],-prefori[n])
        resp_meanori_pref[n,:,1] = np.roll(resp_meanori[n,:,1],-prefori[n])

    # normalize by peak response during still trials
    tempmin,tempmax = resp_meanori_pref[:,:,0].min(axis=1,keepdims=True),resp_meanori_pref[:,:,0].max(axis=1,keepdims=True)
    resp_meanori_pref[:,:,0] = (resp_meanori_pref[:,:,0] - tempmin) / (tempmax - tempmin)
    resp_meanori_pref[:,:,1] = (resp_meanori_pref[:,:,1] - tempmin) / (tempmax - tempmin)

    # resp_meanori_pref
    idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'])
    mean_resp_speedsplit[idx_ses,:,:] = resp_meanori_pref

# ########### Make the figure ##################################################################

redcells    = np.unique(celldata['redcell'])
redcell_labels = ['Nonlabeled','Labeled']
areas       = np.unique(celldata['roi_name'])

fig,axes = plt.subplots(2,2,figsize=(8,8),sharex=True,sharey=True)

for iarea,area in enumerate(areas):
    for ired,redcell in enumerate(redcells):
        ax = axes[iarea,ired]
        idx_neurons = celldata['redcell']==redcell
        idx_neurons = np.logical_and(idx_neurons,celldata['roi_name']==area)
        idx_neurons = np.logical_and(idx_neurons,celldata['tuning_var']>0.1)
        handles = []
        # handles = handles + shaded_error(ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0],center='mean',error='sem',color='black')
        handles.append(shaded_error(ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,0],center='mean',error='sem',color='black'))
        handles.append(shaded_error(ax,x=oris,y=mean_resp_speedsplit[idx_neurons,:,1],center='mean',error='sem',color='red'))
        # ax.plot(oris,np.nanmean(mean_resp_speedsplit[idx_neurons,:,0],axis=0),color='black')
        # plt.fill_between(oris, y-error, y+error)

        # ax.errorbar(oris,np.nanmean(mean_resp_speedsplit[idx_neurons,:,0],axis=0),
                    #  np.nanstd(mean_resp_speedsplit[idx_neurons,:,0],axis=0),color='black')
        # ax.plot(np.nanmean(mean_resp_speedsplit[idx_neurons,:,1],axis=0),color='red')
        ax.legend(handles=handles,labels=['Still','Running'])
        ax.set_xlabel(u'Δ Pref Ori')
        ax.set_xticks(oris,labels=oris,fontsize=8,rotation='vertical')
        ax.set_ylabel('Normalized Response')
        ax.set_title('%s - %s' % (area,redcell_labels[ired]))
plt.tight_layout()

fig.savefig(os.path.join(savedir,'Tuning','Locomotion_V1PM_LabNonLab_' + str(nSessions) + 'sessions.png'), format = 'png')

############################ Compute noise correlations: ###################################

for ises in range(nSessions):
    # get signal correlations:
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    oris            = np.sort(sessions[ises].trialdata['Orientation'].unique())
    ori_counts      = sessions[ises].trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)
    resp_meanori    = np.empty([N,len(oris)])

    for i,ori in enumerate(oris):
        resp_meanori[:,i] = np.nanmean(sessions[ises].respmat[:,sessions[ises].trialdata['Orientation']==ori],axis=1)

    sessions[ises].sig_corr                 = np.corrcoef(resp_meanori)

    prefori                         = oris[np.argmax(resp_meanori,axis=1)]
    sessions[ises].delta_pref       = np.abs(np.subtract.outer(prefori, prefori))

    respmat_res                     = sessions[ises].respmat.copy()

    ## Compute residuals:
    for ori in oris:
        ori_idx     = np.where(sessions[ises].trialdata['Orientation']==ori)[0]
        temp        = np.mean(respmat_res[:,ori_idx],axis=1)
        respmat_res[:,ori_idx] = respmat_res[:,ori_idx] - np.repeat(temp[:, np.newaxis], len(ori_idx), axis=1)

    sessions[ises].noise_corr                   = np.corrcoef(respmat_res)
    
    # sessions[ises].sig_corr[np.eye(N)==1]   = np.nan
    # sessions[ises].noise_corr[np.eye(N)==1]     = np.nan

    idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
    sessions[ises].sig_corr[idx_triu] = np.nan
    sessions[ises].noise_corr[idx_triu] = np.nan
    sessions[ises].delta_pref[idx_triu] = np.nan

    assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
    assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
    assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
    assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)

###################### Plot control figure of signal correlation and  ##############################

sesidx = 0
plt.figure(figsize=(8,5))
plt.imshow(sessions[sesidx].sig_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].sig_corr,15),
           vmax=np.nanpercentile(sessions[sesidx].sig_corr,85))

plt.figure(figsize=(8,5))
plt.imshow(sessions[sesidx].noise_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].noise_corr,5),
           vmax=np.nanpercentile(sessions[sesidx].noise_corr,95))

###################### Compute pairwise neuronal distances: ##############################

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    ## Compute euclidean distance matrix based on soma center:
    sessions[ises].distmat_xyz     = np.zeros((N,N))
    sessions[ises].distmat_xy      = np.zeros((N,N))
    sessions[ises].distmat_rf      = np.zeros((N,N))
    sessions[ises].areamat         = np.empty((N,N),dtype=object)
    sessions[ises].labelmat        = np.empty((N,N),dtype=object)

    x,y,z = sessions[ises].celldata['xloc'],sessions[ises].celldata['yloc'],sessions[ises].celldata['depth']
    b = np.array((x,y,z))
    for i in range(N):
        print(f"\rComputing pairwise distances for neuron {i+1} / {N}",end='\r')
        a = np.array((x[i],y[i],z[i]))
        sessions[ises].distmat_xyz[i,:] = np.linalg.norm(a[:,np.newaxis]-b,axis=0)
        sessions[ises].distmat_xy[i,:] = np.linalg.norm(a[:2,np.newaxis]-b[:2,:],axis=0)

    if 'rf_azimuth' in sessions[ises].celldata:
        rfaz,rfel = sessions[ises].celldata['rf_azimuth'],sessions[ises].celldata['rf_elevation']
        d = np.array((rfaz,rfel))

        for i in range(N):
            c = np.array((rfaz[i],rfel[i]))
            sessions[ises].distmat_rf[i,:] = np.linalg.norm(a[:2,np.newaxis]-b[:2,:],axis=0)

    g = np.meshgrid(sessions[ises].celldata['roi_name'],sessions[ises].celldata['roi_name'])
    sessions[ises].areamat = g[0] + '-' + g[1]

    g = np.meshgrid(sessions[ises].celldata['redcell'].astype(int).astype(str).to_numpy(),
                    sessions[ises].celldata['redcell'].astype(int).astype(str).to_numpy())
    sessions[ises].labelmat = g[0] + '-' + g[1]

    # for i in range(N):
    #     print(f"\rComputing pairwise distances for neuron {i+1} / {N}",end='\r')
    #     for j in range(N):
    #         sessions[ises].distmat_xyz[i,j] = math.dist([sessions[ises].celldata['xloc'][i],sessions[ises].celldata['yloc'][i],sessions[ises].celldata['depth'][i]],
    #                 [sessions[ises].celldata['xloc'][j],sessions[ises].celldata['yloc'][j],sessions[ises].celldata['depth'][j]])
    #         sessions[ises].distmat_xy[i,j] = math.dist([sessions[ises].celldata['xloc'][i],sessions[ises].celldata['yloc'][i]],
    #                 [sessions[ises].celldata['xloc'][j],sessions[ises].celldata['yloc'][j]])
    #         sessions[ises].distmat_rf[i,j] = math.dist([sessions[ises].celldata['rf_azimuth'][i],sessions[ises].celldata['rf_elevation'][i]],
    #                 [sessions[ises].celldata['rf_azimuth'][j],sessions[ises].celldata['rf_elevation'][j]])
    #         sessions[ises].areamat[i,j] = sessions[ises].celldata['roi_name'][i] + '-' + sessions[ises].celldata['roi_name'][j]
    #         sessions[ises].labelmat[i,j] = str(int(sessions[ises].celldata['redcell'][i])) + '-' + str(int(sessions[ises].celldata['redcell'][j]))

    # sessions[ises].distmat_xyz[np.eye(N)==1] = np.nan
    # sessions[ises].distmat_xy[np.eye(N)==1] = np.nan
    idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
    sessions[ises].distmat_xyz[idx_triu] = np.nan
    sessions[ises].distmat_xy[idx_triu] = np.nan
    sessions[ises].distmat_rf[idx_triu] = np.nan
    sessions[ises].areamat[idx_triu] = np.nan
    sessions[ises].labelmat[idx_triu] = np.nan

from scipy.spatial import distance
N = 100
import time

# do stuff


g = np.empty((N,N))

t = time.time()

for i in range(N):
    print(f"\rComputing pairwise distances for neuron {i+1} / {N}",end='\r')
    for j in range(N):
        # sessions[ises].distmat_xyz[i,j] = math.dist([sessions[ises].celldata['xloc'][i],sessions[ises].celldata['yloc'][i],sessions[ises].celldata['depth'][i]],
                # [sessions[ises].celldata['xloc'][j],sessions[ises].celldata['yloc'][j],sessions[ises].celldata['depth'][j]])
        # sessions[ises].distmat_xyz[i,j] = distance.euclidean([sessions[ises].celldata['xloc'][i],sessions[ises].celldata['yloc'][i],sessions[ises].celldata['depth'][i]],
                # [sessions[ises].celldata['xloc'][j],sessions[ises].celldata['yloc'][j],sessions[ises].celldata['depth'][j]])
        
        a = np.array((x[i],y[i],z[i]))
        b = np.array((x[j],y[j],z[j]))
        g[i,j] = np.linalg.norm(a-b)

print(time.time() - t)



t = time.time()

g = np.empty((N,3803))

for i in range(N):
    print(f"\rComputing pairwise distances for neuron {i+1} / {N}",end='\r')
    # for j in range(N):
        # sessions[ises].distmat_xyz[i,j] = math.dist([sessions[ises].celldata['xloc'][i],sessions[ises].celldata['yloc'][i],sessions[ises].celldata['depth'][i]],
                # [sessions[ises].celldata['xloc'][j],sessions[ises].celldata['yloc'][j],sessions[ises].celldata['depth'][j]])
        # sessions[ises].distmat_xyz[i,j] = distance.euclidean([sessions[ises].celldata['xloc'][i],sessions[ises].celldata['yloc'][i],sessions[ises].celldata['depth'][i]],
                # [sessions[ises].celldata['xloc'][j],sessions[ises].celldata['yloc'][j],sessions[ises].celldata['depth'][j]])
        
    a = np.array((x[i],y[i],z[i]))
    # b = np.array((x[j],y[j],z[j]))
    g[i,:] = np.linalg.norm(a[:,np.newaxis]-b,axis=0)

print(time.time() - t)

g[:,0]


                     
# construct dataframe with all pairwise measurements:
df  = pd.DataFrame()

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    tempdf  = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr.flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref.flatten(),
                    'AreaPair': sessions[ises].areamat.flatten(),
                    'DistXYPair': sessions[ises].distmat_xy.flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz.flatten(),
                    'DistRfPair': sessions[ises].distmat_rf.flatten(),
                    'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all')
    df  = pd.concat([df, tempdf], ignore_index=True).reset_index(drop=True)


###################### Noise correlations within and across areas: #########################
plt.figure(figsize=(8,5))
# sns.barplot(data=df,x='AreaPair',y='NoiseCorrelation')
g = df.groupby('AreaPair', as_index=False)['NoiseCorrelation'].mean()
sns.barplot(data=g,x='AreaPair',y='NoiseCorrelation')

############### Relationship anatomical distance and receptive field distance: ##################

df_withinarea = df[(df['AreaPair'].isin(['V1-V1','PM-PM'])) & (df['DistRfPair'].notna()) & (df['DistXYPair'] < 1000)]
# df_withinarea = df[(df['AreaPair'].isin(['PM-PM'])) & (df['DistRfPair'].notna()) & (df['DistXYPair'] < 1000)]

g = sns.displot(df_withinarea, x="DistXYZPair", y="DistRfPair", binwidth=(2, 2), cbar=True,col="AreaPair")
plt.xlim([0,650])
plt.ylim([0,250])
g.set_axis_labels("Anatomical distance \n (approx um)", "RF distance (deg)")

plt.savefig(os.path.join(savedir,'Corr_anat_rf_distance' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')


####################################################################
####### Noise correlations as a function of anatomical distance ####

areapairs = ['V1-V1','PM-V1','PM-PM']

clrs_areapairs = get_clr_area_pairs(areapairs)

plt.figure()

for ises in range(nSessions):
    filter = sessions[ises].celldata['noise_level']<1
    df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                    'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                    'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                    'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                    'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')
    df['DistXYPair'] = df['DistXYPair'].round(-1)
    histdata = df.groupby('DistXYPair', as_index=False)['NoiseCorrelation'].mean()
    histdata = df.groupby(['DistXYPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    for iap,areapair in enumerate(areapairs):
        plt.plot(histdata['DistXYPair'][histdata['AreaPair']==areapair], 
                 histdata['NoiseCorrelation'][histdata['AreaPair']==areapair],
                 color=clrs_areapairs[iap])
    plt.xlabel('Anatomical distance (XY)')
    plt.ylabel('Noise Correlation')
    plt.legend(areapairs)
plt.savefig(os.path.join(savedir,'NoiseCorr_anatomdistance_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

################################################################

plt.figure()
for ises in range(nSessions):
    # filter = sessions[ises].celldata['noise_level']<1
    filter = sessions[ises].celldata['tuning_var']>0
    # filter = sessions[ises].celldata['tuning_var']>0.1
    df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                    'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                    'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                    'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                    'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

    df['DistRfPair']    = df['DistRfPair'].round(-1)
    histdata            = df.groupby('DistRfPair', as_index=False)['NoiseCorrelation'].mean()
    histdata            = df.groupby(['DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    for iap,areapair in enumerate(areapairs):
        plt.plot(histdata['DistRfPair'][histdata['AreaPair']==areapair], 
                 histdata['NoiseCorrelation'][histdata['AreaPair']==areapair],
                 color=clrs_areapairs[iap])
    plt.legend(areapairs)
plt.xlabel('Delta RF')
plt.ylabel('NoiseCorrelation')
plt.xlim([0,200])
plt.ylim([0,0.05])

################################################################

fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    # filter = sessions[ises].celldata['noise_level']<1
    filter = sessions[ises].celldata['tuning_var']>0.1
    # filter = np.logical_and(filter,sessions[ises].celldata['tuning_var']>0.1)
    # filter = sessions[ises].celldata['tuning_var']>0.1
    df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                    'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                    'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                    'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                    'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

    df['DeltaPrefOri'] = np.mod(df['DeltaPrefOri'],180)

    deltapreforis = np.sort(df['DeltaPrefOri'].unique())
    clrs_deltaoris = sns.color_palette('inferno', len(deltapreforis))

    df['DistRfPair']    = df['DistRfPair'].round(-1)

    histdata            = df.groupby(['DeltaPrefOri','DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    for iap,areapair in enumerate(areapairs):
        plt.subplot(1,3,iap+1)
        for idf,deltaori in enumerate(deltapreforis):
            # plt.plot(histdata['DistRfPair'][histdata['AreaPair']==areapair], 
            #      histdata['NoiseCorrelation'][histdata['AreaPair']==areapair],
            #      color=[iap])
            idx = np.logical_and(histdata['AreaPair']==areapair,histdata['DeltaPrefOri']==deltaori)
            plt.plot(histdata['DistRfPair'][idx], 
                 histdata['NoiseCorrelation'][idx],
                 color=clrs_deltaoris[idf])
            
        plt.xlabel('Delta RF')
        plt.ylabel('NoiseCorrelation')
        plt.xlim([0,200])
        plt.ylim([-0.05,0.1])
        plt.title(areapair)
            
    plt.legend(deltapreforis)

################################################################
# Plot Noise correlations as a function of the difference in preferred orientation
# for different percentiles of how strongly tuned neurons are

fig = plt.subplots(1,3,figsize=(12,4))
for ises in range(nSessions):
    tuning_perc_labels = np.linspace(0,100,11)
    tuning_percentiles  = np.percentile(sessions[ises].celldata['tuning_var'],tuning_perc_labels)
    clrs_percentiles    = sns.color_palette('inferno', len(tuning_percentiles))

    histdata            = df.groupby(['DeltaPrefOri','DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    for iap,areapair in enumerate(areapairs):
        plt.subplot(1,3,iap+1)
        for ip in range(len(tuning_percentiles)-1):

            # filter = tuning_percentiles[ip] <= sessions[ises].celldata['tuning_var'] <= tuning_percentiles[ip+1]
            filter = np.logical_and(tuning_percentiles[ip] <= sessions[ises].celldata['tuning_var'],
                                    sessions[ises].celldata['tuning_var'] <= tuning_percentiles[ip+1])
            # filter = sessions[ises].celldata['tuning_var']>0.1
            df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                            'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                            'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                            'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                            'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                            'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                            'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

            # df['DeltaPrefOri'] = np.mod(df['DeltaPrefOri'],180)

            deltapreforis = np.sort(df['DeltaPrefOri'].unique())
            histdata            = df.groupby(['DeltaPrefOri','AreaPair'], as_index=False)['NoiseCorrelation'].mean()

            # plt.plot(histdata['DistRfPair'][histdata['AreaPair']==areapair], 
            #      histdata['NoiseCorrelation'][histdata['AreaPair']==areapair],
            #      color=[iap])
            # idx = np.logical_and(histdata['AreaPair']==areapair,histdata['DeltaPrefOri']==deltaori)
            idx = histdata['AreaPair']==areapair
            plt.plot(histdata['DeltaPrefOri'][idx], 
                    histdata['NoiseCorrelation'][idx],
                    color=clrs_percentiles[ip])
            
        plt.xlabel('Delta Ori')
        plt.ylabel('NoiseCorrelation')
        # plt.xlim([0,200])
        plt.ylim([-0.05,0.1])
        plt.title(areapair)
            
    plt.legend(tuning_perc_labels[1:])

plt.savefig(os.path.join(savedir,'NoiseCorr_tuning_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
# plt.hist(df['DistRfPair'])
# sns.histplot(df['DistRfPair'])

################################################################
plt.figure()
for ises in range(nSessions):
    # filter = sessions[ises].celldata['noise_level']<1
    filter = sessions[ises].celldata['tuning_var']>0.1
    # filter = sessions[ises].celldata['tuning_var']>0.1
    df = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr[filter,:].flatten(),
                    'DeltaPrefOri': sessions[ises].delta_pref[filter,:].flatten(),
                    'AreaPair': sessions[ises].areamat[filter,:].flatten(),
                    'DistXYPair': sessions[ises].distmat_xy[filter,:].flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz[filter,:].flatten(),
                    'DistRfPair': sessions[ises].distmat_rf[filter,:].flatten(),
                    'LabelPair': sessions[ises].labelmat[filter,:].flatten()}).dropna(how='all')

    df['DistRfPair']    = df['DistRfPair'].round(-1)
    histdata            = df.groupby('DistRfPair', as_index=False)['NoiseCorrelation'].mean()
    histdata            = df.groupby(['DistRfPair','AreaPair'], as_index=False)['NoiseCorrelation'].mean()
    plt.plot(histdata['DistRfPair'][histdata['AreaPair']=='V1-V1'], histdata['NoiseCorrelation'][histdata['AreaPair']=='V1-V1'])
    plt.plot(histdata['DistRfPair'][histdata['AreaPair']=='PM-V1'], histdata['NoiseCorrelation'][histdata['AreaPair']=='PM-V1'])
    plt.plot(histdata['DistRfPair'][histdata['AreaPair']=='PM-PM'], histdata['NoiseCorrelation'][histdata['AreaPair']=='PM-PM'])

plt.xlabel('Delta RF')
plt.ylabel('NoiseCorrelation')
plt.xlim([0,200])
plt.ylim([0,0.05])
plt.savefig(os.path.join(savedir,'NoiseCorr_tuning_perArea' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

###################### Noise correlations as a function of pairwise anatomical distance: ####################
fig,axes   = plt.subplots(1,2,figsize=(8,6))

sns.lineplot(x=np.round(df_withinarea['DistXYZPair'],-1),y=df_withinarea['NoiseCorrelation'],hue=df_withinarea['AreaPair'],ax=axes[0])
axes[0].set_xlabel="Pairwise distance XYZ (um)"
# plt.legend(labels=['V1-V1','PM-PM'])
axes[0].set_xlim([-10,600])
axes[0].set_ylim([0,0.13])
# axes[0].set_xlabel("Anatomical distance (approx um)")
axes[0].set_ylabel("Noise Correlation")
axes[0].set_title("Anatomical")

sns.lineplot(x=np.round(df['DistRfPair'],-1),y=df['NoiseCorrelation'],hue=df['AreaPair'],ax=axes[1])
axes[1].set_xlabel="Pairwise RF distance (um)"
axes[1].set_xlim([-10,300])
axes[1].set_ylim([0,0.13])
# axes[1].set_xlabel(['RF distance (ret deg)'])
axes[1].set_ylabel("Noise Correlation")
axes[1].set_title("Receptive Field")

plt.savefig(os.path.join(savedir,'NoiseCorr_anat_rf_distance' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

########################### Noise correlations as a function of pairwise distance: ####################
######################################## Labeled vs unlabeled neurons #################################

fig, axes = plt.subplots(2,2,figsize=(8,7))

areas = ['V1','PM']

for i,iarea in enumerate(areas):
    for j,jarea in enumerate(areas):
        dfarea = df[df['AreaPair']==iarea + '-' + jarea]
        sns.lineplot(ax=axes[i,j],x=np.round(dfarea['DistRfPair'],-1),y=dfarea['NoiseCorrelation'],hue=dfarea['LabelPair'])
        # axes[i,j].set_xlabel="Pairwise distance (um)"
        axes[i,j].set_xlabel="Delta RF (deg)"
        axes[i,j].set_xlim([-10,200])
        axes[i,j].set_ylim([0,0.05])
        axes[i,j].set_title(iarea + '-' + jarea)

plt.savefig(os.path.join(savedir,'NoiseCorr_labeled_RF_distance' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

    

###################### Noise correlations within and across areas: #########################
plt.figure(figsize=(8,5))
# sns.barplot(data=df,x='AreaPair',y='NoiseCorrelation')
g = df.groupby('AreaPair', as_index=False)['NoiseCorrelation'].mean()
# g = df.groupby('AreaPair', as_index=False)['NoiseCorrelation'].std()
sns.barplot(data=g,x='AreaPair',y='NoiseCorrelation')

###################### Noise correlations as a function of pairwise distance: ####################
plt.figure(figsize=(8,5))

g = df.groupby('AreaPair', as_index=False)['NoiseCorrelation'].mean()


# sns.scatterplot(data=df[df['AreaPair']=='V1-V1'],x='DistPair',y='NoiseCorrelation',size=5)
dfV1 = df[df['AreaPair']=='V1-V1']
sns.lineplot(x=np.round(dfV1['DistPair'],-1),y=dfV1['NoiseCorrelation'],color='b')

dfPM = df[df['AreaPair']=='PM-PM']
sns.lineplot(x=np.round(dfPM['DistPair'],-1),y=dfPM['NoiseCorrelation'],color='g')

plt.xlabel="Pairwise distance (um)"
plt.legend(labels=['V1-V1','PM-PM'])
plt.xlim([-10,600])
plt.ylim([0,0.13])

########################### Noise correlations as a function of pairwise distance: ####################
######################################## Labeled vs unlabeled neurons #################################

plt.figure(figsize=(8,5))
fig, axes = plt.subplots(2,2,figsize=(8,5))

areas = ['V1','PM']

for i,iarea in enumerate(areas):
    for j,jarea in enumerate(areas):
        dfarea = df[df['AreaPair']==iarea + '-' + jarea]
        sns.lineplot(ax=axes[i,j],x=np.round(dfarea['DistPair'],-1),y=dfarea['NoiseCorrelation'],hue=dfarea['LabelPair'])
        axes[i,j].set_xlabel="Pairwise distance (um)"
        # plt.legend(labels=['V1-V1','PM-PM'])
        axes[i,j].set_xlim([-10,200])
        axes[i,j].set_ylim([-0.05,0.2])

