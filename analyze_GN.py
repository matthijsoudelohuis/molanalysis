# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import seaborn as sns
os.chdir('C:\\Python\\molanalysis\\')

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from utils.plotting_style import * #get all the fixed color schemes

# from rastermap import Rastermap, utils

# sessions            = filter_sessions(protocols = ['GN'])

savedir = 'C:\\OneDrive\\PostDoc\\Figures\\NoiseRegression\\'


#################################################
# session_list        = np.array([['LPE10883','2023_10_27']])
session_list        = np.array([['LPE10919','2023_11_16']])
sessions            = load_sessions(protocol = 'GN',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')

sesidx = 0

# zscore all the calcium traces:
# calciumdata_z      = st.zscore(calciumdata.copy(),axis=1)

######################################
#Show some traces and some stimuli to see responses:

def show_excerpt_traces_gratings(Session,example_cells=None,trialsel=None):
    
    if example_cells is None:
        example_cells = np.random.choice(Session.calciumdata.shape[1],10)

    if trialsel is None:
        trialsel = [np.random.randint(low=0,high=len(Session.trialdata)-400)]
        trialsel.append(trialsel[0]+40)

    example_tstart  = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    excerpt         = np.array(Session.calciumdata.loc[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)])
    excerpt         = excerpt[:,example_cells]

    min_max_scaler = preprocessing.MinMaxScaler()
    excerpt = min_max_scaler.fit_transform(excerpt)

    # spksselec = spksselec 
    [nframes,ncells] = np.shape(excerpt)

    for i in range(ncells):
        excerpt[:,i] =  excerpt[:,i] + i

    oris        = np.unique(Session.trialdata['centerOrientation'])
    rgba_color  = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(oris)))  
    
    fig, ax = plt.subplots(figsize=[12, 6])
    plt.plot(Session.ts_F[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)],excerpt,linewidth=0.5,color='black')
    # plt.show()

    for i in np.arange(trialsel[0],trialsel[1]):
        ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i],0],1,ncells,alpha=0.3,linewidth=0,
                                facecolor=rgba_color[np.where(oris==Session.trialdata['centerOrientation'][i])]))

    handles= []
    for i,ori in enumerate(oris):
        handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))

    ax.set_xlim([example_tstart,example_tstop])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
    ax.axis('off')


example_cells   = [1250,1230,1257,1551,1559,1616,1645,2006,1925,1972,2178,2110] #PM
example_cells   = [6,23,130,99,361,177,153,413,435]

show_excerpt_traces_gratings(sessions[0],example_cells=example_cells,trialsel=[50,90])
show_excerpt_traces_gratings(sessions[0])

# plt.close('all')


##############################################################################
## Construct tensor: 3D 'matrix' of N neurons by K trials by T time bins
## Parameters for temporal binning
t_pre       = -1    #pre s
t_post      = 2     #post s
binsize     = 0.2   #temporal binsize in s

# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], t_pre, t_post, binsize,method='binmean')

# [tensor,t_axis] = compute_tensor(calciumdata, ts_F, trialdata['tOnset'], t_pre, t_post, binsize,method='interp_lin')
# [tensor,t_axis] = compute_tensor(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'], 
#                                  t_pre, t_post, binsize,method='interp_lin')
# [N,K,T]         = np.shape(tensor) #get dimensions of tensor
# respmat         = tensor[:,:,np.logical_and(t_axis > 0,t_axis < 1)].mean(axis=2)

#Alternative method, much faster:
sessions[sesidx].respmat         = compute_respmat(sessions[0].calciumdata, sessions[0].ts_F, sessions[0].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)
[N,K]           = np.shape(sessions[sesidx].respmat) #get dimensions of response matrix

#hacky way to create dataframe of the runspeed with F x 1 with F number of samples:
temp = pd.DataFrame(np.reshape(np.array(sessions[0].behaviordata['runspeed']),(len(sessions[0].behaviordata['runspeed']),1)))
sessions[sesidx].respmat_runspeed = compute_respmat(temp, sessions[0].behaviordata['ts'], sessions[0].trialdata['tOnset'],
                                   t_resp_start=0,t_resp_stop=1,method='mean')
sessions[sesidx].respmat_runspeed = np.squeeze(sessions[sesidx].respmat_runspeed)

# #### Check if values make sense:
fig,ax = plt.subplots(1,1,figsize=(6,6))
sns.histplot(np.min(sessions[sesidx].respmat,axis=1),ax=ax)
sns.histplot(np.max(sessions[sesidx].respmat,axis=1))

#############################################################################
oris            = np.sort(pd.Series.unique(sessions[sesidx].trialdata['centerOrientation']))
speeds          = np.sort(pd.Series.unique(sessions[sesidx].trialdata['centerSpeed']))
noris           = len(oris) 
nspeeds         = len(speeds)


### Mean response per condition:
resp_mean       = np.empty([N,noris,nspeeds])

for iO,ori in enumerate(oris):
    for iS,speed in enumerate(speeds):
        
        idx_trial = np.logical_and(sessions[0].trialdata['centerOrientation']==ori,sessions[0].trialdata['centerSpeed']==speed)
        resp_mean[:,iO,iS] = np.nanmean(sessions[sesidx].respmat[:,idx_trial],axis=1)

prefori  = np.argmax(resp_meanori,axis=1)

resp_meanori_pref = resp_meanori.copy()
for n in range(N):
    resp_meanori_pref[n,:] = np.roll(resp_meanori[n,:],-prefori[n])

#Sort based on response magnitude:
magresp                 = np.max(resp_meanori,axis=1) - np.min(resp_meanori,axis=1)
arr1inds                = magresp.argsort()
resp_meanori_pref       = resp_meanori_pref[arr1inds[::-1],:]

##### Plot orientation tuned response:
fig, ax = plt.subplots(figsize=(4, 7))
# ax.imshow(resp_meanori_pref, aspect='auto',extent=[0,360,0,N],vmin=-150,vmax=700) 
ax.imshow(resp_meanori_pref, extent=[0,360,0,N],vmin=np.percentile(resp_meanori_pref,5),vmax=np.percentile(resp_meanori_pref,98)) 

plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
ax.set_xlabel('Orientation (deg)')
ax.set_ylabel('Neuron')

# plt.close('all')


########### PCA on trial-averaged responses ############
######### plot result as scatter by orientation ########


colorset    = get_clr_gratingnoise_stimuli()

respmat_zsc = zscore(sessions[sesidx].respmat,axis=1) # zscore for each neuron across trial responses
# respmat_zsc = respmat # zscore for each neuron across trial responses

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(respmat_zsc.T).T #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from N by K to ncomp by K

ori_ind         = [np.argwhere(np.array(sessions[sesidx].trialdata['centerOrientation']) == ori)[:, 0] for ori in oris]
speed_ind       = [np.argwhere(np.array(sessions[sesidx].trialdata['centerSpeed']) == speed)[:, 0] for speed in speeds]

shade_alpha      = 0.2
lines_alpha      = 0.8
# pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

projections = [(0, 1), (1, 2), (0, 2)]
fig, axes = plt.subplots(1, 3, figsize=[7, 3], sharey='row', sharex='row')
for ax, proj in zip(axes, projections):
    for iO, ori in enumerate(oris):                                #plot orientation separately with diff colors
        for iS, speed in enumerate(speeds):                       #plot speed separately with diff colors
            idx = np.intersect1d(ori_ind[iO],speed_ind[iS])
            x = Xp[proj[0],idx]                          #get all data points for this ori along first PC or projection pairs
            y = Xp[proj[1],idx]                          #get all data points for this ori along first PC or projection pairs

            # x = Xp[proj[0],ori_ind[io]]                          #get all data points for this ori along first PC or projection pairs
            # y = Xp[proj[1],ori_ind[io]]                          #and the second
            ax.scatter(x, y, color=colorset[iO,iS,:], s=sessions[sesidx].respmat_runspeed[idx], alpha=0.8)     #each trial is one dot
            ax.set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
            ax.set_ylabel('PC {}'.format(proj[1]+1))

sns.despine(fig=fig, top=True, right=True)
# ax.legend(['%d deg - %d deg/s' % (ori,speed) for ori in unique_oris for speed in unique_speeds],title='Conditions')

################### PCA unsupervised dispaly of noise around center for each condition #################
## split into area 1 and area 2:
idx_V1 = np.where(sessions[sesidx].celldata['roi_name']=='V1')[0]
idx_PM = np.where(sessions[sesidx].celldata['roi_name']=='PM')[0]

X1 = sessions[sesidx].respmat[idx_V1,:]
X2 = sessions[sesidx].respmat[idx_PM,:]

Y1 = np.vstack((sessions[sesidx].trialdata['deltaOrientation'],
               sessions[sesidx].trialdata['deltaSpeed'],
               sessions[sesidx].respmat_runspeed))
Y1 = np.vstack((Y1,np.random.randn(1,K)))

ylabels     = ['Ori','Speed','Running','Random']
arealabels  = ['V1','PM']

# Define neural data parameters
N1,K        = np.shape(X1)
N2          = np.shape(X2)[0]
NY          = np.shape(Y1)[0]

## 
cmap = plt.colormaps['hot']

for iY in range(NY):
    fig, axes = plt.subplots(3, 3, figsize=[9, 9])
    proj = (0, 2)
    # proj = (5, 6)
    for iO, ori in enumerate(oris):                                #plot orientation separately with diff colors
        for iS, speed in enumerate(speeds):                       #plot speed separately with diff colors
            # ax = axes[iO,iS]
            idx         = np.intersect1d(ori_ind[iO],speed_ind[iS])
            
            # Xp          = pca.fit_transform(respmat_zsc[:,idx].T).T #fit pca to response matrix (n_samples by n_features)
            Xp          = pca.fit_transform(X1[:,idx].T).T #fit pca to response matrix (n_samples by n_features)
            #dimensionality is now reduced from N by K to ncomp by K

            x = Xp[proj[0],:]                          #get all data points for this ori along first PC or projection pairs
            y = Xp[proj[1],:]                          #get all data points for this ori along first PC or projection pairs
            
            c = cmap(minmax_scale(Y1[iY,idx], feature_range=(0, 1)))[:,:3]

            # tip_rate = tips.eval("tip / total_bill").rename("tip_rate")
            sns.scatterplot(x=x, y=y, c=c,ax = axes[iO,iS],s=10,legend = False,edgecolor =None)
            plt.title(ylabels[iY])
            # ax.scatter(x, y, color=pal[t], s=25, alpha=0.8)     #each trial is one dot
            # ax.scatter(x, y, color=pal[(iS-1)*len(unique_oris)+iO], s=respmat_runspeed[idx], alpha=0.8)     #each trial is one dot
            axes[iO,iS].set_xlabel('PC {}'.format(proj[0]+1))            #give labels to axes
            axes[iO,iS].set_ylabel('PC {}'.format(proj[1]+1))
    sns.despine(fig=fig, top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'PCA_perStim_color' + ylabels[iY] + '.png'), format = 'png')


#### linear model explaining responses: 
from numpy import linalg

def LM(Y, X, lam=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    Y_hat = X @ B_hat
    return B_hat

def Rss(Y, Y_hat, normed=True):
    """ evaluate (normalized) model error """
    e = Y_hat - Y
    Rss = np.trace(e.T @ e)
    if normed:
        Rss /= Y.shape[0]
    return Rss

def EV(X, u):
    # how much of the variance lies along this dimension?
    # here X is the data matrix (samples x features) and u is the dimension

    return EV

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

kfold = 5
R2_Y_mat    = np.empty((NY,noris,nspeeds))
R2_X1_mat   = np.empty((noris,nspeeds))
weights     = np.empty((NY,N1,noris,nspeeds,kfold)) 

for iO, ori in enumerate(oris): 
    for iS, speed in enumerate(speeds):     
        # ax = axes[iO,iS]
        idx = np.intersect1d(ori_ind[iO],speed_ind[iS])

        X = X1[:,idx].T

        Y = zscore(Y1[:,idx],axis=1).T #to be able to interpret weights in uniform scale

        #Implementing cross validation
        kf  = KFold(n_splits=kfold, random_state=5,shuffle=True)
        
        model = linear_model.Ridge(alpha=0.01)  

        Yhat = np.empty(np.shape(Y))
        for (train_index, test_index),iF in zip(kf.split(X),range(kfold)):
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index,:],Y[test_index,:]
            
            model.fit(X_train,Y_train)

            # Yhat_train  = model.predict(X_train)
            Yhat[test_index,:]   = model.predict(X_test)

            weights[:,:,iO,iS,iF] = model.coef_

        # [iO,iS] = Rss(Y, Yhat)
        for iY in range(NY):
            R2_Y_mat[:,iO,iS] = r2_score(Y, Yhat, multioutput='raw_values')

############ # ############# ############# ############# ############# 
fig, axes   = plt.subplots(1, 4, figsize=[9, 4])
for iY in range(NY):
    sns.heatmap(data=R2_Y_mat[iY,:,:],vmin=0,vmax=1,ax=axes[iY])
    axes[iY].set_title(ylabels[iY])
    axes[iY].set_xticklabels(oris)
    axes[iY].set_yticklabels(speeds)

plt.tight_layout()

#### ################## ######################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
model = PCA(n_components=5)

coefs = np.mean(weights,axis=4) #average over folds

for iO, ori in enumerate(oris): 
    for iS, speed in enumerate(speeds):     
        for iY in range(NY):
            X = X1[:,idx].T

            Y = zscore(Y1[:,idx],axis=1).T #to be able to interpret weights in uniform scale


            # EV(X,u)
            u = coefs[iY,:,iO,iS]
            u = u[:,np.newaxis]


            model.fit(sc.fit_transform(X))
            model.score
            v = model.components_[0,:]
            G = v @ v.T @ X.T @ X

            G = u @ u.T @ X.T @ X
            TSS = np.trace(X.T @ X)

            RSS = np.trace(G)





model.fit(X)

Xcov = np.cov(X.T)

TSS = np.trace(X.T @ X)

# Get variance explained by singular values
explained_variance_ = (S ** 2) / (n_samples - 1)
total_var = explained_variance_.sum()
explained_variance_ratio_ = explained_variance_ / total_var
singular_values_ = S.copy()  # Store the singular values.


for iO, ori in enumerate(oris): 
    for iS, speed in enumerate(speeds):     
        # sns.heatmap(data=R2_Y_mat[])
        # proj_ori    = X_test @ regr.coef_[0,:].T
        # regr.fit(X.T, y_speed)
        # plt.scatter(proj_ori,Yhat_test[:,0])
        # print(r2_score(y_train, Yhat_train))
        print(r2_score(y_test, Yhat_test,multioutput='raw_values'))

        c = np.mean((cmap1(minmax_scale(y_test['deltaOrientation'], feature_range=(0, 1))),
                     cmap2(minmax_scale(y_test['deltaSpeed'], feature_range=(0, 1)))),axis=0)[:,:3]
        sns.scatterplot(x=Yhat_test[:,0], y=Yhat_test[:,1],c=c,ax = axes[iO,iS],legend = False)

        # c = np.mean((cmap1(minmax_scale(y_train['deltaOrientation'], feature_range=(0, 1))),
        #              cmap2(minmax_scale(y_train['deltaSpeed'], feature_range=(0, 1)))),axis=0)[:,:3]
        # sns.scatterplot(x=Yhat_train[:,0], y=Yhat_train[:,1],color=c,ax = axes[iO,iS],legend = False)

        # sns.scatterplot(x=proj_ori, y=proj_speed,color=c,ax = axes[iO,iS],legend = False)
        axes[iO,iS].set_xlabel('delta Ori')            #give labels to axes
        axes[iO,iS].set_ylabel('delta Speed')            #give labels to axes
        axes[iO,iS].set_title('%d deg - %d deg/s' % (ori,speed))       

sns.despine()
plt.tight_layout()

################### Show noise around center for each condition #################

fig, axes = plt.subplots(3, 3, figsize=[9, 9])
proj    = (0, 1)

for iO, ori in enumerate(unique_oris):                                #plot orientation separately with diff colors
    for iS, speed in enumerate(unique_speeds):     
        
        idx     = np.intersect1d(ori_ind[iO],speed_ind[iS])

        X       = respmat_zsc[:,idx]
        y_ori   = sessions[0].trialdata['deltaOrientation'][idx]
        y_speed = sessions[0].trialdata['deltaSpeed'][idx]

        # regr = linear_model.LinearRegression()  
        regr = linear_model.Ridge(alpha=0.001)  
        regr.fit(X.T, y_ori)
        proj_ori    = X.T @ regr.coef_
        regr.fit(X.T, y_speed)
        proj_speed  = X.T @ regr.coef_

        # X = pd.DataFrame({'deltaOrientation': sessions[0].trialdata['deltaOrientation'][idx],
        #         'deltaSpeed': sessions[0].trialdata['deltaSpeed'][idx],
        #         'runSpeed': respmat_runspeed[idx]})
        
        # Y = respmat_zsc[:,idx].T

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=40)

        # regr.fit(X_train,y_train)
        # Yhat_test = regr.predict(X_test)


        # c = np.mean((cmap1(minmax_scale(proj_ori, feature_range=(0, 1))),cmap2(minmax_scale(proj_speed, feature_range=(0, 1)))),axis=0)[:,:3]
        c = np.mean((cmap1(minmax_scale(y_ori, feature_range=(0, 1))),cmap2(minmax_scale(y_speed, feature_range=(0, 1)))),axis=0)[:,:3]

        sns.scatterplot(x=proj_ori, y=proj_speed,c=c,ax = axes[iO,iS],legend = False)
        # sns.scatterplot(x=proj_ori, y=proj_speed,color=c,ax = axes[iO,iS],legend = False)
        axes[iO,iS].set_xlabel('delta Ori')            #give labels to axes
        axes[iO,iS].set_ylabel('delta Speed')            #give labels to axes
        axes[iO,iS].set_title('%d deg - %d deg/s' % (ori,speed))       

sns.despine()
plt.tight_layout()

################## PCA on full session neural data and correlate with running speed

X           = zscore(sessions[0].calciumdata,axis=0)

pca         = PCA(n_components=15) #construct PCA object with specified number of components
Xp          = pca.fit_transform(X) #fit pca to response matrix (n_samples by n_features)
#dimensionality is now reduced from time by N to time by ncomp


## Get interpolated values for behavioral variables at imaging frame rate:
runspeed_F  = np.interp(x=sessions[0].ts_F,xp=sessions[0].behaviordata['ts'],
                        fp=sessions[0].behaviordata['runspeed'])

plotncomps  = 5
Xp_norm     = preprocessing.MinMaxScaler().fit_transform(Xp)
Rs_norm     = preprocessing.MinMaxScaler().fit_transform(runspeed_F.reshape(-1,1))

cmat = np.empty((plotncomps))
for icomp in range(plotncomps):
    cmat[icomp] = pearsonr(x=runspeed_F,y=Xp_norm[:,icomp])[0]

plt.figure()
for icomp in range(plotncomps):
    sns.lineplot(x=sessions[0].ts_F,y=Xp_norm[:,icomp]+icomp,linewidth=0.5)
sns.lineplot(x=sessions[0].ts_F,y=Rs_norm.reshape(-1)+plotncomps,linewidth=0.5,color='k')

plt.xlim([sessions[0].trialdata['tOnset'][500],sessions[0].trialdata['tOnset'][800]])
for icomp in range(plotncomps):
    plt.text(x=sessions[0].trialdata['tOnset'][700],y=icomp+0.25,s='r=%1.3f' %cmat[icomp])

plt.ylim([0,plotncomps+1])

########################################


##############################
# PCA on trial-concatenated matrix:
# Reorder such that tensor is N by K x T (not K by N by T)
# then reshape to N by KxT (each row is now the activity of all trials over time concatenated for one neuron)

mat_zsc     = tensor.transpose((1,0,2)).reshape(N,K*T,order='F') 
mat_zsc     = zscore(mat_zsc,axis=4)

pca               = PCA(n_components=100) #construct PCA object with specified number of components
Xp                = pca.fit_transform(mat_zsc) #fit pca to response matrix

# [U,S,Vt]          = pca._fit_full(mat_zsc,100) #fit pca to response matrix

# [U,S,Vt]          = pca._fit_truncated(mat_zsc,100,"arpack") #fit pca to response matrix

plt.figure()
sns.lineplot(data=pca.explained_variance_ratio_)
plt.xlim([-1,100])
plt.ylim([0,0.15])

