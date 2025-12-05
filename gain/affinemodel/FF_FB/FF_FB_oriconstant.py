#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress

os.chdir('e:\\Python\\molanalysis')

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.plot_lib import * #get all the fixed color schemes
from preprocessing.preprocesslib import assign_layer,assign_layer2

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

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 


#%% 
#%% fit von mises
def vonmises(x,amp,loc,scale):
    return amp * np.exp( (np.cos(x-loc) - 1) / (2 * scale**2) )

from scipy.optimize import curve_fit


#%% fit von mises
def vonmises(x,amp,loc,scale):
    return amp * np.exp( (np.cos(x-loc) - 1) / (2 * scale**2) )

from scipy.optimize import curve_fit

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

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Correlation output:
PO_data             = np.full((narealabelpairs,2,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing preferred orientation'):
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
        
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        #Difference:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        # meanpopact          = np.nanmean(respdata[idx_N2,:],axis=0) - np.nanmean(respdata[idx_N1,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)

        xdata = np.radians(oris)
        PO_data_ses = np.full((N,2),np.nan)
        for iN in range(N):
            try:
                popt1, pcov = curve_fit(vonmises, xdata, meanresp[iN,:,0],p0=[1,xdata[np.argmax(meanresp[iN,:,0])],0.25])
                PO_data_ses[iN,0] = popt1[1]
                # r2_start[i] = r2_score(resp_meanori1[i],vonmises(xdata,popt1[0],popt1[1],popt1[2]))
            except:
                continue

            try:
                popt2, pcov = curve_fit(vonmises, xdata, meanresp[iN,:,1],p0=[1,xdata[np.argmax(meanresp[iN,:,1])],0.25])
                PO_data_ses[iN,1] = popt2[1]
                # r2_end[i]   = r2_score(resp_meanori2[i],vonmises(xdata,popt2[0],popt2[1],popt2[2]))
            except:
                continue

        PO_data[ialp,:,idx_ses] = PO_data_ses[idx_N3]

#%% ########################### Compute tuning metrics: ###################################

#%% 
celldata = pd.concat([ses.celldata for ses in sessions],axis=0)

#%% 
histres2D = 10
histres1D = 2.5
# fig,axes = plt.subplots(1,3,figsize=(6,3),sharex=,sharey=True)
fig,axes = plt.subplots(1,3,figsize=(9,3))

idx_N = rangeresp>0.04
ax = axes[0]
ialp = 0
data = np.degrees(PO_data)
im = ax.hist2d(data[ialp,0,idx_N],data[ialp,1,idx_N],
           bins=np.arange(0,360,histres2D),cmap='magma',density=True,vmax=0.0001)
# bar = plt.colorbar(im[3])
ax.set_yticks(np.arange(0,360,90))
ax.set_xticks(np.arange(0,360,90))
ax.set_xlabel('PO Low')
ax.set_ylabel('PO High')
ax.set_title('FF')

ialp = 1
ax = axes[1]
im = ax.hist2d(data[ialp,0,idx_N],data[ialp,1,idx_N],
           bins=np.arange(0,360,histres2D),cmap='magma',density=True,vmax=0.0001)
# bar = plt.colorbar(im[3])
# ax.set_ylabel('PO High')
ax.set_xlabel('PO Low')

ax.set_title('FB')
ax.set_yticks(np.arange(0,360,90))
ax.set_xticks(np.arange(0,360,90))

PO_drift = np.abs(np.mod(data[:,1,:] - data[:,0,:],360))
ax = axes[2]
sns.histplot(PO_drift[0,idx_N],color='green',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=np.arange(0,190,histres1D))
sns.histplot(PO_drift[1,idx_N],color='purple',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=np.arange(0,190,histres1D))
# sns.histplot(celldata['PO_drift'][idx_N_high],color='purple',element="step",stat="density", 
#              common_norm=False,alpha=0.2,ax=ax)
# ax.set_xlim([0,25])
ax.legend(['FF','FB'],frameon=False,fontsize=8,title='PO difference')
plt.tight_layout()
my_savefig(fig,savedir,'PO_drift_FF_FB', formats = ['png'])
