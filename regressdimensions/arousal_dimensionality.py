

####################################################
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat,construct_behav_matrix_ts_F
from scipy.stats import zscore, pearsonr

from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from utils.plot_lib import * #get all the fixed color schemes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import medfilt

from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_PCA_gratings_3D
from utils.RRRlib import RRR, LM, EV, regress_out_behavior_modulation
from scipy import linalg

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Arousal\\')



# For multiplicative noise, the variability in the firing rate is still described by
# equation 2.1, but the covariance matrix is scaled by the average Žring rates,
# Qij.x/ D ¾2[±ij C c.1 ¡ ±ij/] fi.x/ fj.x/ : (2.3)
# This produces variances that increase as a function of Žring rate and larger
# correlations for neurons with overlapping tuning curves, as seen in the data
# (Lee et al., 1998). From Umakantha et al. 2021. 
# Is gain modulation or multiplicative *shared noise. 

##############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_19']])
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='deconv')
sesidx = 0

for ises in range(nSessions):
    # sessions[ises].videodata['pupil_area']    = medfilt(sessions[ises].videodata['pupil_area'] , kernel_size=25)
    sessions[ises].videodata['motionenergy']  = medfilt(sessions[ises].videodata['motionenergy'] , kernel_size=25)
    sessions[ises].behaviordata['runspeed']   = medfilt(sessions[ises].behaviordata['runspeed'] , kernel_size=51)


########## Construct behavioral variables sampled at imaging frame rate: #####################
Svars,Svarlabels = construct_behav_matrix_ts_F(sessions[sesidx],nvideoPCs=30)
nSvars = len(Svarlabels)

######## Show correlations between different behavioral measures: ##############################
fig = plt.figure()
# sns.heatmap(np.corrcoef(Svars,rowvar=False),xticklabels=Svarlabels,yticklabels=Svarlabels)
sns.heatmap(np.corrcoef(Svars[:,:13],rowvar=False),xticklabels=Svarlabels[:13],yticklabels=Svarlabels[:13])
fig.tight_layout()
fig.savefig(os.path.join(savedir,'Heatmap_Correlations_ArousalVars_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

##### Compute rank of behavioral modulation across neural population activity:

Y = sessions[sesidx].calciumdata.to_numpy()
X = Svars
B_hat = LM(Y, X, lam=30)
EV_rrr = np.empty(nSvars)

# RRR error for all ranks

U, s, Vh = linalg.svd(B_hat)
S = linalg.diagsvd(s,U.shape[0],s.shape[0])

for i,r in enumerate(range(nSvars)):
    print(r)
    L = U[:,:r] @ S[:r,:r]
    W = Vh[:r,:]
    
    B_hat_lr = L @ W
    Y_hat_lr = X @ B_hat_lr

    # B_hat_lr = RRR(Y, X, B_hat, r)
    # Y_hat_lr_test = X @ B_hat_lr
    EV_rrr[i] = EV(Y, Y_hat_lr)

fig = plt.figure(figsize=(4,3))
plt.plot(range(1,nSvars),EV_rrr[1:],color='purple',linewidth=2)
plt.ylabel('Variance Explained')
plt.xlabel('Rank')
plt.axvline(9,linestyle=':',color='black')
plt.ylim([0,0.15])
fig.savefig(os.path.join(savedir,'Rank_EV_deconv_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#### Show original PCA
#Compute average response per trial:
for ises in range(nSessions):
    sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

    sessions[ises].respmat_runspeed = compute_respmat(sessions[ises].behaviordata['runspeed'],
                                                        sessions[ises].behaviordata['ts'], sessions[ises].trialdata['tOnset'],
                                                        t_resp_start=0,t_resp_stop=1,method='mean')

    sessions[ises].respmat_videome = compute_respmat(sessions[ises].videodata['motionenergy'],
                                                    sessions[ises].videodata['timestamps'], sessions[ises].trialdata['tOnset'],
                                                    t_resp_start=0,t_resp_stop=1,method='mean')


fig = plot_PCA_gratings(sessions[sesidx])
fig.savefig(os.path.join(savedir,'PCA_Gratings_Original_deconv_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#### Regress out behaviorally predicted activity using RRR #################################

sessions[sesidx].calciumdata2 = regress_out_behavior_modulation(sessions[sesidx],nvideoPCs = 30,rank=9)
# sessions[sesidx].respmat2 = regress_out_behavior_modulation(sessions[sesidx].respmat,nvideoPCs = 30,rank=5)

#Compute average response per trial:
for ises in range(nSessions):
    sessions[ises].respmat2         = compute_respmat(sessions[ises].calciumdata2, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
                                  t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)

#################### Show PCA again #################################

fig = plot_PCA_gratings(sessions[sesidx])
# fig.savefig(os.path.join(savedir,'PCA_Gratings_RRR5_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'PCA_Gratings_RRR9_deconv' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#################### Show traces of behavior predicted and across area predicted activity: #################################

fig = plot_excerpt(sessions[sesidx],trialsel=None,plot_neural=True,plot_behavioral=False)

trialsel = [800, 1200]
fig = plot_excerpt(sessions[sesidx],trialsel=trialsel,plot_neural=True,plot_behavioral=True,neural_version='traces')



# fig.savefig(os.path.join(savedir,'TraceExcerpt_dF_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'Excerpt_Traces_deconv_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


# from utils.RRRlib import regress_out_behavior_modulation
# from utils.corr_lib import compute_noise_correlation

# ################# With and without regressing out behavioral modulation: #################################
# fig = plot_PCA_gratings(sessions[sesidx])
# fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_RegressOut_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

# from utils.RRRlib import *
# sessions[sesidx].calciumdata2 = regress_out_behavior_modulation(sessions[sesidx],nvideoPCs = 30,rank=15)
# #Compute average response per trial:
# for ises in range(nSessions):
#     sessions[ises].respmat         = compute_respmat(sessions[ises].calciumdata2, sessions[ises].ts_F, sessions[ises].trialdata['tOnset'],
#                                   t_resp_start=0,t_resp_stop=1,method='mean',subtr_baseline=False)
# fig = plot_PCA_gratings(sessions[sesidx])
# fig.savefig(os.path.join(savedir,'PCA','PCA_Gratings_All_RegressOut_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')
