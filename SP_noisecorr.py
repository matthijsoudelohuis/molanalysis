
####################################################
import os
from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic,binned_statistic_2d

from statannotations.Annotator import Annotator

from loaddata.session_info import filter_sessions,load_sessions
# from utils.psth import compute_respmat
# from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
# from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D,plot_excerpt
# from utils.plot_lib import shaded_error
# from utils.RRRlib import regress_out_behavior_modulation
from utils.corr_lib import *

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Neural - Gratings\\')

##############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])

# load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['SP']) 

#   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_data(load_behaviordata=True, load_calciumdata=True,load_videodata=True,calciumversion='deconv')
    
    ##############################################################################

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

############################ Compute noise correlations: ###################################
sessions = compute_noise_correlation(sessions)


###################### Plot control figure of signal and noise corrs ##############################
sesidx = 0
fig = plt.figure(figsize=(8,5))
plt.imshow(sessions[sesidx].noise_corr, cmap='coolwarm',
           vmin=np.nanpercentile(sessions[sesidx].noise_corr,5),
           vmax=np.nanpercentile(sessions[sesidx].noise_corr,95))
plt.savefig(os.path.join(savedir,'NoiseCorrelations','NC_SP_Mat_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')


#%% ##################### Compute pairwise neuronal distances: ##############################
sessions = compute_pairwise_metrics(sessions)

# construct dataframe with all pairwise measurements:
df_allpairs  = pd.DataFrame()

for ises in range(nSessions):
    tempdf  = pd.DataFrame({'NoiseCorrelation': sessions[ises].noise_corr.flatten(),
                    # 'DeltaPrefOri': sessions[ises].delta_pref.flatten(),
                    'AreaPair': sessions[ises].areamat.flatten(),
                    'DistXYPair': sessions[ises].distmat_xy.flatten(),
                    'DistXYZPair': sessions[ises].distmat_xyz.flatten(),
                    'DistRfPair': sessions[ises].distmat_rf.flatten(),
                    'AreaLabelPair': sessions[ises].arealabelmat.flatten(),
                    'LabelPair': sessions[ises].labelmat.flatten()}).dropna(how='all') 
                    #drop all rows that have all nan (diagonal + repeat below daig)
    df_allpairs  = pd.concat([df_allpairs, tempdf], ignore_index=True).reset_index(drop=True)
