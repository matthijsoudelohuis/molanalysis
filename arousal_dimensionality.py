

####################################################
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    os.chdir('t:\\Python\\molanalysis\\')
except:
    os.chdir('e:\\Python\\molanalysis\\')

from loaddata.session_info import filter_sessions,load_sessions
from utils.psth import compute_tensor,compute_respmat
from scipy.stats import zscore, pearsonr

from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from utils.plotting_style import * #get all the fixed color schemes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import medfilt

from utils.explorefigs import excerpt_behavioral


#################################################
# session_list        = np.array([['LPE10883','2023_10_27']])
session_list        = np.array([['LPE10919','2023_11_16']])
sessions            = load_sessions(protocol = 'GN',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=True, calciumversion='dF')

sesidx = 0


# Median filter of data: 
sessions[sesidx].videodata['pupil_area'] = medfilt(sessions[sesidx].videodata['pupil_area'] , kernel_size=25)
sessions[sesidx].videodata['motionenergy'] = medfilt(sessions[sesidx].videodata['motionenergy'] , kernel_size=25)
sessions[sesidx].behaviordata['runspeed'] = medfilt(sessions[sesidx].behaviordata['runspeed'] , kernel_size=51)

S,Slabels = construct_behav_matrix_ts_F(ses,nvideoPCs=nvideoPCs)


def construct_behav_matrix_ts_F(ses,nvideoPCs = 30):
    Slabels = []
    S       = np.empty((len(ses.ts_F),0))
    S       = np.hstack((S,np.expand_dims(np.interp(ses.ts_F.to_numpy(),ses.behaviordata['ts'].to_numpy(), ses.behaviordata['runspeed'].to_numpy()),axis=1)))
    Slabels.append('runspeed')

    fields = ['pupil_area','motionenergy']
    [fields.append('videoPC_' + '%s' % k) for k in range(0,nvideoPCs)]

    for field in fields:
        S       = np.hstack((S,np.expand_dims(np.interp(ses.ts_F.to_numpy(),ses.videodata['timestamps'].to_numpy(), ses.videodata[field].to_numpy()),axis=1)))
        Slabels.append(field)

    return S, Slabels

def regress_out_behavior_modulation(ses,X=None,nvideoPCs = 30):
    S,Slabels = construct_behav_matrix_ts_F(ses,nvideoPCs=nvideoPCs)

    if not X:
        X = ses.calciumdata.to_numpy()

    assert X.shape[0] == S.shape[0],'dimensions of calcium activit and interpolated behavior data do not match'

    Xhat= RRR_wrapper(X,rrrdim=3)


    return 


sns.heatmap(np.corrcoef(S,rowvar=False),xticklabels=Slabels,yticklabels=Slabels)


excerpt_behavioral(sessions[sesidx])