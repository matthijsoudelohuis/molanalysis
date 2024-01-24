

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

# plt.plot(medfilt(data[26016:27016], kernel_size=51),'r',linewidth=0.4)
# plt.plot(data[26016:27016].to_numpy(),'k',linewidth=0.4)

### Create dataframe with relevant variables: 
df = sessions[0].videodata[['pupil_area','motionenergy']]
df['runspeed'] = np.interp(x=sessions[sesidx].videodata['timestamps'],xp=sessions[sesidx].behaviordata['ts'],
                                    fp=sessions[sesidx].behaviordata['runspeed'])

sns.heatmap(df.corr(),vmin=0,vmax=1,annot=True)



excerpt_behavioral(sessions[sesidx])