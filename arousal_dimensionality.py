

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



def plot_norm_trace(x,y,offset=0,clr='k'):
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y[:,np.newaxis])
    handle = plt.plot(x,y + offset,linewidth=0.5,color=clr)[0]
    return handle


def show_excerpt_behavioral_variables(Session,trialsel=None):
    
    if trialsel is None:
        trialsel = [np.random.randint(low=5,high=len(Session.trialdata)-400)]
        trialsel.append(trialsel[0]+40)

    example_tstart  = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    fig, ax = plt.subplots(figsize=[12, 6])

    ts_V    = sessions[0].videodata['timestamps']
    idx_V   = np.logical_and(ts_V>example_tstart,ts_V<example_tstop)
    handles = []
    labels = []

    motionenergy = sessions[0].videodata['motionenergy'][idx_V]
    handles.append(plot_norm_trace(ts_V[idx_V],motionenergy,0,clr='k'))
    labels.append('Motion Energy')

    pupil_area = sessions[0].videodata['pupil_area'][idx_V]
    handles.append(plot_norm_trace(ts_V[idx_V],pupil_area,1,clr='r'))
    labels.append('Pupil Size')

    ts_B    = sessions[0].behaviordata['ts']
    idx_B   = np.logical_and(ts_B>example_tstart,ts_B<example_tstop)

    runspeed = sessions[0].behaviordata['runspeed'][idx_B]
    runspeed = medfilt(runspeed, kernel_size=25)
    handles.append(plot_norm_trace(ts_B[idx_B],runspeed,2,clr='g'))
    labels.append('Running Speed')

    oris            = np.sort(np.unique(Session.trialdata['centerOrientation']))
    speeds          = np.sort(np.unique(Session.trialdata['centerSpeed']))
    clrs,oo     = get_clr_gratingnoise_stimuli(oris,speeds)

    for i in np.arange(trialsel[0],trialsel[1]):
        iO = np.where(oris==Session.trialdata['centerOrientation'][i])
        iS = np.where(speeds==Session.trialdata['centerSpeed'][i])
        ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i],0],1,3,alpha=0.3,linewidth=0,
                                facecolor=clrs[iO,iS,:].flatten()))

    for iO,ori in enumerate(oris):
        for iS,speed in enumerate(speeds):
            ax.add_patch(plt.Rectangle([0,0],1,3,alpha=0.3,linewidth=0,facecolor=clrs[iO,iS,:].flatten()))

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(handles,labels,loc='center right', bbox_to_anchor=(1.25, 0.5),fontsize=12)

    ax.set_xlim([example_tstart,example_tstop])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
    ax.axis('off')

    return fig


show_excerpt_behavioral_variables(sessions[sesidx])