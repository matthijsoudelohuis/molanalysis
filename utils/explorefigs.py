
####################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from utils.plotting_style import * #get all the fixed color schemes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import medfilt


def plot_norm_trace(x,y,offset=0,clr='k'):
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y[:,np.newaxis])
    handle = plt.plot(x,y + offset,linewidth=0.5,color=clr)[0]
    return handle


def excerpt_behavioral(Session,trialsel=None):
    
    if trialsel is None:
        trialsel = [np.random.randint(low=5,high=len(Session.trialdata)-400)]
        trialsel.append(trialsel[0]+40)

    example_tstart  = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    fig, ax = plt.subplots(figsize=[12, 6])

    ts_V    = Session.videodata['timestamps']
    idx_V   = np.logical_and(ts_V>example_tstart,ts_V<example_tstop)
    handles = []
    labels = []

    motionenergy = Session.videodata['motionenergy'][idx_V]
    handles.append(plot_norm_trace(ts_V[idx_V],motionenergy,0,clr='k'))
    labels.append('Motion Energy')

    pupil_area = Session.videodata['pupil_area'][idx_V]
    handles.append(plot_norm_trace(ts_V[idx_V],pupil_area,1,clr='r'))
    labels.append('Pupil Size')

    ts_B    = Session.behaviordata['ts']
    idx_B   = np.logical_and(ts_B>example_tstart,ts_B<example_tstop)

    runspeed = Session.behaviordata['runspeed'][idx_B]
    # runspeed = medfilt(runspeed, kernel_size=25)
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


def plot_stimuli(Session,trialsel,ax):

    #Add stimuli:
    if Session.protocol == 'GR':
        oris        = np.unique(Session.trialdata['Orientation'])
        rgba_color  = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(oris)))  
        
        for i in np.arange(trialsel[0],trialsel[1]):
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i],0],1,1000,alpha=0.3,linewidth=0,
                                    facecolor=rgba_color[np.where(oris==Session.trialdata['Orientation'][i])]))

        handles= []
        for i,ori in enumerate(oris):
            handles.append(ax.add_patch(plt.Rectangle([0,0],1,1000,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))
        ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))
    elif Session.protocol == 'GN':
        print('Need to copy from analyze_GN')
        
    elif Session.protocol == 'IM':
        rgba_color  = plt.get_cmap('prism',lut=np.diff(trialsel)[0])(np.linspace(0, 1, np.diff(trialsel)[0]))  
        
        for i,k in enumerate(np.arange(trialsel[0],trialsel[1])):
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][k],0],
                                       Session.trialdata['tOffset'][k]-Session.trialdata['tOnset'][k],
                                       1000,alpha=0.3,linewidth=0,
                                    facecolor=rgba_color[i,:]))

        # handles= []
        # for i,ori in enumerate(oris):
            # handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))
    return



def excerpt_neural_behavioral(Session,example_cells=None,trialsel=None):
    
    # if example_cells is None:
    #     example_cells = np.random.choice(Session.calciumdata.shape[1],10)

    if example_cells is None:
        areas           = np.unique(Session.celldata['roi_name'])
        labeled         = np.unique(Session.celldata['redcell'])
        nexcells        = 10

        example_cells   = np.empty((len(areas),len(labeled),nexcells))

        for iarea,area in enumerate(areas):
            for ilabel,label in enumerate(labeled):
                example_cells[iarea,ilabel,:] = np.random.choice(np.where(np.logical_and(Session.celldata['roi_name']==area,Session.celldata['redcell']==label))[0],nexcells)
        
        example_cells = np.reshape(example_cells,-1).astype('int32')

    if trialsel is None:
        trialsel = [np.random.randint(low=0,high=len(Session.trialdata)-400)]
        trialsel.append(trialsel[0]+40)

    example_tstart  = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    excerpt         = np.array(Session.calciumdata.loc[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)])
    excerpt         = excerpt[:,example_cells]

    min_max_scaler  = preprocessing.MinMaxScaler()
    excerpt         = min_max_scaler.fit_transform(excerpt)

    ncells = np.shape(excerpt)[1]

    for i in range(ncells):
        excerpt[:,i] =  excerpt[:,i] + i

    fig, ax = plt.subplots(figsize=[12, 6])
    plt.plot(Session.ts_F[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)],excerpt,linewidth=0.5,color='black')

    plot_stimuli(Session,trialsel,ax)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])

    ax.set_xlim([example_tstart,example_tstop])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
    ax.axis('off')

    return fig
