
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
from scipy.stats import zscore
from rastermap import Rastermap, utils

def get_rand_trials(Session):
    trialsel = [np.random.randint(low=5,high=len(Session.trialdata)-100)]
    trialsel.append(trialsel[0]+80)
    return trialsel

def plot_excerpt(Session,trialsel=None,plot_neural=True,plot_behavioral=True,neural_version='traces'):
    if trialsel is None:
        trialsel = get_rand_trials(Session)
    print(trialsel)
    example_tstart  = Session.trialdata['tOffset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    fig, ax = plt.subplots(figsize=[9, 12])
    counter = 0
    if plot_neural:
        if neural_version=='traces':
            counter = plot_neural_traces(Session,ax,trialsel=trialsel,counter=counter)
        elif neural_version=='raster':
            counter = plot_neural_raster(Session,ax,trialsel=trialsel,counter=counter)
        counter -= 1

    if plot_behavioral:
        counter = plot_behavioral_traces(Session,ax,trialsel=trialsel,counter=counter)

    plot_stimuli(Session,trialsel,ax)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])

    ax.set_xlim([example_tstart-10,example_tstop])
    ax.set_ylim([counter-1,1])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
    ax.axis('off')

    return fig


def plot_norm_trace(x,y,offset=0,clr='k'):
    min_max_scaler = preprocessing.MinMaxScaler()
    y = np.array(y)
    y = min_max_scaler.fit_transform(y[:,np.newaxis])
    handle = plt.plot(x,y + offset,linewidth=0.5,color=clr)[0]
    return handle

def plot_stimuli(Session,trialsel,ax):

    #Add stimuli:
    if Session.protocol == 'GR':
        oris        = np.unique(Session.trialdata['Orientation'])
        rgba_color  = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(oris)))  
        
        for i in np.arange(trialsel[0],trialsel[1]):
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i],-1000],1,2000,alpha=0.1,linewidth=0,
                                    facecolor=rgba_color[np.where(oris==Session.trialdata['Orientation'][i])]))

        handles= []
        for i,ori in enumerate(oris):
            handles.append(ax.add_patch(plt.Rectangle([0,0],1,1000,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))
        ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))
    elif Session.protocol == 'GN':
        print('Need to copy from analyze_GN')
        
        oris            = np.sort(np.unique(Session.trialdata['centerOrientation']))
        speeds          = np.sort(np.unique(Session.trialdata['centerSpeed']))
        clrs,oo     = get_clr_gratingnoise_stimuli(oris,speeds)

        for i in np.arange(trialsel[0],trialsel[1]):
            iO = np.where(oris==Session.trialdata['centerOrientation'][i])
            iS = np.where(speeds==Session.trialdata['centerSpeed'][i])
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i],-1000],1,1000,alpha=0.3,linewidth=0,
                                    facecolor=clrs[iO,iS,:].flatten()))

        for iO,ori in enumerate(oris):
            for iS,speed in enumerate(speeds):
                ax.add_patch(plt.Rectangle([0,0],1,3,alpha=0.3,linewidth=0,facecolor=clrs[iO,iS,:].flatten()))

    elif Session.protocol == 'IM':
        # rgba_color  = plt.get_cmap('prism',lut=np.diff(trialsel)[0])(np.linspace(0, 1, np.diff(trialsel)[0]))  
        rgba_color  = sns.color_palette("Set2",np.diff(trialsel)[0])

        for i,k in enumerate(np.arange(trialsel[0],trialsel[1])):
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][k],-1000],
                                       Session.trialdata['tOffset'][k]-Session.trialdata['tOnset'][k],
                                       2000,alpha=0.3,linewidth=0,
                                    # facecolor=rgba_color[i,:]))
                                    facecolor=rgba_color[i]))
        # handles= []
        # for i,ori in enumerate(oris):
            # handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))
    return


def plot_behavioral_traces(Session,ax,trialsel=None,nvideoPCs=10,counter=0):
    
    example_tstart  = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    ts_V            = Session.videodata['timestamps']
    idx_V           = np.logical_and(ts_V>example_tstart,ts_V<example_tstop)
    handles         = []
    labels          = []

    clrs = sns.color_palette('husl', 4)
    clrs = sns.color_palette("crest", nvideoPCs)

    for iPC in range(nvideoPCs):
        motionenergy = Session.videodata['videoPC_%d' %iPC][idx_V]
        handles.append(plot_norm_trace(ts_V[idx_V],motionenergy,counter,clr=clrs[iPC]))
        # labels.append('videoPC%d' %iPC)
        counter -= 1

    ax.text(example_tstart,counter+nvideoPCs/2,'video PCs',fontsize=9,color='black',horizontalalignment='right')

    # motionenergy = Session.videodata['motionenergy'][idx_V]
    # handles.append(plot_norm_trace(ts_V[idx_V],motionenergy,counter,clr='k'))
    # labels.append('Motion Energy')
    # counter -= 1

    pupil_area = Session.videodata['pupil_area'][idx_V]
    handles.append(plot_norm_trace(ts_V[idx_V],pupil_area,counter,clr='purple'))
    # labels.append('Pupil Size')
    ax.text(example_tstart,counter,'Pupil Size',fontsize=9,color='black',horizontalalignment='right')
    counter -= 1

    pupil_area = Session.videodata['pupil_xpos'][idx_V]
    handles.append(plot_norm_trace(ts_V[idx_V],pupil_area,counter,clr='plum'))
    # labels.append('Pupil X-pos')
    ax.text(example_tstart,counter,'Pupil X-pos',fontsize=9,color='black',horizontalalignment='right')
    counter -= 1

    pupil_area = Session.videodata['pupil_ypos'][idx_V]
    handles.append(plot_norm_trace(ts_V[idx_V],pupil_area,counter,clr='plum'))
    # labels.append('Pupil Y-pos')
    ax.text(example_tstart,counter,'Pupil Y-pos',fontsize=9,color='black',horizontalalignment='right')
    counter -= 1

    ts_B    = Session.behaviordata['ts']
    idx_B   = np.logical_and(ts_B>example_tstart,ts_B<example_tstop)

    runspeed = Session.behaviordata['runspeed'][idx_B]

    handles.append(plot_norm_trace(ts_B[idx_B],runspeed,counter,clr='saddlebrown'))
    # labels.append('Running Speed')
    ax.text(example_tstart,counter,'Running Speed',fontsize=9,color='black',horizontalalignment='right')
    counter -= 1

    return counter

def plot_neural_traces(Session,ax,trialsel=None,counter=0):
    
    example_tstart  = Session.trialdata['tOffset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    scaleddata          = np.array(Session.calciumdata)
    min_max_scaler      = preprocessing.MinMaxScaler()
    scaleddata          = min_max_scaler.fit_transform(scaleddata)
    scaleddata          = scaleddata[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)]

    areas           = np.unique(Session.celldata['roi_name'])
    labeled         = np.unique(Session.celldata['redcell'])
    labeltext       = ['unlabeled','labeled',]
    nexcells        = 10

    example_cells   = np.empty((len(areas),len(labeled),nexcells)).astype('int64')

    for iarea,area in enumerate(areas):
        for ilabel,label in enumerate(labeled):
    
            idx = np.where(np.logical_and(Session.celldata['roi_name']==area,Session.celldata['redcell']==label))[0]
            
            excerpt_var = np.var(scaleddata,axis=0)
            example_cells[iarea,ilabel,:] = idx[np.argpartition(excerpt_var[idx], -nexcells)[-nexcells:]]

            # example_cells[iarea,ilabel,:] = idx[np.argpartition(Session.celldata['skew'][idx], -nexcells)[-nexcells:]]
            # example_cells[iarea,ilabel,:] = idx[np.argpartition(Session.celldata['noise_level'][idx], nexcells)[:nexcells]]
    
    clrs                = get_clr_labeled()
    for iarea,area in enumerate(areas):
        for ilabel,label in enumerate(labeled):
            excerpt         = scaleddata[:,example_cells[iarea,ilabel,:]]

            ncells = np.shape(excerpt)[1]

            for i in range(ncells):
                counter -= 1
                ax.plot(Session.ts_F[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)],
                        excerpt[:,i]+counter,linewidth=0.5,color=clrs[ilabel])

            ax.text(example_tstart,counter+ncells/2,area + ' - ' + labeltext[ilabel],fontsize=9,color='black',horizontalalignment='right')


    return counter


def plot_neural_raster(Session,ax,trialsel=None,counter=0):
    
    example_tstart  = Session.trialdata['tOffset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    areas           = np.unique(Session.celldata['roi_name'])
    labeled         = np.unique(Session.celldata['redcell'])
    labeltext       = ['unlabeled','labeled',]


    clrs                = get_clr_labeled()
    for iarea,area in enumerate(areas):
        for ilabel,label in enumerate(labeled):

            idx = np.where(np.logical_and(Session.celldata['roi_name']==area,Session.celldata['redcell']==label))[0]
            ncells = len(idx)

            shrinkfactor = np.sqrt(ncells)

            excerpt         = np.array(Session.calciumdata.loc[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)])
            excerpt         = excerpt[:,idx]

            datamat         = zscore(excerpt.T, axis=1)

            # fit rastermap
            model           = Rastermap(n_PCs=100, n_clusters=50, 
                            locality=0.25, time_lag_window=5).fit(datamat)
            y           = model.embedding # neurons x 1
            isort       = model.isort

            # bin over neurons
            X_embedding     = zscore(utils.bin1d(datamat[isort,:], bin_size=5, axis=0), axis=1)
            # ax.imshow(spks[isort, xmin:xmax], cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
            rasterclrs = ["gray_r","Reds"]
            ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap=rasterclrs[ilabel], aspect="auto", 
                      extent=[example_tstart,example_tstop, counter-ncells/shrinkfactor, counter])
            
            counter -= np.ceil(ncells/shrinkfactor)

            ax.text(example_tstart,counter+ncells/shrinkfactor/2,area + ' - ' + labeltext[ilabel],
                    fontsize=9,color='black',horizontalalignment='right')

    return counter




