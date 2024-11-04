
####################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from utils.plotting_style import *  # get all the fixed color schemes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import medfilt
from scipy.stats import zscore
from rastermap import Rastermap, utils
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import logging

logger = logging.getLogger(__name__)


def get_rand_trials(Session, ntrials=80):
    trialsel = [np.random.randint(low=5, high=len(Session.trialdata)-100)]
    trialsel.append(trialsel[0]+ntrials)
    return trialsel


def plot_excerpt(Session, trialsel=None, neuronsel=None, plot_neural=True, plot_behavioral=True, neural_version='traces'):
    if trialsel is None:
        trialsel = get_rand_trials(Session)
    logger.info('Plotting trials %d to %d' % (trialsel[0], trialsel[1]))
    example_tstart = Session.trialdata['tOffset'][trialsel[0]-1]
    example_tstop = Session.trialdata['tOnset'][trialsel[1]-1]

    fig, ax = plt.subplots(figsize=[9, 12])
    counter = 0
    if plot_neural:
        if neural_version == 'traces':
            counter = plot_neural_traces(
                Session, ax, trialsel=trialsel, neuronsel=neuronsel, counter=counter)
        elif neural_version == 'raster':
            counter = plot_neural_raster(
                Session, ax, trialsel=trialsel, neuronsel=neuronsel, counter=counter)
        counter -= 1

    if plot_behavioral:
        counter = plot_behavioral_traces(
            Session, ax, trialsel=trialsel, counter=counter)

    plot_stimuli(Session, trialsel, ax)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])

    ax.set_xlim([example_tstart-10, example_tstop])
    ax.set_ylim([counter-1, 1])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10,
                  "10 Sec", loc=4, frameon=False))
    ax.axis('off')

    fig.tight_layout()

    return fig


def plot_norm_trace(x, y, offset=0, clr='k'):
    min_max_scaler = preprocessing.MinMaxScaler()
    y = np.array(y)
    y = min_max_scaler.fit_transform(y[:, np.newaxis])
    handle = plt.plot(x, y + offset, linewidth=0.5, color=clr)[0]
    return handle


def plot_stimuli(Session, trialsel, ax):

    # Add stimuli:
    if Session.protocol == 'GR':
        oris = np.unique(Session.trialdata['Orientation'])
        rgba_color = plt.get_cmap('hsv', lut=16)(np.linspace(0, 1, len(oris)))

        for i in np.arange(trialsel[0], trialsel[1]):
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i], -1000], 1, 2000, alpha=0.1, linewidth=0,
                                       facecolor=rgba_color[np.where(oris == Session.trialdata['Orientation'][i])]))

        handles = []
        for i, ori in enumerate(oris):
            handles.append(ax.add_patch(plt.Rectangle(
                [0, 0], 1, 1000, alpha=0.3, linewidth=0, facecolor=rgba_color[i])))
        ax.legend(handles, oris, loc='center right',
                  bbox_to_anchor=(1.25, 0.5))
    elif Session.protocol == 'GN':
        print('Need to copy from analyze_GN')

        oris = np.sort(np.unique(Session.trialdata['centerOrientation']))
        speeds = np.sort(np.unique(Session.trialdata['centerSpeed']))
        clrs, oo = get_clr_gratingnoise_stimuli(oris, speeds)

        for i in np.arange(trialsel[0], trialsel[1]):
            iO = np.where(oris == Session.trialdata['centerOrientation'][i])
            iS = np.where(speeds == Session.trialdata['centerSpeed'][i])
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i], -1000], 1, 1000, alpha=0.3, linewidth=0,
                                       facecolor=clrs[iO, iS, :].flatten()))

        for iO, ori in enumerate(oris):
            for iS, speed in enumerate(speeds):
                ax.add_patch(plt.Rectangle(
                    [0, 0], 1, 3, alpha=0.3, linewidth=0, facecolor=clrs[iO, iS, :].flatten()))

    elif Session.protocol == 'IM':
        # rgba_color  = plt.get_cmap('prism',lut=np.diff(trialsel)[0])(np.linspace(0, 1, np.diff(trialsel)[0]))
        rgba_color = sns.color_palette("Set2", np.diff(trialsel)[0])

        for i, k in enumerate(np.arange(trialsel[0], trialsel[1])):
            ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][k], -1000],
                                       Session.trialdata['tOffset'][k] -
                                       Session.trialdata['tOnset'][k],
                                       2000, alpha=0.3, linewidth=0,
                                       # facecolor=rgba_color[i,:]))
                                       facecolor=rgba_color[i]))
        # handles= []
        # for i,ori in enumerate(oris):
            # handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))
    return


def plot_behavioral_traces(Session, ax, trialsel=None, nvideoPCs=8, counter=0):

    example_tstart = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop = Session.trialdata['tOnset'][trialsel[1]-1]

    ts_V = Session.videodata['ts']
    idx_V = np.logical_and(ts_V > example_tstart, ts_V < example_tstop)
    handles = []
    labels = []

    clrs = sns.color_palette('husl', 4)
    clrs = sns.color_palette("crest", nvideoPCs)

    for iPC in range(nvideoPCs):
        motionenergy = Session.videodata['videoPC_%d' % iPC][idx_V]
        handles.append(plot_norm_trace(
            ts_V[idx_V], motionenergy, counter, clr=clrs[iPC]))
        # labels.append('videoPC%d' %iPC)
        counter -= 1

    ax.text(example_tstart, counter+nvideoPCs/2, 'video PCs',
            fontsize=9, color='black', horizontalalignment='right')

    # motionenergy = Session.videodata['motionenergy'][idx_V]
    # handles.append(plot_norm_trace(
    #     ts_V[idx_V], motionenergy, counter, clr='maroon'))
    # # labels.append('Motion Energy')
    # ax.text(example_tstart, counter, 'video ME', fontsize=9,
    #         color='black', horizontalalignment='right')
    # counter -= 1

    pupil_area = Session.videodata['pupil_area'][idx_V]
    handles.append(plot_norm_trace(
        ts_V[idx_V], pupil_area, counter, clr='purple'))
    # labels.append('Pupil Size')
    ax.text(example_tstart, counter, 'Pupil Size', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    pupil_area = Session.videodata['pupil_xpos'][idx_V]
    handles.append(plot_norm_trace(
        ts_V[idx_V], pupil_area, counter, clr='indigo'))
    # labels.append('Pupil X-pos')
    ax.text(example_tstart, counter, 'Pupil X-pos', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    pupil_area = Session.videodata['pupil_ypos'][idx_V]
    handles.append(plot_norm_trace(
        ts_V[idx_V], pupil_area, counter, clr='plum'))
    # labels.append('Pupil Y-pos')
    ax.text(example_tstart, counter, 'Pupil Y-pos', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    ts_B = Session.behaviordata['ts']
    idx_B = np.logical_and(ts_B > example_tstart, ts_B < example_tstop)

    runspeed = Session.behaviordata['runspeed'][idx_B]

    handles.append(plot_norm_trace(
        ts_B[idx_B], runspeed, counter, clr='saddlebrown'))
    # labels.append('Running Speed')
    ax.text(example_tstart, counter, 'Running Speed', fontsize=9,
            color='black', horizontalalignment='right')
    counter -= 1

    return counter


def plot_neural_traces(Session, ax, trialsel, neuronsel=None, counter=0, nexcells=8):

    example_tstart  = Session.trialdata['tOffset'][trialsel[0]-1]
    example_tstop   = Session.trialdata['tOnset'][trialsel[1]-1]

    scaleddata      = np.array(Session.calciumdata)
    min_max_scaler  = preprocessing.MinMaxScaler()
    scaleddata      = min_max_scaler.fit_transform(scaleddata)
    scaleddata      = scaleddata[np.logical_and(
        Session.ts_F > example_tstart, Session.ts_F < example_tstop)]

    areas           = np.unique(Session.celldata['roi_name'])
    labeled         = np.unique(Session.celldata['redcell'])
    labeltext       = ['unlabeled', 'labeled',]

    if neuronsel is None:
        example_cells   = np.array([])

        for iarea, area in enumerate(areas):
            for ilabel, label in enumerate(labeled):
                idx             = np.where(np.logical_and(Session.celldata['roi_name'] == area, 
                                            Session.celldata['redcell'] == label))[0]
                temp_excells    = np.min((len(idx), nexcells))
                excerpt_var     = np.var(scaleddata, axis=0)
                example_cells    = np.append(example_cells, idx[np.argsort(-excerpt_var[idx])[:temp_excells]])
    else:
        example_cells = np.array(neuronsel)

    clrs = get_clr_labeled()
    for iarea, area in enumerate(areas):
        for ilabel, label in enumerate(labeled):
            example_cells_area_label = example_cells[np.logical_and(Session.celldata['roi_name'][example_cells] == area,
                                                                    Session.celldata['redcell'][example_cells] == label)]

            excerpt = scaleddata[:, example_cells_area_label.astype(int)]

            ncells = np.shape(excerpt)[1]

            for i in range(ncells):
                counter -= 1
                ax.plot(Session.ts_F[np.logical_and(Session.ts_F > example_tstart, Session.ts_F < example_tstop)],
                        excerpt[:, i]+counter, linewidth=0.5, color=clrs[ilabel])

            ax.text(example_tstart, counter+ncells/2, area + ' - ' +
                    labeltext[ilabel], fontsize=9, color='black', horizontalalignment='right')

    return counter


def plot_neural_raster(Session, ax, trialsel, neuronsel=None, counter=0):

    example_tstart = Session.trialdata['tOffset'][trialsel[0]-1]
    example_tstop = Session.trialdata['tOnset'][trialsel[1]-1]

    areas = np.unique(Session.celldata['roi_name'])
    labeled = np.unique(Session.celldata['redcell'])
    labeltext = ['unlabeled', 'labeled',]

    clrs = get_clr_labeled()
    for iarea, area in enumerate(areas):
        for ilabel, label in enumerate(labeled):

            idx = np.where(np.logical_and(
                Session.celldata['roi_name'] == area, Session.celldata['redcell'] == label))[0]
            ncells = len(idx)

            if ncells>0:
                shrinkfactor = np.sqrt(ncells)

                excerpt = np.array(Session.calciumdata.loc[np.logical_and(
                    Session.ts_F > example_tstart, Session.ts_F < example_tstop)])
                excerpt = excerpt[:, idx]

                datamat = zscore(excerpt.T, axis=1)

                # fit rastermap
                model = Rastermap(n_PCs=100, n_clusters=50,
                                locality=0.25, time_lag_window=5).fit(datamat)
                y = model.embedding  # neurons x 1
                isort = model.isort

                # bin over neurons
                X_embedding = zscore(utils.bin1d(
                    datamat[isort, :], bin_size=5, axis=0), axis=1)
                # ax.imshow(spks[isort, xmin:xmax], cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
                rasterclrs = ["gray_r", "Reds"]
                ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap=rasterclrs[ilabel], aspect="auto",
                        extent=[example_tstart, example_tstop, counter-ncells/shrinkfactor, counter])

                counter -= np.ceil(ncells/shrinkfactor)

                ax.text(example_tstart, counter+ncells/shrinkfactor/2, area + ' - ' + labeltext[ilabel],
                        fontsize=9, color='black', horizontalalignment='right')

    return counter


# Function that takes in the tensor and computes the average response for some example neurons:
def plot_tuned_response(calciumdata, trialdata, t_axis, example_cells):
    """
    The plot_tuned_response function is used to visualize the average response of specific neurons to different orientations. It takes in four inputs:
    calciumdata: a 3D tensor containing calcium imaging data for multiple cells, trials, and timepoints.
    trialdata: a pandas DataFrame containing trial information, including orientation.
    t_axis: a 1D array representing the time axis.
    example_cells: a list of cell indices to plot.
    returns: a figure with subplots for each cell and each orientation.
    """
    if calciumdata.ndim != 3:
        raise ValueError("calciumdata must have shape (n_cells, n_trials, n_timepoints)")

    if t_axis.ndim != 1:
        raise ValueError("t_axis must be a 1D array")

    T = len(t_axis)
    oris = np.sort(pd.Series.unique(trialdata['Orientation']))
    resp_meanori = np.empty([len(example_cells), len(oris), T])

    for i, cell in enumerate(example_cells):
        for j, ori in enumerate(oris):
            resp_meanori[i, j, :] = np.nanmean(calciumdata[np.ix_([cell], trialdata['Orientation'] == ori,range(T))], axis=1)

    fig, axs = plt.subplots(len(example_cells), len(oris), figsize=[12, 8], sharex=True, sharey=False)
    axs = axs.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, len(oris)))  # assume this is the color palette used in plot_PCA_gratings
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)),(2,1))

    for i, cell in enumerate(example_cells):
        row_max = 0

        for j, ori in enumerate(oris):
            axs[i * len(oris) + j].plot(t_axis, resp_meanori[i, j, :],color=pal[j])
            # axs[i * len(oris) + j].set_title(f'Cell {cell}, {ori} deg')
            axs[i * len(oris) + j].set_xticks([])
            axs[i * len(oris) + j].set_yticks([])
            # REMOVE axis borders
            axs[i * len(oris) + j].axis('off')
            row_max = max(row_max, np.max(resp_meanori[i, j, :]))
        
        for j in range(len(oris)):
            axs[i * len(oris) + j].set_ylim(top=row_max * 1.1)  # add 10% padding
            # Add vertical dotted line at t=0
            # axs[i * len(oris) + j].axvline(x=0, color='k', linestyle=':', linewidth=1, ymin=axs[i * len(oris) + j].get_ylim()[0], ymax=row_max*0.7)
            axs[i * len(oris) + j].axvline(x=0, color='k', linestyle=':', linewidth=1)

    return fig

def plot_PCA_gratings(ses,size='runspeed',cellfilter=None,apply_zscore=True):
    """
    The plot_PCA_gratings function is used to visualize the first two principal components of a population of neurons' responses to grating stimuli. It takes in three inputs:
    ses: a Session object containing the responses to be analyzed.
    size (optional): a string specifying the size of the scatter plot markers. Default is 'runspeed'.
    cellfilter (optional): a boolean array specifying which cells to include in the analysis. Default is None.
    apply_zscore (optional): a boolean specifying whether to apply a zscore to each neuron's responses. Default is True.
    returns: a figure with subplots for each orientation, with the first two principal components plotted as x and y.
    """
    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########
    respmat = ses.respmat

    if apply_zscore is True:
        respmat = zscore(respmat,axis=1) # zscore for each neuron across trial responses

    if cellfilter is not None:
        respmat = respmat[cellfilter,:]

    pca         = PCA(n_components=15) #construct PCA object with specified number of components
    Xp          = pca.fit_transform(respmat.T).T #fit pca to response matrix (n_samples by n_features)
    #dimensionality is now reduced from N by K to ncomp by K
    
    ori         = ses.trialdata['Orientation']
    oris        = np.sort(pd.Series.unique(ses.trialdata['Orientation']))

    ori_ind = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    shade_alpha = 0.2
    lines_alpha = 0.8
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)), (2, 1))
    if size == 'runspeed':
        sizes = (ses.respmat_runspeed - np.percentile(ses.respmat_runspeed, 5)) / \
            (np.percentile(ses.respmat_runspeed, 95) -
             np.percentile(ses.respmat_runspeed, 5))
    elif size == 'videome':
        sizes = (ses.respmat_videome - np.percentile(ses.respmat_videome, 5)) / \
            (np.percentile(ses.respmat_videome, 95) -
             np.percentile(ses.respmat_videome, 5))

    projections = [(0, 1), (1, 2), (0, 2)]
    projections = [(0, 1), (1, 2)]
    fig, axes = plt.subplots(1, len(projections), figsize=[
                             len(projections)*3, 3], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):
        # plot orientation separately with diff colors
        for t, t_type in enumerate(oris):
            # get all data points for this ori along first PC or projection pairs
            x = Xp[proj[0], ori_ind[t]]
            y = Xp[proj[1], ori_ind[t]]  # and the second
            # ax.scatter(x, y, color=pal[t], s=25, alpha=0.8)     #each trial is one dot
            # each trial is one dot
            ax.scatter(x, y, color=pal[t], s=sizes[ori_ind[t]]*10, alpha=0.8)
            # ax.scatter(x, y, color=pal[t], s=ses.respmat_videome[ori_ind[t]], alpha=0.8)     #each trial is one dot
            ax.set_xlabel('PC {}'.format(proj[0]+1))  # give labels to axes
            ax.set_ylabel('PC {}'.format(proj[1]+1))

        sns.despine(fig=fig, top=True, right=True)
        # ax.legend(oris,title='Ori')

    # Put a legend to the right of the current axis
    ax.legend(oris, title='Orientation', frameon=False, fontsize=6, title_fontsize=8,
              loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    return fig


def plot_PCA_gratings_3D(ses, size='runspeed', export_animation=False, savedir=None,thr_tuning=0):

    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########

    areas = np.unique(ses.celldata['roi_name'])

    ori = ses.trialdata['Orientation']
    oris = np.sort(pd.Series.unique(ses.trialdata['Orientation']))

    ori_ind = [np.argwhere(np.array(ori) == iori)[:, 0] for iori in oris]

    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris)/2)), (2, 1))
    if size == 'runspeed':
        sizes = (ses.respmat_runspeed - np.percentile(ses.respmat_runspeed, 5)) / \
            (np.percentile(ses.respmat_runspeed, 95) -
             np.percentile(ses.respmat_runspeed, 5))
    elif size == 'videome':
        sizes = (ses.respmat_videome - np.percentile(ses.respmat_videome, 5)) / \
            (np.percentile(ses.respmat_videome, 95) -
             np.percentile(ses.respmat_videome, 5))

    fig = plt.figure(figsize=[len(areas)*4, 4])
    # fig,axes = plt.figure(1, len(areas), figsize=[len(areas)*3, 3])

    for iarea, area in enumerate(areas):
        # ax = axes[iarea]
        idx_area = ses.celldata['roi_name'] == area
        idx_tuned = ses.celldata['tuning_var'] > thr_tuning
        idx = np.logical_and(idx_area, idx_tuned)
        # zscore for each neuron across trial responses
        respmat_zsc = zscore(ses.respmat[idx, :], axis=1)

        # construct PCA object with specified number of components
        pca = PCA(n_components=3)
        # fit pca to response matrix (n_samples by n_features)
        Xp = pca.fit_transform(respmat_zsc.T).T
        # dimensionality is now reduced from N by K to ncomp by K

        ax = fig.add_subplot(1, len(areas), iarea+1, projection='3d')
        
        # plot orientation separately with diff colors
        for t, t_type in enumerate(oris):
            # get all data points for this ori along first PC or projection pairs
            x = Xp[0, ori_ind[t]]
            y = Xp[1, ori_ind[t]]  # and the second
            z = Xp[2, ori_ind[t]]  # and the second
            # ax.scatter(x, y, color=pal[t], s=25, alpha=0.8)     #each trial is one dot
            # ax.scatter(x, y, z, color=pal[t], s=ses.respmat_runspeed[ori_ind[t]], alpha=0.8)     #each trial is one dot
            # each trial is one dot
            ax.scatter(x, y, z, color=pal[t], s=sizes[ori_ind[t]]*6, alpha=0.8)
            # ax.scatter(x, y, z,marker='o')     #each trial is one dot
        ax.set_xlabel('PC 1')  # give labels to axes
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_title(area)
        
        ax.grid(False)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Get rid of colored axes planes, remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

            # ax.view_init(elev=-30, azim=45, roll=-45)
        print('Variance Explained (%s) by first 3 components: %2.2f' %
              (area, pca.explained_variance_ratio_.cumsum()[2]))

    if export_animation:
        print("Making animation")
        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 364, 4), interval=100)
        rot_animation.save(os.path.join(
            savedir, 'rotation.gif'), dpi=80, writer='imagemagick')

    return fig


def rotate(angle):
    axes = fig.axes
    for ax in axes:
        ax.view_init(azim=angle)


def plot_PCA_images(ses, size='runspeed'):

    ########### PCA on trial-averaged responses ############
    ######### plot result as scatter by orientation ########

    # zscore for each neuron across trial responses
    respmat_zsc = zscore(ses.respmat, axis=1)
    # respmat_zsc = ses.respmat # zscore for each neuron across trial responses

    # construct PCA object with specified number of components
    pca = PCA(n_components=15)
    # fit pca to response matrix (n_samples by n_features)
    Xp = pca.fit_transform(respmat_zsc.T).T
    # dimensionality is now reduced from N by K to ncomp by K

    # imagid         = ses.trialdata['ImageNumber']
    # imagids        = np.sort(pd.Series.unique(ses.trialdata['ImageNumber']))

    if size == 'runspeed':
        sizes = (ses.respmat_runspeed - np.percentile(ses.respmat_runspeed, 5)) / \
            (np.percentile(ses.respmat_runspeed, 95) -
             np.percentile(ses.respmat_runspeed, 5))
    elif size == 'videome':
        sizes = (ses.respmat_videome - np.percentile(ses.respmat_videome, 5)) / \
            (np.percentile(ses.respmat_videome, 95) -
             np.percentile(ses.respmat_videome, 5))

    colors = sns.color_palette('tab10', np.shape(respmat_zsc)[1])

    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):
        # get all data points for this ori along first PC or projection pairs
        x = Xp[proj[0], :]
        y = Xp[proj[1], :]  # and the second
        ax.scatter(x, y, color=colors, s=sizes*8,
                   alpha=0.3)  # each trial is one dot
        ax.set_xlabel('PC {}'.format(proj[0]+1))  # give labels to axes
        ax.set_ylabel('PC {}'.format(proj[1]+1))

        sns.despine(fig=fig, top=True, right=True)
        # ax.legend(labels=oris)
    plt.tight_layout()
    return fig
