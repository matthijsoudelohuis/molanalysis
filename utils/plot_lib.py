import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings
from utils.plotting_style import *


def shaded_error(x,y,yerror=None,ax=None,center='mean',error='std',color='black',
                 alpha=0.25,linewidth=2,linestyle='-',label=None):
    x = np.array(x)
    y = np.array(y)

    # if np.ndim(y)==1:
    #     y = y[np.newaxis,:]
    if ax is None:
        ax = plt.gca()
        
    if yerror is None:
        if center=='mean':
            ycenter = np.nanmean(y,axis=0)
        elif center=='median':
            ycenter = np.nanmedian(y,axis=0)
        else:
            print('Unknown error type')

        if error=='std':
            yerror = np.nanstd(y,axis=0)
        elif error=='sem':
            yerror = np.nanstd(y,axis=0) / np.sqrt(np.shape(y)[0])
        else:
            print('Unknown error type')
    else:
        ycenter = y
        yerror = np.array(yerror)

    h, = ax.plot(x,ycenter,color=color,linestyle=linestyle,label=label,linewidth=linewidth)
    ax.fill_between(x, ycenter-yerror, ycenter+yerror,color=color,alpha=alpha)

    return h

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

# Define the p-value thresholds array
def get_sig_asterisks(pvalue,return_ns=False):
    if return_ns: 
        pvalue_thresholds = np.array([[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [10000, "ns"]])
    else: 
        pvalue_thresholds = np.array([[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [10000, ""]])
    # Iterate through the thresholds and return the appropriate significance string
    for threshold, asterisks in pvalue_thresholds:
        if pvalue <= float(threshold):
            return asterisks
    # Default return if p-value is greater than 1
    return ""

def add_stim_resp_win(ax,colors=['k','b'],linestyles=['--','--'],linewidth=1):
    ax.axvline(x=0, color=colors[0], linestyle=linestyles[0], linewidth=linewidth)
    ax.axvline(x=20, color=colors[0], linestyle=linestyles[0], linewidth=linewidth)
    ax.axvline(x=25, color=colors[1], linestyle=linestyles[1], linewidth=linewidth)
    ax.axvline(x=45, color=colors[1], linestyle=linestyles[1], linewidth=linewidth)


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    if len(c) != len(x) - 1:
        warnings.warn(
            "The c argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping
    lc.set_array(c)

    return ax.add_collection(lc)

def plot_mean_stim_spatial(ses, sbins, labeled= ['unl','lab'], areas= ['V1','PM','AL','RSP']):
    nlabels     = 2
    nareas      = len(areas)
    clrs_areas  = get_clr_areas(areas)

    fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*3,nlabels*2.5),sharex=True,sharey=True)
    S = len(sbins)
    for ilab,label in enumerate(labeled):
        for iarea, area in enumerate(areas):
            ax      = axes[ilab,iarea]
            idx_N     = np.all((ses.celldata['roi_name']==area, ses.celldata['labeled']==label), axis=0)
            
            nbins_noise     = 5
            C               = nbins_noise + 2
            noise_signal    = ses.trialdata['signal'][ses.trialdata['stimcat']=='N'].to_numpy()
            
            plotdata        = np.empty((C,S))
            idx_T           = ses.trialdata['signal']==0
            plotdata[0,:]   = np.nanmean(np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)
            idx_T           = ses.trialdata['signal']==100
            plotdata[-1,:]   = np.nanmean(np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)

            edges = np.linspace(np.min(noise_signal),np.max(noise_signal),nbins_noise+1)
            centers = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)

            for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                
                idx_T           =  (ses.trialdata['signal']>=low) & (ses.trialdata['signal']<=high)
                plotdata[ibin+1,:]   = np.nanmean(np.nanmean(ses.stensor[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))],axis=0),axis=0)

                plotlabels = np.round(np.hstack((0,centers,100)))
                plotcolors = ['black']  # Start with black
                plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
                plotcolors.append('orange')  # Add orange at the end

            for iC in range(C):
                ax.plot(sbins, plotdata[iC,:], color=plotcolors[iC], label=plotlabels[iC],linewidth=2)

            add_stim_resp_win(ax)

            ax.set_ylim([-0.1,0.75])
            if ilab == 0 and iarea == 0:
                ax.legend(frameon=False,fontsize=6)
            ax.set_xlim([-60,60])
            if ilab == 0:
                ax.set_title(area)
            if ilab == 1:
                ax.set_xlabel('Position relative to stim (cm)')
            if iarea==0:
                ax.set_ylabel('Activity (z)')
                ax.set_yticks([0,0.25,0.5])
    plt.tight_layout()

def proj_tensor(X,W,idx_N,idx_T):
    K               = np.sum(idx_T)
    S               = X.shape[2]
    # W_norm = W[idx_N] / np.linalg.norm(W[idx_N])
    # Z               = np.dot(W_norm, X[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))].reshape((np.sum(idx_N),K*S))).reshape(K,S)
    Z               = np.dot(W[idx_N], X[np.ix_(idx_N,idx_T,np.ones(S).astype(bool))].reshape((np.sum(idx_N),K*S))).reshape(K,S)
    return Z

def get_idx_ttype_lick(trialdata,filter_engaged=True):
    idx_T_conds = np.zeros((trialdata.shape[0],3,2),dtype=bool)
    for itt,tt in enumerate(['C','N','M']):
        for ilr,lr in enumerate([0,1]):
            if filter_engaged: 
                idx_T           = np.all((trialdata['stimcat']==tt,
                                      trialdata['lickResponse']==lr,
                                      trialdata['engaged']==filter_engaged), axis=0)
            else:
                idx_T           = np.all((trialdata['stimcat']==tt,
                                      trialdata['lickResponse']==lr), axis=0)
            idx_T_conds[:,itt,ilr] = idx_T
    return idx_T_conds

def get_idx_noisebins_lick(trialdata,nbins_noise,filter_engaged=True):
    noise_signal    = trialdata['signal'][trialdata['stimcat']=='N'].to_numpy()
    edges           = np.linspace(np.min(noise_signal),np.max(noise_signal),nbins_noise+1)
    centers         = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
    
    idx_T_conds = np.zeros((trialdata.shape[0],nbins_noise,2),dtype=bool)
    for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
        for ilr,lr in enumerate([0,1]):
            if filter_engaged: 
                idx_T           = np.all((trialdata['signal']>=low,
                                      trialdata['signal']<=high,
                                      trialdata['lickResponse']==lr,
                                      trialdata['engaged']==filter_engaged), axis=0)
            else:
                idx_T           = np.all((trialdata['signal']>=low,
                                      trialdata['signal']<=high,
                                      trialdata['lickResponse']==lr), axis=0)
            idx_T_conds[:,ibin,ilr] = idx_T
    return idx_T_conds,centers

def plot_stim_dec_spatial_proj(X, celldata, trialdata, W, sbins, labeled= ['unl','lab'], areas= ['V1','PM','AL','RSP'],filter_engaged=True):
    nlabels     = len(labeled)
    nareas      = len(areas)
    clrs_areas  = get_clr_areas(areas)
    assert X.shape == (len(celldata), len(trialdata), len(sbins)), 'X must be (numcells, numtrials, numbins)'
    assert W.shape[0] == X.shape[0], 'weights must be same shape as firrst dim of X'
    
    nbins_noise     = 5
    C               = nbins_noise + 2

    noise_signal    = trialdata['signal'][trialdata['stimcat']=='N'].to_numpy()
    D               = 2
    linestyles      = [':','-']

    fig,axes = plt.subplots(nlabels,nareas,figsize=(nareas*3,nlabels*2.5),sharex=True,sharey=True)
    S = len(sbins)
    for ilab,label in enumerate(labeled):
        for iarea, area in enumerate(areas):
            ax              = axes[ilab,iarea]
            idx_N           = np.all((celldata['roi_name']==area, celldata['labeled']==label), axis=0)
            N               = np.sum(idx_N)
            
            if N>5:
                plotdata        = np.empty((C,S))
                
                # idx_T_conds  = get_idx_noise_lick(trialdata,nbins_noise,filter_engaged=False)
                
                idx_T_noise,centers     = get_idx_noisebins_lick(trialdata,nbins_noise,filter_engaged=filter_engaged)
                idx_T_ttype     = get_idx_ttype_lick(trialdata,filter_engaged=filter_engaged)

                idx_T_all       = np.concatenate((idx_T_ttype[:,0,:][:,None,:],idx_T_noise,idx_T_ttype[:,-1,:][:,None,:]),axis=1)
                assert np.shape(idx_T_all) == (trialdata.shape[0],C,2)

                plotlabels = np.round(np.hstack((0,centers,100)))
                plotcolors = ['black']  # Start with black
                plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
                plotcolors.append('orange')  # Add orange at the end

                for iC in range(C-1):
                    for iD in range(D):
                        plotdata   = np.nanmean(proj_tensor(X,W,idx_N,idx_T_all[:,iC,iD]),axis=0)
                        ax.plot(sbins, plotdata, color=plotcolors[iC], label=plotlabels[iC],linewidth=2,linestyle=linestyles[iD],)

                        # ax.plot(sbins, plotdata[iC,iD,:], color=plotcolors[iC], label=plotlabels[iC],linewidth=2)
                        # ax.plot(sbins, plotdata[iC,:], color=plotcolors[iC], label=plotlabels[iC],linewidth=2)

                add_stim_resp_win(ax)

                if iarea == 0 and ilab == 0: 
                    leg1 = ax.legend([plt.Line2D([0], [0], color=c, lw=1.5) for c in plotcolors], plotlabels, 
                                ncols=2,frameon=False,fontsize=7,loc='upper left',title='Saliency')
                    ax.add_artist(leg1)
                if iarea == 0 and ilab == 1: 
                    leg2 = ax.legend([plt.Line2D([0], [0], color='k', lw=1.5,ls=l) for l in linestyles],
                                        ['Miss','Hit'], frameon=False,fontsize=7,loc='upper left',title='Response')
                # ax.add_artist(leg1)

                # ax.set_ylim([-0.1,0.75])
                # if ilab == 0 and iarea == 0:
                    # ax.legend(frameon=False,fontsize=6)
                ax.set_xlim([-60,60])
                ax.set_title(area + ' ' + label)
                if ilab == 1:
                    ax.set_xlabel('Position relative to stim (cm)')
                if iarea==0:
                    ax.set_ylabel('Projected Activity (a.u.)')
                    # ax.set_yticks()
            else: 
                ax.axis('off')
    plt.tight_layout()
    return fig


