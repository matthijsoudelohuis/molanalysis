"""
This script contains some plotting functions used in the analysis of mesoscope data
Matthijs Oude Lohuis, 2023-2026, Champalimaud Center
"""

import numpy as np
from operator import itemgetter
from  matplotlib.colors import LinearSegmentedColormap
import matplotlib
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings
from scipy.stats import pearsonr,ttest_rel,ttest_ind
import copy
from statannotations.Annotator import Annotator

desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 14)

def my_savefig(fig,savedir,filename,formats=['png','pdf']):
    for fmt in formats:
        fig.savefig(os.path.join(savedir,filename +  '.' + fmt),format = fmt,dpi=600,bbox_inches='tight',transparent=True)
    # fig.savefig(os.path.join(savedir,filename +  '.png'),format = 'png',dpi=300,bbox_inches='tight',transparent=True)
    # fig.savefig(os.path.join(savedir,filename +  '.pdf'),format = 'pdf',dpi=300,bbox_inches='tight',transparent=True)


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
            # N = np.shape(y)[0]
            N = np.sum(~np.isnan(y),axis=0)
            yerror = np.nanstd(y,axis=0) / np.sqrt(N)

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

def get_sig_asterisks(pvalue, return_ns=False):
    """
    Return a string of asterisks corresponding to the significance level of the p-value.

    Parameters
    ----------
    pvalue : float
        The p-value of the statistical test.
    return_ns : bool, optional
        If `True`, return "ns" for non-significant results. If `False`, return an empty string.
        Default is `False`.

    Returns
    -------
    asterisks : str
        A string of asterisks corresponding to the significance level of the p-value.
    """
    pvalue_thresholds = np.array([[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [10000, "ns" if return_ns else ""]])
    # Iterate through the thresholds and return the appropriate significance string
    for threshold, asterisks in pvalue_thresholds:
        if pvalue <= float(threshold):
            return asterisks
    # Default return if p-value is greater than 1
    return ""

def round_pval(pvalue, return_ns=False):
    """
    """
    pvalue_thresholds = np.array([[1e-4, "0.0001"], [1e-3, "0.001"], [1e-2, "0.01"], [0.05, "0.05"], [10000, "ns" if return_ns else ""]])
    # Iterate through the thresholds and return the appropriate significance string
    for threshold, asterisks in pvalue_thresholds:
        if pvalue <= float(threshold):
            return asterisks

def ax_nticks(ax, n):
    ax.locator_params(axis='x', nbins=n)
    ax.locator_params(axis='y', nbins=n)

def add_stat_annotation(ax, x1, x2, y, p, h=None,**kwargs):
    """
    Add statistical annotation to plot.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to which the annotation will be added.
    x1 : float
        The x-position of the first group.
    x2 : float
        The x-position of the second group.
    y : float
        The y-position of the bar.
    p : float
        The p-value of the statistical test.
    h : float, optional
        The height of the bar. If None, the height will be set to y/10.

    Notes
    -----
    The function draws a line connecting the two groups and adds a label with
    the p-value of the statistical test. The label is centered above the line.
    """
    if h is None:
        h = y / 10
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
    ax.text((x1 + x2) * .5, y + h, get_sig_asterisks(p, return_ns=True),
            ha='center', va='bottom', **kwargs)


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

def add_corr_results(ax, x,y,pos=[0.2,0.1],fontsize=8):
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    r,p = pearsonr(x[~nas], y[~nas])

    print('Correlation (r=%1.2f): p=%.3f' % (r,p))
    if p<0.05:
        ax.text(pos[0],pos[1],'r=%1.2f\np<0.05' % r,transform=ax.transAxes,ha='center',va='center',fontsize=fontsize,color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')
    else: 
        ax.text(pos[0],pos[1],'p=n.s.',transform=ax.transAxes,ha='center',va='center',fontsize=fontsize,color='k')


def add_ind_ttest_results(ax, x,y,pos=[0.2,0.1],fontsize=8):
    # nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_ind(x, y,nan_policy='omit')

    print('Independent t-test: p=%.3f' % (p))
    ax.text(pos[0],pos[1],'p<%s' % round_pval(p,return_ns=True),transform=ax.transAxes,ha='center',va='center',fontsize=fontsize,color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

def add_paired_ttest_results(ax, x,y,pos=[0.2,0.1],fontsize=8):
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])

    print('Paired t-test: p=%.3f' % (p))
    ax.text(pos[0],pos[1],'p<%s' % round_pval(p,return_ns=True),transform=ax.transAxes,ha='center',va='center',fontsize=fontsize,color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

def my_legend_strip(ax):
    leg = ax.get_legend()

    for i,t in enumerate(leg.texts):
        if isinstance(leg.legendHandles[i],matplotlib.lines.Line2D):
            c = leg.legendHandles[i].get_color()
        elif isinstance(leg.legendHandles[i],matplotlib.collections.PathCollection):
            c = leg.legendHandles[i].get_facecolor()
        # c = leg.legendHandles[i].get_facecolor()
        t.set_color(c)
    for handle in leg.legendHandles:
        handle.set_visible(False)
    leg.get_frame().set_visible(False)

def ax_3d_makeup(ax,Xp,lab='PC'):

    nticks = 5
    ax.grid(True)
    ax.set_facecolor('white')
    minperc = 1
    maxperc = 99
    ax.set_xticks(np.linspace(np.percentile(Xp[:,0],minperc),np.percentile(Xp[:,0],maxperc),nticks))
    ax.set_yticks(np.linspace(np.percentile(Xp[:,1],minperc),np.percentile(Xp[:,1],maxperc),nticks))
    ax.set_zticks(np.linspace(np.percentile(Xp[:,2],minperc),np.percentile(Xp[:,2],maxperc),nticks))
    
    ax.set_xlim(np.percentile(Xp[:,0],[minperc,maxperc]))
    ax.set_ylim(np.percentile(Xp[:,1],[minperc,maxperc]))
    ax.set_zlim(np.percentile(Xp[:,2],[minperc,maxperc]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlabel('%s1' % lab, fontsize=6)  # give labels to axes
    ax.set_ylabel('%s2' % lab, fontsize=6)
    ax.set_zlabel('%s3' % lab, fontsize=6)


#function to add colorbar for imshow data and axis
def add_colorbar_outside(im,ax):
    fig = ax.get_figure()
    bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    width = 0.01
    eps = 0.01 #margin between plot and colorbar
    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(im, cax=cax)
    return cbar

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


# Plot the performance across sessions as a function of rank:
def plot_RRR_R2_arealabels(R2_cv,optim_rank,R2_ranks,arealabelpairs,clrs_arealabelpairs,normalize=False):

    nranks          = R2_ranks.shape[2]

    if normalize:
        R2_cv       = copy.deepcopy(R2_cv) #copy array to avoid modifying the original
        optim_rank  = copy.deepcopy(optim_rank)
        R2_ranks    = copy.deepcopy(R2_ranks)

        R2_cv       = R2_cv/R2_cv[0,:][np.newaxis,:]
        optim_rank  = optim_rank/optim_rank[0,:][np.newaxis,:]
        R2_ranks    = np.diff(R2_ranks,axis=2,prepend=0)
        R2_ranks    = R2_ranks/R2_ranks[0,:,:,:,:][np.newaxis,:,:,:,:]

    fig, axes = plt.subplots(1,3,figsize=(9,2.5))
    nSessions = R2_cv.shape[1]
    arealabelpairs2     = [al.replace('-','-\n') for al in arealabelpairs]
    narealabelpairs     = len(arealabelpairs)
    if narealabelpairs==8: 
        statpairs = [(0,1),(0,2),(0,3),
                (4,5),(4,6),(4,7)]
    elif narealabelpairs==4: 
        # statpairs = [(0,1),(0,2),(0,3),(1,3)]
        statpairs = [(0,1),(2,3)]
    elif narealabelpairs==2: 
        statpairs = [(0,1)]
    elif narealabelpairs==6: 
        statpairs = [(0,1),(2,3),(4,5)]
    else: print('Wrong number of arealabelpairs for statistics')

    datatoplot          = np.nanmean(R2_ranks,axis=(3,4))
    axlim               = my_ceil(np.nanmax(np.nanmean(datatoplot,axis=1))*1.1,2)

    ax = axes[0]
    handles = []
    for iapl, arealabelpair in enumerate(arealabelpairs):
        handles.append(shaded_error(np.arange(nranks),datatoplot[iapl,:,:],color=clrs_arealabelpairs[iapl],
                                    alpha=0.25,error='sem',ax=ax))

    ax.legend(handles,arealabelpairs,frameon=False,fontsize=8,loc='lower right')
    ax.set_xlabel('Rank')
    ax.set_ylabel('R2 (cv)')
    # ax.set_yticks([0,0.05,0.1])
    ax.set_ylim([0,axlim])
    ax.set_xlim([0,nranks])
    ax_nticks(ax,5)

    ax=axes[1]
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3 - 0.25,R2_cv[iapl,:],color='k',marker='o',s=8)
        ax.errorbar(iapl+0.2,np.nanmean(R2_cv[iapl,:]),np.nanstd(R2_cv[iapl,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

    ax.set_ylabel('R2 (cv)')
    ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),2)])
    ax.set_xticks(range(narealabelpairs),labels=[])

    testdata = R2_cv
    testdata = testdata[:,~np.isnan(testdata).any(axis=0)]

    df = pd.DataFrame({'R2':  testdata.flatten(),
                    'arealabelpair':np.repeat(np.arange(narealabelpairs),np.shape(testdata)[1])})
    
    annotator = Annotator(ax, statpairs, data=df, x="arealabelpair", y='R2', order=np.arange(narealabelpairs))
    # annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False)
    annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False,comparisons_correction="holm-bonferroni")
    annotator.apply_and_annotate()

    ax=axes[2]
    for iapl, arealabelpair in enumerate(arealabelpairs):
        ax.scatter(np.ones(nSessions)*iapl + np.random.rand(nSessions)*0.3 - 0.25,optim_rank[iapl,:],color='k',marker='o',s=10)
        ax.errorbar(iapl+0.2,np.nanmean(optim_rank[iapl,:]),np.nanstd(optim_rank[iapl,:])/np.sqrt(nSessions),color=clrs_arealabelpairs[iapl],marker='o',zorder=10)

    ax.set_xticks(range(narealabelpairs),labels=[])
    # ax.set_ylabel('Number of dimensions')
    ax.set_yticks(np.arange(0,14,2))
    ax.set_ylim([0,my_ceil(np.nanmax(optim_rank),0)+1])
    # ax.set_title('Dimensionality')

    testdata = optim_rank
    testdata = testdata[:,~np.isnan(testdata).any(axis=0)]

    df = pd.DataFrame({'R2':  testdata.flatten(),
                    'arealabelpair':np.repeat(np.arange(narealabelpairs),np.shape(testdata)[1])})

    annotator = Annotator(ax, statpairs, data=df, x="arealabelpair", y='R2', order=np.arange(narealabelpairs))
    # annotator.configure(test='Wilcoxon', text_format='star', loc='inside',verbose=False)
    annotator.configure(test='t-test_paired', text_format='star', loc='inside',verbose=False,comparisons_correction="holm-bonferroni")
    annotator.apply_and_annotate()

    ax.set_ylabel('Rank')

    ax.set_xlabel('Population pair')
    ax.set_xticks(range(narealabelpairs))

    sns.despine(top=True,right=True,offset=3)
    axes[1].set_xticklabels(arealabelpairs2,fontsize=7)
    axes[2].set_xticklabels(arealabelpairs2,fontsize=7)
    fig.tight_layout()
    return fig


# Plot the performance across sessions as a function of rank:
def plot_RRR_R2_arealabels_paired(R2_cv,optim_rank,R2_ranks,arealabelpairs,clrs_arealabelpairs,normalize=False):

    nranks              = R2_ranks.shape[2]
    nSessions           = R2_cv.shape[1]
    arealabelpairs2     = [al.replace('-','-\n') for al in arealabelpairs]
    narealabelpairs     = len(arealabelpairs)

    meanrankdata          = np.nanmean(R2_ranks,axis=(3,4))
    if normalize:
        R2_cv       = copy.deepcopy(R2_cv) #copy array to avoid modifying the original
        optim_rank  = copy.deepcopy(optim_rank)
        R2_ranks    = copy.deepcopy(R2_ranks)

        R2_cv       = R2_cv/R2_cv[0,:][np.newaxis,:]
        optim_rank  = optim_rank/optim_rank[0,:][np.newaxis,:]
        
        meanrankdata    = np.diff(meanrankdata,axis=2,prepend=0)
        meanrankdata    = meanrankdata/meanrankdata[0,:,:][np.newaxis,:,:]

    axlim               = my_ceil(np.nanmax(np.nanmean(meanrankdata,axis=1))*1.1,2)

    fig, axes = plt.subplots(1,3,figsize=(7.5,2.5))

    ax = axes[0]
    handles = []
    for iapl, arealabelpair in enumerate(arealabelpairs):
        handles.append(shaded_error(np.arange(nranks),meanrankdata[iapl,:,:],color=clrs_arealabelpairs[iapl],
                                    alpha=0.25,error='sem',ax=ax))

    ax.legend(handles,arealabelpairs,frameon=False,fontsize=8,loc='lower right')
    ax.set_xlabel('Rank')
    ax.set_ylabel('R2 (cv)')
    # ax.set_yticks(np.arange(0,0.3,0.05))
    ax.set_yticks(np.arange(0,0.3,0.025))
    ax.set_xticks(np.arange(0,20,5))
    ax.set_ylim([0,axlim])
    ax.set_xlim([0,nranks])
    # ax.set_ylim([0,0.15])

    ax=axes[1]
    ax.scatter(R2_cv[0,:],R2_cv[1,:],color=clrs_arealabelpairs[0],marker='o',s=10)
    
    temp = np.reshape(R2_ranks,np.shape(R2_ranks)[:3] + (np.prod(np.shape(R2_ranks)[3:]),))
    temp = np.nanstd(temp,axis=3)
    R2_cv_error = np.full(np.shape(R2_cv),np.nan)
    for i,j in np.ndindex(np.shape(R2_cv)):
        # print(i,j)
        if np.isnan(optim_rank[i,j]):
            continue
        R2_cv_error[i,j] = temp[i,j,int(optim_rank[i,j])]
    ax.errorbar(R2_cv[0,:],R2_cv[1,:], xerr=R2_cv_error[0,:],yerr=R2_cv_error[1,:],color=clrs_arealabelpairs[0],
                marker='o',elinewidth=0.5,capsize=0,capthick=0,
                fmt='none',markersize=5)

    ax.plot([0,1],[0,1],color='k',linestyle='--',linewidth=0.5)
    ax.set_xlabel(arealabelpairs[0])
    ax.set_ylabel(arealabelpairs[1])
    add_paired_ttest_results(ax,R2_cv[0,:],R2_cv[1,:],pos=[0.7,0.1],fontsize=10)
    ax.set_title('R2 (cv)',fontsize=10)
    ax.set_xticks(np.arange(0,0.3,0.1))
    ax.set_yticks(np.arange(0,0.3,0.1))
    ax.set_xlim([0,my_ceil(np.nanmax(R2_cv),2)])
    ax.set_ylim([0,my_ceil(np.nanmax(R2_cv),2)])

    ax=axes[2]
    ax.scatter(optim_rank[0,:],optim_rank[1,:],color=clrs_arealabelpairs[0],marker='o',s=10)
    ax.plot([0,20],[0,20],color='k',linestyle='--',linewidth=0.5)
    ax.set_xlabel(arealabelpairs[0])
    ax.set_ylabel(arealabelpairs[1])
    add_paired_ttest_results(ax,optim_rank[0,:],optim_rank[1,:],pos=[0.7,0.1],fontsize=10)
    ax.set_xticks(np.arange(0,20,5))
    ax.set_yticks(np.arange(0,20,5))
    ax.set_xlim([0,my_ceil(np.nanmax(optim_rank),0)+1])
    ax.set_ylim([0,my_ceil(np.nanmax(optim_rank),0)+1])
    ax.set_title('Rank',fontsize=10)

    sns.despine(top=True,right=True,offset=3)
    fig.tight_layout()
    return fig

################################################################
## Series of function that spit out lists of colors for different combinations of 
## areas, protocols, mice, stimuli, etc. 

def get_clr_areas(areas):
    palette       = {'V1'  : sns.xkcd_rgb['seaweed'],
                    'PM' : sns.xkcd_rgb['barney'],
                    'AL' : sns.xkcd_rgb['clear blue'],
                    'RSP' : sns.xkcd_rgb['orangered']}
    return itemgetter(*areas)(palette)

def sort_areas(areas):
    areas_order = ['V1', 'PM', 'AL', 'RSP']
    areas = [x for x in areas_order if x in areas]
    return areas

def get_clr_area_pairs(areapairs):
    palette       = {'V1-V1'  : sns.xkcd_rgb['seaweed'],
                    'PM-V1' : sns.xkcd_rgb['peacock blue'],
                    'V1-PM' : sns.xkcd_rgb['orangered'],
                    'PM-PM' : sns.xkcd_rgb['barney'],
                    'V1-AL' : sns.xkcd_rgb['scarlet'],
                    'V1-RSP' : sns.xkcd_rgb['grape'],
                    'PM-AL' : sns.xkcd_rgb['cerulean'],
                    'PM-RSP' : sns.xkcd_rgb['violet'],
                    'RSP-AL' : sns.xkcd_rgb['khaki'],
                    ' ' : sns.xkcd_rgb['black']}
    return itemgetter(*areapairs)(palette)

def get_clr_labeled():
    # clrs            = ['black','red']
    return ['gray','indianred']

def arealabeled_to_figlabels(arealabeled):
    # arealabeled_fig = np.array(np.shape(arealabeled),dtype=object)
    table       = {'V1unl': "$V1_{ND}$",
        'V1lab' : "$V1_{PM}$",
        'PMunl' : "$PM_{ND}$",
        'PMlab' : "$PM_{V1}$",
        'V1_UNL': "$V1_{ND}$",
        'V1_LAB' : "$V1_{PM}$",
        'PM_UNL' : "$PM_{ND}$",
        'PM_LAB' : "$PM_{V1}$",
        'ALunl':    "$AL_{ND}$",
        'ALlab' :   "$AL_{PM}$",
        'RSPunl' :  "$RSP_{ND}$",
        'RSPlab' :  "$RSP_{PM}$"}
    return itemgetter(*arealabeled)(table)

def get_clr_deltaoris(deltaoris,version=90):
    # c = ["darkred","darkgreen"]
    # v = [0,1.]
    # l = list(zip(v,c))
    # cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
    # colors = cmap((45-np.mod(deltaoris,90))/45)
    # c = ["darkgreen","darkblue","darkred"]
    c = ["darkred","darkblue","darkgreen"]
    v = [0,.5,1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('gbr',l, N=256)

    # cmap = sns.color_palette('viridis', as_cmap=True)
    # colors = cmap((90-np.mod(deltaoris,180))/90)
    if version == 90: 
        colors = cmap(np.abs(90-deltaoris)/90)
    elif version == 180:
        colors = cmap(np.abs(180-deltaoris)/180)
        
    return colors

def get_clr_labelpairs(pairs):
    palette       = {'unl-unl': sns.xkcd_rgb['grey'],
        'unl-lab' : sns.xkcd_rgb['rose'],
        'lab-unl' : sns.xkcd_rgb['orange'],
        'lab-lab' : sns.xkcd_rgb['red'],
        ' ' : sns.xkcd_rgb['black']}
    # palette       = {'0-0': sns.xkcd_rgb['grey'],
    #     '0-1' : sns.xkcd_rgb['rose'],
    #     '1-0' : sns.xkcd_rgb['rose'],
    #     '1-1' : sns.xkcd_rgb['red']}
    return itemgetter(*pairs)(palette)

def get_clr_area_labeled(area_labeled):
    palette       = {'V1unl': sns.xkcd_rgb['seaweed'],
        'V1lab' : sns.xkcd_rgb['rose'],
        'PMunl' : sns.xkcd_rgb['barney'],
        'PMlab' : sns.xkcd_rgb['red'],
        'V1_UNL': sns.xkcd_rgb['seaweed'],
        'V1_LAB' : sns.xkcd_rgb['rose'],
        'PM_UNL' : sns.xkcd_rgb['barney'],
        'PM_LAB' : sns.xkcd_rgb['red'],
        'ALunl': sns.xkcd_rgb['clear blue'],
        'ALlab' : sns.xkcd_rgb['burnt orange'],
        'RSPunl' : sns.xkcd_rgb['sienna'],
        'RSPlab' : sns.xkcd_rgb['crimson']}
    return itemgetter(*area_labeled)(palette)

def get_clr_arealabelpairs(arealabelpairs):
    palette       = {
        'V1lab-V1unl-PMunlL2/3': '#9933FF',
        'V1lab-V1unl-PMunlL5':  '#9933FF',
        'PMlab-PMunl-V1unlL2/3': '#00CC99',
        'PMlab-PMunl-V1unlL5':  '#00CC99',
                    }

    return itemgetter(*arealabelpairs)(palette)

def get_clr_area_low_high():
    c = np.array([["#C489FF", "#7C00F9"],  # V1lab-V1unl-PMunl
                   ["#7BFFDE", "#009A74"]])  # PMlab-PMunl-V1unl
    return c

def get_clr_area_labelpairs(area_labelpairs):
    palette       = {
        # 'V1unl-V1unl': sns.xkcd_rgb['mint'],
        # 'V1unl-V1lab': sns.xkcd_rgb['light green'],
        # 'V1lab-V1lab': sns.xkcd_rgb['chartreuse'],
        'V1unl-V1unl': sns.xkcd_rgb['green'],
        'V1unl-V1lab': sns.xkcd_rgb['dark cyan'],
        'V1lab-V1lab': sns.xkcd_rgb['forest green'],

        'PMunl-PMunl': sns.xkcd_rgb['lilac'],
        'PMunl-PMlab': sns.xkcd_rgb['orchid'],
        'PMlab-PMlab': sns.xkcd_rgb['plum'],
        
        'V1unl-PMunl': sns.xkcd_rgb['tangerine'],
        'V1unl-PMlab': sns.xkcd_rgb['orange brown'],
        'V1lab-PMunl': sns.xkcd_rgb['reddish orange'],
        'V1lab-PMlab': sns.xkcd_rgb['crimson'],
        
        'PMunl-V1unl': sns.xkcd_rgb['dark sea green'],
        'PMunl-V1lab': sns.xkcd_rgb['minty green'],
        'PMlab-V1unl': sns.xkcd_rgb['teal blue'],
        'PMlab-V1lab': sns.xkcd_rgb['true blue'],

        'V1unl-ALunl': sns.xkcd_rgb['teal'],
        'V1lab-ALunl': sns.xkcd_rgb['neon blue'],

        'V1unl-RSPunl': sns.xkcd_rgb['peach'],
        'V1lab-RSPunl': sns.xkcd_rgb['powder blue'],

        'PMunl-ALunl': sns.xkcd_rgb['teal'],
        'PMlab-ALunl': sns.xkcd_rgb['neon blue'],

        'PMunl-RSPunl': sns.xkcd_rgb['peach'],
        'PMlab-RSPunl': sns.xkcd_rgb['powder blue'],
                    }
    
    return itemgetter(*area_labelpairs)(palette)

def get_clr_layerpairs(pairs):
    palette       = {'L2/3-L2/3': sns.xkcd_rgb['teal'],
        'L2/3-L4': sns.xkcd_rgb['neon blue'],
        'L2/3-L5': sns.xkcd_rgb['lilac'],
        'L4-L2/3': sns.xkcd_rgb['peach'],
        'L4-L4': sns.xkcd_rgb['powder blue'],
        'L4-L5': sns.xkcd_rgb['navy'],
        'L5-L2/3': sns.xkcd_rgb['deep purple'],
        'L5-L4': sns.xkcd_rgb['light grey'],
        'L5-L5': sns.xkcd_rgb['royal blue'],
        ' ' : sns.xkcd_rgb['black']}
    return itemgetter(*pairs)(palette)

def get_clr_arealayerpairs(pairs):
    palette       = {
        'V1L2/3-PML2/3': sns.xkcd_rgb['teal'],
        'V1L2/3-PML5': sns.xkcd_rgb['lilac'],
        'V1L5-V1L2/3': sns.xkcd_rgb['forest green'],
        'V1L2/3-V1L5': sns.xkcd_rgb['apple green'],
        'PML5-PML2/3': sns.xkcd_rgb['deep purple'],
        'PML5-V1L2/3': sns.xkcd_rgb['royal blue'],
        'PML2/3-V1L2/3': sns.xkcd_rgb['powder blue'],
        'PML2/3-PML5': sns.xkcd_rgb['navy'],
        ' ' : sns.xkcd_rgb['black']}
    return itemgetter(*pairs)(palette)

def get_clr_recombinase(enzymes):
    palette       = {'non': 'gray',
        'cre' : 'orangered',
        'flp' : 'indianred'}
    return itemgetter(*enzymes)(palette)

def get_clr_protocols(protocols):
    palette       = {'GR': sns.xkcd_rgb['pinky red'],
                    'GN': sns.xkcd_rgb['bright blue'],
                    'SP' : sns.xkcd_rgb['coral'],
                    'RF' : sns.xkcd_rgb['emerald'],
                    'IM' : sns.xkcd_rgb['very dark green'],
                    'DM' : sns.xkcd_rgb['grape'],
                    'DN' : sns.xkcd_rgb['emerald'],
                    'DP' : sns.xkcd_rgb['neon blue']
}
    return itemgetter(*protocols)(palette)

def get_clr_thrbins(Z):
    """
    Returns colors for threshold bins. The colors are chosen such that they are
    sorted by increasing brightness. The first color is black, the last color is
    orange.
    ----------
    Z : number of conditions (catch, max + nbins_noise).
    -------
    clrs_Z :         List of colors for the threshold bins.
    labels_Z :         List of labels for the threshold bins.
    """
    # plotcolors = [sns. sns.color_palette("inferno",C)
    clrs_Z = ['black']  # Start with black
    clrs_Z += sns.color_palette("magma", n_colors=Z-2)  # Add 5 colors from the magma palette
    clrs_Z.append('orange')  # Add orange at the end

    if Z == 5: 
        labels_Z = ['catch','sub','thr','sup','max']
    elif Z == 6:
        labels_Z = ['catch','sub','thr','sup','max']
    elif Z == 7:
        labels_Z = ['catch','imp','sub','thr','sup','sat','max']

    return clrs_Z,labels_Z

def get_clr_stimuli_vr(stimuli):
    stims           = ['A','B','C','D']
    clrs            = sns.color_palette('husl', 4)
    palette         = {stims[i]: clrs[i] for i in range(len(stims))}
    return itemgetter(*stimuli)(palette)

def get_clr_stimuli_vr_norm(stimuli):
    stims           = [0,1,2,3]
    clrs            = sns.color_palette('husl', 4)
    palette         = {stims[i]: clrs[i] for i in range(len(stims))}
    return itemgetter(*stimuli)(palette)

def get_clr_stimuli_vr(stimuli):
    stims           = ['A','B','C','D']
    clrs            = sns.color_palette('husl', 4)
    palette         = {stims[i]: clrs[i] for i in range(len(stims))}
    return itemgetter(*stimuli)(palette)

def get_clr_blocks(blocks):
    # clrs            = ['#ff8274','#74f1ff']
    clrs            = sns.color_palette('Greys', 2)
    return clrs

def get_clr_gratingnoise_stimuli(oris,speeds):
    cmap1        = np.array([[160,0,255], #oris
                            [0,255,115],
                            [255,164,0]])
    cmap1       = cmap1 / 255
    cmap2       = np.array([0.2,0.6,1]) #speeds

    clrs = np.empty((3,3,3))
    labels = np.empty((3,3),dtype=object)
    for iO,ori in enumerate(oris):
        for iS,speed in enumerate(speeds):
            clrs[iO,iS,:] = cmap1[iO,] * cmap2[iS]
            labels[iO,iS] = '%d deg - %d deg/s' % (ori,speed)     

    clrs            = np.reshape(sns.color_palette('dark', 9),(3,3,3))
    # cmap1           = plt.colormaps['tab10']((0,0.5,1))[:,:3]
    # cmap2           = plt.colormaps['Accent']((0,0.5,1))[:,:3]

    # clrs = np.empty((3,3,3))
    # for i in range(3):
    #     for j in range(3):
    #         clrs[i,j,:] = np.mean((cmap1[i,:],cmap2[j,:]),axis=0)

    # clrs = clrs - np.min(clrs)
    # clrs = clrs / np.max(clrs)
    
    return clrs,labels


def get_clr_GN_svars(labels):
    palette       = {'Ori': '#2e17c4',
                     'Grating Ori': '#2e17c4',
                'Grating Speed' : '#17c42e',
                'Speed' : '#17c42e',
                'STF'   : '#17c42e',
                'Running' : '#c417ad',
                'Locomotion' : '#c417ad',
                'RunSpeed' : '#c417ad',
                'Random' : '#021011',
                'videoME' : '#adc417',
                'MotionEnergy' : '#adc417'}
    return itemgetter(*labels)(palette)


def get_clr_outcome(outcomes):
    palette       = {'CR': '#070f7a',
                    'MISS' : '#806900',
                    'HIT' : '#0b7a07',
                    'FA' : '#7a070b'}
    # palette       = {'CR': '#89A6FA',
    #         'MISS' : '#FADB89',
    #         'HIT' : '#89FA95',
    #         'FA' : '#FA89AD'}
    # palette       = {'CR': '#0026C7',
    #             'MISS' : '#C79400',
    #             'HIT' : '#00C722',
    #             'FA' : '#C70028'}
    return itemgetter(*outcomes)(palette)

def get_clr_psy(signals):
    # clrs            = sns.color_palette('Blues', len(signals))
    # clrs            = sns.color_palette('plasma', len(signals))
    clrs            = sns.color_palette('inferno', len(signals))
    # palette         = {stims[i]: clrs[i] for i in range(len(stims))}
    return clrs

def get_clr_animal_id(animal_ids):
    # clrs            = sns.color_palette('inferno', len(signals))

    clrs = sns.color_palette(palette='tab10', n_colors=len(animal_ids))

    # animal_ids  = np.array(['LPE10884', 'LPE11081', 'LPE11086', 'LPE11495', 'LPE11622',
    #    'LPE11623', 'LPE11997', 'LPE11998', 'LPE12013', 'LPE12223',
    #    'LPE12385'], dtype=object)
    
    return clrs
