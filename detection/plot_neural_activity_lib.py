

####################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import copy
# from sklearn import preprocessing
# from utils.plotting_style import * #get all the fixed color schemes
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# from scipy.signal import medfilt
# from scipy.stats import zscore
# from rastermap import Rastermap, utils
# from sklearn.decomposition import PCA
# import matplotlib.animation as animation

######################## Function to plot snakestyle heatmaps per stim per area #####################

def plot_snake_area(data,sbins,stimtypes=['C','N','M'],sort='peakloc'):
    if sort=='peakloc': #Sort the neurons based on location of peak response:
        sortidx     = np.argsort(-np.nanargmax(np.nanmean(data,axis=2),axis=1))
    elif sort=='stimresp': #Sort the neurons based on peak response:
        sortidx     = np.argsort(np.nanmean(np.nanmean(data[:,(sbins>=0) & (sbins<=20),:],axis=2),axis=1))
    
    data        = data[sortidx,:,:]
    Narea       = np.shape(data)[0]
    X, Y        = np.meshgrid(sbins, range(Narea)) #Construct X Y positions of the heatmaps:

    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
    for iTT in range(len(stimtypes)):
        plt.subplot(1,3,iTT+1)
        c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'bwr',
                           vmin=-np.percentile(data,99),vmax=np.percentile(data,99))
        # c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.25)
        # c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.5)
        plt.title(stimtypes[iTT],fontsize=11)
        if iTT==0:
            plt.ylabel('nNeurons',fontsize=10)
        else:
            axes[iTT].set_yticks([])
        plt.xlabel('Pos. relative to stim (cm)',fontsize=9)
        plt.xlim([-80,60])
        plt.ylim([0,Narea])
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.04, 0.4])
    fig.colorbar(c, cax=cbar_ax,label='Activity (z)')
    
    return fig


##################### Function to plot activity across trials for individual neurons #####################

def plot_snake_neuron_stimtypes(data,sbins,trialdata,stimtypes=['C','N','M']):
    # sortidx     = np.argsort(-np.nanargmax(np.nanmean(data,axis=2),axis=1))
    # data        = data[sortidx,:,:]
    Ntrials         = np.shape(data)[0]

    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
    for iTT,stimtype in enumerate(stimtypes):
        plt.subplot(1,3,iTT+1)
        idx = trialdata['stimcat']==stimtype
        Ntrials = sum(idx)
        X, Y            = np.meshgrid(sbins, range(Ntrials)) #Construct X Y positions of the heatmaps:

        c = plt.pcolormesh(X,Y,data[idx,:], cmap = 'bwr',
                           vmin=-np.nanpercentile(data,99),vmax=np.nanpercentile(data,99))
        plt.title(stimtypes[iTT],fontsize=11)
        plt.ylabel('Trial number',fontsize=10)
        plt.xlabel('Pos. relative to stim (cm)',fontsize=9)
        plt.xlim([-80,80])
        plt.ylim([0,Ntrials])
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.04, 0.4])
    fig.colorbar(c, cax=cbar_ax,label='Activity (z)')

    return fig

def plot_snake_neuron_sortnoise(data,sbins,ses):
    
    sortvars        = ['signal','runspeed','lickresponse']

    trialidx        = ses.trialdata['stimcat']=='N'
    # trialidx = np.logical_and(trialidx,ses.trialdata['engaged']==1)

    Ntrials         = np.sum(trialidx)

    fig, axes       = plt.subplots(nrows=2,ncols=len(sortvars),figsize=(10,8))
    
    for ivar,sortvar in enumerate(sortvars):
        plt.subplot(2,len(sortvars),ivar+1)
        binidx = np.logical_and(sbins>=0,sbins<=20)
        y = np.nanmean(data[np.ix_(trialidx,binidx)],axis=1).reshape(-1, 1)
        if sortvar=='signal':
            x=ses.trialdata['signal'][trialidx].to_numpy().reshape(-1, 1)
        elif sortvar=='lickresponse':
            x=ses.trialdata['lickResponse'][trialidx].to_numpy().reshape(-1, 1)
        elif sortvar=='runspeed':
            x=ses.respmat_runspeed[:,trialidx].squeeze().reshape(-1, 1)
        # plt.scatter(ses.trialdata['signal'][trialidx],y,s=10,c='k')
        # sns.regplot(x, y, ci=None)

        model2 = LinearRegression()
        model2.fit(x, y)
        r2 = model2.score(x, y)

        plt.scatter(x, y,color='g')
        plt.plot(x, model2.predict(x),color='k')
        plt.title('%s (R2 = %1.2f)' % (sortvar,r2),fontsize=11)

        plt.subplot(2,len(sortvars),ivar+1+len(sortvars))
        if sortvar=='signal':
            sortidx     = np.argsort(ses.trialdata['signal'][trialidx]).to_numpy()
        elif sortvar=='lickresponse':
            sortidx     = np.argsort(ses.trialdata['nLicks'][trialidx]).to_numpy()
        elif sortvar=='runspeed':
            sortidx     = np.argsort(ses.respmat_runspeed[:,trialidx]).squeeze()

        plotdata        = data[trialidx,:]
        plotdata        = data[sortidx,:]

        Ntrials         = sum(trialidx)
        X, Y            = np.meshgrid(sbins, range(Ntrials)) #Construct X Y positions of the heatmaps:

        c = plt.pcolormesh(X,Y,plotdata, cmap = 'bwr',
                           vmin=-np.nanpercentile(data,99),vmax=np.nanpercentile(data,99))

        if ivar==0:
            plt.ylabel('Trial number',fontsize=10)
        else:
            axes[1,ivar].set_yticks([])

        plt.xlabel('Pos. relative to stim (cm)',fontsize=9)
        plt.xlim([-80,80])
        plt.ylim([0,Ntrials])
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.04, 0.4])
    fig.colorbar(c, cax=cbar_ax,label='Activity (z)')

    return fig


def plot_mean_activity_example_neurons(data,sbins,ses,example_cell_ids):
    
    vars        = ['signal','hitmiss','runspeed']
    # sortvars        = ['signal','hit/miss','runspeed','lickresponse']

    T          = len(ses.trialdata)
    N          = len(example_cell_ids)
    S          = len(sbins)
    fig, axes  = plt.subplots(nrows=N,ncols=len(vars),figsize=(3*len(vars),2*N),sharey='row',sharex=True)
    
    for ivar,uvar in enumerate(vars):

        for iN,cell_id in enumerate(example_cell_ids):
            uN = np.where(ses.celldata['cell_id']==cell_id)[0][0]
            if uvar=='signal':
                nbins_noise     = 5
                C               = nbins_noise + 2
                noise_signal    = ses.trialdata['signal'][ses.trialdata['stimcat']=='N'].to_numpy()
                
                plotdata        = np.empty((C,S))
                plotdata[0,:]   = np.nanmean(data[uN,ses.trialdata['signal']==0,:],axis=0)
                plotdata[-1,:]  = np.nanmean(data[uN,ses.trialdata['signal']==100,:],axis=0)

                edges = np.linspace(np.min(noise_signal),np.max(noise_signal),nbins_noise+1)
                centers = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)

                for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                    # print(low,high)
                    idx = (ses.trialdata['signal']>=low) & (ses.trialdata['signal']<=high)
                    plotdata[ibin+1,:] = np.mean(data[uN,idx,:],axis=0)
                    
                plotlabels = np.round(np.hstack((0,centers,100)))
                # plotcolors = np.hstack(('k',np.linspace(0,1,nbins_noise),'r'))
                plotcolors = sns.color_palette("inferno",C)
                
                # plotcolors = [sns. sns.color_palette("inferno",C)
                plotcolors = ['black']  # Start with black
                plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
                plotcolors.append('orange')  # Add orange at the end

                # print(plotcolors)

            elif uvar=='hitmiss':

                C               = 2
                noise_trials        = ses.trialdata['stimcat']=='N'

                # unoise          = np.unique(ses.trialdata['signal'][noise_trials].to_numpy())
                usignals        = np.unique(ses.trialdata['signal'].to_numpy())
                
                plotdata        = np.empty((C,S))
                
                temp            = copy.deepcopy(data[uN,:,:])

                for isig,usig in enumerate(usignals):
                    temp[ses.trialdata['signal']==usig,:] -= np.nanmean(temp[ses.trialdata['signal']==usig,:],axis=0,keepdims=True)

                plotdata[0,:]  = np.nanmean(temp[(ses.trialdata['lickResponse']==0) & (noise_trials),:],axis=0)
                plotdata[1,:]  = np.nanmean(temp[(ses.trialdata['lickResponse']==1) & (noise_trials),:],axis=0)
                
                plotlabels = ['Miss','Hit']
                # plotcolors = np.hstack(('k',np.linspace(0,1,nbins_noise),'r'))
                plotcolors = sns.color_palette("husl",C)

            elif uvar=='runspeed':
                
                C               = 5
                # noise_signal    = ses.trialdata['signal'][ses.trialdata['stimcat']=='N'].to_numpy()
                
                plotdata        = np.empty((C,S))

                edges = np.nanquantile(ses.runPSTH,np.linspace(0,1,C+1),axis=None)
                centers = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)

                for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                    # print(low,high)
                    idx         = np.logical_and(ses.runPSTH>=low,ses.runPSTH<=high)
                    temp        = copy.deepcopy(data[uN,:,:])
                    # Compute the mean along axis=0 for elements where idx is True
                    masked_data = np.where(idx, temp, np.nan)  # Replace False with NaN
                    plotdata[ibin,:] = np.nanmean(masked_data, axis=0)  # Compute the mean ignoring NaN
                    
                plotlabels = np.round(centers)
                # plotcolors = np.hstack(('k',np.linspace(0,1,nbins_noise),'r'))
                plotcolors = sns.color_palette("inferno",C)
        
            ax = axes[iN,ivar]
            
            for iC in range(C):
                ax.plot(sbins, plotdata[iC,:], color=plotcolors[iC], label=plotlabels[iC],linewidth=2)
            if iN==0:
                ax.legend(loc='upper left',fontsize=6)

            if iN==N-1:
                ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
                ax.set_xticks([-75,-50,-25,0,25,50,75])
            else:
                ax.set_xticklabels([])
            ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
            ax.axvline(x=20, color='k', linestyle='--', linewidth=1)
            ax.axvline(x=25, color='b', linestyle='--', linewidth=1)
            ax.axvline(x=45, color='b', linestyle='--', linewidth=1)
            ax.set_xlim([-80,60])

            # plt.ylim([0,Ntrials])
        
    # fig.subplots_adjust(right=0.88)
    # cbar_ax = fig.add_axes([0.91, 0.3, 0.04, 0.4])
    # fig.colorbar(c, cax=cbar_ax,label='Activity (z)')
    plt.tight_layout()
    return fig