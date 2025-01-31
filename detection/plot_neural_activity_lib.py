

####################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import copy

from utils.plot_lib import *
from utils.plotting_style import * #get all the fixed color schemes
# from sklearn import preprocessing
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
    elif sort=='stimwin': #Sort the neurons based on peak response in the stim window:
        sortidx     = np.argsort(np.nanmean(np.nanmean(data[:,(sbins>=0) & (sbins<=20),:],axis=2),axis=1))
    elif sort=='respwin': #Sort the neurons based on peak response
        sortidx     = np.argsort(np.nanmean(np.nanmean(data[:,(sbins>=25) & (sbins<=45),:],axis=2),axis=1))

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

######################## Function to plot snakestyle heatmaps per stim per area #####################

def plot_snake_allareas(data,sbins,arealabels,trialtypes=['C','N','M'],sort='peakloc'):
    uareas      = np.unique(arealabels)
    uareas      = sort_areas(uareas)
    nareas      = len(uareas)
    ntrialtypes = len(trialtypes)
    datalim     = my_ceil(np.percentile(data,99),1)
    fig, axes = plt.subplots(nrows=nareas,ncols=ntrialtypes,figsize=(ntrialtypes*2.5,nareas*2.5))
    for iarea,area in enumerate(uareas):
        idx_N       = arealabels==area
        Ncells      = np.sum(idx_N)
        X, Y        = np.meshgrid(sbins, range(Ncells)) #Construct X Y positions of the heatmaps:

        temp        = data[idx_N,:,:]
        if sort=='peakloc': #Sort the neurons based on location of peak response:
            sortidx     = np.argsort(-np.nanargmax(np.nanmean(temp,axis=2),axis=1))
        elif sort=='stimwin': #Sort the neurons based on peak response in the stim window:
            sortidx     = np.argsort(np.nanmean(np.nanmean(temp[:,(sbins>=0) & (sbins<=20),:],axis=2),axis=1))
        elif sort=='respwin': #Sort the neurons based on peak response
            sortidx     = np.argsort(np.nanmean(np.nanmean(temp[:,(sbins>=25) & (sbins<=45),:],axis=2),axis=1))
            
        temp        = temp[sortidx,:,:]

        for iTT in range(len(trialtypes)):
            ax = axes[iarea,iTT]
           

            c = ax.pcolormesh(X,Y,temp[:,:,iTT], cmap = 'bwr',
                            vmin=-datalim,vmax=datalim)
            # c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.25)
            if iarea==0:
                ax.set_title(trialtypes[iTT],fontsize=10)
            if iTT==0:
                # ax.set_ylabel('%s \n nNeurons' % area,fontsize=10)
                ax.set_ylabel('%s' % area,fontsize=10)
                ax.set_yticks([0,Ncells])
            else:
                ax.set_yticks([])
            if iarea==nareas-1:
                ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
                ax.set_xticks([-50,-25,0,25,50])
            else:
                ax.set_xticks([])
            ax.set_xlim([-80,60])
            add_stim_resp_win(ax)
            ax.set_ylim([0,Ncells])
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.94, 0.73, 0.03, 0.1])
    cbar = fig.colorbar(c, cax=cbar_ax,ticks=[-datalim,0,datalim])
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Activity (z)',labelpad=-60)
    
   # plt.tight_layout()
    
    return fig

######################## Function to plot snakestyle heatmaps per stim per animal #####################

def plot_snake_allanimals(data,sbins,animallabels,trialtypes=['C','N','M'],sort='peakloc'):
    uanimals        = np.unique(animallabels)
    nanimals        = len(uanimals)
    ntrialtypes     = len(trialtypes)
    datalim         = my_ceil(np.percentile(data,99),1)

    fig, axes   = plt.subplots(nrows=nanimals,ncols=ntrialtypes,figsize=(ntrialtypes*2.5,nanimals*2.5))
    for ianimal,uanimal in enumerate(uanimals):
        idx_N       = animallabels==uanimal
        Ncells      = np.sum(idx_N)
        X, Y        = np.meshgrid(sbins, range(Ncells)) #Construct X Y positions of the heatmaps:

        temp        = data[idx_N,:,:]
        if sort=='peakloc': #Sort the neurons based on location of peak response:
            sortidx     = np.argsort(-np.nanargmax(np.nanmean(temp,axis=2),axis=1))
        elif sort=='stimwin': #Sort the neurons based on peak response in the stim window:
            sortidx     = np.argsort(np.nanmean(np.nanmean(temp[:,(sbins>=0) & (sbins<=20),:],axis=2),axis=1))
        elif sort=='respwin': #Sort the neurons based on peak response
            sortidx     = np.argsort(np.nanmean(np.nanmean(temp[:,(sbins>=25) & (sbins<=45),:],axis=2),axis=1))
            
        temp        = temp[sortidx,:,:]

        for iTT in range(len(trialtypes)):
            ax = axes[ianimal,iTT]
            c = ax.pcolormesh(X,Y,temp[:,:,iTT], cmap = 'bwr',
                            vmin=-datalim,vmax=datalim)
            # c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.25)
            # c = plt.pcolormesh(X,Y,data[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.5)
            if ianimal==0:
                ax.set_title(trialtypes[iTT],fontsize=10)
            if iTT==0:
                # ax.set_ylabel('%s \n nNeurons' % area,fontsize=10)
                ax.set_ylabel('%s' % uanimal,fontsize=10)
                ax.set_yticks([0,Ncells])
            else:
                ax.set_yticks([])
            if ianimal==nanimals-1:
                ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
                ax.set_xticks([-50,-25,0,25,50])
            else:
                ax.set_xticks([])
            ax.set_xlim([-80,60])
            add_stim_resp_win(ax)
            ax.set_ylim([0,Ncells])
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.94, 0.73, 0.03, 0.1])
    cbar = fig.colorbar(c, cax=cbar_ax,ticks=[-datalim,0,datalim])
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Activity (z)',labelpad=-50)
    
   # plt.tight_layout()
    
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
                noise_trials    = ses.trialdata['stimcat']=='N'

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

                edges           = np.nanquantile(ses.runPSTH,np.linspace(0,1,C+1),axis=None)
                centers         = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
                temp            = copy.deepcopy(data[uN,:,:])

                usignals        = np.unique(ses.trialdata['signal'].to_numpy())
                for isig,usig in enumerate(usignals):
                    temp[ses.trialdata['signal']==usig,:] -= np.nanmean(temp[ses.trialdata['signal']==usig,:],axis=0,keepdims=True)

                for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                    # print(low,high)
                    idx         = np.logical_and(ses.runPSTH>=low,ses.runPSTH<=high)
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
                ax.legend(loc='upper left',fontsize=6,frameon=False,title=uvar)

            if iN==N-1:
                ax.set_xlabel('Pos. relative to stim (cm)',fontsize=9)
                ax.set_xticks([-75,-50,-25,0,25,50,75])
                ax.set_xticklabels([-75,-50,-25,0,25,50,75])
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


def plot_noise_activity_example_neurons(ses,example_cell_ids):
    
    data = copy.deepcopy(ses.respmat)

    T          = len(ses.trialdata)
    N          = len(example_cell_ids)

    # fig, axes  = plt.subplots(nrows=N,ncols=len(vars),figsize=(3*len(vars),2*N),sharey='row',sharex=True)
    fig, axes  = plt.subplots(nrows=3,ncols=3,figsize=(9,9),sharey='row',sharex=True)
    
    for iN,cell_id in enumerate(example_cell_ids[:10]):
        ax      = axes[iN//3,iN%3]
        uN      = np.where(ses.celldata['cell_id']==cell_id)[0][0]

        lickresp    = [0,1]
        D           = len(lickresp)

        sigtype     = 'signal'

        nbins_noise     = 5
        C               = nbins_noise + 2
        noise_signal    = ses.trialdata['signal'][ses.trialdata['stimcat']=='N'].to_numpy()
        
        # plotdata        = np.empty((C,S))
        # plotdata[0,:]   = np.nanmean(data[uN,ses.trialdata['signal']==0],axis=0)
        # plotdata[-1,:]  = np.nanmean(data[uN,ses.trialdata['signal']==100],axis=0)

        min_ntrials = 5

        edges = np.linspace(np.min(noise_signal),np.max(noise_signal),nbins_noise+1)
        centers = np.stack((edges[:-1],edges[1:]),axis=1).mean(axis=1)
        centers = np.r_[0,centers,100]
        plotlabels = np.round(np.hstack((0,centers,100)))
        # plotcolors = np.hstack(('k',np.linspace(0,1,nbins_noise),'r'))
        plotcolors = sns.color_palette("inferno",C)
        
        # plotcolors = [sns. sns.color_palette("inferno",C)
        plotcolors = ['blue']  # Start with black
        # plotcolors += sns.color_palette("magma", n_colors=nbins_noise)  # Add 5 colors from the magma palette
        plotcolors.append('red')  # Add orange at the end
        plotlabels = ['miss','hit']
        markerstyles = ['o','o']
        
        data_sig_hit_mean = np.empty((C,D))
        handles = []
        for ilr,lr in enumerate(lickresp):
            #Catch trials
            idx_T           = np.all((ses.trialdata['signal']==0, 
                                        ses.trialdata['lickResponse']==lr,
                                        ses.trialdata['engaged']==1), axis=0)
            data_sig_hit_mean[0,ilr]        = np.nanmean(ses.respmat[uN,idx_T],axis=0)
            #Max trials
            idx_T           = np.all((ses.trialdata['signal']==100,
                                        ses.trialdata['lickResponse']==lr,
                                        ses.trialdata['engaged']==1), axis=0)
            data_sig_hit_mean[-1,ilr]        = np.nanmean(ses.respmat[uN,idx_T],axis=0)

            for ibin,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                idx_T           = np.all((ses.trialdata[sigtype]>=low,
                                        ses.trialdata[sigtype]<=high,
                                        ses.trialdata['lickResponse']==lr,
                                        ses.trialdata['engaged']==1), axis=0)
                if np.sum(idx_T)>=min_ntrials:
                    data_sig_hit_mean[ibin+1,ilr]        = np.nanmean(ses.respmat[uN,idx_T],axis=0)

            # ax.plot(centers,data_sig_hit_mean[:,ilr], color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
            h, = ax.plot(centers[1:-1],data_sig_hit_mean[1:-1,ilr], color=plotcolors[ilr], label=plotlabels[ilr],linewidth=2)
            handles.append(h)
            idx_T           = np.all((ses.trialdata['stimcat']=='N', 
                                        ses.trialdata['lickResponse']==lr,
                                        ses.trialdata['engaged']==1), axis=0)
            ax.scatter(ses.trialdata['signal'][idx_T],ses.respmat[uN,idx_T],marker=markerstyles[ilr],color=plotcolors[ilr],s=5,alpha=0.5)
            ax.set_title('%s' % cell_id,fontsize=11)
        if iN==0:
            ax.legend(handles,plotlabels,loc='upper left',fontsize=11,frameon=False)
    
        # fig.subplots_adjust(right=0.88)
        # cbar_ax = fig.add_axes([0.91, 0.3, 0.04, 0.4])
        # fig.colorbar(c, cax=cbar_ax,label='Activity (z)')
    plt.tight_layout()
    return fig



