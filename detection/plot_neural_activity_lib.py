

####################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# from sklearn import preprocessing
# from utils.plotting_style import * #get all the fixed color schemes
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# from scipy.signal import medfilt
# from scipy.stats import zscore
# from rastermap import Rastermap, utils
# from sklearn.decomposition import PCA
# import matplotlib.animation as animation

######################## Function to plot snakestyle heatmaps per stim per area #####################

def plot_snake_area(snakeplot,sbins,stimtypes=['C','N','M']):
    #Sort the neurons based on location of peak response:
    sortidx     = np.argsort(-np.nanargmax(np.nanmean(snakeplot,axis=2),axis=1))
    snakeplot   = snakeplot[sortidx,:,:]
    Narea       = np.shape(snakeplot)[0]
    X, Y        = np.meshgrid(sbins, range(Narea)) #Construct X Y positions of the heatmaps:

    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
    for iTT in range(len(stimtypes)):
        plt.subplot(1,3,iTT+1)
        c = plt.pcolormesh(X,Y,snakeplot[:,:,iTT], cmap = 'bwr',
                           vmin=-np.percentile(snakeplot,99),vmax=np.percentile(snakeplot,99))
        # c = plt.pcolormesh(X,Y,snakeplot[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.25)
        # c = plt.pcolormesh(X,Y,snakeplot[:,:,iTT], cmap = 'viridis',vmin=-0.25,vmax=1.5)
        plt.title(stimtypes[iTT],fontsize=11)
        if iTT==0:
            plt.ylabel('nNeurons',fontsize=10)
        else:
            axes[iTT].set_yticks([])
        plt.xlabel('Pos. relative to stim (cm)',fontsize=9)
        plt.xlim([-80,80])
        plt.ylim([0,Narea])
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.04, 0.4])
    fig.colorbar(c, cax=cbar_ax,label='Activity (z)')
    
    return fig


##################### Function to plot activity across trials for individual neurons #####################

def plot_snake_neuron_stimtypes(data,sbins,trialdata,stimtypes=['C','N','M']):
    # sortidx     = np.argsort(-np.nanargmax(np.nanmean(snakeplot,axis=2),axis=1))
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
