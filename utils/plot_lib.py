import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from operator import itemgetter
import matplotlib.pyplot as plt

def shaded_error(ax,x,y,center='mean',error='std',color='black'):
    # shaded_error(ax,x=oris,ydata=mean_resp_speedsplit[idx_neurons,:,0],errortype='std',color='black')
    if center=='mean':
        ycenter = np.nanmean(y,axis=0)
    elif center=='median':
        ycenter = np.nanmedian(y,axis=0)
    else:
        print('Unknown error type')

    if error=='std':
        yerror = np.nanstd(y,axis=0)
    elif error=='sem':
        yerror = np.nanstd(y,axis=0) / np.sqrt(np.shape(y)[1])
    else:
        print('Unknown error type')

    h, = ax.plot(x,ycenter,color=color)
    ax.fill_between(x, ycenter-yerror, ycenter+yerror,color=color,alpha=0.2)

    # ax.errorbar(oris,np.nanmean(mean_resp_speedsplit[idx_neurons,:,0],axis=0),
    #              np.nanstd(mean_resp_speedsplit[idx_neurons,:,0],axis=0),color='black')


    return h