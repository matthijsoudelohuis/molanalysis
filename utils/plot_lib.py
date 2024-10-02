import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def shaded_error(ax,x,y,yerror=None,center='mean',error='std',color='black',linestyle='-'):
    x = np.array(x)
    y = np.array(y)

    # if np.ndim(y)==1:
    #     y = y[np.newaxis,:]

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
            yerror = np.nanstd(y,axis=0) / np.sqrt(np.shape(y)[1])
        else:
            print('Unknown error type')
    else:
        ycenter = y
        yerror = np.array(yerror)

    h, = ax.plot(x,ycenter,color=color,linestyle=linestyle)
    ax.fill_between(x, ycenter-yerror, ycenter+yerror,color=color,alpha=0.2)

    return h

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

# Define the p-value thresholds array
def get_sig_asterisks(pvalue):
    pvalue_thresholds = np.array([[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, ""]])
    # Iterate through the thresholds and return the appropriate significance string
    for threshold, asterisks in pvalue_thresholds:
        if pvalue <= float(threshold):
            return asterisks
    # Default return if p-value is greater than 1
    return ""