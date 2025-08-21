
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from scipy.stats import zscore
from scipy.stats import linregress

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
session_list        = np.array([['LPE12223','2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = 'SP')

# Load proper data and compute average trial responses:                      
# sessions[0].load_data(load_calciumdata=True,calciumversion='deconv')

# nSessions = 5

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    # sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='deconv',keepraw=True)
    sessions[ises].load_data(load_calciumdata=True,calciumversion='deconv')
    data                = zscore(sessions[ises].calciumdata.to_numpy(), axis=0)
    poprate             = np.nanmean(data,axis=1)
    sessions[ises].celldata['popcoupling'] = [np.corrcoef(data[:,i],poprate)[0,1] for i in range(np.shape(data)[1])]

#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


#%% 
celldata['rf_sz_F'] = celldata['rf_sx_F'] * celldata['rf_sy_F']
celldata['rf_sz_Fneu'] = celldata['rf_sx_Fneu'] * celldata['rf_sy_Fneu']
xvar = 'rf_sx_F'
# xvar = 'rf_sy_F'
xvar = 'rf_sz_F'
# xvar = 'rf_sz_Fneu'
yvar = 'popcoupling'

rf_lim = 500
# rf_lim = 20
# idx_N_filter = (celldata['rf_r2_F']>0.2) & (celldata['rf_sz_F']<rf_lim)
idx_N = np.all((celldata['rf_r2_F']>0.2,
                       celldata[xvar]<rf_lim,
                    #    celldata['roi_name']=='V1'
                       ),axis=0)

fig,ax = plt.subplots(1,1,figsize=(4,4))
sns.scatterplot(data=celldata[idx_N],y=yvar,x=xvar,ax=ax,marker='.',
                color='black',alpha=0.5)
                # hue='roi_name',color='black',alpha=0.5)

x = celldata[xvar][idx_N]
y = celldata[yvar][idx_N]

mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])

# Plot regression line
xs = np.array([x.min(),x.max()])
ys = slope * xs + intercept
ax.plot(xs,ys,'r')
# Plot r value and p value
ax.text(0.6,0.7,'r=%1.2f, p=%1.2e' % (r_value,p_value),transform=plt.gca().transAxes)

ax.set_xlim([0,rf_lim])
ax.set_ylim([-0.1,0.5])
sns.despine(fig=fig,trim=True,top=True,right=True,offset= 5)

#%% 

sns.displot(celldata, x=xvar, y=yvar, hue="roi_name")

