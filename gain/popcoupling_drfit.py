
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import linregress

from loaddata.session_info import filter_sessions,load_sessions
from utils.gain_lib import *
from utils.tuning import *

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################

sessions,nSessions   = filter_sessions(protocols = ['GR'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% 

session_list        = np.array([['LPE11086_2024_01_05']])
session_list        = np.array([['LPE12223_2024_06_10']])

sessions,nSessions   = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:                      
for ises in range(nSessions):
    # sessions[ises].load_data(load_behaviordata=False, load_calciumdata=True,calciumversion='dF')
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)
    # sessions[ises].load_data(load_calciumdata=True,calciumversion='deconv')
    # data                = zscore(sessions[ises].calciumdata.to_numpy(), axis=0)
    # poprate             = np.nanmean(data,axis=1)
    # sessions[ises].celldata['popcoupling'] = [np.corrcoef(data[:,i],poprate)[0,1] for i in range(np.shape(data)[1])]

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)
sessions = compute_tuning_wrapper(sessions)

#%% 

#%% ########################### Compute tuning metrics: ###################################
# sessions = compute_tuning_wrapper(sessions)
ntrials_block   = 400
uoris           = np.unique(sessions[0].trialdata['Orientation'])

for ises in range(nSessions):
    sessions[ises].celldata['PO_start'] = compute_prefori(sessions[ises].respmat[:,:ntrials_block],
                                                        sessions[ises].trialdata['Orientation'][:ntrials_block])

    sessions[ises].celldata['PO_end']   = compute_prefori(sessions[ises].respmat[:,-ntrials_block:],
                                                        sessions[ises].trialdata['Orientation'][-ntrials_block:])
    
    # sessions[ises].delta_pref       = np.abs(np.mod(np.subtract.outer(prefori, prefori),180))


    # sessions[ises].celldata['PO_drift'] = np.abs(sessions[ises].celldata['PO_end'] - sessions[ises].celldata['PO_start'])
    # sessions[ises].celldata['PO_drift'] = np.abs(np.mod(sessions[ises].celldata['PO_end'],180) - np.mod(sessions[ises].celldata['PO_start'],180))

    dpref = sessions[ises].celldata['PO_end'] - sessions[ises].celldata['PO_start']
    sessions[ises].celldata['PO_drift']= np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

#%% ########################### Compute tuning metrics: ###################################
# sessions = compute_tuning_wrapper(sessions)
ntrials_block   = 400
uoris           = np.unique(sessions[0].trialdata['Orientation'])

for ises in range(nSessions):
    N  = len(sessions[ises].celldata)
    #Compute signal correlation on first versus last trials:
    trialfilter                     = np.concatenate((np.ones(ntrials_block,dtype=bool),
                                                      np.zeros(len(sessions[ises].trialdata)-ntrials_block,dtype=bool)),
                                                      axis=0)
    resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
    trialfilter                     = np.concatenate((np.zeros(len(sessions[ises].trialdata)-ntrials_block,dtype=bool),
                                                      np.ones(ntrials_block,dtype=bool)),
                                                      axis=0)
    resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
    
    sessions[ises].celldata['sig_corr_startend'] = [np.corrcoef(resp_meanori1[i], resp_meanori2[i])[0,1] for i in range(N)]


#%% fit von mises
def vonmises(x,amp,loc,scale):
    return amp * np.exp( (np.cos(x-loc) - 1) / (2 * scale**2) )

from scipy.optimize import curve_fit

#%% ########################### Compute tuning metrics: ###################################
# sessions = compute_tuning_wrapper(sessions)
ntrials_block   = 400
uoris           = np.unique(sessions[0].trialdata['Orientation'])

for ises in tqdm(range(nSessions),total=nSessions):
    N  = len(sessions[ises].celldata)
    #Compute signal correlation on first versus last trials:
    trialfilter                     = np.concatenate((np.ones(ntrials_block,dtype=bool),
                                                      np.zeros(len(sessions[ises].trialdata)-ntrials_block,dtype=bool)),
                                                      axis=0)
    resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
    trialfilter                     = np.concatenate((np.zeros(len(sessions[ises].trialdata)-ntrials_block,dtype=bool),
                                                      np.ones(ntrials_block,dtype=bool)),
                                                      axis=0) 
    resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=~trialfilter)
    
    resp_meanori1 = zscore(resp_meanori1,axis=1)
    resp_meanori2 = zscore(resp_meanori2,axis=1)
    xdata = np.radians(uoris)

    # sessions[ises].celldata['PO_start'] = []
    # sessions[ises].celldata['PO_end']   = []
    PO_start = np.full((N),np.nan)
    PO_end   = np.full((N),np.nan)
    r2_start = np.full((N),np.nan)
    r2_end   = np.full((N),np.nan)
    for i in range(N):
        try:
            popt1, pcov = curve_fit(vonmises, xdata, resp_meanori1[i],p0=[1,xdata[np.argmax(resp_meanori1[i])],0.25])
            PO_start[i] = popt1[1]
            r2_start[i] = r2_score(resp_meanori1[i],vonmises(xdata,popt1[0],popt1[1],popt1[2]))
        except:
            continue

        try:
            popt2, pcov = curve_fit(vonmises, xdata, resp_meanori2[i],p0=[1,xdata[np.argmax(resp_meanori2[i])],0.25])
            PO_end[i]   = popt2[1]
            r2_end[i]   = r2_score(resp_meanori2[i],vonmises(xdata,popt2[0],popt2[1],popt2[2]))
        except:
            continue
    
    sessions[ises].celldata['PO_start'] = np.degrees(PO_start)
    sessions[ises].celldata['PO_end']   = np.degrees(PO_end)
    sessions[ises].celldata['R2_fit']   = np.nanmean(np.array([r2_start,r2_end]),axis=0)

    dpref = sessions[ises].celldata['PO_end'] - sessions[ises].celldata['PO_start']
    sessions[ises].celldata['PO_drift']= np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

#%% 
celldata = pd.concat([ses.celldata for ses in sessions],axis=0)

#%% 
percthr = 50
histres = 5
fig,axes = plt.subplots(1,4,figsize=(12,3))

ax = axes[0]
idx_N_low = np.all((celldata['noise_level']<20,
                    # celldata['roi_name']=='V1',
                    celldata['pop_coupling']<np.percentile(celldata['pop_coupling'],100-percthr),
                    celldata['tuning_var']>0.1,
                    # celldata['R2_fit']>0.5,
                    ),axis=0)
im = ax.hist2d(celldata['PO_start'][idx_N_low],celldata['PO_end'][idx_N_low],
           bins=np.arange(0,360,histres),cmap='magma',density=True,vmax=0.0001)
# bar = plt.colorbar(im[3])

ax.set_xlabel('Start')
ax.set_ylabel('End')
ax.set_title('PO drift (low coupling)')

ax = axes[1]
idx_N_high = np.all((celldata['noise_level']<20,
                    # celldata['roi_name']=='V1',
                    celldata['tuning_var']>0.1,
                    # celldata['R2_fit']>0.5,
                    celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],percthr)
                    ),axis=0)
im = ax.hist2d(celldata['PO_start'][idx_N_high],celldata['PO_end'][idx_N_high],
           bins=np.arange(0,360,histres),cmap='magma',density=True,vmax=0.0001)

# ax.set_xlabel('Pref Ori start')
ax.set_ylabel('End')
ax.set_title('PO drift (high coupling)')

ax = axes[2]
sns.histplot(celldata['PO_drift'][idx_N_low],color='green',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax)
sns.histplot(celldata['PO_drift'][idx_N_high],color='purple',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax)
ax.set_xlim([0,25])
ax.legend(['low','high'],frameon=False,fontsize=8,title='Pop. coupling')

ax = axes[3]
xvar = 'pop_coupling'
yvar = 'PO_drift'

clrs = sns.color_palette('dark',nSessions)
corrvals = np.empty((nSessions,))
r_values = np.empty((nSessions,))
p_values = np.empty((nSessions,))
for ises in range(nSessions):

    idx_N = np.all((sessions[ises].celldata['noise_level']<20,
                        sessions[ises].celldata['roi_name']=='V1',
                        # sessions[ises].celldata['R2_fit']>0.5,
                        # sessions[ises].celldata['gOSI']>0.5,
                        sessions[ises].celldata['tuning_var']>0.1,
                        # sessions[ises].celldata['PO_drift']<25,
                        ),axis=0)

    sns.scatterplot(data=sessions[ises].celldata[idx_N],y=yvar,x=xvar,ax=ax,marker='.',
                    color=clrs[ises],alpha=0.2)

    x = sessions[ises].celldata[xvar][idx_N]
    y = sessions[ises].celldata[yvar][idx_N]

    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_values[ises], p_values[ises], std_err = linregress(x[mask], y[mask])

    # Plot regression line
    xs = np.array([x.min(),x.max()])
    ys = slope * xs + intercept
    ax.plot(xs,ys,color=clrs[ises],alpha=0.5)

ax.text(0.6,0.7,'r=%1.2f, std=%1.2f' % (np.nanmean(r_values),np.nanstd(r_values)),transform=plt.gca().transAxes)
print('%d/%d sessions significant' % (np.sum(p_values<0.05),nSessions))

idx_N = np.all((celldata['noise_level']<20,
                        celldata['roi_name']=='V1',
                        celldata['tuning_var']>0.1,
                        # celldata['R2_fit']>0.5,
                            # celldata['PO_drift']<25,
                        ),axis=0)

x = celldata[xvar][idx_N]
y = celldata[yvar][idx_N]

mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
# Plot regression line
xs = np.array([x.min(),x.max()])
ys = slope * xs + intercept
ax.plot(xs,ys,color='black',lw=2,linestyle='--') #,color=clrsises],alpha=0.5)

ax.text(0.6,0.6,'r=%1.2f, p=%1.2e' % (r_value,p_value),transform=plt.gca().transAxes)
ax.set_ylim([0,25])
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset= 5)
my_savefig(fig,savedir,'PO_drift_vs_popcoupling_%dsessions' % (nSessions),formats=['png'])

#%% 
percthr = 50
fig,axes = plt.subplots(1,1,figsize=(4,3))
ax = axes
xvar = 'pop_coupling'
yvar = 'sig_corr_startend'

clrs = sns.color_palette('dark',nSessions)
corrvals = np.empty((nSessions,))
r_values = np.empty((nSessions,))
p_values = np.empty((nSessions,))
for ises in range(nSessions):

    idx_N = np.all((sessions[ises].celldata['noise_level']<20,
                        sessions[ises].celldata['roi_name']=='V1',
                        # sessions[ises].celldata['R2_fit']>0.5,
                        # sessions[ises].celldata['gOSI']>0.5,
                        sessions[ises].celldata['tuning_var']>0.1,
                        ),axis=0)

    sns.scatterplot(data=sessions[ises].celldata[idx_N],y=yvar,x=xvar,ax=ax,marker='.',
                    color=clrs[ises],alpha=0.5)

    x = sessions[ises].celldata[xvar][idx_N]
    y = sessions[ises].celldata[yvar][idx_N]

    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_values[ises], p_values[ises], std_err = linregress(x[mask], y[mask])

    # Plot regression line
    xs = np.array([x.min(),x.max()])
    ys = slope * xs + intercept
    ax.plot(xs,ys,color=clrs[ises],alpha=0.5)

ax.text(0.4,0.1,'Mean: r=%1.2f, std=%1.2f' % (np.nanmean(r_values),np.nanstd(r_values)),transform=plt.gca().transAxes)
print('%d/%d sessions significant' % (np.sum(p_values<0.05),nSessions))

idx_N = np.all((celldata['noise_level']<20,
                        celldata['roi_name']=='V1',
                        celldata['tuning_var']>0.1,
                        # celldata['R2_fit']>0.5,
                        ),axis=0)

x = celldata[xvar][idx_N]
y = celldata[yvar][idx_N]

mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
# Plot regression line
xs = np.array([x.min(),x.max()])
ys = slope * xs + intercept
ax.plot(xs,ys,color='black',lw=2,linestyle='--') #,color=clrsises],alpha=0.5)

ax.text(0.4,0.02,'r=%1.2f, p=%1.2e' % (r_value,p_value),transform=plt.gca().transAxes)
ax.set_ylim([0,1])
ax.set_xlabel('Population Coupling')
ax.set_ylabel('Signal Correlation\n(Start-End)')
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset= 5)
my_savefig(fig,savedir,'SigCorr_vs_popcoupling_%dsessions' % (nSessions),formats=['png'])


