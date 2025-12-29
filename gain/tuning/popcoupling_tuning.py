
#%% Import libs:
import os, math, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir('e:\\Python\\molanalysis')
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import linregress
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions,load_sessions
from utils.gain_lib import *
from utils.tuning import *
from utils.pair_lib import compute_pairwise_anatomical_distance

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\SharedGain'

#%% #############################################################################

sessions,nSessions   = filter_sessions(protocols = ['GR'])

sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Remove sessions with too much drift in them:
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessions_in_list    = np.where(~sessiondata['session_id'].isin(['LPE12013_2024_05_02','LPE10884_2023_10_20','LPE09830_2023_04_12']))[0]
sessions            = [sessions[i] for i in sessions_in_list]
nSessions           = len(sessions)

#%% 

session_list        = np.array([['LPE11086_2024_01_05']])
session_list        = np.array([['LPE12223_2024_06_10']])
session_list        = np.array([['LPE12223_2024_06_10','LPE11086_2024_01_05','LPE10919_2023_11_06']])

sessions,nSessions  = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv',keepraw=False)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pairwise_anatomical_distance(sessions)

#%% 
####### #     # #     # ### #     #  #####     ######  ###  #####  ####### ######  
   #    #     # ##    #  #  ##    # #     #    #     #  #  #     #    #    #     # 
   #    #     # # #   #  #  # #   # #          #     #  #  #          #    #     # 
   #    #     # #  #  #  #  #  #  # #  ####    #     #  #   #####     #    ######  
   #    #     # #   # #  #  #   # # #     #    #     #  #        #    #    #   #   
   #    #     # #    ##  #  #    ## #     #    #     #  #  #     #    #    #    #  
   #     #####  #     # ### #     #  #####     ######  ###  #####     #    #     # 


#%% Compute tuning for non running trials only:
for ises in tqdm(range(len(sessions)),desc= 'Computing tuning metrics: '):
    if sessions[ises].sessiondata['protocol'].isin(['GR'])[0]:
        idx_K = sessions[ises].respmat_runspeed<1
        # idx_K = sessions[ises].respmat_runspeed>2
        sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat[:,idx_K],
                                                        sessions[ises].trialdata['Orientation'][idx_K])

#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Figure
nbins = 5
uoris = np.unique(celldata['pref_ori'])
binedges_popcoupling   = np.percentile(celldata['pop_coupling'],np.linspace(0,100,nbins+1))
clrs_popcoupling = sns.color_palette('magma',nbins)

fig, ax = plt.subplots(1,1,figsize=(4,3.5))
for i in range(len(binedges_popcoupling)-1):
    idx_N = np.all((celldata['pop_coupling']>binedges_popcoupling[i],
                  celldata['pop_coupling']<binedges_popcoupling[i+1]),axis=0)
    
    ax.plot(uoris,np.histogram(celldata['pref_ori'][idx_N],bins=np.arange(0,360+22.5,22.5))[0],
            color=clrs_popcoupling[i])
  
ax.set_xlim([0,337.5])
ax.set_xticks(np.arange(0,360+22.5,45))
ax.set_xlabel('Preferred orientation')
ax.set_ylabel('Count')
ax.legend(['0-20%','20-40%','40-60%','60-80%','80-100%'],
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
                    reverse=True,fontsize=7,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')

sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
# my_savefig(fig,savedir,'Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])

#%% 
fig, ax = plt.subplots(1,1,figsize=(4,4),subplot_kw={'projection': 'polar'})

#repeat 0 to 360
uoris = np.unique(celldata['pref_ori'])
uoris = np.append(uoris,360)

for i in range(len(binedges_popcoupling)-1):
    idx_N = np.all((celldata['pop_coupling']>binedges_popcoupling[i],
                  celldata['pop_coupling']<binedges_popcoupling[i+1]),axis=0)
    histdata = np.histogram(celldata['pref_ori'][idx_N],bins=np.arange(0,360+22.5,22.5))[0]
    histdata = np.append(histdata,histdata[0]) #repeat 0 to 360
    ax.plot(np.deg2rad(uoris),histdata,
            color=clrs_popcoupling[i])
# ax.set_rticks([0,5,10,15,20,25])
ax.set_theta_zero_location("N")

ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.set_yticklabels([])
ax.tick_params(axis='y', which='major', pad=10)
ax.set_title('Count',pad=10)
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
ax.legend(['0-20%','20-40%','40-60%','60-80%','80-100%'],
                    reverse=True,fontsize=7,frameon=False,bbox_to_anchor=(1.25,0.9), 
                    title='pop. coupling',loc='upper center')
my_savefig(fig,savedir,'Polar_Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])

#%% 
#######  #####  ### 
#     # #     #  #  
#     # #        #  
#     #  #####   #  
#     #       #  #  
#     # #     #  #  
#######  #####  ### 


#%% Figure of orientation selectivity index:
nbins = 5
tuning_metric = 'OSI'
# tuning_metric = 'gOSI'
# tuning_metric = 'tuning_var'
maxnoiselevel = 20

data = np.full((nbins,nSessions),np.nan)
for ises,ses in enumerate(sessions):
    binedges_popcoupling   = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,nbins+1))
    for ibin in range(len(binedges_popcoupling)-1):
        idx_N = np.all((ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
                    ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1],
                    # ses.celldata['roi_name']=='V1',
                    ses.celldata['noise_level']<maxnoiselevel),axis=0)
        
        # idx_N = np.all((ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
        #             ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1]),axis=0)
        data[ibin,ises] = np.mean(ses.celldata[tuning_metric][idx_N])

fig, ax = plt.subplots(1,1,figsize=(4,3.5))

meandata = np.mean(data,axis=1)
errordata = np.std(data,axis=1) /  np.sqrt(nSessions)
ax.errorbar(np.arange(nbins),meandata,errordata,color='k',linewidth=2)
ax.set_xticks(np.arange(nbins))
ax.set_ylim([0,0.7])
ax.set_xlabel('Coupling bins')
ax.set_ylabel(tuning_metric)
# ax.legend(['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'],
#                     reverse=True,fontsize=7,frameon=False,title='pop. coupling',bbox_to_anchor=(1.05,1), loc='upper left')
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_%s_tuning_%dGRsessions' % (tuning_metric,nSessions),formats=['png'])


#%% 
####### #     # #     # ### #     #  #####     ######  #     #    ######  ####### ######     ######     #    ####### ####### 
   #    #     # ##    #  #  ##    # #     #    #     #  #   #     #     # #     # #     #    #     #   # #      #    #       
   #    #     # # #   #  #  # #   # #          #     #   # #      #     # #     # #     #    #     #  #   #     #    #       
   #    #     # #  #  #  #  #  #  # #  ####    ######     #       ######  #     # ######     ######  #     #    #    #####   
   #    #     # #   # #  #  #   # # #     #    #     #    #       #       #     # #          #   #   #######    #    #       
   #    #     # #    ##  #  #    ## #     #    #     #    #       #       #     # #          #    #  #     #    #    #       
   #     #####  #     # ### #     #  #####     ######     #       #       ####### #          #     # #     #    #    ####### 


# single von Mises kernel
def vonmises(x, amp, loc, kappa):
    return amp * np.exp(kappa * (np.cos(x - loc) - 1))

# double von Mises: peaks π apart, but with different amplitudes
def vonmises_double(x, amp1, amp2, loc, kappa):
    loc = np.mod(loc, 2*np.pi)  # keep loc in [0, 2π)
    return (amp1 * np.exp(kappa * (np.cos(x - loc) - 1)) +
            amp2 * np.exp(kappa * (np.cos(x - (loc + np.pi)) - 1)))

#%% Generate synthetic test data and show double von mises fit with curve_fit:
x_data = np.linspace(0, 2*np.pi, 200)
y_true = vonmises_double(x_data, amp1=3, amp2=1.5, loc=np.pi/4, kappa=5)
y_data = y_true + 0.2 * np.random.randn(len(x_data))  # add noise

# fit with curve_fit
p0 = [2, 2, np.pi/2, 2]   # initial guess: [amp1, amp2, loc, kappa]
params, cov = curve_fit(vonmises_double, x_data, y_data, p0=p0)

print("Fitted parameters:")
print("amp1=%.3f, amp2=%.3f, loc=%.3f, kappa=%.3f" % tuple(params))

# plot results
plt.figure(figsize=(6,4))
plt.scatter(x_data, y_data, s=10, alpha=0.6, label="data")
plt.plot(x_data, y_true, 'k--', lw=2, label="ground truth")
plt.plot(x_data, vonmises_double(x_data, *params), 'r-', lw=2, label="fit")
plt.xlabel("x (radians)")
plt.ylabel("Response")
plt.legend()
plt.show()

#%% 
ex_sesid            = 'LPE12223_2024_06_10'
example_neuron      = 'LPE12223_2024_06_10_'

ises            = np.where(np.array(sessions.keys())==ex_sesid)[0][0]
ineuron         = np.where(np.array(sessions[ises].celldata['cell_id'])==example_neuron)[0][0]

percentile      = 50
uoris           = np.unique(sessions[0].trialdata['Orientation'])

poprate             = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)

#Compute signal correlation on first versus last trials:
trialfilter                     = poprate > np.percentile(poprate,percentile)
resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
trialfilter                     = poprate < np.percentile(poprate,percentile)
resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)

# resp_meanori1   = zscore(resp_meanori1,axis=1)
# resp_meanori2   = zscore(resp_meanori2,axis=1)

resp_meanori1   -= np.min(resp_meanori1,axis=1,keepdims=True)
resp_meanori2   -= np.min(resp_meanori2,axis=1,keepdims=True)

resp_meanori1   /= np.max(resp_meanori1,axis=1,keepdims=True)
resp_meanori2   /= np.max(resp_meanori2,axis=1,keepdims=True)

xdata           = np.radians(uoris)

# plt.plot(resp_meanori1[7,:])
# plt.plot(resp_meanori2[7,:])

popt1, pcov = curve_fit(vonmises_double, xdata, resp_meanori1[i],
                        p0=[0.5,0.5,xdata[np.argmax(resp_meanori1[i])]+np.random.randn()*0.05,1],
                        bounds=([0,0,0,0], [1,1,2*np.pi,50]))

# # plot results
# plt.figure(figsize=(6,4))
# plt.scatter(xdata, resp_meanori1[i], s=10, alpha=0.6, label="data")
# # plt.plot(xdata, y_true, 'k--', lw=2, label="ground truth")
# plt.plot(x_data, vonmises_double(x_data, *popt1), 'r-', lw=2, label="fit")
# plt.xlabel("x (radians)")
# plt.ylabel("Response")
# plt.legend()
# plt.show()

ydata_low = vonmises_double(xdata,*popt1)

popt1, pcov = curve_fit(vonmises_double, xdata, resp_meanori2[i],
                    p0=[0.5,0.5,xdata[np.argmax(resp_meanori2[i])]+np.random.randn()*0.05,1],
                        bounds=([0,0,0,0], [1,1,2*np.pi,50]))
ydata_high = vonmises_double(xdata,*popt1)


#%% ########################### Compute tuning across population rate: ###################################
# Do von Mises fit for each neuron with low and high population activity
percentile      = 50
uoris           = np.unique(sessions[0].trialdata['Orientation'])

for ises in tqdm(range(nSessions),total=nSessions):
    Nses  = len(sessions[ises].celldata)

    poprate             = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)

    #Compute signal correlation on first versus last trials:
    trialfilter                     = poprate > np.percentile(poprate,percentile)
    resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
    trialfilter                     = poprate < np.percentile(poprate,percentile)
    resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)

    # resp_meanori1   = zscore(resp_meanori1,axis=1)
    # resp_meanori2   = zscore(resp_meanori2,axis=1)

    resp_meanori1   -= np.min(resp_meanori1,axis=1,keepdims=True)
    resp_meanori2   -= np.min(resp_meanori2,axis=1,keepdims=True)

    resp_meanori1   /= np.max(resp_meanori1,axis=1,keepdims=True)
    resp_meanori2   /= np.max(resp_meanori2,axis=1,keepdims=True)

    xdata           = np.radians(uoris)

    # plt.plot(resp_meanori1[7,:])
    # plt.plot(resp_meanori2[7,:])
    
    # fitparams = np.full((Nses,2,3),np.nan)
    # sessions[ises].celldata['PO_low'] = []
    # sessions[ises].celldata['PO_high']   = []
    # PO_low = np.full((N),np.nan)
    # PO_high   = np.full((N),np.nan)
    # r2_start = np.full((N),np.nan)
    # r2_end   = np.full((N),np.nan)
    for col in ['VM_amp_low','VM_loc_low','VM_sca_low','VM_r2_low','VM_amp_high','VM_loc_high','VM_sca_high','VM_r2_high']:
        sessions[ises].celldata[col] = np.nan

    for i in range(Nses):
        try:
            # popt1, pcov = curve_fit(vonmises, xdata, resp_meanori1[i],p0=[1,xdata[np.argmax(resp_meanori1[i])],0.25])
            
            popt1, pcov = curve_fit(vonmises_double, xdata, resp_meanori1[i],
                                    p0=[0.5,0.5,xdata[np.argmax(resp_meanori1[i])]+np.random.randn()*0.05,1],
                                    bounds=([0,0,0,0], [1,1,2*np.pi,50]))

            # # generate synthetic test data
            # x_data = np.linspace(0, 2*np.pi, 200)
            # y_data = vonmises_double(x_data, *popt1)

            # # plot results
            # plt.figure(figsize=(6,4))
            # plt.scatter(xdata, resp_meanori1[i], s=10, alpha=0.6, label="data")
            # # plt.plot(xdata, y_true, 'k--', lw=2, label="ground truth")
            # plt.plot(x_data, vonmises_double(x_data, *popt1), 'r-', lw=2, label="fit")
            # plt.xlabel("x (radians)")
            # plt.ylabel("Response")
            # plt.legend()
            # plt.show()

            # fitparams[i,0,1:3] = popt1
            # fitparams[i,1,3] = r2_score(resp_meanori1[i],vonmises(xdata,popt1[0],popt1[1],popt1[2]))
            sessions[ises].celldata.loc[i,'VM_amp_low']       = popt1[0]
            sessions[ises].celldata.loc[i,'VM_loc_low']       = np.degrees(popt1[2])
            sessions[ises].celldata.loc[i,'VM_kappa_low']     = popt1[3]
            # sessions[ises].celldata.loc[i,'VM_r2_low']        = r2_score(resp_meanori1[i],vonmises(xdata,popt1[0],popt1[1],popt1[2]))
            sessions[ises].celldata.loc[i,'VM_r2_low']        = r2_score(resp_meanori1[i],vonmises_double(xdata,*popt1))
        except:
            continue

        try:
            popt1, pcov = curve_fit(vonmises_double, xdata, resp_meanori2[i],
                                p0=[0.5,0.5,xdata[np.argmax(resp_meanori2[i])]+np.random.randn()*0.05,1],
                                    bounds=([0,0,0,0], [1,1,2*np.pi,50]))
            # popt1, pcov = curve_fit(vonmises, xdata, resp_meanori2[i],p0=[1,xdata[np.argmax(resp_meanori2[i])],0.25])
            sessions[ises].celldata.loc[i,'VM_amp_high']       = popt1[0]
            sessions[ises].celldata.loc[i,'VM_loc_high']       = np.degrees(popt1[2])
            sessions[ises].celldata.loc[i,'VM_kappa_high']     = popt1[3]
            # sessions[ises].celldata.loc[i,'VM_r2_high']        = r2_score(resp_meanori2[i],vonmises(xdata,popt1[0],popt1[1],popt1[2]))
            sessions[ises].celldata.loc[i,'VM_r2_high']        = r2_score(resp_meanori2[i],vonmises_double(xdata,*popt1))

            # fitparams[i,1,:3] = popt1
            # fitparams[i,1,3] = r2_score(resp_meanori1[i],vonmises(xdata,popt1[0],popt1[1],popt1[2]))
            # popt2, pcov = curve_fit(vonmises, xdata, resp_meanori2[i],p0=[1,xdata[np.argmax(resp_meanori2[i])],0.25])
            # PO_high[i]   = popt2[1]
            # r2_end[i]   = r2_score(resp_meanori2[i],vonmises(xdata,popt2[0],popt2[1],popt2[2]))
        except:
            continue
    
    dpref = sessions[ises].celldata['VM_loc_high'] - sessions[ises].celldata['VM_loc_low']
    sessions[ises].celldata['PO_drift'] = np.min(np.array([np.abs(dpref+360),np.abs(dpref+180),np.abs(dpref-0),np.abs(dpref-180),np.abs(dpref-360)]),axis=0)

#%% 
celldata = pd.concat([ses.celldata for ses in sessions],axis=0)

#%% 
percthr     = 50
histres1d   = 1 #binsize for 1d histogrram of PO change
histres2d   = 10 #binsize for 2d plot
r2_thr      = 0.3
fig,axes = plt.subplots(1,4,figsize=(12,3))

ax = axes[0]
idx_N_low = np.all((celldata['noise_level']<20,
                    # celldata['roi_name']=='V1',
                    celldata['pop_coupling']<np.percentile(celldata['pop_coupling'],100-percthr),
                    # celldata['tuning_var']>0.1,
                    celldata['VM_r2_low']>r2_thr,
                    ),axis=0)
im = ax.hist2d(celldata['VM_loc_low'][idx_N_low],celldata['VM_loc_high'][idx_N_low],
           bins=np.arange(0,360,histres2d),cmap='magma',density=True,vmax=0.0001)
# bar = plt.colorbar(im[3])

ax.set_xlabel('Low Population Activity')
ax.set_ylabel('High Population Activity')
ax.set_title('PO change (low coupling)')

ax = axes[1]
idx_N_high = np.all((celldata['noise_level']<20,
                    # celldata['roi_name']=='V1',
                    # celldata['tuning_var']>0.1,
                    celldata['VM_r2_high']>r2_thr,
                    celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],percthr)
                    ),axis=0)
im = ax.hist2d(celldata['VM_loc_low'][idx_N_high],celldata['VM_loc_high'][idx_N_high],
           bins=np.arange(0,360,histres2d),cmap='magma',density=True,vmax=0.0001)

# ax.set_xlabel('Pref Ori start')
ax.set_ylabel('High Population Activity')
ax.set_title('PO change (high coupling)')

ax = axes[2]
sns.histplot(celldata['PO_drift'][idx_N_low],color='green',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=np.arange(0,180,histres1d))
sns.histplot(celldata['PO_drift'][idx_N_high],color='purple',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=np.arange(0,180,histres1d))
ax.set_xlim([0,90])
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
                        # sessions[ises].celldata['tuning_var']>0.1,
                        ((sessions[ises].celldata['VM_r2_low'] + sessions[ises].celldata['VM_r2_high'])/2)>r2_thr,
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
n_neg = np.sum((p_values<0.05) & (r_values<0))
n_pos = np.sum((p_values<0.05) & (r_values>0))
print('%d/%d sessions negative corr, %d/%d sessions positive corr' % (n_neg,nSessions,n_pos,nSessions))

idx_N = np.all((celldata['noise_level']<20,
                        celldata['roi_name']=='V1',
                        # celldata['tuning_var']>0.1,
                        # celldata['R2_fit']>0.5,
                        ((celldata['VM_r2_low'] + celldata['VM_r2_high'])/2)>r2_thr,
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
my_savefig(fig,savedir,'PO_change_poprate_vs_popcoupling_%dGRsessions' % (nSessions),formats=['png'])

#%%
idx_N_low = np.all((celldata['noise_level']<20,
                    celldata['roi_name']=='V1',
                    celldata['pop_coupling']<np.percentile(celldata['pop_coupling'],100-percthr),
                    ((celldata['VM_r2_low'] + celldata['VM_r2_high'])/2)>r2_thr,
                    ),axis=0)
idx_N_high = np.all((celldata['noise_level']<20,
                    celldata['roi_name']=='V1',
                    ((celldata['VM_r2_low'] + celldata['VM_r2_high'])/2)>r2_thr,
                    celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],percthr)
                    ),axis=0)

#%%
histres = 0.5
bins = np.logspace(np.log10(0.1),np.log10(50),50)
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
sns.histplot(celldata['VM_kappa_low'][idx_N_low],color='green',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
sns.histplot(celldata['VM_kappa_high'][idx_N_low],color='purple',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
ax.legend(['low','high'],frameon=False,fontsize=8,title='Pop. coupling')

sns.histplot(celldata['VM_kappa_low'][idx_N_high],color='red',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
sns.histplot(celldata['VM_kappa_high'][idx_N_high],color='blue',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=bins)
# ax.set_xlim([0,90])
ax.legend(['low coupling - low pop. rate','low coupling - high pop. rate',
           'high coupling - low pop. rate','high coupling - high pop. rate'],
           frameon=False,fontsize=8,bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xscale('log')
ax.set_xlabel('Kappa (tuning narrowness)')
sns.despine(fig=fig,trim=True,offset=5,ax=ax)
my_savefig(fig,savedir,'Tuningwidth_poprate_vs_popcoupling_%dGRsessions' % (nSessions),formats=['png'])











#%% 
####### ####### #     # ######  ####### ######     #    #          ######  #######  #####  ######  ####### #     #  #####  ####### 
   #    #       ##   ## #     # #     # #     #   # #   #          #     # #       #     # #     # #     # ##    # #     # #       
   #    #       # # # # #     # #     # #     #  #   #  #          #     # #       #       #     # #     # # #   # #       #       
   #    #####   #  #  # ######  #     # ######  #     # #          ######  #####    #####  ######  #     # #  #  #  #####  #####   
   #    #       #     # #       #     # #   #   ####### #          #   #   #             # #       #     # #   # #       # #       
   #    #       #     # #       #     # #    #  #     # #          #    #  #       #     # #       #     # #    ## #     # #       
   #    ####### #     # #       ####### #     # #     # #######    #     # #######  #####  #       ####### #     #  #####  ####### 


#%%  Load data properly:                      
for ises in range(nSessions):
    sessions[ises].load_tensor(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                # calciumversion='dF',keepraw=False)
                                calciumversion='deconv',keepraw=False)

#%%
t_axis = sessions[ises].t_axis
for ises in range(nSessions):
    sessions[ises].respmat = np.nanmean(sessions[ises].tensor[:,:,(t_axis>0) & (t_axis<1.5)], axis=2)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)
sessions = compute_tuning_wrapper(sessions)

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Figure
nbins                   = 8
binedges_popcoupling    = np.percentile(celldata['pop_coupling'],np.linspace(0,100,nbins+1))
clrs_popcoupling        = sns.color_palette('magma',nbins)

ustim, istimeses, stims  = np.unique(sessions[ises].trialdata['Orientation'], \
        return_index=True, return_inverse=True)


N                       = len(celldata)
Nstimuli                = len(ustim)
Nrepet                  = int(len(sessions[ises].trialdata)/Nstimuli)
ntimebins               = len(t_axis)
tensor_meanori          = np.empty((N,Nstimuli,ntimebins))

prefori     = (sessions[ises].celldata['pref_ori'].values//22.5).astype(int)
data        = sessions[ises].tensor[:,np.argsort(stims),:]
data        -= np.mean(data)
data        = data.reshape((N, Nrepet, Nstimuli, ntimebins), order='F')
data        = np.mean(data,axis=1)

for n in range(N):
    data[n,:,:] = np.roll(data[n,:,:],-prefori[n],axis=0)

#%% 
fig, axes = plt.subplots(nbins,Nstimuli,figsize=(Nstimuli*2,nbins*2),sharex=True,sharey=True)
# ylims = [-0.1,0.6]
for icbin in range(len(binedges_popcoupling)-1):
    idx_N   = np.all((celldata['pop_coupling']>binedges_popcoupling[icbin],
                  celldata['pop_coupling']<binedges_popcoupling[icbin+1]),axis=0)
    
    for istim in range(Nstimuli):
        ax = axes[icbin,istim]
        ax.plot(t_axis,np.mean(data[idx_N,istim,:],axis=0),linewidth=2,
                color=clrs_popcoupling[icbin])

        ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
        ax.axis('off')
    # ax.set_xlim([0,337.5])
    # ax.set_xticks(np.arange(0,360+22.5,45))
    # ax.set_xlabel('Preferred orientation'
# ax.set_ylim(ylims)
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_tensor_resp_tuning_%dGRsessions' % nSessions,formats=['png'])

#%% 
fig, ax = plt.subplots(1,1,figsize=(3,2),sharex=True,sharey=True)

ylims = [-0.1,0.6]
for icbin in range(len(binedges_popcoupling)-1):
    idx_N   = np.all((celldata['pop_coupling']>binedges_popcoupling[icbin],
                  celldata['pop_coupling']<binedges_popcoupling[icbin+1]),axis=0)
    
    tuning_curve = np.nanmean(data[np.ix_(idx_N,range(Nstimuli),(t_axis>0) & (t_axis<1.5))],axis=(0,2)) 
    tuning_curve = np.nanmean(data[np.ix_(idx_N,range(Nstimuli),(t_axis>0) & (t_axis<0.75))],axis=(0,2)) 
    # - np.nanmean(data[np.ix_(idx_N,range(Nstimuli),t_axis<0)],axis=(0,2))
    - np.nanmean(data)

    tuning_curve = np.nanmean(data[:,:,(t_axis>0) & (t_axis<0.75)],axis=(2)) - np.nanmean(data[:,:,(t_axis<0)],axis=(2))
    tuning_curve = np.nanmean(tuning_curve[idx_N,:],axis=0)
    # - np.nanmean(data[np.ix_(idx_N,range(Nstimuli),t_axis<0)],axis=(0,2))

    ax.plot(uoris,tuning_curve,linewidth=2,color=clrs_popcoupling[icbin])

ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
ax.set_xticks(np.arange(0,360+22.5,45))
ax.set_xlabel('Preferred orientation')
plt.tight_layout()
sns.despine(fig=fig,trim=True,top=True,right=True,offset=3)
my_savefig(fig,savedir,'Popcoupling_tuning_%dGRsessions' % nSessions,formats=['png'])






#%%
####### #     #    #    #     # ######  #       #######    #     # ####### #     # ######  ####### #     #  #####  
#        #   #    # #   ##   ## #     # #       #          ##    # #       #     # #     # #     # ##    # #     # 
#         # #    #   #  # # # # #     # #       #          # #   # #       #     # #     # #     # # #   # #       
#####      #    #     # #  #  # ######  #       #####      #  #  # #####   #     # ######  #     # #  #  #  #####  
#         # #   ####### #     # #       #       #          #   # # #       #     # #   #   #     # #   # #       # 
#        #   #  #     # #     # #       #       #          #    ## #       #     # #    #  #     # #    ## #     # 
####### #     # #     # #     # #       ####### #######    #     # #######  #####  #     # ####### #     #  #####  


ises = 0

#%% Multiplicative tuning for different coupled neurons: 
nPopCouplingBins    = 5
nPopRateBins        = 5

binedges_popcoupling   = np.percentile(sessions[ises].celldata['pop_coupling'],np.linspace(0,100,nPopCouplingBins+1))

poprate             = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)
binedges_poprate    = np.percentile(poprate,np.linspace(0,100,nPopRateBins+1))

stims    = sessions[ises].trialdata['Orientation'].to_numpy()
ustim   = np.unique(stims)
nstim   = len(ustim)

# respmat = sessions[idx_GR].respmat
respmat = zscore(sessions[ises].respmat,axis=1)

N = np.shape(sessions[ises].respmat)[0]
resp_meanori    = np.empty([N,16])
for istim,stim in enumerate(ustim):
    resp_meanori[:,istim] = np.nanmean(respmat[:,sessions[ises].trialdata['Orientation']==stim],axis=1)
prefori  = np.argmax(resp_meanori,axis=1)

meandata = np.full((N,nPopRateBins,nstim),np.nan)
stddata  = np.full((N,nPopRateBins,nstim),np.nan)

for iPopRateBin in range(nPopRateBins):
# ax = axes[d]
    data    = respmat
    for istim,stim in enumerate(ustim):
        idx_T = np.all((stims == stim,
                        poprate>binedges_poprate[iPopRateBin],
                        poprate<=binedges_poprate[iPopRateBin+1]),axis=0)
        # idx_T = np.all((stims == stim,
        #                 poprate>=-1000,
        #                 poprate<=1000),axis=0)
        meandata[:,iPopRateBin,istim] = np.mean(respmat[:,idx_T],axis=1)
        stddata[:,iPopRateBin,istim] = np.std(respmat[:,idx_T],axis=1)

    # sm = np.roll(sm,shift=-prefori,axis=1)
    for n in range(N):
        meandata[n,iPopRateBin,:] = np.roll(meandata[n,iPopRateBin,:],-prefori[n])
        stddata[n,iPopRateBin,:] = np.roll(stddata[n,iPopRateBin,:],-prefori[n])

#%% 
clrs_popcoupling    = sns.color_palette('viridis',nPopCouplingBins)

fig,axes = plt.subplots(1,nPopCouplingBins,figsize=(15,2.5),sharey=True,sharex=True)
for iPopCouplingBin in range(nPopCouplingBins):
    ax = axes[iPopCouplingBin]
    idx_popcoupling = np.all((sessions[ises].celldata['OSI']>0,
                            sessions[ises].celldata['pop_coupling']>binedges_popcoupling[iPopCouplingBin],
                            sessions[ises].celldata['pop_coupling']<=binedges_popcoupling[iPopCouplingBin+1]),axis=0)
    for iPopRateBin in range(nPopRateBins):
        ax.plot(np.mean(meandata[idx_popcoupling,iPopRateBin,:],axis=0),color=clrs_popcoupling[iPopRateBin],
                linewidth=2)
    ax.set_xticks(np.arange(0,len(ustim),2),labels=ustim[::2],fontsize=7)
    # ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
    ax.set_xlabel('Orientation',fontsize=9)
    # ax.set_ylabel('Neuron',fontsize=9)
    ax.tick_params(axis='x', labelrotation=45)
sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
# my_savefig(fig,savedir,'SP_coupling_vs_GR_tunedresp_%s' % (sessions[ises].session_id), formats = ['png'])

#%% 
sessions[ises].poprate = np.nanmean(zscore(sessions[ises].respmat.T, axis=0),axis=1)
resp = zscore(sessions[ises].respmat.T,axis=0)
resp = sessions[ises].respmat.T

#%% Fit affine model:
# sessions = fitAffine_GR_singleneuron_full(sessions,radius=500)

sessions = fitAffine_GR_singleneuron_split(sessions,radius=500)

# fitAffine_GR_singleneuron_split

#%% Identify example cell with high population coupling and orientation tuning:
#%% Get good unmodulated cell: 
idx_examples = np.all((sessions[ises].celldata['aff_r2_grfull']>np.percentile(sessions[ises].celldata['aff_r2_grfull'],80),
                       sessions[ises].celldata['aff_alpha_grfull']<np.percentile(sessions[ises].celldata['aff_alpha_grfull'],50),
                       sessions[ises].celldata['aff_beta_grfull']<np.percentile(sessions[ises].celldata['aff_beta_grfull'],50),
                       ),axis=0)

print(ses.celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(ses.celldata['cell_id'][idx_examples])

#%% Get good multiplicatively modulated cells: 
idx_examples = np.all((sessions[ises].celldata['aff_r2_grfull']>np.percentile(sessions[ises].celldata['aff_r2_grfull'],80),
                       sessions[ises].celldata['aff_alpha_grfull']>np.percentile(sessions[ises].celldata['aff_alpha_grfull'],80),
                       sessions[ises].celldata['aff_beta_grfull']<np.percentile(sessions[ises].celldata['aff_beta_grfull'],50),
                       ),axis=0)

print(sessions[ises].celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(sessions[ises].celldata['cell_id'][idx_examples])
# example_neuron      = 'LPE13959_2025_02_24_3_0120'

#%% Get good additively modulated cells: 
idx_examples = np.all((sessions[ises].celldata['aff_r2_grfull']>np.percentile(sessions[ises].celldata['aff_r2_grfull'],80),
                       sessions[ises].celldata['aff_alpha_grfull']<np.percentile(sessions[ises].celldata['aff_alpha_grfull'],20),
                       sessions[ises].celldata['aff_beta_grfull']>np.percentile(sessions[ises].celldata['aff_beta_grfull'],90),
                       ),axis=0)

print(sessions[ises].celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(sessions[ises].celldata['cell_id'][idx_examples])
# example_cell      = 'LPE10919_2023_11_06_0_0014'
# example_cell      = 'LPE10919_2023_11_06_1_0021'
# example_cell      = 'LPE10919_2023_11_06_5_0441'

#%% 
# pal = sns.color_palette('husl', len(oris))
pal = np.tile(sns.color_palette('husl', 8), (2, 1))

# clrs_stimuli    = sns.color_palette('viridis',8)
fig,ax = plt.subplots(1,1,figsize=(3.5,3))
idx_N = np.where(sessions[ises].celldata['cell_id']==example_cell)[0][0]

for istim,stim in enumerate(ustim[:8]):
    idx_T = np.mod(sessions[ises].trialdata['Orientation'],180)==stim
    ax.scatter(sessions[ises].poprate[idx_T],resp[idx_T,idx_N],
               color=pal[istim],s=0.5)
    x = sessions[ises].poprate[idx_T]
    y = resp[idx_T,idx_N]
    b = linregress(x, y)
    
    xp = np.linspace(np.percentile(sessions[ises].poprate,0.5),
                     np.percentile(sessions[ises].poprate,99.5),100)
    ax.plot(xp,b[0]*xp+b[1],color=pal[istim],linestyle='-',linewidth=2)

ax.set_xlim(np.percentile(sessions[ises].poprate,[0.1,99.9]))
ax.set_ylim(np.percentile(resp[:,idx_N],[0.1,99.9]))
# ax.plot(np.mean(meandata[example_cell,:,:],axis=0),color='k',linewidth=2)
ax.set_ylabel('Response',fontsize=10)
# ax.set_xticks(np.arange(0,len(ustim),2),labels=ustim[::2],fontsize=7)
ax.set_xlabel('Population rate',fontsize=10)
# ax.tick_params(axis='x', labelrotation=45)
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
# my_savefig(fig,savedir,'Example_cell_%s' % (sessions[ises].celldata['cell_id'][example_cell]), formats = ['png'])

#%%
# example_cell      = np.random.choice(sessions[ises].celldata['cell_id'][idx_examples])
idx_N = np.where(sessions[ises].celldata['cell_id']==example_cell)[0][0]

nPopRateBins = 5
binedges_poprate = np.percentile(sessions[ises].poprate,np.linspace(0,99,nPopRateBins+1))
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
x = np.mod(sessions[ises].trialdata['Orientation'],180)
y = sessions[ises].poprate
z = resp[:,idx_N]
stimcond = np.array(x//22.5).astype(int)
# c = pal[sessions[ises].trialdata['stimCond'].astype(int)]
c = pal[stimcond]
ax.scatter(x,y,z,c=c,s=0.5)

# xp = np.linspace(np.percentile(sessions[ises].poprate,0.5),
#                      np.percentile(sessions[ises].poprate,99.5),100)
oris            = np.unique(np.mod(sessions[ises].trialdata['Orientation'],180))
for iqpoprate in range(nPopRateBins):
    resp_meanori    = np.empty([len(oris)])

    for i,ori in enumerate(oris):
        idx_T = np.all((
                    np.mod(sessions[ises].trialdata['Orientation'],180)==ori,
                    y>binedges_poprate[iqpoprate],
                    y<=binedges_poprate[iqpoprate+1]),axis=0)
        # resp_meanori[i] = np.nanmean(resp[example_cell,idx_T])
        resp_meanori[i] = np.nanmean(resp[idx_T,idx_N])
    ax.plot(oris,np.repeat(np.mean([binedges_poprate[iqpoprate],binedges_poprate[iqpoprate+1]]),len(oris)),resp_meanori,
                color='k',linestyle='-',linewidth=2)
    for i,ori in enumerate(oris):
        ax.plot(oris[i],np.mean([binedges_poprate[iqpoprate],binedges_poprate[iqpoprate+1]]),resp_meanori[i],
                color=pal[i,:],linewidth=0,marker='o',markersize=5)
    
    # tuning_curve = mean_resp_gr(sessions[ises],trialfilter=idx_T)[0]
    # tuning_curve = np.mean(z[idx_T],axis=0)
    # b = linregress(x[idx_T], z[idx_T])
    # ax.plot(xp,b[0]*xp+b[1],binedges_poprate[iqpoprate],color=pal[istim],linestyle='-',linewidth=2)
ax.set_xticks(np.arange(0,180+22.5,22.5))
ax.set_xlabel('Stimulus orientation (deg)',fontsize=10)
ax.set_ylabel('Pop. rate (z-score)',fontsize=10)
ax.set_zlabel('Response (deconv.)',fontsize=10)
ax.set_xlim([0,160])
ax.set_ylim(np.percentile(sessions[ises].poprate,[0.1,99.5]))
ax.set_zlim(np.percentile(resp[:,idx_N],[0.1,99]))
fig.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'Example_cell_3D_Ori_PopRate_Response_%s' % (example_cell), formats = ['png'])






#%% Get good unmodulated cell: 
idx_examples = np.all((sessions[ises].celldata['aff_r2_grsplit']>np.percentile(sessions[ises].celldata['aff_r2_grsplit'],80),
                       sessions[ises].celldata['aff_alpha_grsplit']<np.percentile(sessions[ises].celldata['aff_alpha_grsplit'],50),
                       sessions[ises].celldata['aff_beta_grsplit']<np.percentile(sessions[ises].celldata['aff_beta_grsplit'],50),
                       ),axis=0)

print(ses.celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(ses.celldata['cell_id'][idx_examples])

#%% Get good multiplicatively modulated cells: 
idx_examples = np.all((sessions[ises].celldata['aff_r2_grsplit']>np.percentile(sessions[ises].celldata['aff_r2_grsplit'],80),
                       sessions[ises].celldata['aff_alpha_grsplit']>np.percentile(sessions[ises].celldata['aff_alpha_grsplit'],80),
                       sessions[ises].celldata['aff_beta_grsplit']<np.percentile(sessions[ises].celldata['aff_beta_grsplit'],50),
                       ),axis=0)

print(sessions[ises].celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(sessions[ises].celldata['cell_id'][idx_examples])
# example_neuron      = 'LPE13959_2025_02_24_3_0120'

#%% Get good additively modulated cells: 
idx_examples = np.all((sessions[ises].celldata['aff_r2_grsplit']>np.percentile(sessions[ises].celldata['aff_r2_grsplit'],80),
                       sessions[ises].celldata['aff_alpha_grsplit']<np.percentile(sessions[ises].celldata['aff_alpha_grsplit'],50),
                       sessions[ises].celldata['aff_beta_grsplit']>np.percentile(sessions[ises].celldata['aff_beta_grsplit'],80),
                       ),axis=0)

print(sessions[ises].celldata['cell_id'][idx_examples])

example_cell      = np.random.choice(sessions[ises].celldata['cell_id'][idx_examples])
# example_cell      = 'LPE10919_2023_11_06_0_0014'
# example_cell      = 'LPE10919_2023_11_06_1_0021'
# example_cell      = 'LPE10919_2023_11_06_5_0441'

#%% Merge celldata from all sessions
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%%
fig,ax = plt.subplots(1,1,figsize=(4.5,4))

idx_N = np.all((celldata['noise_level']<20,
                # celldata['gOSI']>0.3,
                celldata['OSI']>0.5,
                # celldata['tuning_var']>0.05,
                ),axis=0)

# sns.scatterplot(data=celldata[celldata['aff_r2_grsplit']],x='aff_alpha_grsplit',y= 'aff_beta_grsplit',alpha=0.25,s=10,ax=ax)
sns.scatterplot(data=celldata[idx_N],x='aff_alpha_grsplit',y= 'aff_beta_grsplit',alpha=0.5,s=8,ax=ax,color='k')
ax.set_xlim(np.percentile(celldata['aff_alpha_grsplit'],[0.01,99.99]))
ax.set_ylim(np.percentile(celldata['aff_beta_grsplit'],[0.01,99.99]))
ax.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
ax.axvline(x=1,color='k',linestyle='--',linewidth=0.5)

b = linregress(celldata['aff_alpha_grsplit'][idx_N], celldata['aff_beta_grsplit'][idx_N])
ax.text(0.6,0.6,'r=%1.2f, p=%s' % (b[2],get_sig_asterisks(b[3])),transform=plt.gca().transAxes)

sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'Corr_Alpha_Beta_GR_RateSplit_%d' % nSessions, formats = ['png'])

#%%
fig,ax = plt.subplots(1,1,figsize=(4.5,4))
# sns.scatterplot(data=celldata[idx_N],x='tuning_var',y= 'aff_alpha_grsplit',alpha=0.5,s=8,ax=ax,color='k')
# sns.scatterplot(data=celldata[idx_N],x='gOSI',y= 'aff_alpha_grsplit',alpha=0.5,s=8,ax=ax,color='k')
# sns.scatterplot(data=celldata[idx_N],x='OSI',y= 'aff_alpha_grsplit',alpha=0.5,s=8,ax=ax,color='k')

# fig,ax = plt.subplots(1,1,figsize=(4.5,4))
sns.scatterplot(data=celldata[idx_N],x='tuning_var',y= 'aff_beta_grsplit',alpha=0.5,s=8,ax=ax,color='k')
# sns.scatterplot(data=celldata[idx_N],x='gOSI',y= 'aff_beta_grsplit',alpha=0.5,s=8,ax=ax,color='k')
# sns.scatterplot(data=celldata[idx_N],x='OSI',y= 'aff_beta_grsplit',alpha=0.5,s=8,ax=ax,color='k')


#%%





#%% Show split by low and high population rate: 
# clrs_stimuli    = sns.color_palette('viridis',8)
fig,ax = plt.subplots(1,1,figsize=(3.5,3))
idx_N = np.where(sessions[ises].celldata['cell_id']==example_cell)[0][0]

ustim,_,stims  = np.unique(sessions[ises].trialdata['Orientation'],return_index=True,return_inverse=True)

perc        = 50

# mean_resp_speedsplit = np.empty((nCells,nOris,2))
sessions[ises].poprate = np.nanmean(zscore(sessions[ises].respmat, axis=1), axis=0)

idx_low    = sessions[ises].poprate<=np.percentile(sessions[ises].poprate,perc)
idx_high   = sessions[ises].poprate>np.percentile(sessions[ises].poprate,100-perc)

meanresp    = np.empty([len(oris),2])
for i,ori in enumerate(oris):
    meanresp[i,0] = np.nanmean(sessions[ises].respmat[idx_N,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_low)])
    meanresp[i,1] = np.nanmean(sessions[ises].respmat[idx_N,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_high)])
meanresp -= np.min(meanresp)

sns.regplot(x=meanresp[:,0],
            y=meanresp[:,1],
            ax=ax,color='black',scatter_kws={'s':10})
ax.plot([0,1e10],[0,1e10],color='k',linestyle='--',linewidth=1)

ax.set_xlim([0,np.max(meanresp)*1.1])
ax.set_ylim([0,np.max(meanresp)*1.1])
ax_nticks(ax,5)

ax.set_xlabel('Low population rate',fontsize=10)
ax.set_ylabel('High population rate',fontsize=10)
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,savedir,'Example_cell_GR_RateSplit_%s' % (example_cell), formats = ['png'])

