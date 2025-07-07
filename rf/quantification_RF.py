"""
This script analyzes receptive field position across V1 and PM in 2P Mesoscope recordings
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
os.chdir('e:\\Python\\molanalysis')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from loaddata.session_info import filter_sessions,load_sessions
from utils.rf_lib import *
from loaddata.get_data_folder import get_local_drive
from utils.corr_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\RF mapping\\RF_quantification')

#%% ################### Loading the data ##############################

session_list        = np.array([['LPE11086','2024_01_10']])
session_list        = np.array([['LPE12013','2024_05_02']])
session_list        = np.array([['LPE09830','2023_04_12']])

#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                # ['LPE11998','2024_05_02'], #GN
                                # ['LPE12013','2024_05_02'], #GN
                                # ['LPE12013','2024_05_07'], #GN
                                # ['LPE11086','2023_12_15'], #GR
                                ['LPE10919','2023_11_06']]) #GR

session_list        = np.array([['LPE12223','2024_06_08'],
                                ['LPE12223','2024_06_10']])

sessions,nSessions = load_sessions(protocol = 'GR',session_list=session_list)
sessions,nSessions = load_sessions(protocol = 'RF',session_list=session_list)

#%% 
sessions,nSessions = filter_sessions(protocols = ['GR','GN'],session_rf=True,filter_areas=['V1','PM'])
# sessions,nSessions = filter_sessions(protocols = ['SP','IM'])
# sessions,nSessions = filter_sessions(protocols = ['RF'],filter_areas=['V1','PM'])

#%% 
r2_thr = 0.2 #R2 of the 2D gaussian fit
r2_thr = 0.1 #R2 of the 2D gaussian fit

#%% Interpolation of receptive fields:
sessions = compute_pairwise_anatomical_distance(sessions)

sessions = smooth_rf(sessions,r2_thr=r2_thr,radius=50,rf_type='Fneu',mincellsFneu=5)
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smooth_with_Fsig(sessions) 

sessions = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')
# sessions = compute_pairwise_delta_rf(sessions,rf_type='F')


#%% Show fraction of receptive fields per session before any corrections:
for rf_type in ['F','Fneu','Fsmooth']:
    [fig,rf_frac_F] = plot_RF_frac(sessions,rf_type=rf_type,r2_thr=r2_thr)
    fig.savefig(os.path.join(savedir,'RF_fraction_%s' % rf_type  + '.png'), format = 'png')
# fig.savefig(os.path.join(savedir,'RF_quantification','RF_fraction_F_IMincluded' + '.png'), format = 'png')

#%% ##################### Retinotopic mapping within V1 and PM #####################
rf_type = 'Fneu'
rf_type = 'Fsmooth'
for ises in range(nSessions):
# for ises in [1,5,12]:
    fig = plot_rf_plane(sessions[ises].celldata,r2_thr=r2_thr,rf_type=rf_type) 
    fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_plane_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '.png'), format = 'png')
    # fig.savefig(os.path.join(savedir,'RF_planes','V1_PM_plane_allcells_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '.png'), format = 'png')

#%%
# sessions = exclude_outlier_rf(sessions) 
rf_type = 'F'
#Show fraction of receptive fields per session after filtering out scattered neurons: 
[fig,rf_frac_F] = plot_RF_frac(sessions,rf_type=rf_type,r2_thr=r2_thr)
# fig.savefig(os.path.join(savedir,'RF_quantification','RF_fraction_F_filter' + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'RF_quantification','RF_fraction_out%s_%s' % (rf_type,sessions[ises].sessiondata['session_id'][0]) + '.png'), format = 'png')

#%% 
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\RF mapping\\RF_planes')

#%% 
ises = 22
fig = plot_rf_plane(sessions[ises].celldata,r2_thr=r2_thr,rf_type='F') 
fig = plot_rf_plane(sessions[ises].celldata,r2_thr=r2_thr,rf_type='Fsmooth') 

#%% ########### Plot locations of receptive fields as on the screen ##############################
rf_type = 'Fsmooth'
rf_type = 'F'
for ises in range(nSessions):
# for ises in [5]:
    fig = plot_rf_screen(sessions[ises].celldata,r2_thr=r2_thr,rf_type=rf_type) 
    fig.savefig(os.path.join(savedir,'RF_planes','RF_gauss_screen_' + rf_type + '_' + sessions[ises].sessiondata['session_id'][0] +  rf_type + '.png'), format = 'png')

#%% Show distribution of delta receptive fields across areas: 
sessions = compute_pairwise_delta_rf(sessions,rf_type='Fsmooth')
sessions = compute_pairwise_delta_rf(sessions,rf_type='F')

#%% Make a figure with each session is one line for each of the areapairs a histogram of distmat_rf:
areapairs = ['V1-V1','PM-PM','V1-PM']
savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\RF mapping')
# r2_thr = 0.2
fig = plot_delta_rf_across_sessions(sessions,areapairs,r2_thr=r2_thr,rf_type='F')
fig.savefig(os.path.join(savedir,'DeltaRF_Areapairs_%dsessions_' % nSessions + '.png'), format = 'png')

#%% Make a histogram of delta receptive fields across V1 and PM based on labeling
areapairs = ['V1-V1','PM-PM','V1-PM']
projpairs = ['unl-unl','lab-lab']

fig = plot_delta_rf_projections(sessions,areapairs,projpairs,filter_near=False)
fig.savefig(os.path.join(savedir,'DeltaRF_Projpairs_nearfilter_%dsessions_' % nSessions + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'DeltaRF_Projpairs_%dsessions_' % nSessions + '.png'), format = 'png')


#%% 
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

fig,axes   = plt.subplots(1,1,figsize=(5,5))
plt.hist(celldata['rf_r2_Fneu'],color='gray',bins=np.arange(0,1,step=0.025))

#%% ## remove any double cells (for example recorded in both GR and RF)
celldata    = celldata.drop_duplicates(subset='cell_id', keep="first")
celldata    = celldata[~np.isnan(celldata['rf_az_Fneu'])]

r2data      = celldata['rf_r2_F']
dev_az      = np.abs(celldata['rf_az_F'] - celldata['rf_az_Fneu'])
dev_el      = np.abs(celldata['rf_el_F'] - celldata['rf_el_Fneu'])
dev_rf      = np.sqrt(dev_az**2 + dev_el**2)

xticks      = np.arange(0,1,step=0.1)

alphaval    = 0.2
areas   = ['V1','PM']
spat_dims = ['az','el','rf']
clrs_areas = get_clr_areas(areas)
fig,axes   = plt.subplots(2,3,figsize=(10,7))
thrs = [25,50]
for iarea,area in enumerate(areas):
    
    for ispat_dim,spat_dim in enumerate(spat_dims):
        idx = celldata['roi_name'] == area

        if spat_dim == 'az':
            axes[iarea,ispat_dim].scatter(r2data[idx],dev_az[idx],s=3,alpha=alphaval,c=clrs_areas[iarea])
            axes[iarea,ispat_dim].set_ylim([0,100])

        elif spat_dim == 'el':
            axes[iarea,ispat_dim].scatter(r2data[idx],dev_el[idx],s=3,alpha=alphaval,c=clrs_areas[iarea])
            axes[iarea,ispat_dim].set_ylim([0,100])
        
        elif spat_dim == 'rf':
            axes[iarea,ispat_dim].scatter(r2data[idx],dev_rf[idx],s=3,alpha=alphaval,c=clrs_areas[iarea])
            axes[iarea,ispat_dim].set_ylim([0,100])
            axes[iarea,ispat_dim].axhline(y=thrs[iarea],xmin=0,xmax=1,color='black',linewidth=1,linestyle='--')
            axes[iarea,ispat_dim].text(y=thrs[iarea]+5,x=7,s='Scatter thr',fontsize=9)

        axes[iarea,ispat_dim].set_xticks(-np.log10(xticks),labels=xticks,fontsize=6)
        axes[iarea,ispat_dim].set_title(area + ' ' + spat_dim)
        axes[iarea,ispat_dim].set_xlim([2,10])
        axes[iarea,ispat_dim].set_ylabel('RF Scatter')

plt.tight_layout()

fig.savefig(os.path.join(savedir,'RF_quantification','RF_scatter_vs_pval' + '.png'), format = 'png')

#%% ##################### How far are individual cells from neuropil #####################
fig,axes = plt.subplots(1,4,figsize=(8,2))

axes[0].hist(celldata['rf_az_F'][celldata['roi_name']=='V1'] - celldata['rf_az_Fneu'][celldata['roi_name']=='V1'],
             bins=np.arange(-200,200,step=5),density=True,color='green')
axes[0].set_title('V1 - Azimuth')
axes[1].hist(celldata['rf_el_F'][celldata['roi_name']=='V1'] - celldata['rf_el_Fneu'][celldata['roi_name']=='V1'],
             bins=np.arange(-200,200,step=5),density=True,color='green')
axes[1].set_title('V1 - Elevation')

axes[2].hist(celldata['rf_az_F'][celldata['roi_name']=='PM'] - celldata['rf_az_Fneu'][celldata['roi_name']=='PM'],
             bins=np.arange(-200,200,step=5),density=True,color='purple')
axes[2].set_title('PM - Azimuth')
axes[3].hist(celldata['rf_el_F'][celldata['roi_name']=='PM'] - celldata['rf_el_Fneu'][celldata['roi_name']=='PM'],
             bins=np.arange(-200,200,step=5),density=True,color='purple')
axes[3].set_title('PM - Elevation')
plt.suptitle('RF Scatter vs Neuropil')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'RF_jitter_hist' + '.png'), format = 'png')

#%% 
fig,axes = plt.subplots(1,2,figsize=(8,3))
pvals = [0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]
pvals = 0.1**np.arange(10)    
jitterdata = np.empty((2,2,len(pvals)))
for ip,p in enumerate(pvals):
    idx = np.logical_and(celldata['roi_name']=='V1',celldata['rf_p_F']<p)
    # jitterdata[0,0,ip] = np.std(dev_az[idx])
    jitterdata[0,0,ip] = np.nanmedian(dev_az[idx])
    idx = np.logical_and(celldata['roi_name']=='PM',celldata['rf_p_F']<p)
    # jitterdata[0,1,ip] = np.std(dev_az[idx])
    jitterdata[0,1,ip] = np.nanmedian(dev_az[idx])
    idx = np.logical_and(celldata['roi_name']=='V1',celldata['rf_p_F']<p)
    # jitterdata[1,0,ip] = np.std(dev_el[idx])
    jitterdata[1,0,ip] = np.nanmedian(dev_el[idx])
    idx = np.logical_and(celldata['roi_name']=='PM',celldata['rf_p_F']<p)
    # jitterdata[1,1,ip] = np.std(dev_el[idx])
    jitterdata[1,1,ip] = np.nanmedian(dev_el[idx])

axes[0].plot(-np.log10(pvals),jitterdata[0,0,:],linewidth=2,linestyle='-',color='green')
axes[0].plot(-np.log10(pvals),jitterdata[0,1,:],linewidth=2,linestyle='-',color='purple')
axes[1].plot(-np.log10(pvals),jitterdata[1,0,:],linewidth=2,linestyle='-',color='green')
axes[1].plot(-np.log10(pvals),jitterdata[1,1,:],linewidth=2,linestyle='-',color='purple')
for ax in axes: 
    ax.set_xticks(-np.log10(pvals),labels=np.log10(pvals),fontsize=6)
    ax.set_xlabel("P-value threshold")
    ax.set_ylabel('RF jitter')
    ax.set_xlim([0,9])
    ax.set_ylim([0,50])
    ax.axvline(-np.log10(sig_thr),color='k',linestyle='--')
    ax.text(-np.log10(sig_thr)+0.05,40,'p = ' + str(sig_thr),fontsize=10)
axes[0].set_title('Azimuth')
axes[1].set_title('Elevation')

plt.tight_layout()

plt.savefig(os.path.join(savedir,'RF_jitter_vs_pval' + '.png'), format = 'png')

#%% 
sessions = compute_pairwise_metrics(sessions)

# ###### Fit gradient of RF as a function of spatial location of somata:

# # r2 = interp_rf(sessions,sig_thr=0.01,show_fit=True)

# ###### Smooth RF with local good fits (spatial location of somata): ######
# for ises in range(nSessions):
#     fig = plot_rf_plane(sessions[ises].celldata,r2_thr=r2_thr) 
#     fig.savefig(os.path.join(savedir,'V1_PM_azimuth_elevation_inplane_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

# smooth_rf(sessions,sig_thr=0.001,radius=100)

# for ises in range(nSessions):
#     fig = plot_rf_plane(sessions[ises].celldata,sig_thr=1) 
#     fig.savefig(os.path.join(savedir,'V1_PM_azimuth_elevation_inplane_smooth_' + sessions[ises].sessiondata['session_id'][0] + '.png'), format = 'png')

# ###

## Combine cell data from all loaded sessions to one dataframe:
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
celldata    = celldata.drop_duplicates(subset='cell_id', keep="first")

###################### Retinotopic mapping within V1 and PM #####################

###################### RF size difference between V1 and PM #####################
order = [0,1] #for statistical testing purposes
pairs = [(0,1)]

order = ['V1','PM'] #for statistical testing purposes
pairs = [('V1','PM')]
fig,ax   = plt.subplots(1,1,figsize=(3,4))

sns.violinplot(data=celldata,y="rf_sz_F",x="roi_name",palette=['blue','red'],ax=ax)

annotator = Annotator(ax, pairs, data=celldata, x="roi_name", y="rf_sz_F", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

ax.set_xlabel('area')
ax.set_ylabel('RF size\n(squared degrees)')

plt.savefig(os.path.join(savedir,'V1_PM_rf_size_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

###################### Include only neurons nearby labeled cells #####################
rads = np.arange(600)
fracincl = np.empty((nSessions,600))

for ises in range(nSessions):
    for radius in rads:
        idx = filter_nearlabeled(sessions[ises],radius=radius)
        fracincl[ises,radius] = np.sum(idx) / len(sessions[ises].celldata)

fig = plt.figure(figsize=(4,3))
plt.plot(rads,fracincl.T,linewidth=2,alpha=0.5)
plt.plot(rads,np.median(fracincl,axis=0),linewidth=3,color='black')
plt.xlabel(u"Dist. from labeled neuron \u03bcm")
plt.ylabel('Included data')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'Filter_NearLabeled_%d_Sessions' % nSessions + '.png'))

#%% 
for ses in sessions:
    ses.celldata['rf_size'] = np.abs(ses.celldata['rf_sx_Fgaussrf_sx_Fgauss'] * ses.celldata['rf_sy_Fgauss'])
celldata    = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Make a figure with boxplots with the 'rf_sx_Fgauss' and 'rf_sy_Fgauss' in celldata where only neurons with rf_r2_Fgauss is higher than a threshold, set at 0.2 and separate by area:
areas = ['V1','PM']
clrs_areas = get_clr_areas(areas)
fig,ax   = plt.subplots(1,2,figsize=(6,3))
sns.boxplot(data=celldata[celldata['rf_r2_Fgauss'] > 0.2],x="roi_name",y="rf_sx_Fgauss",palette=clrs_areas,
            ax=ax[0],order=areas,showfliers=False)
sns.boxplot(data=celldata[celldata['rf_r2_Fgauss'] > 0.2],x="roi_name",y="rf_sy_Fgauss",palette=clrs_areas,
            ax=ax[1],order=areas,showfliers=False)

ax[0].set_xlabel('area')
ax[0].set_ylabel('RF size\n(azimuth)')
ax[0].set_ylim([0,50])

ax[1].set_xlabel('area')
ax[1].set_ylabel('RF size\n(elevation)')
ax[1].set_ylim([0,50])

plt.tight_layout()
fig.savefig(os.path.join(savedir,'V1_PM_rf_xysize_boxplot' + '.png'), format = 'png')

#%% Print medians:
for area in areas:
    print('Median RF size for %s:' % area)
    print('  Azimuth: %.2f degrees' % np.median(celldata[np.logical_and(celldata['roi_name'] == area,celldata['rf_r2_Fgauss'] > 0.2)]['rf_sx_Fgauss']))
    print('  Elevation: %.2f degrees' % np.median(celldata[np.logical_and(celldata['roi_name'] == area,celldata['rf_r2_Fgauss'] > 0.2)]['rf_sy_Fgauss']))
    print('')

#%% Make a figure with boxplots with the 'rf_size' in celldata where only neurons with rf_r2_Fgauss is higher than a threshold, set at 0.2 and separate by area:
areas = ['V1','PM']
clrs_areas = get_clr_areas(areas)
fig,ax   = plt.subplots(1,1,figsize=(3,3))
sns.boxplot(data=celldata[celldata['rf_r2_Fgauss'] > 0.2],x="roi_name",y="rf_size",palette=clrs_areas,
            ax=ax,order=areas,showfliers=False)
ax.set_xlabel('area')
ax.set_ylabel('RF size\n(squared degrees)')
plt.tight_layout()
ax.set_ylim([0,1000])
fig.savefig(os.path.join(savedir,'V1_PM_rf_size_boxplot' + '.png'), format = 'png')

#%% Print medians:
for area in areas:
    print('Median RF size for %s:' % area)
    print('  %.2f squared degrees' % np.median(celldata[np.logical_and(celldata['roi_name'] == area,celldata['rf_r2_Fgauss'] > 0.2)]['rf_size']))
    print('')
