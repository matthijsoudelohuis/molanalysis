"""
This script analyzes receptive field position across V1 and PM in 2P Mesoscope recordings
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

####################################################
import math,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from loaddata.session_info import filter_sessions,load_sessions

### TODO:
# append for multiple sessions to dataframe with pairwise measurements
# filter and compute only for cells with receptive field? Probably not
# compute distance in x,y,z, but also only x,y for RF distance plot

savedir = 'E:\\OneDrive\\PostDoc\\Figures\\Neural - RF\\RF_quantification\\'

#################### Loading the data ##############################
# 
# sessions            = filter_sessions(protocols = ['SP'])

session_list        = np.array([['LPE09830','2023_04_10']])
session_list        = np.array([['LPE10885','2023_10_20']])
sessions            = load_sessions(protocol = 'IM',session_list=session_list,load_behaviordata=True, 
                                    load_calciumdata=True, load_videodata=False, calciumversion='dF')


# ## Combine cell data from all loaded sessions to one dataframe:
# celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#stupid filters because suite2p sometimes outputs this first roi as a good cell:
idx_filter = (sessions[0].celldata['npix']>5) & (sessions[0].celldata['skew']>0.01) & (sessions[0].celldata['xloc']<512) & (sessions[0].celldata['yloc']<512)

celldata = sessions[0].celldata[idx_filter]
calciumdata = sessions[0].calciumdata.iloc[:,np.where(idx_filter)[0]]

celldata.iloc[celldata['roi_name']=='ROI 2',celldata.columns=='roi_name'] = 'V1'
# ## remove any double cells (for example recorded in both GR and RF)
# celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

###################### Retinotopic mapping within V1 and PM #####################

fig,axes        = plt.subplots(2,2,figsize=(12,12))

areas           = ['V1','PM'] 

vars            = ['rf_azimuth','rf_elevation']

for i in range(2):
     for j in range(2):
        sns.scatterplot(data = celldata[celldata['roi_name']==areas[j]],x='xloc',y='yloc',hue=vars[i],ax=axes[i,j],palette='gist_rainbow')
        
        box = axes[i,j].get_position()
        axes[i,j].set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])  # Shrink current axis's height by 10% on the bottom
        axes[i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5))        # Put a legend next to current axis
        axes[i,j].set_xlabel('')
        axes[i,j].set_ylabel('')
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
        axes[i,j].set_xlim([0,512])
        axes[i,j].set_ylim([0,512])
        axes[i,j].set_title(areas[j] + ' - ' + vars[i],fontsize=15)

plt.savefig(os.path.join(savedir,'V1_PM_azimuth_elevation_inplane_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

###################### RF size difference between V1 and PM #####################
order = [0,1] #for statistical testing purposes
pairs = [(0,1)]

order = ['V1','PM'] #for statistical testing purposes
pairs = [('V1','PM')]
fig,ax   = plt.subplots(1,1,figsize=(3,4))

sns.violinplot(data=celldata,y="rf_size",x="roi_name",palette=['blue','red'],ax=ax)

annotator = Annotator(ax, pairs, data=celldata, x="roi_name", y="rf_size", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

ax.set_xlabel('area')
ax.set_ylabel('RF size\n(squared degrees)')

plt.savefig(os.path.join(savedir,'V1_PM_rf_size_' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

############################## Noise correlations #################
[T,N]       = np.shape(calciumdata) #get dimensions of data matrix

noise_corr = np.corrcoef(calciumdata.to_numpy().T)

## Compute euclidean distance matrix based on soma center:
distmat_xyz     = np.zeros((N,N))
distmat_xy     = np.zeros((N,N))
distmat_rf      = np.zeros((N,N))
areamat         = np.empty((N,N),dtype=object)
labelmat        = np.empty((N,N),dtype=object)

for i in range(N):
    print(f"\rComputing pairwise distances for neuron {i+1} / {N}",end='\r')
    for j in range(N):
        distmat_xyz[i,j] = math.dist([sessions[0].celldata['xloc'][i],sessions[0].celldata['yloc'][i],sessions[0].celldata['depth'][i]],
                [sessions[0].celldata['xloc'][j],sessions[0].celldata['yloc'][j],sessions[0].celldata['depth'][j]])
        distmat_xy[i,j] = math.dist([sessions[0].celldata['xloc'][i],sessions[0].celldata['yloc'][i]],
                [sessions[0].celldata['xloc'][j],sessions[0].celldata['yloc'][j]])
        distmat_rf[i,j] = math.dist([sessions[0].celldata['rf_azimuth'][i],sessions[0].celldata['rf_elevation'][i]],
                [sessions[0].celldata['rf_azimuth'][j],sessions[0].celldata['rf_elevation'][j]])
        areamat[i,j] = sessions[0].celldata['roi_name'][i] + '-' + sessions[0].celldata['roi_name'][j]
        labelmat[i,j] = str(int(sessions[0].celldata['redcell'][i])) + '-' + str(int(sessions[0].celldata['redcell'][j]))


#Just a check that this works: should only show values in upper triangle of noise corr matrix:
noise_corr2   = np.triu(noise_corr,k=1) #keep only upper triangular part
plt.figure()
plt.imshow(noise_corr2,vmin=-0.1,vmax=0.1)

# construct dataframe with all pairwise measurements:
idx_triu = np.tri(N,N,k=0)==0 #index only upper triangular part
df = pd.DataFrame({'NoiseCorrelation': noise_corr[idx_triu].flatten(),
                'AreaPair': areamat[idx_triu].flatten(),
                'DistXYPair': distmat_xy[idx_triu].flatten(),
                'DistXYZPair': distmat_xyz[idx_triu].flatten(),
                'DistRfPair': distmat_rf[idx_triu].flatten(),
                'LabelPair': labelmat[idx_triu].flatten()})


############### Relationship anatomical distance and receptive field distance: ##################

df_withinarea = df[(df['AreaPair'].isin(['V1-V1','PM-PM'])) & (df['DistRfPair'].notna()) & (df['DistXYPair'] < 1000)]

g = sns.displot(df_withinarea, x="DistLocPair", y="DistRfPair", binwidth=(2, 2), cbar=True,col="AreaPair")
plt.xlim([0,650])
plt.ylim([0,250])
g.set_axis_labels("Anatomical distance \n (approx um)", "RF distance (deg)")

plt.savefig(os.path.join(savedir,'Corr_anat_rf_distance' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

###################### Noise correlations within and across areas: #########################
plt.figure(figsize=(8,5))
sns.barplot(data=df,x='AreaPair',y='NoiseCorrelation')

###################### Noise correlations as a function of pairwise anatomical distance: ####################
fig,axes   = plt.subplots(1,2,figsize=(8,6))

sns.lineplot(x=np.round(df_withinarea['DistXYZPair'],-1),y=df_withinarea['NoiseCorrelation'],hue=df_withinarea['AreaPair'],ax=axes[0])
axes[0].set_xlabel="Pairwise distance XYZ (um)"
# plt.legend(labels=['V1-V1','PM-PM'])
axes[0].set_xlim([-10,600])
axes[0].set_ylim([0,0.13])
# axes[0].set_xlabel("Anatomical distance (approx um)")
axes[0].set_ylabel("Noise Correlation")
axes[0].set_title("Anatomical")

sns.lineplot(x=np.round(df['DistRfPair'],-1),y=df['NoiseCorrelation'],hue=df['AreaPair'],ax=axes[1])
axes[1].set_xlabel="Pairwise RF distance (um)"
axes[1].set_xlim([-10,300])
axes[1].set_ylim([0,0.13])
# axes[1].set_xlabel(['RF distance (ret deg)'])
axes[1].set_ylabel("Noise Correlation")
axes[1].set_title("Receptive Field")

plt.savefig(os.path.join(savedir,'NoiseCorr_anat_rf_distance' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

########################### Noise correlations as a function of pairwise distance: ####################
######################################## Labeled vs unlabeled neurons #################################

fig, axes = plt.subplots(2,2,figsize=(8,7))

areas = ['V1','PM']

for i,iarea in enumerate(areas):
    for j,jarea in enumerate(areas):
        dfarea = df[df['AreaPair']==iarea + '-' + jarea]
        sns.lineplot(ax=axes[i,j],x=np.round(dfarea['DistRfPair'],-1),y=dfarea['NoiseCorrelation'],hue=dfarea['LabelPair'])
        # axes[i,j].set_xlabel="Pairwise distance (um)"
        axes[i,j].set_xlabel="Delta RF (deg)"
        axes[i,j].set_xlim([-10,200])
        axes[i,j].set_ylim([-0.005,0.025])
        axes[i,j].set_title(iarea + '-' + jarea)

plt.savefig(os.path.join(savedir,'NoiseCorr_labeled_RF_distance' + sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')



