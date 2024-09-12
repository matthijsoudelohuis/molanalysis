# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center

I have a dataset in which I have activity of neurons in primary visual cortex (V1) 
and secondary higher visual area posteromedial (PM) in awake headfixed mice. The mice 
are presented with a range of visual stimuli including oriented gratings and natural 
images. The neurons in V1 that project to PM are labeled with tdTomato (labeled) and 
neurons in PM that project to V1 are also labeled. I want to explore the differences 
in the activity between the labeled and unlabeled neurons in each area in an unbiased
manner. For example, by looking at the shared activity patterns with PCA and the 
differences in the weights of the PCA components. However, the PCA dimensions of 
each session in which different neurons are recorded will be uninterpretable relative 
to the weights of other sessions. What are ways in which one can get an unbiased but 
interpretable understanding of what types of activity are different between labeled 
and unlabeled neurons and taking multiple different sessions into account?
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import get_local_drive

from sklearn import preprocessing
from scipy.stats import zscore
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statannotations.Annotator import Annotator
from scipy.stats import kstest

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_rel

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning, compute_prefori
from utils.plotting_style import * #get all the fixed color schemes
from utils.explorefigs import plot_PCA_gratings,plot_PCA_images

savedir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\PCA - Images and gratings\\')

#%% #############################################################################
session_list        = np.array([['LPE10919','2023_11_06']])
# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE09830','2023_04_10']])
# session_list        = np.array([['LPE09830','2023_04_10'],
#                                 ['LPE09830','2023_04_12']])
session_list        = np.array([['LPE11086','2024_01_05']])

session_list        = np.array([['LPE11086','2023_12_16']])
session_list        = np.array([['LPE12223','2024_06_11']])
session_list        = np.array([['LPE11998','2024_05_08']])



#Sessions with good receptive field mapping in both V1 and PM:
session_list        = np.array([['LPE09665','2023_03_21'], #GR
                                ['LPE10884','2023_10_20'], #GR
                                # ['LPE11998','2024_05_02'], #GN
                                # ['LPE12013','2024_05_02'], #GN
                                # ['LPE12013','2024_05_07'], #GN
                                ['LPE10919','2023_11_06']]) #GR

#%% Load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'IM',session_list=session_list)
# sessions,nSessions   = load_sessions(protocol = 'GR',session_list=session_list)
# sessions,nSessions   = filter_sessions(protocols = ['GR'],filter_areas=['V1','PM'])

#%% Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion='deconv')

sesidx = 2

#%% Plot PCA

#%% #### 
sesidx = 0
fig = plot_PCA_gratings(sessions[sesidx])
fig = plot_PCA_images(sessions[sesidx])

# fig.savefig(os.path.j0in(savedir,'PCA','PCA_3D_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% Fit PCA

ses         = sessions[sesidx]

areas       = np.array(['V1','PM'])
labeled     = np.array(['unl','lab'])
ses.celldata['labeled'] = labeled[ses.celldata['redcell'].astype(int)]

respmat = ses.respmat

respmat = zscore(respmat,axis=1) # zscore for each neuron across trial responses

pca         = PCA(n_components=25) #construct PCA object with specified number of components
Xp          = pca.fit_transform(respmat.T).T #fit pca to response matrix (n_samples by n_features)
Xc          = pca.components_ #
#dimensionality is now reduced from N by K to ncomp by K

#%% 
nPCstoplot  = 5

fig, axes = plt.subplots(1, nPCstoplot, figsize=[nPCstoplot*3, 2], sharey='row')

for iPC in range(nPCstoplot):
    ax = axes[iPC]
    for iarea,area in enumerate(areas):
        for ilabel,label in enumerate(labeled):
            idx = np.logical_and(ses.celldata['labeled']==label,ses.celldata['roi_name']==area)
            # ax.scatter(Xc[iPC,idx],Xc[iPC,idx],color=get_clr_area_labeled([area + label]),s=25,alpha=0.8)
            # ax.hist(Xc[iPC,idx],Xc[iPC,idx],color=get_clr_area_labeled([area + label]),s=25,alpha=0.8)
            sns.histplot(Xc[iPC,idx],color=get_clr_area_labeled([area + label]),linewidth=1,label=area+label,alpha=0.8,
                         element="step",stat="count",fill=False,ax=ax)
            ax.legend(loc='upper right',frameon=False,fontsize=7)
        pval = kstest(Xc[iPC,np.logical_and(ses.celldata['labeled']==labeled[0],ses.celldata['roi_name']==area)],
                Xc[iPC,np.logical_and(ses.celldata['labeled']==labeled[1],ses.celldata['roi_name']==area)])[1]
        if pval<0.05:
            ax.text(0.03,50+10*iarea,s='*%s' %(area),fontsize=6)
    pval = kstest(Xc[iPC,np.logical_and(ses.celldata['labeled']==labeled[0],ses.celldata['roi_name']==areas[0])],
                Xc[iPC,np.logical_and(ses.celldata['labeled']==labeled[0],ses.celldata['roi_name']==areas[1])])[1]
    if pval<0.05:
        ax.text(0.03,40,s='*V1-PM',fontsize=6)

plt.tight_layout()

#%% 

areas       = np.array(['V1','PM'])
labeled     = np.array(['unl','lab'])

nPCs        = 25
pca         = PCA(n_components=nPCs) #construct PCA object with specified number of components


#%% Load sessions lazy: 
sessions,nSessions   = load_sessions(protocol = 'IM',session_list=session_list)
sessions,nSessions   = filter_sessions(protocols = ['GR','IM','GN'],filter_areas=['V1','PM'])
sessions,nSessions   = filter_sessions(protocols = ['IM'],filter_areas=['V1','PM'])

#%% Load proper data and compute PCA weights:                      
for ises,ses in tqdm(enumerate(sessions),desc='loading sessions and computing PCA weights'):    # iterate over sessions
    ses.load_data(load_calciumdata=True,calciumversion='dF')
    ses.celldata['labeled'] = labeled[ses.celldata['redcell'].astype(int)]
    fit          = pca.fit(zscore(ses.calciumdata,axis=0)) #fit pca to response matrix (n_samples by n_features)
    ses.components = fit.components_ # zscore for each neuron across trial responses

#%% Perform statistical test (ks test) on PCA weight distribution by area and label:   
pval_label = np.full((nSessions,25,2),np.nan)
pval_area = np.full((nSessions,25,2),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='testing PCA Weights'):    # iterate over sessions
    for iPC in range(nPCs): # loop over PC dimensions
        for iarea,area in enumerate(areas): # loop over areas
            if np.any(np.logical_and(ses.celldata['labeled']==labeled[1],ses.celldata['roi_name']==area)):
                pval = kstest(ses.components[iPC,np.logical_and(ses.celldata['labeled']==labeled[0],ses.celldata['roi_name']==area)],
                            ses.components[iPC,np.logical_and(ses.celldata['labeled']==labeled[1],ses.celldata['roi_name']==area)])[1]
                pval_label[ises,iPC,iarea] = pval
        for ilabel,label in enumerate(labeled): # loop over areas
            idx_V1 = np.logical_and(ses.celldata['labeled']==label,ses.celldata['roi_name']==areas[0])
            idx_PM = np.logical_and(ses.celldata['labeled']==label,ses.celldata['roi_name']==areas[1])
            if np.any(idx_V1) and np.any(idx_PM):
                pval = kstest(ses.components[iPC,idx_V1],ses.components[iPC,idx_PM])[1]
                pval_area[ises,iPC,ilabel] = pval

pval_label[np.isnan(pval_label)] = 1
pval_area[np.isnan(pval_area)] = 1

#%% A histogram of the fraction of sessions that has a significant pvalue (<0.05) per area per PCs. 
# On the x-axis is the PC and bars are split based on the area
clrs_areas = get_clr_areas(areas)
pthr = 0.05

fig,ax = plt.subplots(1,1,figsize=(8,3))
sns.barplot(data=pd.DataFrame({'PCs':np.tile(range(nPCs),2),'Area':np.repeat(areas, nPCs),'Sig. Freq.':np.mean(pval_label<pthr,axis=0).flatten()}),
            x='PCs',y='Sig. Freq.',hue='Area',palette=clrs_areas,ax=ax)
ax.set_ylabel('Fraction session pval < 0.05')
ax.set_xlabel('PCs')
ax.set_ylim([0,1])
ax.legend(loc='upper right',frameon=False)
ax.set_title('KS test labeled vs unlabeled weights')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'pval_label_%1.2f.png' % pthr),dpi=300)

#%% A histogram of the fraction of sessions that has a significant pvalue (<0.05) per area per PCs. 
# On the x-axis is the PC and bars are split based on the area
pthr = 0.05
clrs_labeled = get_clr_labeled()

fig,ax = plt.subplots(1,1,figsize=(8,3))
sns.barplot(data=pd.DataFrame({'PCs':np.tile(range(nPCs),2),'Area':np.repeat(areas, nPCs),'Sig. Freq.':np.mean(pval_area<pthr,axis=0).flatten()}),
            x='PCs',y='Sig. Freq.',hue='Area',palette=clrs_areas,ax=ax)
ax.set_ylabel('Fraction session pval < 0.05')
ax.set_xlabel('PCs')
ax.set_ylim([0,1])
ax.legend(loc='upper right',frameon=False)
ax.set_title('KS test labeled vs unlabeled weights')
plt.tight_layout()
fig.savefig(os.path.join(savedir,'pval_area_%1.2f.png' % pthr),dpi=300)

#%% Ward Clustering per session:

n_clusters      = 6
nPCs            = 25

for ises,ses in tqdm(enumerate(sessions),desc='Hierarchical Clustering'):    # iterate over sessions

    ses.celldata['area_labeled']    = ses.celldata['roi_name'] + ses.celldata['labeled']
    area_labeled                    = np.array(['V1unl', 'V1lab', 'PMunl', 'PMlab'], dtype=object)
    clrs_area_labeled               = get_clr_area_labeled(area_labeled)

    # Assuming `neural_data` is a (neurons x features) array for a session
    # Step 1: Dimensionality Reduction
    pca = PCA(n_components=nPCs)  # Reduce to 10 components
    reduced_data = pca.fit_transform(zscore(ses.calciumdata,axis=0).T)

    # Step 2: Compute distance matrix for clustering
    distance_matrix = linkage(reduced_data, method='ward')  # Ward's method for hierarchical clustering

    # # Step 3: Plot the dendrogram
    # plt.figure(figsize=(10, 7))
    # dendrogram(distance_matrix)
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Neuron Index')
    # plt.ylabel('Distance')

    # Step 3: Apply Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = cluster.fit_predict(reduced_data)

    # Show clustering results:
    palettes        = ['tab10',clrs_area_labeled]
    hues            = [labels,ses.celldata['area_labeled']]
    hue_orders      = [range(n_clusters),area_labeled]
    projs           = [(0, 1),(2, 3)]

    fig,axes = plt.subplots(2,2,figsize=(8,8))
    for iproj,proj in enumerate(projs):
        for ipalette,palette in enumerate(palettes):
            ax = axes[iproj,ipalette]
            sns.scatterplot(x=reduced_data[:, proj[0]], y=reduced_data[:, proj[1]], 
                            hue=hues[ipalette], hue_order=hue_orders[ipalette], palette=palette,ax = ax)
            ax.set_title('Ward clustering in PC space')
            ax.set_xlabel('Principal Component %d' % proj[0])
            ax.set_ylabel('Principal Component %d' % proj[1])

    # Visualize clustering
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'WardClusters_ScatterPC_Area_Labeled_%s.png' % ses.sessiondata['session_id'][0]),dpi=300)

    # Show occupancy of each cluster by neuron types:
    df = pd.DataFrame({'cluster':labels,'area_labeled':ses.celldata['area_labeled']})
    df = df.groupby(['cluster','area_labeled']).size().unstack(fill_value=0)
    df = df.div(df.sum(axis=1), axis=0)

    # Show a stacked bar plot for each cluster with which type of neurons compose the cluster (neuron type = area_labeled)
    fig,ax = plt.subplots(figsize=(3,2))
    bottom = np.zeros(n_clusters)

    # for clus in range(n_clusters):
    for iarea_label,area_label in enumerate(area_labeled):
        if area_label in df:
            p = ax.bar(x=range(n_clusters),height=df.loc[:,area_label], bottom=bottom,color=clrs_area_labeled[iarea_label])
            bottom += df.loc[:,area_label]

    ax.set_xticks(range(n_clusters))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Fraction of neurons')

    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'WardCluster_Occupancey_Area_Labeled_%s.png' % ses.sessiondata['session_id'][0]),dpi=300)



#%% Estimate the dimensionality by area and label:   

outmat = np.full((nSessions,2,2,3),np.nan)

for ises,ses in tqdm(enumerate(sessions),desc='Estimating dimensionality'):    # iterate over sessions
    for iarea,area in enumerate(areas):
        np.random.seed(1)
        idx_labeled             = np.where(np.logical_and(ses.celldata['roi_name']==area,ses.celldata['labeled']=='lab'))[0]
        idx_unlabeled           = np.where(np.logical_and(ses.celldata['roi_name']==area,ses.celldata['labeled']=='unl'))[0]
        idx_unlabeled_sample    = np.random.choice(np.where(idx_unlabeled)[0],len(idx_labeled),replace=False)

        if np.any(idx_labeled):
            for ilabel,label in enumerate(labeled):
                if label == 'unl':
                    idx = idx_unlabeled_sample
                else: 
                    idx = idx_labeled

                X = ses.calciumdata.iloc[:,idx]
                X = zscore(X,axis=0)
                pca = PCA(n_components=min(200,X.shape[1]))
                pca.fit(X)
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                n_components_90 = np.argmax(cumvar>0.9)+1
                # pval_label[ises,iarea,ilabel,0] = pca.explained_variance_ratio_[0]
                outmat[ises,iarea,ilabel,0] = n_components_90
                outmat[ises,iarea,ilabel,1] = n_components_90 / X.shape[1]

                # We center the data and compute the sample covariance matrix.
                n_samples = X.shape[0]
                cov_matrix = np.dot(X.T, X) / n_samples
                # Compute the participation ratio
                eigenvalues = np.array([np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_])
                outmat[ises,iarea,ilabel,2] = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)

#%% Plot results: participation ratio per area and label
# Plot the mean and standard error of the mean of the participation ratio for each area and label and 
# test statistically significant difference between labeled and unlabeled within area with a paired t-test. 
metrics = ['NComponents 90%','Fraction components 90%','Participation Ratio']

for im,metric in enumerate(metrics):
    datamat = outmat[:,:,:,im].reshape(nSessions,-1)

    fig,ax = plt.subplots(1,1,figsize=(4,3))
    plt.plot([0,1],datamat[:,:2].T)#,color=clrs_area_labeled)
    ax.text(0.15, np.nanpercentile(datamat[:,:2],80), 'p = %.3f' % ttest_rel(datamat[:,0],datamat[:,1],nan_policy='omit')[1], fontsize=9)
    plt.plot([2,3],datamat[:,2:].T)#,color=clrs_area_labeled)
    ax.text(2.15,  np.nanpercentile(datamat[:,:2],80), 'p = %.3f' % ttest_rel(datamat[:,2],datamat[:,3],nan_policy='omit')[1], fontsize=9)

    ax.set_ylabel('%s' % metric)
    ax.set_xticks(range(4),labels=area_labeled)
    ax.set_title('%s per area and label' % metric)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,'Dimensionality_%s.png' % metric), format = 'png')

