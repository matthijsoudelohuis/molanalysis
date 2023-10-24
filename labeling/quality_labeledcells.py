# -*- coding: utf-8 -*-
"""
This script analyzes the quality of the recordings and their relation 
to various factors such as depth of recording, being labeled etc.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loaddata.session_info import filter_sessions,load_sessions
from statannotations.Annotator import Annotator

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn import preprocessing

protocol            = ['VR']
# protocol            = ['GR']
protocol            = ['GR','VR','IM','RF','SP']
protocol            = ['GR','VR','IM']

# sessions            = filter_sessions(protocol,only_animal_id=['LPE09830','LPE09665'])
# sessions            = filter_sessions(protocol,only_animal_id=['LPE09829'])
sessions            = filter_sessions(protocol)

## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

###################### Calcium trace skewness for labeled vs unlabeled cells:
order = [0,1] #for statistical testing purposes
pairs = [(0,1)]

fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF","meanF_chan2"]
fields = ["skew","noise_level","event_rate","radius","npix_soma"]

nfields = len(fields)
fig,axes   = plt.subplots(1,nfields,figsize=(12,4))

for i in range(nfields):
    sns.violinplot(data=celldata,y=fields[i],x="redcell",palette=['gray','red'],ax=axes[i])
    axes[i].set_ylim(np.nanpercentile(celldata[fields[i]],[0.1,99.9]))

    annotator = Annotator(axes[i], pairs, data=celldata, x="redcell", y=fields[i], order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    axes[i].set_xlabel('labeled')
    axes[i].set_ylabel('')
    axes[i].set_title(fields[i])

df2 = celldata.groupby(['redcell'])['redcell'].count()

labelcounts = celldata.groupby(['redcell'])['redcell'].count().to_numpy()
plt.suptitle('Quality comparison labeled ({0}) vs unlabeled ({1}) cells'.format(labelcounts[1],labelcounts[0]))
plt.tight_layout()


df = celldata[["skew","noise_level","npix_soma",
               "meanF","meanF_chan2","event_rate","redcell"]]
sns.pairplot(data=df, hue="redcell")

sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap='bwr')


# ###################### Calcium trace skewness for labeled vs unlabeled cells:
# sns.violinplot(data=celldata,y="skew",x="redcell",palette='Accent',ax=ax1)
# annotator = Annotator(ax1, pairs, data=celldata, x="redcell", y="skew", order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
# annotator.apply_and_annotate()

# ###################### Calcium trace noise level for labeled vs unlabeled cells:
# sns.violinplot(y = celldata[celldata["noise_level"]<1.5]["noise_level"],x = celldata["redcell"],palette='Accent',ax=ax2)
# annotator = Annotator(ax2, pairs, data=celldata, x="redcell", y="noise_level", order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
# annotator.apply_and_annotate()

# ###################### Calcium trace skewness for labeled vs unlabeled cells:
# sns.violinplot(data=celldata,y="event_rate",x = "redcell",palette='Accent',ax=ax3)
# annotator = Annotator(ax3, pairs, data=celldata, x="redcell", y="event_rate", order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
# annotator.apply_and_annotate()



###################### Noise level for labeled vs unlabeled cells:

## plot precentage of labeled cells as a function of depth:
# sns.barplot(x='depth', y='redcell', data=celldata[celldata['roi_name'].isin(['V1','PM'])], estimator=lambda y: sum(y==1)*100.0/len(y))
sns.lineplot(data=celldata[celldata['roi_name'].isin(['V1','PM'])],x='depth', y='redcell', estimator=lambda y: sum(y==1)*100.0/len(y))
# sns.lineplot(data=celldata,x='depth', y='redcell', hue='roi_name',estimator=lambda y: sum(y==1)*100.0/len(y),palette='Accent')
plt.ylabel('% labeled cells')

#Plot fraction of labeled cells across areas of recordings: 
sns.barplot(x='roi_name', y='redcell', data=celldata, estimator=lambda x: sum(x==1)*100.0/len(x),palette='Accent')
plt.ylabel('% labeled cells')

## plot number of cells per plane across depths:
sns.histplot(data=celldata, x='depth',hue='roi_name',palette='Accent')

## plot quality of cells per plane across depths with skew:
# sns.lineplot(data=celldata, x="depth",y=celldata['skew'],estimator='mean')
sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['skew'],estimator='mean')

## plot quality of cells per plane across depths with noise level:
# sns.lineplot(data=celldata, x="depth",y=celldata['noise_level'],estimator='mean')
sns.lineplot(x=np.round(celldata["depth"],-1),y=celldata['noise_level'],estimator='mean')
plt.ylim([0,0.3])

##### 

#Show total counts across all planes:
fig = plt.figure(figsize=[5, 4])
sns.histplot(data=celldata,x="redcell_prob",hue='redcell',stat='count',linestyle='solid',binwidth=0.025,element='step')

#Show for each plane separately, hack = split by depths assuming each plane has different micrometer of depth
fig = plt.figure(figsize=[5, 4])

for i,depth in enumerate(np.unique(celldata['depth'])):
    sns.kdeplot(data=celldata[celldata['depth'] == depth],x="redcell_prob",hue='redcell',linestyle='solid')

#################### 

plt.figure(figsize=(5,4))
# plt.scatter(redcell[iscell[:,0]==1,1],noise_level[iscell[:,0]==1],s=8,c='k')
sns.scatterplot(data=celldata,x='redcell_prob',y='noise_level',s=8,c='k')
plt.xlabel('red cell probability')
plt.ylabel('noise level ')

plt.figure(figsize=(5,4))
sns.scatterplot(data=celldata,x='event_rate',y='noise_level',s=8,c='k')
plt.xlabel('event rate')
plt.ylabel('noise level ')


########### Show calcium traces: #############################################################

session_list        = np.array([['LPE09830','2023_04_12']])
sessions            = load_sessions(protocol = 'GR',session_list=session_list,load_calciumdata=True)


def show_excerpt_traces_labeled(Session,example_cells=None,trialsel=None):
    
    if example_cells is None:
        example_cells = np.random.choice(Session.calciumdata.shape[1],10)

    if trialsel is None:
        trialsel = [np.random.randint(low=0,high=len(Session.trialdata)-400)]
        trialsel.append(trialsel[0]+40)

    example_tstart = Session.trialdata['tOnset'][trialsel[0]-1]
    example_tstop = Session.trialdata['tOnset'][trialsel[1]-1]

    excerpt         = np.array(Session.calciumdata.loc[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)])
    excerpt         = excerpt[:,example_cells]

    min_max_scaler = preprocessing.MinMaxScaler()
    excerpt = min_max_scaler.fit_transform(excerpt)

    # spksselec = spksselec 
    [nframes,ncells] = np.shape(excerpt)

    for i in range(ncells):
        excerpt[:,i] =  excerpt[:,i] + i

    oris        = np.unique(Session.trialdata['Orientation'])
    rgba_color  = plt.get_cmap('hsv',lut=16)(np.linspace(0, 1, len(oris)))  
    
    labeled = Session.celldata['redcell'][example_cells].to_numpy().astype(int)
    temp = ['k','r']
    colors = [temp[labeled[i]] for i in range(len(example_cells))]

    fig, ax = plt.subplots(figsize=[12, 6])
    # plt.plot(Session.ts_F[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)],excerpt,linewidth=0.5,color='black')
    # plt.plot(Session.ts_F[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)],excerpt,linewidth=0.5,color=colors)

    for i in range(ncells):
        plt.plot(Session.ts_F[np.logical_and(Session.ts_F>example_tstart,Session.ts_F<example_tstop)],excerpt[:,i],linewidth=0.5,color=colors[i])

    for i in np.arange(trialsel[0],trialsel[1]):
        ax.add_patch(plt.Rectangle([Session.trialdata['tOnset'][i],0],1,ncells,alpha=0.3,linewidth=0,
                                facecolor=rgba_color[np.where(oris==Session.trialdata['Orientation'][i])]))

    handles= []
    for i,ori in enumerate(oris):
        handles.append(ax.add_patch(plt.Rectangle([0,0],1,ncells,alpha=0.3,linewidth=0,facecolor=rgba_color[i])))

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(handles,oris,loc='center right', bbox_to_anchor=(1.25, 0.5))

    ax.set_xlim([example_tstart,example_tstop])

    ax.add_artist(AnchoredSizeBar(ax.transData, 10, "10 Sec",loc=4,frameon=False))
    ax.axis('off')


#randomly select X labeled and Y unlabeled cells from V1 to show
n_ex = 6
skew_thr = 50
example_cells = np.concatenate((np.random.choice(np.where(np.all((sessions[0].celldata['skew']>np.nanpercentile(sessions[0].celldata['skew'],skew_thr),
                    sessions[0].celldata['roi_name']=='PM',
                    sessions[0].celldata['redcell']==1),axis=0))[0],n_ex),
                    np.random.choice(np.where(np.all((sessions[0].celldata['skew']>np.nanpercentile(sessions[0].celldata['skew'],skew_thr),
                    sessions[0].celldata['roi_name']=='PM',
                    sessions[0].celldata['redcell']==0),axis=0))[0],n_ex)))

show_excerpt_traces_labeled(sessions[0],example_cells=example_cells)


