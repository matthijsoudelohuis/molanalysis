#%%
import os
os.chdir('e:\\Python\\molanalysis')
from loaddata.get_data_folder import *

from labeling.tdTom_labeling_cellpose import gen_red_images,proc_labeling_session
from loaddata.get_data_folder import get_rawdata_drive
from preprocessing.locate_rf import locate_rf_session, optim_resp_win

#%% 
animal_id          = 'LPE11086' #If empty than all animals in folder will be processed
sessiondate         = '2024_01_10'
rawdatadir = get_rawdata_drive([animal_id],protocols=['RF'])

t_starts        = np.array([0, 0.2, 0.4])
t_stops         = np.array([0.5, 0.75, 1])

fracdata        = np.empty((len(t_starts),len(t_stops)))
logsumdata      = np.empty((len(t_starts),len(t_stops)))
ncells          = 2

for istart,t_resp_start in enumerate(t_starts):
    for istop,t_resp_stop in enumerate(t_stops):
        df = optim_resp_win(rawdatadir,animal_id,sessiondate,t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,
                            iplane=0,ncells=ncells)
        fracdata[istart,istop] = df.count()['RF_azim'] / ncells
        logsumdata[istart,istop] = np.sum(-np.log10(df['RF_p']))

# df = sparse_noise_STA(rawdatadir,animal_id,sessiondate,t_resp_start=t_resp_start,t_resp_stop=t_resp_stop,
                            # iplane=4,ncells=200)

#%% 
fig,axes = plt.subplots(1,2,figsize=(10,5))
im = axes[0].imshow(fracdata,interpolation='none',origin='lower',vmin=0,vmax=1)
axes[0].set_xticks(np.arange(len(t_stops)),labels=t_stops)
axes[0].set_yticks(np.arange(len(t_starts)),labels=t_starts)
axes[0].set_xlabel('Resp Win Stop')
axes[0].set_ylabel('Resp Win Start')
cbar = fig.colorbar(im,ax=axes[0],shrink=0.5,aspect=5,pad=0.05)
cbar.set_label('Fraction of cells with RF')
im = axes[1].imshow(logsumdata/logsumdata.max(),interpolation='none',origin='lower')
axes[1].set_xticks(np.arange(len(t_stops)),labels=t_stops)
axes[1].set_yticks(np.arange(len(t_starts)),labels=t_starts)
axes[1].set_xlabel('Resp Win Stop')
axes[1].set_ylabel('Resp Win Start')
cbar = fig.colorbar(im,ax=axes[1],shrink=0.5,aspect=5,pad=0.05)
cbar.set_label('Cumulative significance of RF (norm)')
plt.tight_layout()
