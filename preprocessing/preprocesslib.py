"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of preprocessing functions that take raw data
(behavior, task, microscope, video) and preprocess them. Is called by preprocess_main.
Principally the data is integrated with additional info and stored for pandas dataframe usage

Banners: https://textkool.com/en/ascii-art-generator?hl=default&vl=default&font=Old%20Banner&text=DETECTION%20TASK

"""

import os, math
from pathlib import Path
import pandas as pd
import numpy as np
from natsort import natsorted
from datetime import datetime
import scipy
from scipy.stats import zscore
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import scipy.stats as st

from utils.twoplib import get_meta
from loaddata.get_data_folder import get_data_folder
from labeling.tdTom_labeling_cellpose import proc_labeling_plane
from tqdm.auto import tqdm

"""
  #####  #######  #####   #####  ### ####### #     # ######     #    #######    #    
 #     # #       #     # #     #  #  #     # ##    # #     #   # #      #      # #   
 #       #       #       #        #  #     # # #   # #     #  #   #     #     #   #  
  #####  #####    #####   #####   #  #     # #  #  # #     # #     #    #    #     # 
       # #             #       #  #  #     # #   # # #     # #######    #    ####### 
 #     # #       #     # #     #  #  #     # #    ## #     # #     #    #    #     # 
  #####  #######  #####   #####  ### ####### #     # ######  #     #    #    #     # 
"""

def proc_sessiondata(rawdatadir,animal_id,sessiondate,protocol):
    """ preprocess general information about this mouse and session """
    
    #Init sessiondata dataframe:
    sessiondata     = pd.DataFrame()

    #Populate sessiondata with information regarding this session:
    sessiondata         = sessiondata.assign(animal_id = [animal_id])
    sessiondata         = sessiondata.assign(sessiondate = [sessiondate])
    sessiondata         = sessiondata.assign(session_id = [animal_id + '_' + sessiondate])
    sessiondata         = sessiondata.assign(experimenter = ["Matthijs Oude Lohuis"])
    sessiondata         = sessiondata.assign(species = ["Mus musculus"])
    sessiondata         = sessiondata.assign(lab = ["Petreanu Lab"])
    sessiondata         = sessiondata.assign(institution = ["Champalimaud Research"])
    sessiondata         = sessiondata.assign(preprocessdate = [datetime.now().strftime("%Y_%m_%d")])
    sessiondata         = sessiondata.assign(protocol = [protocol])

    sessions_overview_VISTA = pd.read_excel(os.path.join(get_data_folder(),'VISTA_Sessions_Overview.xlsx'))
    # sessions_overview_VR    = pd.read_excel(os.path.join(rawdatadir,'VR_Sessions_Overview.xlsx'))
    sessions_overview_DE    = pd.read_excel(os.path.join(get_data_folder(),'DE_Sessions_Overview.xlsx'))
    sessions_overview_AKS   = pd.read_excel(os.path.join(get_data_folder(),'AKS_Sessions_Overview.xlsx'))

    if np.any(np.logical_and(sessions_overview_VISTA["sessiondate"] == sessiondate,sessions_overview_VISTA["protocol"] == protocol)):
        sessions_overview = sessions_overview_VISTA
    elif np.any(np.logical_and(sessions_overview_DE["sessiondate"] == sessiondate,sessions_overview_DE["protocol"] == protocol)):
        sessions_overview = sessions_overview_DE
    elif np.any(np.logical_and(sessions_overview_AKS["sessiondate"] == sessiondate,sessions_overview_AKS["protocol"] == protocol)):
        sessions_overview = sessions_overview_AKS
    else: 
        print('Session not found in excel session overview')
        return sessiondata
    
    idx =   (sessions_overview["sessiondate"] == sessiondate) & \
            (sessions_overview["animal_id"] == animal_id) & \
            (sessions_overview["protocol"] == protocol)
    if np.any(idx):
        sessiondata         = pd.merge(sessiondata,sessions_overview.loc[idx],'inner') #Copy all the data from the excel to sessiondata dataframe
        age_in_days         = (datetime.strptime(sessiondata['sessiondate'][0], "%Y_%m_%d") - datetime.strptime(sessiondata['DOB'][0], "%Y_%m_%d")).days
        sessiondata         = sessiondata.assign(age_in_days = [age_in_days]) #Store the age in days at time of the experiment
    
        expr_in_days         = (datetime.strptime(sessiondata['sessiondate'][0], "%Y_%m_%d") - datetime.strptime(sessiondata['DOV'][0], "%Y_%m_%d")).days
        sessiondata         = sessiondata.assign(expression_in_days = [expr_in_days]) #Store the age in days at time of the experiment

    else: 
        print('Session not found in excel session overview')

    return sessiondata

"""
 ######  ####### #     #    #    #     # ### ####### ######  ######     #    #######    #    
 #     # #       #     #   # #   #     #  #  #     # #     # #     #   # #      #      # #   
 #     # #       #     #  #   #  #     #  #  #     # #     # #     #  #   #     #     #   #  
 ######  #####   ####### #     # #     #  #  #     # ######  #     # #     #    #    #     # 
 #     # #       #     # #######  #   #   #  #     # #   #   #     # #######    #    ####### 
 #     # #       #     # #     #   # #    #  #     # #    #  #     # #     #    #    #     # 
 ######  ####### #     # #     #    #    ### ####### #     # ######  #     #    #    #     # 
"""

def proc_behavior_passive(rawdatadir,sessiondata):
    """ preprocess all the behavior data for one session: running """
    
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    harpdata_file   = list(filter(lambda a: 'harp' in a, filenames)) #find the harp files
    harpdata_file   = list(filter(lambda a: 'csv'  in a, harpdata_file)) #take the csv file, not the rawharp bin
    
    #Get data, construct dataframe and modify a bit:
    behaviordata        = pd.read_csv(os.path.join(sesfolder,harpdata_file[0]),skiprows=0)
    behaviordata.columns = ["rawvoltage","ts","zpos","runspeed"] #rename the columns
    behaviordata        = behaviordata.drop(columns="rawvoltage") #remove rawvoltage, not used

    #Remove segments with double sampling (here something went wrong)
    #Remove that piece that has timestamps overlapping, find by finding first timestamp again greater than reset
    idx = np.where(np.diff(behaviordata['ts'])<0)[0]
    restartidx = []
    for i in idx:
        restartidx.append(np.where(behaviordata['ts'] > behaviordata['ts'][i])[0][0])

    for i,r in zip(idx,restartidx):
        # print(i,r)
        behaviordata.drop(behaviordata.loc[i:r].index,inplace=True)
        print('Removed double sampled harp data with duration %1.2f seconds' % ((r-i)/1000))

    behaviordata = behaviordata.reset_index(drop=True)

    #subsample data 10 times (from 1000 to 100 Hz)
    behaviordata = behaviordata.iloc[::10, :].reset_index(drop=True) 

    #Filter slightly to get rid of transient at wheel turn and 0.5hz LED blink artefact:
    # idx = np.arange(2000,9000)
    # plt.plot(behaviordata['runspeed'][idx])
    behaviordata['runspeed'] = gaussian_filter(behaviordata['runspeed'], sigma=21)
    # plt.plot(behaviordata['runspeed'][idx])

    # Some checks:
    if sessiondata['session_id'][0] not in ['LPE09665_2023_03_15','LPE09665_2023_03_20']:
        assert(np.allclose(np.diff(behaviordata['ts']),1/100,rtol=0.2)) #timestamps ascending and around sampling rate
    runspeed = behaviordata['runspeed'][1000:].to_numpy()
    assert(np.all(runspeed > -50) and all(runspeed < 100)) #running speed (after initial phase) within reasonable range

    behaviordata['session_id']     = sessiondata['session_id'][0]

    return behaviordata

"""
  #####  ######     #    ####### ### #     #  #####   #####  
 #     # #     #   # #      #     #  ##    # #     # #     # 
 #       #     #  #   #     #     #  # #   # #       #       
 #  #### ######  #     #    #     #  #  #  # #  ####  #####  
 #     # #   #   #######    #     #  #   # # #     #       # 
 #     # #    #  #     #    #     #  #    ## #     # #     # 
  #####  #     # #     #    #    ### #     #  #####   #####  
  """

def proc_GR(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
    
    #Checks:
    nOris = len(pd.unique(trialdata['Orientation']))
    assert(nOris==8 or nOris == 16) #8 or 16 distinct orientations
    ori_counts = trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(all(ori_counts > 50) and all(ori_counts < 400)) #between 50 and 400 repetitions

    junk,junk,oriconds  = np.unique(trialdata['Orientation'],return_index=True,return_inverse=True)
    trialdata['stimCond']    = oriconds

    assert(np.allclose(trialdata['tOffset'] - trialdata['tOnset'],0.75,atol=0.1)) #stimulus duration all around 0.75s
    assert(np.allclose(np.diff(trialdata['tOnset']),2,atol=0.1)) #total trial duration all around 2s

    trialdata['session_id']     = sessiondata['session_id'][0]
    sessiondata['ntrials']      = len(trialdata) #add number of trials
    trialdata['trial_id']       = np.array([sessiondata['session_id'][0] + '_' + '%04.0f' % k for k in range(0,len(trialdata))])

    return sessiondata,trialdata


def proc_GN(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
    
    trialdata['Speed'] = trialdata['TF'] / trialdata['SF']

    #Checks:
    CenterOris  = np.array([30,90,150]);        #orientations degrees
    # CenterTF    = np.array([1.5,3, 6]);        #Hz #earlier version
    # CenterSF    = np.array([0.12,0.06,0.03]);  #cpd #earlier version

    CenterTF    = np.array([1, 2.5, 4]);        #Hz #v4
    CenterSF    = np.array([5/30,3/30,1/30]);   #cpd #v4
    CenterSpeed = CenterTF / CenterSF           #speed is TF / SF

    trialdata['centerOrientation'] = '' #add column with the center orientation for this stimulus (wiht noise around this)
    trialdata['centerTF'] = ''
    trialdata['centerSF'] = ''
    # trialdata['centerSpeed'] = ''
    for k in range(len(trialdata)): #for every trial get what center the ori, TF and SF where closest to:
        trialdata.iloc[k,trialdata.columns.get_loc("centerOrientation")] = CenterOris[np.abs((CenterOris - trialdata.Orientation[k])).argmin()]
        trialdata.iloc[k,trialdata.columns.get_loc("centerTF")] = CenterTF[np.abs((CenterTF - trialdata.TF[k])).argmin()]
        trialdata.iloc[k,trialdata.columns.get_loc("centerSF")] = CenterSF[np.abs((CenterSF - trialdata.SF[k])).argmin()]
    
    trialdata['centerSpeed'] = trialdata['centerTF'] / trialdata['centerSF']
    
    junk,junk,oriconds  = np.unique(trialdata['centerOrientation'],return_index=True,return_inverse=True)
    junk,junk,speedconds  = np.unique(trialdata['centerSpeed'],return_index=True,return_inverse=True)
    trialdata['oriCond']     = oriconds
    trialdata['speedCond']   = speedconds
    trialdata['stimCond']    = oriconds + speedconds*3

    # define the noise relative to the center:  
    trialdata['deltaOrientation']   = trialdata['Orientation'] - trialdata['centerOrientation'] 
    trialdata['deltaTF']            = trialdata['TF'] - trialdata['centerTF']
    trialdata['deltaSF']            = trialdata['SF'] - trialdata['centerSF']
    trialdata['deltaSpeed']         = trialdata['Speed'] - trialdata['centerSpeed']
    trialdata['logdeltaSpeed']      = np.log10(trialdata['Speed']) - np.log10(trialdata['centerSpeed'].to_numpy().astype('float64'))

    #Checks:
    assert(all(np.isin(trialdata['centerSpeed'],CenterSpeed))),'grating speed not in originally programmed stimulus speeds'
    ori_counts = trialdata.groupby(['centerOrientation','centerSpeed'])['centerOrientation'].count().to_numpy()
    assert(all(ori_counts > 100) and all(ori_counts < 400)) #between 100 and 400 repetitions for each stimulus
    assert(np.allclose(trialdata['tOffset'] - trialdata['tOnset'],0.75,atol=0.1)) #stimulus duration all around 0.75s
    assert(np.allclose(np.diff(trialdata['tOnset']),2,atol=0.1)) #total trial duration all around 2s

    trialdata['session_id']     = sessiondata['session_id'][0]
    sessiondata['ntrials']      = len(trialdata) #add number of trials
    trialdata['trial_id']       = np.array([sessiondata['session_id'][0] + '_' + '%04.0f' % k for k in range(0,len(trialdata))])

    return sessiondata,trialdata

"""
 ### #     #    #     #####  #######  #####  
  #  ##   ##   # #   #     # #       #     # 
  #  # # # #  #   #  #       #       #       
  #  #  #  # #     # #  #### #####    #####  
  #  #     # ####### #     # #             # 
  #  #     # #     # #     # #       #     # 
 ### #     # #     #  #####  #######  #####  
                                             
"""

def proc_IM(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
    
    trialdata['session_id']     = sessiondata['session_id'][0]
    sessiondata['ntrials']      = len(trialdata) #add number of trials
    trialdata['trial_id']       = np.array([sessiondata['session_id'][0] + '_' + '%04.0f' % k for k in range(0,len(trialdata))])

    return sessiondata,trialdata

def proc_MV(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
    
    trialdata['session_id']     = sessiondata['session_id'][0]
    sessiondata['ntrials']      = len(trialdata) #add number of trials

    assert trialdata['ismei'].dtype == 'int64'
    assert len(np.unique(trialdata['ismei'][trialdata['ismei']!=0])) >= 10
    assert trialdata['mei_cellid'].dtype == 'object'
    assert len(np.unique(trialdata['mei_cellid'])) >= 10

    trialdata['ismei']          = trialdata['ismei'].astype('int64')
    trialdata['mei_cellid']     = trialdata['mei_cellid']
    trialdata['trial_id']       = np.array([sessiondata['session_id'][0] + '_' + '%04.0f' % k for k in range(0,len(trialdata))])

    return sessiondata,trialdata



"""
######  ####### ####### #######  #####  ####### ### ####### #     #    #######    #     #####  #    # 
#     # #          #    #       #     #    #     #  #     # ##    #       #      # #   #     # #   #  
#     # #          #    #       #          #     #  #     # # #   #       #     #   #  #       #  #   
#     # #####      #    #####   #          #     #  #     # #  #  #       #    #     #  #####  ###    
#     # #          #    #       #          #     #  #     # #   # #       #    #######       # #  #   
#     # #          #    #       #     #    #     #  #     # #    ##       #    #     # #     # #   #  
######  #######    #    #######  #####     #    ### ####### #     #       #    #     #  #####  #    # 
"""

def proc_task(rawdatadir,sessiondata):
    """ preprocess all the trial, stimulus and behavior data for one behavior VR session """
    
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    #Init output dataframes:
    trialdata       = pd.DataFrame()
    behaviordata    = pd.DataFrame()

    #Process behavioral data:
    filenames       = os.listdir(sesfolder)
    
    harpdata_file   = list(filter(lambda a: 'harp' in a, filenames)) #find the harp files
    harpdata_file   = list(filter(lambda a: 'csv'  in a, harpdata_file)) #take the csv file, not the rawharp bin
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]))
    
    trialdata['lickResponse'] = trialdata['lickResponse'].astype(int)
    trialdata = trialdata.rename(columns={'Reward': 'rewardAvailable'})
    trialdata['rewardGiven']  = np.logical_and(trialdata['rewardAvailable'],trialdata['lickResponse']).astype('int')
    trialdata = trialdata.rename(columns={'trialType': 'signal'})
    
    for col in trialdata.columns:
        trialdata = trialdata.rename(columns={col: col[0].lower() + col[1:]})

    # if np.isin(sessiondata['sessiondate'][0],['2023_11_27','2023_11_28','2023_11_30']):
    #     trialdata['signal'] = trialdata['signal']*100

    # Signal values: 
    assert(np.all(np.logical_and(trialdata['signal'] >= 0,trialdata['signal'] <= 100))), 'not all signal values are between 0 and 100'
    assert(np.any(trialdata['signal'] > 1)), 'signal values do not exceed 1'
    assert(np.isin(np.unique(trialdata['stimRight'])[0],['Ori45','Ori135','A','B','C','D','E','F','G'])), 'Unknown stimulus presented'
    if os.path.exists(os.path.join(sesfolder,"suite2p")):  
        assert(len(np.unique(trialdata['stimRight']))==1), 'more than one stimulus appears to be presented in this recording session'

    assert sessiondata['stim'][0] == np.unique(trialdata['stimRight'])[0], 'Stimulus in overview does not match stimulus in trialdata'

    if sessiondata['stim'][0] == 'B':
        sessiondata.at[0,'stim'] = 'F'
        trialdata.loc[:,'stimRight'] = trialdata.loc[:,'stimLeft'] = 'F'

    #Give stimulus category type 
    trialdata['stimcat']            =  ''
    idx                             = trialdata[trialdata['signal']==100].index
    trialdata.loc[idx,'stimcat']    = 'M'
    idx                             = trialdata[trialdata['signal']==0].index
    trialdata.loc[idx,'stimcat']    = 'C'

    if sessiondata.protocol[0] == 'DM':
        assert np.all(np.isin(trialdata['signal'],[0,100])), 'Max protocol with intermediate saliencies'

    elif sessiondata.protocol[0] == 'DP':
        nconds = len(np.unique(trialdata['signal']))
        assert(nconds>3 and nconds<6), 'too many or too few conditions for psychometric protocol'
        idx         = ~np.isin(trialdata['signal'],[0,100])
        trialdata.iloc[idx,trialdata.columns.get_loc('stimcat')]    = 'P'

    elif sessiondata.protocol[0] == 'DN':
        idx         = ~np.isin(trialdata['signal'],[0,100])
        sigs        = np.unique(trialdata['signal'][idx])
        
        if 'signal_center' in sessiondata:
            sigrange    = [sessiondata['signal_center'][0]-sessiondata['signal_range'][0]/2,
                    sessiondata['signal_center']+sessiondata['signal_range']/2]
        else:
            sessiondata['signal_center'] = np.mean(sigs).round()
            sessiondata['signal_range'] = [np.max(sigs) - np.min(sigs)]

        assert np.all(np.logical_and((sigs>=sessiondata['signal_center'][0]-sessiondata['signal_range'][0]/2),
                        np.all(sigs<=sessiondata['signal_center'][0]+sessiondata['signal_range'][0]/2))),'outside range'
        assert(len(sigs)>5), 'no signal jitter observed'
        assert sessiondata['signal_center'][0]==np.mean(sigs).round(), 'center of noise does not match overview'
        assert sessiondata['signal_range'][0]==np.max(sigs) - np.min(sigs), 'noise range does not match overview'

        trialdata['signal_jitter']          = ''
        trialdata.loc[trialdata.index[idx],'signal_jitter']     = trialdata.loc[trialdata.index[idx],'signal'].to_numpy() - sessiondata['signal_center'].to_numpy()
        
        trialdata.iloc[idx,trialdata.columns.get_loc('stimcat')]    = 'N'
    else:
        print('unknown protocol abbreviation')

    assert ~np.any(trialdata['stimcat'].isnull()), 'stimulus category labeling error, unknown stimstrength'

    #Get behavioral data, construct dataframe and modify a bit:
    behaviordata        = pd.read_csv(os.path.join(sesfolder,harpdata_file[0]),skiprows=0)
    if np.any(behaviordata.filter(regex='Item')):
        behaviordata.columns = ["rawvoltage","ts","trialNumber","zpos","runspeed","lick","reward"] #rename the columns
    behaviordata = behaviordata.drop(columns="rawvoltage") #remove rawvoltage, not used
    behaviordata = behaviordata.rename(columns={'timestamp': 'ts','trialnumber': 'trialNumber'}) #rename consistently
    # behaviordata = behaviordata[behaviordata['trialNumber'] <= np.max(trialdata['trialNumber'])]
    behaviordata.loc[behaviordata.index[behaviordata['trialNumber'] > np.max(trialdata['trialNumber'])],'trialNumber'] = np.max(trialdata['trialNumber'])

    ## Licks, get only timestamps of onset of discrete licks
    lickactivity    = np.diff(behaviordata['lick'].to_numpy().astype('int')) #behaviordata lick is True whenever tongue in ROI>threshold
    lick_ts         = behaviordata['ts'][np.append(lickactivity==1,False)].to_numpy()
    
    ## Rewards, same as licks, get only onset of reward as timestamps
    rewardactivity = np.diff(behaviordata['reward'].to_numpy()) #behaviordata lick is True whenever tongue in ROI>threshold
    reward_ts      = behaviordata['ts'][np.append(rewardactivity>0,False)].to_numpy()
    
    ## Modify position indices to be all in overall space, not per trial:
    # for trialdata fields: trialStart, trialEnd, stimStart, stimEnd, rewardZoneStart, rewardZoneEnd
    # for behaviordata fields: zpos
    trialdata['trialStart_k']   = 0 #always zero
    trialdata['trialEnd_k']     = 0 #length of that trial
    for k in range(len(trialdata)):
        triallength = max(behaviordata.loc[behaviordata.index[behaviordata['trialNumber']==k+1],'zpos'])
        trialdata.loc[k,'trialEnd_k'] = triallength
    trialdata['tStart'] = np.concatenate(([behaviordata['ts'][0]],trialdata['tEnd'].to_numpy()[:-1]))

    behaviordata['zpos_k'] = behaviordata['zpos']
    # behaviordata['zpos_tot'] = behaviordata['zpos']
    for k in range(1,len(trialdata)):
        behaviordata.loc[behaviordata.index[behaviordata['trialNumber']==k+1],'zpos'] += np.cumsum(trialdata['trialEnd_k'])[k-1]
    
    #correct for postponing problems with stim end if lick time out
    trialdata['stimEnd']              = trialdata['stimStart'] + 20 

    # Copy the original data to new fields that have values relative to that trial:
    trialdata['stimStart_k']          = trialdata['stimStart']
    trialdata['stimEnd_k']            = trialdata['stimEnd']
    trialdata['rewardZoneStart_k']    = trialdata['rewardZoneStart']
    trialdata['rewardZoneEnd_k']      = trialdata['rewardZoneEnd']
    
    # Create or overwrite fields with overall z position:
    trialdata['trialStart']             = np.cumsum(np.concatenate(([0],trialdata['trialEnd_k'][:-1].to_numpy())))
    trialdata['trialEnd']               = trialdata['trialEnd_k'] + trialdata['trialStart']
    trialdata['stimStart']              = trialdata['stimStart_k'] + trialdata['trialStart']
    trialdata['stimEnd']                = trialdata['stimEnd_k'] + trialdata['trialStart']
    trialdata['rewardZoneStart']        = trialdata['rewardZoneStart_k'] + trialdata['trialStart']
    trialdata['rewardZoneEnd']          = trialdata['rewardZoneEnd_k'] + trialdata['trialStart']

    # k = 10
    # idx = behaviordata.index[behaviordata['trialNumber']==k+1]
    # plt.figure()
    # plt.plot(behaviordata.loc[idx,'ts'],behaviordata.loc[idx,'zpos_k'])
    # plt.scatter(trialdata.loc[k,'tStart'],trialdata.loc[k,'trialStart_k'],s=50)
    # plt.figure()
    # plt.plot(behaviordata.loc[idx,'ts'],behaviordata.loc[idx,'zpos'])
    # plt.scatter(trialdata.loc[k,'tStart'],trialdata.loc[k,'trialStart'],s=50)

    sessiondata['stimLength']       = np.mean(trialdata['stimEnd'] - trialdata['stimStart'])
    sessiondata['rewardZoneOffset'] = np.mean(trialdata['rewardZoneStart'] - trialdata['stimStart'])
    sessiondata['rewardZoneLength'] = np.mean(trialdata['rewardZoneEnd'] - trialdata['rewardZoneStart'])
    stable_stimzone                 = np.allclose(sessiondata['stimLength'], trialdata['stimEnd'] - trialdata['stimStart'], rtol=1e-05)
    stable_rewzone                  = np.allclose(sessiondata['rewardZoneOffset'], trialdata['rewardZoneStart'] - trialdata['stimStart'], rtol=1e-05)
    stable_rewzonelength            = np.allclose(sessiondata['rewardZoneLength'], trialdata['rewardZoneEnd'] - trialdata['rewardZoneStart'], rtol=1e-05)
    sessiondata['stable_params']    = np.all((stable_stimzone,stable_rewzone,stable_rewzonelength))
    if not sessiondata['stable_params'][0]:
        print('Variable stimulus or reward zone within session detected\n')

    g = trialdata[['trialStart','stimStart','stimEnd','rewardZoneEnd','trialEnd']].to_numpy().flatten()
    assert np.all(np.diff(g)>=0), 'trial event ordering issue'

    #Subsample the data, don't need this high 1000 Hz resolution
    behaviordata = behaviordata.iloc[::10, :].copy().reset_index(drop=True) #subsample data 10 times (to 100 Hz)
    
    behaviordata['lick']    = False #init to false and set to true for sample of first lick or rew
    behaviordata['reward']  = False
    
    #Add the lick times again to the subsampled dataframe:
    for lick in lick_ts:
        behaviordata.loc[behaviordata.index[np.argmax(lick<behaviordata['ts'])],'lick'] = True
    print("%d licks" % np.sum(behaviordata['lick'])) #Give output to check if reasonable

    if datetime.strptime(sessiondata['sessiondate'][0],"%Y_%m_%d") >= datetime(2024, 4, 15):
        sessiondata['minLicks'] = 3
    else: 
        sessiondata['minLicks'] = 1

    #Assert that licks are not inside the reward zone for trials in which no lick response was recorded:
    for k in range(len(trialdata)):
        idx_rewzone     = np.logical_and(behaviordata['zpos']>trialdata['rewardZoneStart'][k],behaviordata['zpos']<trialdata['rewardZoneEnd'][k])
        idx             = np.logical_and(behaviordata['lick'],idx_rewzone)#.flatten()
        if np.sum(idx)>=sessiondata['minLicks'][0] and trialdata['lickResponse'][k]==False:
            print('%d lick(s) registered in reward zone of trial %d with lickResponse==false' % (np.sum(idx),k))
    
    #Add the reward times again to the subsampled dataframe:
    for reward in reward_ts: #set only the first timestamp of reward to True, to have single indices
        behaviordata.loc[behaviordata.index[np.argmax(reward<behaviordata['ts'])],'reward'] = True
    
    #Add the timestamps of entering and exiting stimulus zone:
    trialdata['tStimStart'] = ''
    trialdata['tStimEnd'] = ''
    for t in range(len(trialdata)):
        idx             = np.where(behaviordata['zpos'] >= trialdata.loc[t, 'stimStart'])[0][0]
        trialdata.loc[trialdata.index[t], 'tStimStart'] =  behaviordata.loc[idx,'ts']
        
        idx             = np.where(behaviordata['zpos'] >= trialdata.loc[trialdata.index[t], 'stimEnd'])[0][0]
        trialdata.loc[trialdata.index[t], 'tStimEnd'] =  behaviordata.loc[idx,'ts']
    assert ~np.any(trialdata['tStimEnd']<trialdata['tStimStart']), 'Stimulus end earlier than stimulus start'
    assert ~np.any(trialdata['stimEnd']<trialdata['stimStart']), 'Stimulus end earlier than stimulus start'

    #Compute the timestamp and spatial location of the reward being given and store in trialdata:
    trialdata['tReward'] = pd.Series(dtype='float')
    trialdata['sReward'] = pd.Series(dtype='float')
    for k in range(len(trialdata)):
        idx_k           = behaviordata['reward']
        idx_rewzone     = np.logical_and(behaviordata['zpos']>=trialdata['rewardZoneStart'][k],behaviordata['zpos']<=trialdata['rewardZoneEnd'][k])
        idx             = np.logical_and(idx_k,idx_rewzone) #.flatten()
        if np.any(idx):
            trialdata.loc[k,'tReward'] = behaviordata['ts'].iloc[np.where(idx)[0][0]]
            trialdata.loc[k,'sReward'] = behaviordata['zpos'].iloc[np.where(idx)[0][0]]
     
    if ~np.all(trialdata['tReward'][trialdata['rewardGiven']==1]):
        print('a rewarded trial has no timestamp of reward' % trialdata['tReward'][trialdata['rewardGiven']==1].isnull().count())
    if np.any(trialdata['tReward'][trialdata['rewardGiven']==0]):
        print('%d unrewarded trials have timestamp of reward (manual?)' % trialdata['tReward'][trialdata['rewardGiven']==0].count())

    # Compute reward rate (fraction of possible rewarded trials that are rewarded) 
    # with sliding window for engagement index:
    sliding_window = 24
    rewardrate_thr = 0.3
    trialdata['engaged'] = 1
    # for t in range(sliding_window,len(trialdata)):
    for t in range(len(trialdata)):
        idx = np.intersect1d(np.arange(len(trialdata)),np.arange(t-sliding_window/2,t+sliding_window/2))
        hitrate = np.sum(trialdata['rewardGiven'][idx]) / np.sum(trialdata['rewardAvailable'][idx])
        if hitrate < rewardrate_thr:
            trialdata.loc[trialdata.index[t],'engaged'] = 0

    print("%d total rewards" % np.sum(behaviordata['reward'])) #Give output to check if reasonable
    print("%d rewarded trials" % trialdata['tReward'].count()) #Give output to check if reasonable

    behaviordata['session_id']  = sessiondata['session_id'][0] #Store unique session_id
    trialdata['session_id']     = sessiondata['session_id'][0]
    sessiondata['ntrials']      = len(trialdata) #add number of trials
    trialdata['trial_id']       = np.array([sessiondata['session_id'][0] + '_' + '%04.0f' % k for k in range(0,len(trialdata))])

    return sessiondata, trialdata, behaviordata

"""
 #     # ### ######  ####### ####### ######     #    #######    #    
 #     #  #  #     # #       #     # #     #   # #      #      # #   
 #     #  #  #     # #       #     # #     #  #   #     #     #   #  
 #     #  #  #     # #####   #     # #     # #     #    #    #     # 
  #   #   #  #     # #       #     # #     # #######    #    ####### 
   # #    #  #     # #       #     # #     # #     #    #    #     # 
    #    ### ######  ####### ####### ######  #     #    #    #     # 
"""

def proc_videodata(rawdatadir,sessiondata,behaviordata,keepPCs=30):
    
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')

    filenames       = os.listdir(sesfolder)

    avi_file        = list(filter(lambda a: '.avi' in a, filenames)) #find the trialdata file
    csv_file        = list(filter(lambda a: 'cameracsv' in a, filenames)) #find the trialdata file

    csvdata         = pd.read_csv(os.path.join(sesfolder,csv_file[0]))
    nts             = len(csvdata)
    ts              = csvdata['Item2'].to_numpy()

    videodata       = pd.DataFrame(data = ts, columns = ['ts'])

    #Check that the number of frames is ballpark range of what it should be based on framerate and session duration:
    framerate = np.round(1/np.mean(np.diff(ts))).astype(int)
    sessiondata['video_fs'] = framerate

    assert np.isin(framerate,[10,15,30,60,66]), 'Error! Frame rate is not 10, 15, 30, 60 Hz, something wrong with triggering'
    sesdur = behaviordata.loc[behaviordata.index[-1],'ts']  - behaviordata.loc[behaviordata.index[0],'ts'] 
    assert np.isclose(nts,sesdur * framerate,rtol=0.01), f"Number of trials {nts} is not close enough to session duration {sesdur} * framerate {framerate} = {sesdur * framerate}"
    #Check that frame rate matches interframe interval:
    assert np.isclose(1/framerate,np.mean(np.diff(ts)),rtol=0.01)
    #Check that inter frame interval does not take on crazy values:
    # issues_ts = np.logical_or(np.diff(ts[1:-1])<1/framerate/3,np.diff(ts[1:-1])>1/framerate*2)
    issues_ts = np.concatenate(([False],np.logical_or(np.diff(ts[1:-1])<1/framerate/3,
                                                  np.diff(ts[1:-1])>1/framerate*2),[False,False]))

    if np.any(issues_ts):
        print('Interpolating %d video timestamp issues' % np.sum(issues_ts))
        # Interpolate samples where timestamps are off:
        ts[issues_ts] = np.interp(np.where(issues_ts)[0],np.where(~issues_ts)[0],ts[~issues_ts])
        
        #Go through another loop:
        issues_ts = np.concatenate(([False],np.logical_or(np.diff(ts[1:-1])<1/framerate/3,
                                                  np.diff(ts[1:-1])>1/framerate*2),[False,False]))
        ts[issues_ts] = np.interp(np.where(issues_ts)[0],np.where(~issues_ts)[0],ts[~issues_ts])
        #Go through another loop:
        issues_ts = np.concatenate(([False],np.logical_or(np.diff(ts[1:-1])<1/framerate/3,
                                                  np.diff(ts[1:-1])>1/framerate*2),[False,False]))
        ts[issues_ts] = np.interp(np.where(issues_ts)[0],np.where(~issues_ts)[0],ts[~issues_ts])

    # assert ~np.any(np.logical_or(np.diff(ts[1:-1])<1/framerate/3,np.diff(ts[1:-1])>1/framerate*2))
    assert np.sum(np.logical_or(np.diff(ts[1:-1])<1/framerate/3,np.diff(ts[1:-1])>1/framerate*2))<150,'Error! Too many video timestamp issues'

    videodata['zpos'] = np.interp(x=videodata['ts'],xp=behaviordata['ts'],
                                    fp=behaviordata['zpos'])               

    #Load FaceMap data: 
    facemapfile =  list(filter(lambda a: '_proc' in a, filenames)) #find the processed facemap file
    if facemapfile and len(facemapfile)==1 and os.path.exists(os.path.join(sesfolder,facemapfile[0])):
        # facemapfile = "W:\\Users\\Matthijs\\Rawdata\\NSH07422\\2023_03_13\\SP\\Behavior\\SP_NSH07422_camera_2023-03-13T16_44_07_proc.npy"
        
        proc = np.load(os.path.join(sesfolder,facemapfile[0]),allow_pickle=True).item()
        
        assert len(proc['motion'][0])==0,'multivideo performed, should not be done'
        
        roi_types = [proc['rois'][i]['rtype'] for i in range(len(proc['rois']))]
        assert 'motion SVD' in roi_types,'motion SVD missing, _proc file does not contain motion svd roi'
        assert nts==len(proc['motion'][1]),'not the same number of timestamps as frames'

        videodata['motionenergy']   = proc['motion'][1]
        PC_labels                   = list('videoPC_' + '%s' % k for k in range(0,keepPCs))
        videodata = pd.concat([videodata,pd.DataFrame(proc['motSVD'][1][:,:keepPCs],columns=PC_labels)],axis=1)
        
        #Pupil data:
        if 'Pupil' not in roi_types: 
            print('Pupil ROI missing (perhaps video too dark)')
            videodata['pupil_area'] = videodata['pupil_ypos'] = videodata['pupil_xpos'] = ''
        else:
            videodata['pupil_area']   = proc['pupil'][0]['area_smooth']
            videodata['pupil_ypos']   = proc['pupil'][0]['com'][:,0]
            videodata['pupil_xpos']   = proc['pupil'][0]['com'][:,1]

            #remove outlier data (poor pupil fits):
            xpos = zscore(videodata['pupil_xpos'])
            ypos = zscore(videodata['pupil_ypos'])
            area = zscore(videodata['pupil_area'])

            idx = np.any((np.abs(xpos)>5,np.abs(ypos)>5,np.abs(area)>5,videodata['pupil_area']==0),axis=0)
            print('set %1.4f percent of video frames with pupil fit outlier samples to nan \n' % (np.sum(idx) / len(videodata['pupil_xpos'])))
            videodata.iloc[idx,videodata.columns.get_loc("pupil_xpos")] = np.nan
            videodata.iloc[idx,videodata.columns.get_loc("pupil_ypos")] = np.nan
            videodata.iloc[idx,videodata.columns.get_loc("pupil_area")] = np.nan
    else:
        print(f'#######################  Could not locate facemapdata... in {filenames}')

    videodata['session_id']  = sessiondata['session_id'][0]

    return videodata

"""
 ### #     #    #     #####  ### #     #  #####  
  #  ##   ##   # #   #     #  #  ##    # #     # 
  #  # # # #  #   #  #        #  # #   # #       
  #  #  #  # #     # #  ####  #  #  #  # #  #### 
  #  #     # ####### #     #  #  #   # # #     # 
  #  #     # #     # #     #  #  #    ## #     # 
 ### #     # #     #  #####  ### #     #  #####  
"""

def proc_imaging(sesfolder, sessiondata, filter_good_cells=True):
    """ integrate preprocessed calcium imaging data """
    
    suite2p_folder = os.path.join(sesfolder,"suite2p")
    
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])

    # load ops of plane0:
    ops                = np.load(os.path.join(plane_folders[0], 'ops.npy'), allow_pickle=True).item()
    
    # read metadata from tiff (just take first tiff from the filelist
    # metadata should be same for all if settings haven't changed during differernt protocols
    localtif = os.path.join(sesfolder,sessiondata.protocol[0],'Imaging',
                             os.listdir(os.path.join(sesfolder,sessiondata.protocol[0],'Imaging'))[0])
    if os.path.exists(ops['filelist'][0]):
        meta, meta_si   = get_meta(ops['filelist'][0])
    elif os.path.exists(localtif):
        meta, meta_si   = get_meta(localtif)
    meta_dict       = dict() #convert to dictionary:
    for line in meta_si:
        meta_dict[line.split(' = ')[0]] = line.split(' = ')[1]
   
    #put some general information in the sessiondata  
    sessiondata = sessiondata.assign(nplanes = ops['nplanes'])
    sessiondata = sessiondata.assign(roi_xpix = ops['Lx'])
    sessiondata = sessiondata.assign(roi_ypix = ops['Ly'])
    sessiondata = sessiondata.assign(nchannels = ops['nchannels'])
    sessiondata = sessiondata.assign(fs = ops['fs'])
    sessiondata = sessiondata.assign(date_suite2p = ops['date_proc'])
    sessiondata = sessiondata.assign(microscope = ['2p-ram Mesoscope'])
    sessiondata = sessiondata.assign(laser_wavelength = ['920'])
    sessiondata = sessiondata.assign(calcium_indicator = ['GCaMP6s'])
    
    #Add information about the imaging from scanimage:  
    sessiondata = sessiondata.assign(SI_pz_constant             = float(meta_dict['SI.hBeams.lengthConstants']))
    sessiondata = sessiondata.assign(SI_pz_Fraction             = float(meta_dict['SI.hBeams.powerFractions']))
    sessiondata = sessiondata.assign(SI_pz_power                = float(meta_dict['SI.hBeams.powers']))
    sessiondata = sessiondata.assign(SI_pz_adjust               = meta_dict['SI.hBeams.pzAdjust'])
    sessiondata = sessiondata.assign(SI_pz_reference            = float(meta_dict['SI.hStackManager.zPowerReference']))
    
    sessiondata = sessiondata.assign(SI_motioncorrection        = bool(meta_dict['SI.hMotionManager.correctionEnableXY']))
    sessiondata = sessiondata.assign(SI_linePeriod              = float(meta_dict['SI.hRoiManager.linePeriod']))
    sessiondata = sessiondata.assign(SI_linesPerFrame           = float(meta_dict['SI.hRoiManager.linesPerFrame']))
    sessiondata = sessiondata.assign(SI_pixelsPerLine           = float(meta_dict['SI.hRoiManager.pixelsPerLine']))
    sessiondata = sessiondata.assign(SI_scanFramePeriod         = float(meta_dict['SI.hRoiManager.scanFramePeriod']))
    sessiondata = sessiondata.assign(SI_volumeFrameRate         = float(meta_dict['SI.hRoiManager.scanFrameRate']))
    sessiondata = sessiondata.assign(SI_frameRate               = float(meta_dict['SI.hRoiManager.scanVolumeRate']))
    sessiondata = sessiondata.assign(SI_bidirectionalscan       = bool(meta_dict['SI.hScan2D.bidirectional']))
    
    sessiondata = sessiondata.assign(SI_fillFractionSpatial     = float(meta_dict['SI.hScan2D.fillFractionSpatial']))
    sessiondata = sessiondata.assign(SI_fillFractionTemporal    = float(meta_dict['SI.hScan2D.fillFractionTemporal']))
    sessiondata = sessiondata.assign(SI_flybackTimePerFrame     = float(meta_dict['SI.hScan2D.flybackTimePerFrame']))
    sessiondata = sessiondata.assign(SI_flytoTimePerScanfield   = float(meta_dict['SI.hScan2D.flytoTimePerScanfield']))
    sessiondata = sessiondata.assign(SI_linePhase               = float(meta_dict['SI.hScan2D.linePhase']))
    sessiondata = sessiondata.assign(SI_scanPixelTimeMean       = float(meta_dict['SI.hScan2D.scanPixelTimeMean']))
    sessiondata = sessiondata.assign(SI_scannerFrequency        = float(meta_dict['SI.hScan2D.scannerFrequency']))
    sessiondata = sessiondata.assign(SI_actualNumSlices         = int(meta_dict['SI.hStackManager.actualNumSlices']))
    sessiondata = sessiondata.assign(SI_numFramesPerVolume      = int(meta_dict['SI.hStackManager.numFramesPerVolume']))

    ## Get trigger data to align timestamps:
    filenames         = os.listdir(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior'))
    triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
    triggerdata       = pd.read_csv(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior',triggerdata_file[0]),skiprows=1).to_numpy()
    #skip the first row because is init of the variable in BONSAI
    [ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

    # getting numer of ROIs
    nROIs = len(meta['RoiGroups']['imagingRoiGroup']['rois'])
    #Find the names of the rois:
    roi_area    = [meta['RoiGroups']['imagingRoiGroup']['rois'][i]['name'] for i in range(nROIs)]
    
    #Find the depths of the planes for each roi:
    roi_depths = np.array([],dtype=int)
    roi_depths_idx = np.array([],dtype=int)

    for i in range(nROIs):
        zs = np.array([meta['RoiGroups']['imagingRoiGroup']['rois'][i]['zs']]).flatten()
        roi_depths = np.append(roi_depths,zs)
        roi_depths_idx = np.append(roi_depths_idx,np.repeat(i,len(zs)))
    
    #get all the depths of the planes in order of imaging:
    plane_zs    = np.array(meta_dict['SI.hStackManager.zs'].replace('[','').replace(']','').split(' ')).astype('float64')

    #Find the roi to which each plane belongs:
    plane_roi_idx = np.array([roi_depths_idx[np.where(roi_depths == plane_zs[i])[0][0]] for i in range(ops['nplanes'])])

    for iplane,plane_folder in tqdm(enumerate(plane_folders), desc="Processing plane", total=len(plane_folders)):
    # for iplane,plane_folder in enumerate(plane_folders[:1]):
        print('processing plane %s / %s' % (iplane+1,ops['nplanes']))

        ops                 = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()
        
        [ts_plane, protocol_frame_idx_plane] = align_timestamps(sessiondata, ops, triggerdata)

        chan2_prob          = np.load(os.path.join(plane_folder, 'redcell.npy'))
        iscell              = np.load(os.path.join(plane_folder, 'iscell.npy'))
        stat                = np.load(os.path.join(plane_folder, 'stat.npy'), allow_pickle=True)
        
        if os.path.exists(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane)):
            redcell_seg         = np.load(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane), allow_pickle=True).item()
            masks_cp_red        = redcell_seg['masks']
            Nredcells_plane     = len(np.unique(masks_cp_red))-1 # number of labeled cells overall, minus 1 because 0 for all nonlabeled pixels
            redcell             = proc_labeling_plane(iplane,plane_folder,showcells=False,overlap_threshold=0.5)
        else: 
            print(f'\n\n Warning: cellpose results not found in {plane_folder}, setting labeling to zero\n\n')
            redcell             = np.zeros((len(iscell),3))
            Nredcells_plane     = 0

        ncells_plane              = len(iscell)
        
        celldata_plane            = pd.DataFrame()

        celldata_plane            = celldata_plane.assign(iscell        = iscell[:,0])
        celldata_plane            = celldata_plane.assign(iscell_prob   = iscell[:,1])

        celldata_plane            = celldata_plane.assign(skew          = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(radius        = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(npix_soma     = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(npix          = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(xloc          = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(yloc          = np.empty([ncells_plane,1]))

        for k in range(0,ncells_plane):
            celldata_plane['skew'][k] = stat[k]['skew']
            celldata_plane['radius'][k] = stat[k]['radius']
            celldata_plane['npix_soma'][k] = stat[k]['npix_soma']
            celldata_plane['npix'][k] = stat[k]['npix']
            celldata_plane['xloc'][k] = stat[k]['med'][0] / 512 * 600
            celldata_plane['yloc'][k] = stat[k]['med'][1] / 512 * 600
        
        celldata_plane['redcell']           = redcell[:,0]
        celldata_plane['frac_of_ROI_red']   = redcell[:,1]
        celldata_plane['frac_red_in_ROI']   = redcell[:,2]
        celldata_plane['chan2_prob']        = chan2_prob[:,1]
        celldata_plane['nredcells']         = Nredcells_plane
        
        celldata_plane['plane_idx']     = iplane
        celldata_plane['roi_idx']       = plane_roi_idx[iplane]
        celldata_plane['plane_in_roi_idx']       = np.where(np.where(plane_roi_idx==plane_roi_idx[iplane])[0] == iplane)[0][0]
        celldata_plane['roi_name']      = roi_area[plane_roi_idx[iplane]]
        celldata_plane['depth']         = plane_zs[iplane] - sessiondata['ROI%d_dura' % (plane_roi_idx[iplane]+1)][0]
        #compute power at this plane: formula: P = P0 * exp^((z-z0)/Lz)
        celldata_plane['power_mw']      = sessiondata['SI_pz_power'][0]  * math.exp((plane_zs[iplane] - sessiondata['SI_pz_reference'][0])/sessiondata['SI_pz_constant'][0])

        redcelllabels                   = np.array(['unl','lab']) #Give redcells a string label
        celldata_plane['labeled']       = celldata_plane['redcell'].astype(int).apply(lambda x: redcelllabels[x])
        celldata_plane['arealabel']     = celldata_plane['roi_name'] + celldata_plane['labeled']

        if os.path.exists(os.path.join(plane_folder, 'RF_Fgauss.npy')):
            RF_Fgauss = np.load(os.path.join(plane_folder, 'RF_Fgauss.npy'))
            celldata_plane['rf_az_F']   = RF_Fgauss[:,0]
            celldata_plane['rf_el_F']   = RF_Fgauss[:,1]
            celldata_plane['rf_sx_F']   = RF_Fgauss[:,2]
            celldata_plane['rf_sy_F']   = RF_Fgauss[:,3]
            celldata_plane['rf_r2_F']   = RF_Fgauss[:,4]

        if os.path.exists(os.path.join(plane_folder, 'RF_Fneugauss.npy')):
            RF_Fneugauss = np.load(os.path.join(plane_folder, 'RF_Fneugauss.npy'))
            celldata_plane['rf_az_Fneu']   = RF_Fneugauss[:,0]
            celldata_plane['rf_el_Fneu']   = RF_Fneugauss[:,1]
            celldata_plane['rf_sx_Fneu']   = RF_Fneugauss[:,2]
            celldata_plane['rf_sy_Fneu']   = RF_Fneugauss[:,3]
            celldata_plane['rf_r2_Fneu']   = RF_Fneugauss[:,4]

        if os.path.exists(os.path.join(plane_folder, 'RF_Fsmooth.npy')):
            RF_Fsmooth = np.load(os.path.join(plane_folder, 'RF_Fsmooth.npy'))
            celldata_plane['rf_az_Fsmooth']   = RF_Fsmooth[:,0]
            celldata_plane['rf_el_Fsmooth']   = RF_Fsmooth[:,1]
            celldata_plane['rf_sx_Fsmooth']   = RF_Fsmooth[:,2]
            celldata_plane['rf_sy_Fsmooth']   = RF_Fsmooth[:,3]
            celldata_plane['rf_r2_Fsmooth']   = RF_Fsmooth[:,4]

        # OLD RF estimates loading:
        # if os.path.exists(os.path.join(plane_folder, 'RF_F.npy')):
        #     RF_F = np.load(os.path.join(plane_folder, 'RF_F.npy'))
        #     celldata_plane['rf_az_F']   = RF_F[:,0]
        #     celldata_plane['rf_el_F']   = RF_F[:,1]
        #     celldata_plane['rf_sz_F']   = RF_F[:,2]
        #     celldata_plane['rf_p_F']    = RF_F[:,3]
            
        # if os.path.exists(os.path.join(plane_folder, 'RF_Fneu.npy')):
        #     RF_Fneu = np.load(os.path.join(plane_folder, 'RF_Fneu.npy'))
        #     celldata_plane['rf_az_Fneu']   = RF_Fneu[:,0]
        #     celldata_plane['rf_el_Fneu']   = RF_Fneu[:,1]
        #     celldata_plane['rf_sz_Fneu']   = RF_Fneu[:,2]
        #     celldata_plane['rf_p_Fneu']    = RF_Fneu[:,3]

        # if os.path.exists(os.path.join(plane_folder, 'RF_Favg.npy')):
        #     RF_Favg = np.load(os.path.join(plane_folder, 'RF_Favg.npy'))
        #     celldata_plane['rf_az_Favg']   = RF_Favg[:,0]
        #     celldata_plane['rf_el_Favg']   = RF_Favg[:,1]
        #     celldata_plane['rf_sz_Favg']   = RF_Favg[:,2]
        #     celldata_plane['rf_p_Favg']    = RF_Favg[:,3]

        # if os.path.exists(os.path.join(plane_folder, 'RF_Fblock.npy')):
        #     RF_Fblock = np.load(os.path.join(plane_folder, 'RF_Fblock.npy'))
        #     assert(np.shape(RF_Fblock)==(256,6)), 'problem with dimensions of Fblock'
        #     # RF_Fblock[:,1] = RF_Fblock[:,1] 
        #     # vec_elevation       = [-16.7,50.2] #bottom and top of screen displays

        #     distblock = np.sqrt((celldata_plane['xloc'].to_numpy()[:,None] - RF_Fblock[:,5][None,:])**2 + 
        #                         (celldata_plane['yloc'].to_numpy()[:,None] - RF_Fblock[:,4][None,:])**2)
        #     celldata_plane['rf_az_Fblock']   = RF_Fblock[np.argmin(distblock,axis=1),0]
        #     celldata_plane['rf_el_Fblock']   = RF_Fblock[np.argmin(distblock,axis=1),1]
        #     celldata_plane['rf_sz_Fblock']   = RF_Fblock[np.argmin(distblock,axis=1),2]
        #     celldata_plane['rf_p_Fblock']    = RF_Fblock[np.argmin(distblock,axis=1),3]
         
        ##################### load suite2p activity outputs:
        F                   = np.load(os.path.join(plane_folder, 'F.npy'), allow_pickle=True)
        F_chan2             = np.load(os.path.join(plane_folder, 'F_chan2.npy'), allow_pickle=True)
        Fneu                = np.load(os.path.join(plane_folder, 'Fneu.npy'), allow_pickle=True)
        spks                = np.load(os.path.join(plane_folder, 'spks.npy'), allow_pickle=True)
        
        #Correct overcorrected fluorescence by shifting fluorescence to at least the neuropil:
        idx = np.percentile(F,5,axis=1)<0
        offset = np.percentile(Fneu[idx,:],5,axis=1,keepdims=True) - np.percentile(F[idx,:],5,axis=1,keepdims=True)
        F[idx,:] = F[idx,:] + offset
        
        # If ROIs were manually added, no FChan2 might be made in earlier versions of suite2p
        if np.shape(F_chan2)[0] < np.shape(F)[0]:
            print('ROIs were manually added in suite2p, fabricating red channel data...')
            F_chan2     = np.vstack((F_chan2, np.tile(F_chan2[[-1],:], 1)))

        #If there are bad frames then interpolate these values with median values:
        if os.path.exists(os.path.join(sesfolder,'bad_frames.npy')):
            bad_frames              = np.load(os.path.join(sesfolder,'bad_frames.npy'))
            F[:,bad_frames]         = np.median(F,axis=1,keepdims=True)
            F_chan2[:,bad_frames]   = np.median(F_chan2,axis=1,keepdims=True)
            Fneu[:,bad_frames]      = np.median(Fneu,axis=1,keepdims=True)
            spks[:,bad_frames]      = np.median(spks,axis=1,keepdims=True)

        # Correct neuropil and compute dF/F: (Rupprecht et al. 2021)
        dF     = calculate_dff(F, Fneu,coeff_Fneu=0.7,prc=10) #see function below

        # Compute average fluorescence on green and red channels:
        celldata_plane['meanF']         = np.mean(F, axis=1)
        celldata_plane['meanF_chan2']   = np.mean(F_chan2, axis=1)

        # Calculate the noise level of the cells ##### Rupprecht et al. 2021 Nat Neurosci.
        celldata_plane['noise_level'] = np.nanmedian(np.abs(np.diff(dF, axis=-1)), axis=-1) / np.sqrt(ops['fs']) * 100
        # celldata_plane['noise_level'] = np.median(np.abs(np.diff(dF,axis=1)),axis=1)/np.sqrt(ops['fs'])

        #Count the number of events by taking number of stretches with z-scored activity above 2:
        zF              = st.zscore(dF.copy(),axis=1)
        nEvents         = np.sum(np.diff(np.ndarray.astype(zF > 2,dtype='uint8'))==1,axis=1)
        event_rate      = nEvents / (ops['nframes'] / ops['fs'])
        celldata_plane['event_rate'] = event_rate

        F           = F[:,protocol_frame_idx_plane==1].transpose()
        F_chan2     = F_chan2[:,protocol_frame_idx_plane==1].transpose()
        Fneu        = Fneu[:,protocol_frame_idx_plane==1].transpose()
        spks        = spks[:,protocol_frame_idx_plane==1].transpose()
        dF          = dF[:,protocol_frame_idx_plane==1].transpose()

        # if imaging was aborted during scanning of a volume, later planes have less frames
        # Compensate by duplicating last value
        if np.shape(F)[0]==len(ts_master):
            pass       #do nothing, shapes match
        elif np.shape(F)[0]==len(ts_master)-1: #copy last timestamp of array
            F           = np.vstack((F, np.tile(F[[-1],:], 1)))
            F_chan2     = np.vstack((F_chan2, np.tile(F_chan2[[-1],:], 1)))
            Fneu        = np.vstack((Fneu, np.tile(Fneu[[-1],:], 1)))
            spks        = np.vstack((spks, np.tile(spks[[-1],:], 1)))
            dF          = np.vstack((dF, np.tile(dF[[-1],:], 1)))
        else:
            print("Problem with timestamps and imaging frames")
 
        #construct dataframe with activity by cells: give unique cell_id as label:
        # cell_ids            = list(sessiondata['session_id'][0] + '_' + '%s' % iplane + '_' + '%04.0f' % k for k in range(0,ncells_plane))
        cell_ids            = np.array([sessiondata['session_id'][0] + '_' + '%s' % iplane + '_' + '%04.0f' % k for k in range(0,ncells_plane)])
        #store cell_ids in celldata:
        celldata_plane['cell_id']         = cell_ids

        if os.path.exists(os.path.join(plane_folder, 'Fmatch.mat')):
            print(f"Writing overlapping cell IDs")
            celldata_plane = proc_roimatchpub(os.path.join(plane_folder, 'Fmatch.mat'),
                                              sessiondata,celldata_plane)
            
        #Filter out neurons with mean fluorescence below threshold:
        meanF_thresh                                    = 25
        iscell[celldata_plane['meanF']<meanF_thresh,0]  = 0

        #Filter only good cells
        if filter_good_cells:
            celldata_plane  = celldata_plane[iscell[:,0]==1]
            cell_ids        = cell_ids[np.where(iscell[:,0]==1)[0]]
            F               = F[:,iscell[:,0]==1]
            F_chan2         = F_chan2[:,iscell[:,0]==1]
            Fneu            = Fneu[:,iscell[:,0]==1]
            spks            = spks[:,iscell[:,0]==1]
            dF              = dF[:,iscell[:,0]==1]

        if iplane == 0: #if first plane then init dataframe, otherwise append
            celldata = celldata_plane.copy()
        else:
            celldata = pd.concat([celldata,celldata_plane])
        
        #Save both deconvolved and fluorescence data:
        dFdata_plane                    = pd.DataFrame(dF, columns=cell_ids)
        dFdata_plane['ts']              = ts_master    #add timestamps
        deconvdata_plane                = pd.DataFrame(spks, columns=cell_ids)   
        deconvdata_plane['ts']          = ts_master    #add timestamps
        Fchan2data_plane                = pd.DataFrame(F_chan2, columns=cell_ids)
        Fchan2data_plane['ts']          = ts_master    #add timestamps
        #Fchan2data is not saved but average across neurons, see below
        
        if iplane == 0:
            dFdata = dFdata_plane.copy()
            deconvdata = deconvdata_plane.copy()
            Fchan2data = Fchan2data_plane.copy()
        else:
            dFdata = dFdata.merge(dFdata_plane)
            deconvdata = deconvdata.merge(deconvdata_plane)
            Fchan2data = Fchan2data.merge(Fchan2data_plane)
    
    celldata.reset_index(inplace=True,drop=True) #remove index based on within plane idx

    #If ROI is unnamed, replace if ROI_1/V1 combi, ROI_2/PM combi, otherwise error:
    if celldata['roi_name'].str.contains('ROI').any():
        if celldata['roi_name'].isin(['PM']).any():
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI_2','V1')
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI 2','V1')
            print('Unnamed ROI in scanimage inferred to be V1')
        if celldata['roi_name'].isin(['V1']).any():
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI_1','PM')
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI 1','PM')
            print('Unnamed ROI in scanimage inferred to be PM')
        assert not celldata['roi_name'].str.contains('ROI').any(),'unknown area'

    #Assign layers to the cells based on recording depth and area:
    celldata = assign_layer(celldata)
    
    #Add recombinase enzym label to red cells:
    labelareas = ['V1','PM']
    for area in labelareas:
        temprecombinase =  area + '_recombinase'
        celldata.loc[celldata['roi_name']==area,'recombinase'] = sessiondata[temprecombinase].to_list()[0]
    celldata.loc[celldata['redcell']==0,'recombinase'] = 'non' #set all nonlabeled cells to 'non'

    ## identify moments of large tdTomato fluorescence change across the session:
    tdTom_absROI        = np.abs(st.zscore(Fchan2data,axis=0)) #get zscored tdtom fluo for rois and take absolute
    Fchan2data          = pd.DataFrame(st.zscore(np.mean(tdTom_absROI,axis=1)),columns=['Fchan2']) #average across ROIs and zscore again

    Ftsdata             = pd.DataFrame(dFdata['ts'], columns=['ts'])
    dFdata              = dFdata.drop('ts',axis=1) #ts was used for alignment, drop, saved separately (Ftsdata)
    deconvdata          = deconvdata.drop('ts',axis=1) 
    assert(np.shape(dFdata)[1]==np.shape(celldata)[0]), '# of cells unequal in cell data and fluo data'

    celldata['session_id']      = sessiondata['session_id'][0] #add session id to celldata as identifier

    return sessiondata,celldata,dFdata,deconvdata,Ftsdata,Fchan2data


"""
 #     # ####### #       ######  ####### ######  ####### #     # #     #  #####   #####  
 #     # #       #       #     # #       #     # #       #     # ##    # #     # #     # 
 #     # #       #       #     # #       #     # #       #     # # #   # #       #       
 ####### #####   #       ######  #####   ######  #####   #     # #  #  # #        #####  
 #     # #       #       #       #       #   #   #       #     # #   # # #             # 
 #     # #       #       #       #       #    #  #       #     # #    ## #     # #     # 
 #     # ####### ####### #       ####### #     # #        #####  #     #  #####   #####  
"""

# np.unique(protocol_tif_nframes)
# np.argwhere(protocol_tif_nframes==7)
# triggerdata       = pd.read_csv(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior',triggerdata_file[0]),skiprows=1).to_numpy()

# pos = 1310
# plt.plot(protocol_tif_nframes[pos-2:pos+4])
# plt.plot(np.diff(triggerdata[:,1])[pos-2:pos+4])

def align_timestamps(sessiondata, ops, triggerdata):
    # get idx of frames belonging to this protocol:
    protocol_tifs           = list(filter(lambda x: sessiondata['protocol'][0] in x, ops['filelist']))
    
    protocol_tif_idx        = np.array([i for i, x in enumerate(ops['filelist']) if x in protocol_tifs])
    #get the number of frames for each of the files belonging to this protocol:
    protocol_tif_nframes    = ops['frames_per_file'][protocol_tif_idx]
    
    protocol_frame_idx = []
    for i in np.arange(len(ops['filelist'])):
        if i in protocol_tif_idx:
            protocol_frame_idx = np.append(protocol_frame_idx,np.repeat(True,ops['frames_per_file'][i]))
        else:
           protocol_frame_idx = np.append(protocol_frame_idx,np.repeat(False,ops['frames_per_file'][i]))
    
    protocol_nframes = sum(protocol_frame_idx).astype('int') #the number of frames acquired in this protocol

    if sessiondata['session_id'][0] == 'LPE12013_2024_04_29': #insert two extra triggers that were aberrant
        pos = 782
        nframes = protocol_tif_nframes[pos]
        triggerdata = np.insert(triggerdata,[pos+1],values=[pos+1,triggerdata[pos,1]+nframes/ops['fs']],axis=0)
        
        pos = 1038
        nframes = protocol_tif_nframes[pos]
        triggerdata = np.insert(triggerdata,[pos+1],values=[pos+1,triggerdata[pos,1]+nframes/ops['fs']],axis=0)
        
        triggerdata[795,1] = triggerdata[795,1] + 0.05
    elif sessiondata['session_id'][0] == 'LPE11998_2024_04_29': #insert two extra triggers that were aberrant
        pos = 1310
        nframes = protocol_tif_nframes[pos]
        triggerdata = np.insert(triggerdata,[pos+1],values=[pos+1,triggerdata[pos,1]+nframes/ops['fs']],axis=0)
        
    ## Get trigger information:
    nTriggers = np.shape(triggerdata)[0]
    nTiffFiles = len(protocol_tif_idx)
    if nTriggers-1 == nTiffFiles:
        triggerdata = triggerdata[1:,:]
        if datetime.strptime(sessiondata['sessiondate'][0],"%Y_%m_%d") > datetime(2024, 1, 16):
            print('First trigger missed, problematic with trigger at right VDAQ channel after Feb 2024')
    elif nTriggers-2 == nTiffFiles:
        triggerdata = triggerdata[2:,:]
        print('First two triggers missed, too slow for scanimage acquisition system')
    nTriggers = np.shape(triggerdata)[0]
    assert nTiffFiles==nTriggers,"Not the same number of tiffs as triggers"

    timestamps = np.empty(protocol_nframes) #init empty array for the timestamps

    #set the timestamps by interpolating the timestamps from the trigger moment to the next:
    for i in np.arange(nTriggers):
        startidx    = sum(protocol_tif_nframes[0:i]) 
        endidx      = startidx + protocol_tif_nframes[i]
        if i>0:
            start_ts    = np.max((timestamps[startidx-1] + 1/ops['fs'],triggerdata[i,1]))
        else: 
            start_ts    = triggerdata[i,1]
        tempts      = np.linspace(start_ts,start_ts+(protocol_tif_nframes[i]-1)*1/ops['fs'],num=protocol_tif_nframes[i])
        timestamps[startidx:endidx] = tempts
    
    # from sklearn.linear_model import LinearRegression
    # reg = LinearRegression().fit(np.reshape(temp,(-1,1)),np.reshape(triggerdata[:,1],(-1,1)))
    # tempts2      = reg.predict(np.reshape(np.arange(protocol_nframes),(-1,1)))

    #Verification of alignment:
    idx         = np.append([0],np.cumsum(protocol_tif_nframes[:]).astype('int64')-1)
    reconstr    = timestamps[idx]
    target      = triggerdata[:,1]
    diffvec     = reconstr[0:len(target)] - target
    h           = np.diff(timestamps)

    if any(h<0) or any(h>1) or any(diffvec>0) or any(diffvec<-1):
        print('Problem with aligning trigger timestamps to imaging frames')
    
    #Check that all interframe intervals are relatively close to imaging frame rate:
    assert np.allclose(1/sessiondata['fs'],np.diff(timestamps,axis=0),rtol=0.3),'Ifis too dissimilar to imaging frame rate'

    return timestamps, protocol_frame_idx


def list_tifs(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            filepath = root + os.sep + name
            if filepath.endswith(".tif"):
                r.append(os.path.join(root, name))
    return r

# Helper function for delta F/F0:
def calculate_dff(F, Fneu, coeff_Fneu=0.7, prc=10): #Rupprecht et al. 2021
    # correct trace for neuropil contamination:
    Fc = F - coeff_Fneu * Fneu + np.median(Fneu,axis=1,keepdims=True)
    # Establish baseline as percentile of corrected trace (50 is median)
    F0 = np.percentile(Fc,prc,axis=1,keepdims=True)
    #Compute dF / F0:
    dFF = (Fc - F0) / F0
    return dFF

def plot_pupil_dist(videodata):
    fig,axes  = plt.subplots(1,3,figsize=(9,3))

    xpos = zscore(videodata['pupil_xpos'])
    ypos = zscore(videodata['pupil_ypos'])
    area = zscore(videodata['pupil_area'])
    axes[0].scatter(xpos,ypos,s=5,alpha=0.1)
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[1].scatter(xpos,area,s=5,alpha=0.1)
    axes[1].set_xlabel('Area')
    axes[1].set_ylabel('X Position')
    axes[2].scatter(ypos,area,s=5,alpha=0.1)
    axes[2].set_xlabel('Area')
    axes[2].set_ylabel('Y Position')
    plt.tight_layout()
    return

def assign_layer(celldata):
    celldata['layer'] = ''

    layers = {
        'V1': {
            'L2/3': (0, 200),
            'L4': (200, 275),
            'L5': (275, np.inf)
        },
        'PM': {
            'L2/3': (0, 200),
            'L4': (200, 275),
            'L5': (275, np.inf)
        },
        'AL': {
            'L2/3': (0, 200),
            'L4': (200, 275),
            'L5': (275, np.inf)
        },
        'RSP': {
            'L2/3': (0, 300),
            'L5': (300, np.inf)
        }
    }

    # layers = {
    #     'V1': {
    #         'L2/3': (0, 250),
    #         'L4': (250, 350),
    #         'L5': (350, np.inf)
    #     },
    #     'PM': {
    #         'L2/3': (0, 250),
    #         'L4': (250, 325),
    #         'L5': (325, np.inf)
    #     },
    #     'AL': {
    #         'L2/3': (0, 250),
    #         'L4': (250, 325),
    #         'L5': (325, np.inf)
    #     },
    #     'RSP': {
    #         'L2/3': (0, 300),
    #         'L5': (300, np.inf)
    #     }
    # }

    for roi, layerdict in layers.items():
        for layer, (mindepth, maxdepth) in layerdict.items():
            idx = celldata[(celldata['roi_name'] == roi) & (mindepth <= celldata['depth']) & (celldata['depth'] < maxdepth)].index
            celldata.loc[idx, 'layer'] = layer
    
    assert(celldata['layer'].notnull().all()), 'problematice assignment of layer based on ROI and depth'
    
    #References: 
    # V1: 
    # Niell & Stryker, 2008 Journal of Neuroscience
    # Gilman, et al. 2017 eNeuro
    # RSC/PM:
    # Zilles 1995 Rat cortex areal and laminar structure

    return celldata

def add_session_bounds(sessiondata,data):
    if 'trialNumber' in data or 'TrialNumber' in data:
        if 'tOnset' in data:
            sessiondata['tStart']       = np.min(data['tOnset']) - 3 #add session start timestamp
            sessiondata['tEnd']         = np.max(data['tOffset']) + 3 #add session stop timestamp
        elif 'tStart' in data:
            sessiondata['tStart']       = np.min(data['tStart']) - 3 #add session start timestamp 
            sessiondata['tEnd']         = np.max(data['tEnd']) + 3 #add session start timestamp
        else: 
            raise ValueError('add_session_bounds: data must have either trialNumber, tOnset, tStart, or ts column')
    else: 
        sessiondata['tStart']       = np.min(data['ts']) #add session start timestamp 
        sessiondata['tEnd']         = np.max(data['ts']) #add session start timestamp 

    return sessiondata

def trim_session_bounds(sessiondata,data):
    # trim data to tStart and tEnd:
    data = data[data['ts']>sessiondata['tStart'][0]].reset_index(drop=True)
    data = data[data['ts']<sessiondata['tEnd'][0]].reset_index(drop=True)
    return data

def proc_roimatchpub(matfname,sessiondata,celldata_plane):

    # rawdatadir_mei      = "M:\\RawData\\"
    # animal_id_mei       = 'LPE12385' #If empty than all animals in folder will be processed
    # sessiondate_mei     = '2024_06_16'

    # matfname = os.path.join(rawdatadir_mei,animal_id_mei,sessiondate_mei,'suite2p','plane0','Fmatch.mat')

    data            = scipy.io.loadmat(matfname)
    mapping         = data['roiMatchData']['allSessionMapping'][0][0]
    nMatchingCells  = np.shape(mapping)[0]

    orig_Fall       = str(data['roiMatchData']['allRois'][0][0][0][0][0])
    tempfname       = orig_Fall.replace('Fall.mat','iscell.npy')
    iscell_ref      = np.load(tempfname)
    ncells_orig     = np.shape(iscell_ref)[0]

    _,_,animal_id_ref,sessiondate_ref,_,iplane_ref,_ = orig_Fall.split('\\')
    iplane_ref              = int(iplane_ref.split('plane')[1])
    old_cell_ids            = np.array([animal_id_ref + '_' + sessiondate_ref + '_' + '%s' % iplane_ref + '_' + '%04.0f' % k for k in range(0,ncells_orig)])

    new_cell_ids            = np.array([sessiondata['session_id'][0] + '_' + '%s' % celldata_plane['plane_idx'][0] + '_' + '%04.0f' % k for k in range(0,len(celldata_plane))])
    #filter only iscell cells
    old_cell_ids_iscell     = old_cell_ids[iscell_ref[:,0]==1]
    new_cell_ids_iscell     = new_cell_ids[celldata_plane['iscell']==1]
    #now get the cell ids for those that are matched
    old_cell_ids_both       = old_cell_ids_iscell[mapping[:,0]]
    new_cell_ids_both       = new_cell_ids_iscell[mapping[:,1]]
    #make a 2D array with columns having the cell ids of the ref and mei session:
    cell_id_mapping         = np.column_stack((old_cell_ids_both,new_cell_ids_both))
    # cell_id_mapping       = np.column_stack((old_cell_ids_iscell[mapping[:,0]],new_cell_ids_iscell[mapping[:,1]]))

    _,comm1,comm2 = np.intersect1d(celldata_plane['cell_id'],cell_id_mapping[:,1],return_indices=True)
    # finding the common values in celldata_plane['cell_id'] and cell_id_mapping[:,1]
    # comm1 are the indices out of all new cell_ids that are in the list of matched neurons
    # comm2 are the indices in the list of matching cell_ids

    celldata_plane['ref_cell_id'] = '' #add the ref cell id to the dataframe
    celldata_plane.loc[comm1,'ref_cell_id'] = old_cell_ids_both[comm2]
    
    assert (celldata_plane['ref_cell_id'] != '').sum() == nMatchingCells, 'problematic assignment of ref_cell_id'
    
    return celldata_plane
