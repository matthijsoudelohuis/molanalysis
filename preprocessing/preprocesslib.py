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
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from utils.twoplib import get_meta
import scipy.stats as st
from loaddata.get_data_folder import get_data_folder
from scipy.stats import zscore
from labeling.tdTom_labeling_cellpose import proc_labeling_plane

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
    # assert(all(ori_counts > 50) and all(ori_counts < 400)) #between 50 and 400 repetitions

    assert(np.allclose(trialdata['tOffset'] - trialdata['tOnset'],0.75,atol=0.1)) #stimulus duration all around 0.75s
    assert(np.allclose(np.diff(trialdata['tOnset']),2,atol=0.1)) #total trial duration all around 2s

    trialdata['session_id']     = sessiondata['session_id'][0]

    return trialdata


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

    return trialdata

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

    return trialdata



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
            sessiondata['signal_center']=np.mean(sigs).round()
            sessiondata['signal_range']=[np.max(sigs) - np.min(sigs)]

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
    lickactivity    = np.diff(behaviordata['lick']) #behaviordata lick is True whenever tongue in ROI>threshold
    lick_ts         = behaviordata['ts'][np.append(lickactivity==1,False)].to_numpy()
    
    ## Rewards, same as licks, get only onset of reward as timestamps
    rewardactivity = np.diff(behaviordata['reward'])
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
    assert np.allclose(sessiondata['stimLength'], trialdata['stimEnd'] - trialdata['stimStart'], rtol=1e-05)
    # assert np.allclose(sessiondata['rewardZoneOffset'], trialdata['rewardZoneStart'] - trialdata['stimStart'], rtol=1e-05)
    assert np.allclose(sessiondata['rewardZoneLength'], trialdata['rewardZoneEnd'] - trialdata['rewardZoneStart'], rtol=1e-05)
    g = trialdata[['trialStart','stimStart','stimEnd','rewardZoneEnd','trialEnd']].to_numpy().flatten()
    assert np.all(np.diff(g)>=0), 'trial event ordering issue'

    #Subsample the data, don't need this high 1000 Hz resolution
    behaviordata = behaviordata.iloc[::10, :].copy().reset_index(drop=True) #subsample data 10 times (to 100 Hz)
    
    behaviordata['lick']    = False #init to false and set to true for sample of first lick or rew
    behaviordata['reward']  = False
    
    #Add the lick times and reward times again to the subsampled dataframe:
    for lick in lick_ts:
        behaviordata.loc[behaviordata.index[np.argmax(lick<behaviordata['ts'])],'lick'] = True
    print("%d licks" % np.sum(behaviordata['lick'])) #Give output to check if reasonable

    if datetime.strptime(sessiondata['sessiondate'][0],"%Y_%m_%d") >= datetime(2024, 4, 16):
        sessiondata['minLicks'] = 3
    else: 
        sessiondata['minLicks'] = 1

    #Assert that licks are not inside the reward zone for trials in which no lick response was recorded:
    for k in range(len(trialdata)):
        idx_rewzone     = np.logical_and(behaviordata['zpos']>trialdata['rewardZoneStart'][k],behaviordata['zpos']<trialdata['rewardZoneEnd'][k])
        idx             = np.logical_and(behaviordata['lick'],idx_rewzone)#.flatten()
        if np.sum(idx)>=sessiondata['minLicks'][0] and trialdata['lickResponse'][k]==False:
            print('%d lick(s) registered in reward zone of trial %d with lickResponse==false' % (np.sum(idx),k))

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

    assert np.isin(framerate,[10,15,30]), 'Error! Frame rate is not 10, 15 or 30 Hz, something wrong with triggering'
    sesdur = behaviordata.loc[behaviordata.index[-1],'ts']  - behaviordata.loc[behaviordata.index[0],'ts'] 
    assert np.isclose(nts,sesdur * framerate,rtol=0.01)
    #Check that frame rate matches interframe interval:
    assert np.isclose(1/framerate,np.mean(np.diff(ts)),rtol=0.01)
    #Check that inter frame interval does not take on crazy values:
    # issues_ts = np.logical_or(np.diff(ts[1:-1])<1/framerate/3,np.diff(ts[1:-1])>1/framerate*2)
    issues_ts = np.concatenate(([False],np.logical_or(np.diff(ts[1:-1])<1/framerate/3,
                                                  np.diff(ts[1:-1])>1/framerate*2),[False,False]))
    if np.any(issues_ts):
        print('Interpolating %d video timestamp issues' % np.sum(issues_ts))
        ts[issues_ts] = np.interp(np.where(issues_ts)[0],np.where(~issues_ts)[0],ts[~issues_ts])
        # Interpolate samples where timestamps are off:
    assert ~np.any(np.logical_or(np.diff(ts[1:-1])<1/framerate/3,np.diff(ts[1:-1])>1/framerate*2))

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

            idx = np.logical_or(np.abs(xpos)>5,np.abs(ypos)>5 ,np.abs(area)>5)
            print('set %1.4f percent of video frames with pupil fit outlier samples to nan \n' % (np.sum(idx) / len(videodata['pupil_xpos'])))
            videodata.iloc[idx,videodata.columns.get_loc("pupil_xpos")] = np.nan
            videodata.iloc[idx,videodata.columns.get_loc("pupil_ypos")] = np.nan
            videodata.iloc[idx,videodata.columns.get_loc("pupil_area")] = np.nan
    else:
        print('#######################  Could not locate facemapdata...')


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

def proc_imaging(sesfolder, sessiondata):
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

    for iplane,plane_folder in enumerate(plane_folders):
    # for iplane,plane_folder in enumerate(plane_folders[:1]):
        print('processing plane %s / %s' % (iplane+1,ops['nplanes']))

        ops                 = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()
        
        [ts_plane, protocol_frame_idx_plane] = align_timestamps(sessiondata, ops, triggerdata)

        iscell              = np.load(os.path.join(plane_folder, 'iscell.npy'))
        stat                = np.load(os.path.join(plane_folder, 'stat.npy'), allow_pickle=True)
        
        if os.path.exists(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane)):
            redcell_seg         = np.load(os.path.join(plane_folder,'redim_plane%d_seg.npy' %iplane), allow_pickle=True).item()
            masks_cp_red        = redcell_seg['masks']
            Nredcells_plane     = len(np.unique(masks_cp_red))-1 # number of labeled cells overall, minus 1 because 0 for all nonlabeled pixels
            redcell = proc_labeling_plane(iplane,plane_folder,showcells=False,overlap_threshold=0.5)
        else: 
            print('\n\n Warning: cellpose results not found, setting labeling to zero\n\n')
            redcell             = np.zeros((len(iscell),3))
            Nredcells_plane     = 0

        ncells_plane              = len(iscell)
        
        celldata_plane            = pd.DataFrame()

        celldata_plane            = celldata_plane.assign(iscell        = iscell[:,0])
        celldata_plane            = celldata_plane.assign(iscell_prob   = iscell[:,1])

        celldata_plane            = celldata_plane.assign(skew          = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(chan2_prob    = np.empty([ncells_plane,1]))
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
            celldata_plane['xloc'][k] = stat[k]['med'][0]
            celldata_plane['yloc'][k] = stat[k]['med'][1]
        
        celldata_plane['redcell']           = redcell[:,0]
        celldata_plane['frac_of_ROI_red']   = redcell[:,1]
        celldata_plane['frac_red_in_ROI']   = redcell[:,2]
        celldata_plane['nredcells']         = Nredcells_plane

        celldata_plane['plane_idx']     = iplane
        celldata_plane['roi_idx']       = plane_roi_idx[iplane]
        celldata_plane['plane_in_roi_idx']       = np.where(np.where(plane_roi_idx==plane_roi_idx[iplane])[0] == iplane)[0][0]
        celldata_plane['roi_name']      = roi_area[plane_roi_idx[iplane]]
        celldata_plane['depth']         = plane_zs[iplane] - sessiondata['ROI%d_dura' % (plane_roi_idx[iplane]+1)][0]
        #compute power at this plane: formula: P = P0 * exp^((z-z0)/Lz)
        celldata_plane['power_mw']      = sessiondata['SI_pz_power'][0]  * math.exp((plane_zs[iplane] - sessiondata['SI_pz_reference'][0])/sessiondata['SI_pz_constant'][0])

        if os.path.exists(os.path.join(plane_folder, 'RF.npy')):
            RF = np.load(os.path.join(plane_folder, 'RF.npy'))
            celldata_plane['rf_azimuth']    = RF[0,:]
            celldata_plane['rf_elevation']  = RF[1,:]
            celldata_plane['rf_size']       = RF[2,:]
            celldata_plane['rf_p']          = RF[3,:]

        ##################### load suite2p activity outputs:
        F                   = np.load(os.path.join(plane_folder, 'F.npy'), allow_pickle=True)
        F_chan2             = np.load(os.path.join(plane_folder, 'F_chan2.npy'), allow_pickle=True)
        Fneu                = np.load(os.path.join(plane_folder, 'Fneu.npy'), allow_pickle=True)
        spks                = np.load(os.path.join(plane_folder, 'spks.npy'), allow_pickle=True)
        
        if np.shape(F_chan2)[0] < np.shape(F)[0]:
            print('ROIs were manually added in suite2p, fabricating red channel data...')
            F_chan2     = np.vstack((F_chan2, np.tile(F_chan2[[-1],:], 1)))

        # Correct neuropil and compute dF/F: (Rupprecht et al. 2021)
        dF     = calculate_dff(F, Fneu,coeff_Fneu=0.7,prc=10) #see function below

        # Compute average fluorescence on green and red channels:
        celldata_plane['meanF']         = np.mean(F, axis=1)
        celldata_plane['meanF_chan2']   = np.mean(F_chan2, axis=1)

        # Calculate the noise level of the cells ##### Rupprecht et al. 2021 Nat Neurosci.
        celldata_plane['noise_level'] = np.median(np.abs(np.diff(dF,axis=1)),axis=1)/np.sqrt(ops['fs'])

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

        #Filter only good cells
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
        dFdata_plane['timestamps']      = ts_master    #add timestamps
        deconvdata_plane                = pd.DataFrame(spks, columns=cell_ids)   
        deconvdata_plane['timestamps']  = ts_master    #add timestamps
        Fchan2data_plane                = pd.DataFrame(F_chan2, columns=cell_ids)
        Fchan2data_plane['timestamps']  = ts_master    #add timestamps
        #Fchan2data is not saved but average across neurons, see below
        
        if iplane == 0:
            dFdata = dFdata_plane.copy()
            deconvdata = deconvdata_plane.copy()
            Fchan2data = Fchan2data_plane.copy()
        else:
            dFdata = dFdata.merge(dFdata_plane)
            deconvdata = deconvdata.merge(deconvdata_plane)
            Fchan2data = Fchan2data.merge(Fchan2data_plane)
    
    #If ROI is unnamed, replace if ROI_1/V1 combi, ROI_2/PM combi, otherwise error:
    if celldata['roi_name'].str.contains('ROI').any():
        print('An imaging area was not named in scanimage')
        if celldata['roi_name'].isin(['PM']).any():
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI_2','V1')
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI 2','V1')
        if celldata['roi_name'].isin(['V1']).any():
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI_1','PM')
            celldata['roi_name'] = celldata['roi_name'].str.replace('ROI 1','PM')
        assert not celldata['roi_name'].str.contains('ROI').any(),'unknown area'

    #Add recombinase enzym label to red cells:
    labelareas = ['V1','PM']
    for area in labelareas:
        temprecombinase =  area + '_recombinase'
        celldata.loc[celldata['roi_name']==area,'recombinase'] = sessiondata[temprecombinase].to_list()[0]
    celldata.loc[celldata['redcell']==0,'recombinase'] = 'non' #set all nonlabeled cells to 'non'

    # Correct for suite2p artefact where first roi is a cell, but metrics are wrong. Based on skew or npix_soma:
    # celldata.iloc[np.where(celldata['npix_soma']<5)[0],celldata.columns.get_loc('iscell')] = 0
    # celldata.iloc[np.where(celldata['skew']<5)[0],celldata.columns.get_loc('iscell')] = 0

    ## identify moments of large tdTomato fluorescence change across the session:
    tdTom_absROI    = np.abs(st.zscore(Fchan2data,axis=0)) #get zscored tdtom fluo for rois and take absolute
    tdTom_meanZ     = st.zscore(np.mean(tdTom_absROI,axis=1)) #average across ROIs and zscore again
    
    dFdata['F_chan2']           = tdTom_meanZ.to_numpy() #store in dFdata and deconvdata
    deconvdata['F_chan2']       = tdTom_meanZ.to_numpy()

    celldata['session_id']      = sessiondata['session_id'][0]
    dFdata['session_id']        = sessiondata['session_id'][0]
    deconvdata['session_id']    = sessiondata['session_id'][0]


    return sessiondata,celldata,dFdata,deconvdata


"""
 #     # ####### #       ######  ####### ######  ####### #     # #     #  #####   #####  
 #     # #       #       #     # #       #     # #       #     # ##    # #     # #     # 
 #     # #       #       #     # #       #     # #       #     # # #   # #       #       
 ####### #####   #       ######  #####   ######  #####   #     # #  #  # #        #####  
 #     # #       #       #       #       #   #   #       #     # #   # # #             # 
 #     # #       #       #       #       #    #  #       #     # #    ## #     # #     # 
 #     # ####### ####### #       ####### #     # #        #####  #     #  #####   #####  
"""

def align_timestamps(sessiondata, ops, triggerdata):
    # get idx of frames belonging to this protocol:
    protocol_tifs           = list(filter(lambda x: sessiondata['protocol'][0] in x, ops['filelist']))
    # if sessiondata['session_id'][0] == 'LPE11622_2024_03_07' and sessiondata['protocol'][0] == 'SP4':
        # protocol_tifs = protocol_tifs[306:]

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

    ## Get trigger information:
    nTriggers = np.shape(triggerdata)[0]
    nTiffFiles = len(protocol_tif_idx)
    if nTriggers-1 == nTiffFiles:
        triggerdata = triggerdata[1:,:]
        if datetime.strptime(sessiondata['sessiondate'][0],"%Y_%m_%d") > datetime(2024, 1, 1):
            print('First trigger missed, problematic with trigger at right VDAQ channel in 2024')

    elif nTriggers-2 == nTiffFiles:
        triggerdata = triggerdata[2:,:]
        print('First two triggers missed, too slow for scanimage acquisition system')
    nTriggers = np.shape(triggerdata)[0]
    assert nTiffFiles==nTriggers,"Not the same number of tiffs as triggers"

    timestamps = np.empty([protocol_nframes,1]) #init empty array for the timestamps

    #set the timestamps by interpolating the timestamps from the trigger moment to the next:
    for i in np.arange(nTriggers):
        startidx    = sum(protocol_tif_nframes[0:i]) 
        endidx      = startidx + protocol_tif_nframes[i]
        start_ts    = triggerdata[i,1]
        tempts      = np.linspace(start_ts,start_ts+(protocol_tif_nframes[i]-1)*1/ops['fs'],num=protocol_tif_nframes[i])
        timestamps[startidx:endidx,0] = tempts
        
    #Verification of alignment:
    idx         = np.append([0],np.cumsum(protocol_tif_nframes[:]).astype('int64')-1)
    reconstr    = timestamps[idx,0]
    target      = triggerdata[:,1]
    diffvec     = reconstr[0:len(target)] - target
    h           = np.diff(timestamps[:,0])
    if any(h<0) or any(h>1) or any(diffvec>0) or any(diffvec<-1):
        print('Problem with aligning trigger timestamps to imaging frames')
        
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