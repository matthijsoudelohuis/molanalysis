"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script contains a series of preprocessing functions that take raw data
(behavior, task, microscope, video) and preprocess them. Is called by preprocess_main.
Principally the data is integrated with additional info and stored for pandas dataframe usage
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

    sessions_overview_VISTA = pd.read_excel(os.path.join(rawdatadir,'VISTA_Sessions_Overview.xlsx'))
    sessions_overview_VR    = pd.read_excel(os.path.join(rawdatadir,'VR_Sessions_Overview.xlsx'))

    if np.any(np.logical_and(sessions_overview_VISTA["sessiondate"] == sessiondate,sessions_overview_VISTA["protocol"] == protocol)):
        sessions_overview = sessions_overview_VISTA
    elif np.any(np.logical_and(sessions_overview_VR["sessiondate"] == sessiondate,sessions_overview_VR["protocol"] == protocol)):
        sessions_overview = sessions_overview_VR
    else: 
        print('Session not found in excel session overview')
    
    idx =   (sessions_overview["sessiondate"] == sessiondate) & \
            (sessions_overview["animal_id"] == animal_id) & \
            (sessions_overview["protocol"] == protocol)
    if np.any(idx):
        sessiondata         = pd.merge(sessiondata,sessions_overview.loc[idx],'inner') #Copy all the data from the excel to sessiondata dataframe
        age_in_days         = (datetime.strptime(sessiondata['sessiondate'][0], "%Y_%m_%d") - datetime.strptime(sessiondata['DOB'][0], "%Y_%m_%d")).days
        sessiondata         = sessiondata.assign(age_in_days = [age_in_days]) #Store the age in days at time of the experiment
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

    # Piece of debug code for chunks
    # prepost = 900 #sometimes data is saved double: delete ths overlapping part
    # import matplotlib.pyplot as plt
    # for i in idx:
    #     plt.figure()
    #     plt.plot(behaviordata['ts'][i-prepost:i+prepost].to_numpy())
    #     plt.plot(behaviordata['ts'][i],'k.')
    #     # plt.plot(behaviordata['runspeed'][i-prepost:i+prepost])
    #     plt.plot(behaviordata['ts'][i-prepost:i+prepost],behaviordata['runspeed'][i-prepost:i+prepost]) 
    #     plt.plot(behaviordata['ts'][i],behaviordata['runspeed'][i],'k.') 

    #subsample data 10 times (to 100 Hz)
    behaviordata = behaviordata.iloc[::10, :].reset_index(drop=True) 

    #some checks:
    # if sessiondata['session_id'][0] not in ['LPE09665_2023_03_15','LPE09665_2023_03_20']:
    # assert(np.allclose(np.diff(behaviordata['ts']),1/100,rtol=0.1)) #timestamps ascending and around sampling rate
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
 ######  #######    #     #    #    ######  ######  ### #     #  #####  
 #     # #          ##   ##   # #   #     # #     #  #  ##    # #     # 
 #     # #          # # # #  #   #  #     # #     #  #  # #   # #       
 ######  #####      #  #  # #     # ######  ######   #  #  #  # #  #### 
 #   #   #          #     # ####### #       #        #  #   # # #     # 
 #    #  #          #     # #     # #       #        #  #    ## #     # 
 #     # #          #     # #     # #       #       ### #     #  #####  
"""

def proc_RF(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    
    filenames       = os.listdir(sesfolder)
    
    log_file        = list(filter(lambda a: 'log' in a, filenames)) #find the trialdata file
    
    #RF_log.bin
    #The vector saved is long GridSize(1)xGridSize(2)x(RunTime/Duration)
    #where RunTime is the total display time of the Bonsai programme.
    #The file format is .binary data with int8 data format
    with open(os.path.join(sesfolder,log_file[0]) , 'rb') as fid:
        grid_array = np.fromfile(fid, np.int8)
    
    xGrid           = 52
    yGrid           = 13
    nGrids          = 1800
    
    nGrids_emp = int(len(grid_array)/xGrid/yGrid)
    if nGrids_emp != nGrids:
        if np.isclose(len(grid_array)/xGrid/yGrid,nGrids,atol=1):
            nGrids          = nGrids_emp
            print('\n####### One grid too many or too few.... Correcting for it.\n')
        else:
            print('\n####### Problem with number of grids in receptive field mapping\n')

    grid_array                      = np.reshape(grid_array, [nGrids,xGrid,yGrid])
    grid_array                      = np.transpose(grid_array, [1,2,0])
    grid_array = np.rot90(grid_array, k=1, axes=(0,1))
    
    grid_array[grid_array==-1]       = 1
    grid_array[grid_array==0]       = -1
    grid_array[grid_array==-128]    = 0
    
    # fig, ax = plt.subplots(figsize=(7, 3))
    # ax.imshow(grid_array[:,:,0], aspect='auto',cmap='gray')
    # ax.imshow(grid_array[:,:,-1], aspect='auto',cmap='gray')
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file

    if not len(trialdata_file)==0 and os.path.exists(os.path.join(sesfolder,trialdata_file[0])):
        trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
        RF_timestamps   = trialdata.iloc[:,1].to_numpy()

    else: ## Get trigger data to align ts_master:
        print('Interpolating timestamps because trigger data is missing for the receptive field stimuli')
        triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
        triggerdata       = pd.read_csv(os.path.join(sesfolder,triggerdata_file[0]),skiprows=2).to_numpy()
        
        #rework from last timestamp: triggerdata[1,-1]
        RF_timestamps = np.linspace(triggerdata[-1,1]-nGrids*0.5, triggerdata[-1,1], num=nGrids, endpoint=True)
    
    return grid_array,RF_timestamps

"""
 #     # ######     #######    #     #####  #    # 
 #     # #     #       #      # #   #     # #   #  
 #     # #     #       #     #   #  #       #  #   
 #     # ######        #    #     #  #####  ###    
  #   #  #   #         #    #######       # #  #   
   # #   #    #        #    #     # #     # #   #  
    #    #     #       #    #     #  #####  #    # 
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
    
    #Code stimuli simply as A, B, C, etc.:
    trialdata = trialdata.replace('stim','', regex=True)

    trialdata['lickResponse'] = trialdata['lickResponse'].astype(int)
    trialdata = trialdata.rename(columns={'Reward': 'rewardAvailable'})
    trialdata['rewardGiven']  = np.logical_and(trialdata['rewardAvailable'],trialdata['lickResponse']).astype('int')

    #If rewarded stimulus is on the left, context = 1, right = 0
    blocklen = 50
    temp = np.tile(np.concatenate([np.ones(blocklen),np.zeros(blocklen)]),100).astype('int')
    if np.argmax(trialdata['stimLeft'] == sessiondata['gostim'][0])<blocklen:    #identify first block left or right:
        trialdata['context'] = temp[:len(trialdata)]
    else:
        trialdata['context'] = 1-temp[:len(trialdata)]
        
    #Add which trial number it is relative to block switch:
    temp        = np.tile(range(blocklen),100).astype('int')
    trialdata['n_in_block']         = temp[:len(trialdata)]
    
    #Add what the stimuli are normalized across mice: here 0 is the rewarded stimulus, 
    # and the rest relative to that one (wrapped numbering, so with C=reward, C=0,D=1,A=2,B=3)
    temp    = np.array([ord(trialdata['stimLeft'][itrial]) - 65 for itrial in range(len(trialdata))])
    trialdata['stimLeft_norm'] = np.mod(temp - (ord(sessiondata['gostim'][0]) - 65),4)
    temp    = np.array([ord(trialdata['stimRight'][itrial]) - 65 for itrial in range(len(trialdata))])
    trialdata['stimRight_norm'] = np.mod(temp - (ord(sessiondata['gostim'][0]) - 65),4)
    
    #Get behavioral data, construct dataframe and modify a bit:
    behaviordata        = pd.read_csv(os.path.join(sesfolder,harpdata_file[0]),skiprows=0)
    behaviordata.columns = ["rawvoltage","ts","trialnum","zpos","runspeed","lick","reward"] #rename the columns
    behaviordata = behaviordata.drop(columns="rawvoltage") #remove rawvoltage, not used
    
    ## Licks
    lickactivity    = np.diff(behaviordata['lick'])
    lick_ts         = behaviordata['ts'][np.append(lickactivity==1,False)].to_numpy()
    
    ## Rewards
    rewardactivity = np.diff(behaviordata['reward'])
    reward_ts      = behaviordata['ts'][np.append(rewardactivity>0,False)].to_numpy()
    
    ## Reconstruct total position:
    trialdata['TrialEnd'] = 0
    for t in range(len(trialdata)):
        trialdata.loc[trialdata.index[trialdata['TrialNumber']==t+1],'TrialEnd'] = max(behaviordata.loc[behaviordata.index[behaviordata['trialnum']==t],'zpos'])

    behaviordata['zpos_tot'] = behaviordata['zpos']
    for t in range(len(trialdata)):
        behaviordata.loc[behaviordata.index[behaviordata['trialnum']==t+1],'zpos_tot'] += np.cumsum(trialdata['TrialEnd'])[t]

    trialdata['StimStart_tot']          = trialdata['StimStart'] + np.cumsum(trialdata['TrialEnd'])
    trialdata['StimEnd_tot']            = trialdata['StimEnd'] + np.cumsum(trialdata['TrialEnd'])
    trialdata['RewardZoneStart_tot']    = trialdata['RewardZoneStart'] + np.cumsum(trialdata['TrialEnd'])
    trialdata['RewardZoneEnd_tot']      = trialdata['RewardZoneEnd'] + np.cumsum(trialdata['TrialEnd'])
    
    #Subsample the data, don't need this resolution
    behaviordata = behaviordata.iloc[::10, :].copy().reset_index(drop=True) #subsample data 10 times (to 100 Hz)
    
    behaviordata['lick']    = False #init to false and set to true for sample of first lick or rew
    behaviordata['reward']  = False
    
    #Now add the lick times and reward times again to the subsampled dataframe:
    for lick in lick_ts:
        # behaviordata['lick'][np.argmax(lick<behaviordata['ts'])] = True
        behaviordata.loc[behaviordata.index[np.argmax(lick<behaviordata['ts'])],'lick'] = True
    
    print("%d licks" % np.sum(behaviordata['lick'])) #Give output to check if reasonable

    trialdata['tStart'] = np.concatenate(([behaviordata['ts'][0]],trialdata['tEnd'][1:]))

    #Add the timestamps of entering and exiting stimulus zone:
    trialdata['tStimStart'] = ''
    trialdata['tStimEnd'] = ''
    for t in range(len(trialdata)):
        idx             = behaviordata['trialnum']==t+1
        z_temp          = behaviordata.loc[behaviordata.index[idx],'zpos'].to_numpy()
        ts_temp         = behaviordata.loc[behaviordata.index[idx],'ts'].to_numpy()
        try:
            tStimStart      = ts_temp[np.where(z_temp >= trialdata.loc[trialdata.index[t], 'StimStart'])[0][0]]
        except:
            tStimStart        = trialdata.loc[trialdata.index[t], 'tStart']
            print('Stimulus start later than trial end')
        trialdata.loc[trialdata.index[t], 'tStimStart'] = tStimStart
        
        try:
            tStimEnd        = ts_temp[np.where(z_temp >= trialdata.loc[trialdata.index[t], 'StimEnd'])[0][0]]
        except:
            tStimEnd        = trialdata.loc[trialdata.index[t], 'tEnd']
            print('Stimulus end later than trial end')
        trialdata.loc[trialdata.index[t], 'tStimEnd'] = tStimEnd

    for reward in reward_ts: #set only the first timestamp of reward to True, to have single indices
        behaviordata.loc[behaviordata.index[np.argmax(reward<behaviordata['ts'])],'reward'] = True
    
    #Compute the timestamp and spatial location of the reward being given and store in trialdata:
    trialdata['tReward'] = np.nan
    trialdata['sReward'] = np.nan
    for t in range(len(trialdata)-1):
        idx = np.logical_and(behaviordata['reward'],[behaviordata['trialnum']==t+1]).flatten()
        if np.any(idx):
            trialdata.loc[trialdata.index[t],'tReward'] = behaviordata['ts'].iloc[np.where(idx)[0][0]]
            trialdata.loc[trialdata.index[t],'sReward'] = behaviordata['zpos'].iloc[np.where(idx)[0][0]]

    #Compute reward rate (fraction of GO trials rewarded) with sliding window for engagement index:
    sliding_window = 24
    rewardrate_thr = 0.3
    trialdata['engaged'] = 1
    # for t in range(sliding_window,len(trialdata)):
    for t in range(len(trialdata)):
        idx = np.intersect1d(np.arange(len(trialdata)),np.arange(t-sliding_window/2,t+sliding_window/2))
        hitrate = np.sum(trialdata['rewardGiven'][idx]) / np.sum(trialdata['rewardAvailable'][idx])
        if hitrate < rewardrate_thr:
            trialdata.loc[trialdata.index[t],'engaged'] = 0

    assert(np.all(~np.isnan(trialdata['tReward'][trialdata['rewardGiven']==1]))) #check all rewarded trials have timestamp of reward

    # print("%d rewards" % len(reward_ts)) #Give output to check if reasonable
    print("%d rewards" % np.sum(behaviordata['reward'])) #Give output to check if reasonable
    print("%d rewards in unique trials" % trialdata['tReward'].count()) #Give output to check if reasonable

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

    videodata       = pd.DataFrame(data = ts, columns = ['timestamps'])

    #Check that the number of frames is ballpark range of what it should be based on framerate and session duration:
    
    if datetime.strptime(sessiondata['sessiondate'][0],"%Y_%m_%d") > datetime(2023, 3, 30):
        
        framerate = 30
        sesdur = behaviordata.loc[behaviordata.index[-1],'ts']  - behaviordata.loc[behaviordata.index[0],'ts'] 
        assert np.isclose(nts,sesdur * framerate,rtol=3)
        #Check that frame rate matches interframe interval:
        assert np.isclose(1/framerate,np.mean(np.diff(ts)),rtol=0.01)
        #Check that inter frame interval does not take on crazy values:
        assert ~np.any(np.logical_or(np.diff(ts[1:-1])<1/framerate/3,np.diff(ts[1:-1])>1/framerate*2))
    else:
        framerate = np.round(1/np.mean(np.diff(ts)))
        idx = np.where(np.diff(ts[1:-1])<1/framerate/3)[0] + 2
        for i in idx:
            ts[i] = np.mean((ts[i-1],ts[i+1]))
        idx = np.where(np.diff(ts[1:-1])>1/framerate*2)[0] + 2
        for i in idx:
            ts[i] = np.mean((ts[i-1],ts[i+1]))
    
    #Load FaceMap data: 
    facemapfile =  list(filter(lambda a: '_proc' in a, filenames)) #find the processed facemap file
    if facemapfile and len(facemapfile)==1 and os.path.exists(facemapfile[0]):
        # facemapfile = "W:\\Users\\Matthijs\\Rawdata\\NSH07422\\2023_03_13\\SP\\Behavior\\SP_NSH07422_camera_2023-03-13T16_44_07_proc.npy"
        # facemapfile = "W:\\Users\\Matthijs\\Rawdata\\LPE09829\\2023_03_29\\VR\\Behavior\\VR_LPE09829_camera_2023-03-29T15_32_29_proc.npy"
        
        proc = np.load(facemapfile[0],allow_pickle=True).item()
        
        assert(len(proc['motion'][0])==0,'multivideo performed, should not be done')
        assert(len(proc['rois'])==2,'designed to analyze 2 rois, pupil and motion svd, _proc file contains a different #rois')
        assert(all(x in [proc['rois'][i]['rtype'] for i in range(2)] for x in ['motion SVD', 'Pupil']),'roi type error')

        videodata['motionenergy']   = proc['motion'][1]
        PC_labels                   = list('videoPC_' + '%s' % k for k in range(0,keepPCs))
        videodata = pd.concat([videodata,pd.DataFrame(proc['motSVD'][1][:,:keepPCs],columns=PC_labels)],axis=1)
        
        #Pupil data:
        videodata['pupil_area']   = proc['pupil'][0]['area_smooth']
        videodata['pupil_ypos']   = proc['pupil'][0]['com'][:,0]
        videodata['pupil_xpos']   = proc['pupil'][0]['com'][:,1]
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
    triggerdata       = pd.read_csv(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior',triggerdata_file[0]),skiprows=2).to_numpy()
    #skip two rows because the first is init of the variable, and the second????
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
    plane_zs    = np.array(meta_dict['SI.hStackManager.zs'].replace('[','').replace(']','').split(' ')).astype('int')
    
    #Find the roi to which each plane belongs:
    plane_roi_idx = np.array([roi_depths_idx[np.where(roi_depths == plane_zs[i])[0][0]] for i in range(ops['nplanes'])])

    for iplane,plane_folder in enumerate(plane_folders):
    # for iplane,plane_folder in enumerate(plane_folders[:1]):
        print('processing plane %s / %s' % (iplane+1,ops['nplanes']))

        ops                 = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()
        
        [ts_plane, protocol_frame_idx_plane] = align_timestamps(sessiondata, ops, triggerdata)

        iscell              = np.load(os.path.join(plane_folder, 'iscell.npy'))
        stat                = np.load(os.path.join(plane_folder, 'stat.npy'), allow_pickle=True)
        redcell             = np.load(os.path.join(plane_folder, 'redcell.npy'), allow_pickle=True)

        ncells_plane              = len(iscell)
        
        celldata_plane            = pd.DataFrame()

        celldata_plane            = celldata_plane.assign(iscell = iscell[:,0])
        celldata_plane            = celldata_plane.assign(iscell_prob = iscell[:,1])

        celldata_plane            = celldata_plane.assign(skew          = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(chan2_prob    = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(radius        = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(npix_soma     = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(npix          = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(xloc          = np.empty([ncells_plane,1]))
        celldata_plane            = celldata_plane.assign(yloc          = np.empty([ncells_plane,1]))

        for k in range(1,ncells_plane):
            celldata_plane['skew'][k] = stat[k]['skew']
            celldata_plane['radius'][k] = stat[k]['radius']
            celldata_plane['npix_soma'][k] = stat[k]['npix_soma']
            celldata_plane['npix'][k] = stat[k]['npix']
            celldata_plane['xloc'][k] = stat[k]['med'][0]
            celldata_plane['yloc'][k] = stat[k]['med'][1]
        
        celldata_plane['redcell_prob']  = redcell[:,1]
        celldata_plane['redcell']       = redcell[:,0]

        celldata_plane['plane_idx']     = iplane
        celldata_plane['roi_idx']       = plane_roi_idx[iplane]
        celldata_plane['plane_in_roi_idx']       = np.where(np.where(plane_roi_idx==plane_roi_idx[iplane])[0] == iplane)[0][0]
        celldata_plane['roi_name']      = roi_area[plane_roi_idx[iplane]]
        celldata_plane['depth']         = plane_zs[iplane] - sessiondata['ROI%d_dura' % (plane_roi_idx[iplane]+1)][0]
        #compute power at this plane: formula: P = P0 * exp^((z-z0)/Lz)
        celldata_plane['power_mw']      = sessiondata['SI_pz_power'][0]  * math.exp((plane_zs[iplane] - sessiondata['SI_pz_reference'][0])/sessiondata['SI_pz_constant'][0])

        if os.path.exists(os.path.join(plane_folder, 'RF.npy')):
            RF = np.load(os.path.join(plane_folder, 'F.npy'))
            celldata_plane['rf_azimuth']    = RF[:,0]
            celldata_plane['rf_elevation']  = RF[:,1]
            celldata_plane['rf_size']       = RF[:,2]

        ##################### load suite2p activity outputs:
        F                   = np.load(os.path.join(plane_folder, 'F.npy'), allow_pickle=True)
        F_chan2             = np.load(os.path.join(plane_folder, 'F_chan2.npy'), allow_pickle=True)
        Fneu                = np.load(os.path.join(plane_folder, 'Fneu.npy'), allow_pickle=True)
        spks                = np.load(os.path.join(plane_folder, 'spks.npy'), allow_pickle=True)
        
        #!#!@#!$!#%#%#!$@!#%!^%@!$!#$ 

        if np.shape(F_chan2)[0] < np.shape(F)[0]:
            F_chan2     = np.vstack((F_chan2, np.tile(F_chan2[[-1],:], 1))).sleknf()
#!#!@#!$!#%#%#!$@!#%!^%@!$!#$ 

        # Compute dF/F:
        dF     = calculate_dff(F, Fneu,prc=10) #see function below

        # Compute average fluorescence on green and red channels:
        celldata_plane['meanF']         = np.mean(F, axis=1)
        celldata_plane['meanF_chan2']   = np.mean(F_chan2, axis=1)

        # Calculate the noise level of the cells ##### Rupprecht et al. 2021 Nat Neurosci.
        noise_level         = np.median(np.abs(np.diff(dF,axis=1)),axis=1)/np.sqrt(ops['fs'])
        celldata_plane['noise_level'] = noise_level

        #Count the number of events by taking stretches with z-scored activity above 2:
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
        cell_ids            = list(sessiondata['session_id'][0] + '_' + '%s' % iplane + '_' + '%s' % k for k in range(0,ncells_plane))
        
        #store cell_ids in celldata:
        celldata_plane['cell_id']         = cell_ids

        if iplane == 0: #if first plane then init dataframe, otherwise append
            celldata = celldata_plane.copy()
        else:
            celldata = celldata.append(celldata_plane)
            
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
    
    # Correct for suite2p artefact if never opened suite2p to set 0th cell to not iscell based on npix size:
    celldata.iloc[np.where(celldata['npix']==1)[0],celldata.columns.get_loc('iscell')] = 0

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
    assert np.shape(protocol_tif_nframes)[0]==nTriggers,"Not the same number of tiffs as triggers"

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
def calculate_dff(F, Fneu, prc=20): #Rupprecht et al. 2021
    # correct trace for neuropil contamination:
    Fc = F - 0.7 * Fneu + np.median(Fneu,axis=1,keepdims=True)
    # Establish baseline as percentile of corrected trace (50 is median)
    F0 = np.percentile(Fc,prc,axis=1,keepdims=True)
    #Compute dF / F0: 
    dFF = (Fc - F0) / F0
    return dFF

def calc_dF(F: np.ndarray, baseline: str, win_baseline: float,
               sig_baseline: float, fs: float, prctile_baseline: float = 8) -> np.ndarray:
    """ preprocesses fluorescence traces for spike deconvolution

    baseline-subtraction with window 'win_baseline'
    
    Parameters
    ----------------

    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence

    baseline : str
        setting that describes how to compute the baseline of each trace

    win_baseline : float
        window (in seconds) for max filter

    sig_baseline : float
        width of Gaussian filter in seconds

    fs : float
        sampling rate per plane

    prctile_baseline : float
        percentile of trace to use as baseline if using `constant_prctile` for baseline
    
    Returns
    ----------------

    F : float, 2D array
        size [neurons x time], baseline-corrected fluorescence

    """
    win = int(win_baseline*fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(F,    [0., sig_baseline])
        Flow = minimum_filter1d(Flow,    win)
        Flow = maximum_filter1d(Flow,    win)
    elif baseline == 'constant':
        Flow = gaussian_filter(F,    [0., sig_baseline])
        Flow = np.amin(Flow)
    elif baseline == 'constant_prctile':
        Flow = np.percentile(F, prctile_baseline, axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.

    F = F - Flow

    return F

 # dF              = F - 0.7*Fneu
        # dF              = calc_dF(dF, ops['baseline'], ops['win_baseline'], 
        #                         ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])