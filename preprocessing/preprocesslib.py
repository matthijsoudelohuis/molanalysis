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
# from ScanImageTiffReader import ScanImageTiffReader as imread
from twoplib import get_meta

        
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
    sessiondata         = sessiondata.assign(sex = ["Male"])
    sessiondata         = sessiondata.assign(lab = ["Petreanu Lab"])
    sessiondata         = sessiondata.assign(institution = ["Champalimaud Research"])
    sessiondata         = sessiondata.assign(preprocessdate = [datetime.now().strftime("%Y_%m_%d")])
    sessiondata         = sessiondata.assign(protocol = [protocol])

    if protocol in ['IM','GR','RF','SP']:
        sessions_overview = pd.read_excel(os.path.join(rawdatadir,'VISTA_Sessions_Overview.xlsx'))
    elif protocol in ['VR']: 
        sessions_overview = pd.read_excel(os.path.join(rawdatadir,'VR_Sessions_Overview.xlsx'))

    idx = np.where(np.logical_and(sessions_overview["sessiondate"] == sessiondate,sessions_overview["protocol"] == protocol))[0]
    sessiondata = pd.merge(sessiondata,sessions_overview.loc[idx])

    age_in_days = (datetime.strptime(sessiondata['sessiondate'][0], "%Y_%m_%d") - datetime.strptime(sessiondata['DOB'][0], "%Y_%m_%d")).days
    
    sessiondata         = sessiondata.assign(age_in_days = [age_in_days])

    return sessiondata

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
    behaviordata = behaviordata.drop(columns="rawvoltage") #remove rawvoltage, not used
    behaviordata = behaviordata.iloc[::10, :].reset_index(drop=True) #subsample data 10 times (to 100 Hz)

    return behaviordata

def proc_GR(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
    
    return trialdata

def proc_IM(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
    
    return trialdata

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
    
    grid_array                      = np.reshape(grid_array, [nGrids,xGrid,yGrid])
    grid_array                      = np.transpose(grid_array, [1,2,0])
    grid_array = np.rot90(grid_array, k=1, axes=(0,1))
    
    grid_array[grid_array==-1]       = 1
    grid_array[grid_array==0]       = -1
    grid_array[grid_array==-128]    = 0
    
    # fig, ax = plt.subplots(figsize=(7, 3))
    # ax.imshow(grid_array[:,:,0].transpose(), aspect='auto',cmap='gray')
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file

    if not len(trialdata_file)==0 and os.path.exists(os.path.join(sesfolder,trialdata_file[0])):
        trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
        RF_timestamps   = trialdata.iloc[:,1].to_numpy()

    else: ## Get trigger data to align ts_master:
        triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
        triggerdata       = pd.read_csv(os.path.join(sesfolder,triggerdata_file[0]),skiprows=2).to_numpy()
        
        #rework from last timestamp: triggerdata[1,-1]
        RF_timestamps = np.linspace(triggerdata[-1,1]-nGrids*0.5, triggerdata[-1,1], num=nGrids, endpoint=True)
    
    return grid_array,RF_timestamps

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
    
    #If rewarded stimulus is on the left, context = 1, right = 0
    blocklen = 50
    temp = np.repeat(np.concatenate([np.ones(blocklen),np.zeros(blocklen)]),100).astype('int')
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
    
    #Subsample the data, don't need this resolution
    behaviordata = behaviordata.iloc[::10, :].copy().reset_index(drop=True) #subsample data 10 times (to 100 Hz)
    
    behaviordata['lick'] = False
    behaviordata['reward'] = False
    
    #Now add the lick times and reward times again to the subsampled dataframe:
    for lick in lick_ts:
        # behaviordata['lick'][np.argmax(lick<behaviordata['ts'])] = True
        behaviordata.loc[behaviordata.index[np.argmax(lick<behaviordata['ts'])],'lick'] = True
        
    # print("%d licks" % len(lick_ts)) #Give output to check if reasonable
    print("%d licks" % np.sum(behaviordata['lick'])) #Give output to check if reasonable

    for reward in reward_ts:
        # behaviordata['reward'][np.argmax(reward<behaviordata['ts'])] = True
        behaviordata.loc[behaviordata.index[np.argmax(reward<behaviordata['ts'])],'reward'] = True
    
    # print("%d rewards" % len(reward_ts)) #Give output to check if reasonable
    print("%d rewards" % np.sum(behaviordata['reward'])) #Give output to check if reasonable

    return sessiondata, trialdata, behaviordata

def proc_imaging(sesfolder, sessiondata):
    """ integrate preprocessed calcium imaging data """
    
    suite2p_folder = os.path.join(sesfolder,"suite2p")
    
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])

    # load ops of plane0:
    ops                = np.load(os.path.join(plane_folders[0], 'ops.npy'), allow_pickle=True).item()
    
    # read metadata from tiff (just take first tiff from the filelist
    # metadata should be same for all if settings haven't changed during differernt protocols
    meta, meta_si   = get_meta(ops['filelist'][0])
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

    #lists, not really necessary:
    # sessiondata = sessiondata.assign(SI_zdepths                 = meta_dict['SI.hStackManager.zs'])
    # sessiondata = sessiondata.assign(SI_axesPosition            = meta_dict['SI.hMotors.axesPosition'])

    ## Get trigger data to align timestamps:
    filenames           = os.listdir(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior'))
    triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
    triggerdata       = pd.read_csv(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior',triggerdata_file[0]),skiprows=2).to_numpy()

    [ts_master, protocol_frame_idx_master] = align_timestamps(sessiondata, ops, triggerdata)

    # getting numer of ROIs
    nROIs = len(meta['RoiGroups']['imagingRoiGroup']['rois'])
    #Find the names of the rois:
    roi_area    = [meta['RoiGroups']['imagingRoiGroup']['rois'][i]['name'] for i in range(nROIs)]
    #Find the depths of the planes for each roi:
    roi_depths  = np.array([meta['RoiGroups']['imagingRoiGroup']['rois'][i]['zs'] for i in range(nROIs)]) #numpy array of depths for each roi
    #get all the depths of the planes in order of imaging:
    plane_zs    = np.array(meta_dict['SI.hStackManager.zs'].replace('[','').replace(']','').split(' ')).astype('int')
    #Find the roi to which each plane belongs:
    plane_roi_idx  = np.array([np.where(roi_depths == plane_zs[i])[0][0] for i in range(8)])

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

        if iplane == 0: #if first plane then init dataframe, otherwise append
            celldata = celldata_plane.copy()
        else:
            celldata = celldata.append(celldata_plane)
            
        #load suite2p activity outputs:
        F                   = np.load(os.path.join(plane_folder, 'F.npy'), allow_pickle=True)
        Fneu                = np.load(os.path.join(plane_folder, 'Fneu.npy'), allow_pickle=True)
        spks                = np.load(os.path.join(plane_folder, 'spks.npy'), allow_pickle=True)
        # Compute dF/F:
        dF              = F - 0.7*Fneu
        dF              = calc_dF(dF, ops['baseline'], ops['win_baseline'], 
                                ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])
        
        F = F[:,protocol_frame_idx_plane==1].transpose()
        Fneu = Fneu[:,protocol_frame_idx_plane==1].transpose()
        spks = spks[:,protocol_frame_idx_plane==1].transpose()
        dF = dF[:,protocol_frame_idx_plane==1].transpose()
        
        # if imaging was aborted during scanning of a volume, later planes have less frames
        # Compensate by duplicating last value
        if np.shape(F)[0]==len(ts_master):
            pass       #do nothing, shapes match
        elif np.shape(F)[0]==len(ts_master)-1: #copy last timestamp of array
            F           = np.vstack((F, np.tile(F[[-1],:], 1)))
            Fneu        = np.vstack((Fneu, np.tile(Fneu[[-1],:], 1)))
            spks        = np.vstack((spks, np.tile(spks[[-1],:], 1)))
            dF          = np.vstack((dF, np.tile(dF[[-1],:], 1)))
        else:
            print("Problem with timestamps and imaging frames")
 
        #construct dataframe with activity by cells: give unique cell_id as label:
        cell_ids            = list(sessiondata['session_id'][0] + '_' + '%s' % iplane + '_' + '%s' % k for k in range(0,ncells_plane))
        # calciumdata_plane   = pd.DataFrame(F, columns=cell_ids)
        calciumdata_plane   = pd.DataFrame(spks, columns=cell_ids)
        calciumdata_plane['timestamps']   = ts_master    #add timestamps

        
        if iplane == 0:
            calciumdata = calciumdata_plane.copy()
        else:
            calciumdata = calciumdata.merge(calciumdata_plane)
            
    #Finally set which cells are labeled with tdTomato: 
    # celldata["labeled"] = celldata['chan2_prob'] > 0.75

    return sessiondata,celldata,calciumdata

def align_timestamps(sessiondata, ops, triggerdata):
    # get idx of frames belonging to this protocol:
    protocol_tifs       = list(filter(lambda x: sessiondata['protocol'][0] in x, ops['filelist']))
    protocol_tif_idx    = np.array([i for i, x in enumerate(ops['filelist']) if x in protocol_tifs])
    
    protocol_tif_nframes    = ops['frames_per_file'][protocol_tif_idx]
    
    protocol_frame_idx = []
    for i in np.arange(len(ops['filelist'])):
        if i in protocol_tif_idx:
            protocol_frame_idx = np.append(protocol_frame_idx,np.repeat(True,ops['frames_per_file'][i]))
        else:
           protocol_frame_idx = np.append(protocol_frame_idx,np.repeat(False,ops['frames_per_file'][i]))
    
    protocol_nframes = sum(protocol_frame_idx).astype('int')
    
    ## Get trigger information:
    nTriggers = np.shape(triggerdata)[0]
    # timestamps = np.empty([ops['nframes'],1]) #init empty array for the timestamps
    timestamps = np.empty([protocol_nframes,1]) #init empty array for the timestamps

    for i in np.arange(nTriggers):
        startidx    = sum(protocol_tif_nframes[0:i]) 
        endidx      = startidx + protocol_tif_nframes[i]
        start_ts    = triggerdata[i,1]
        tempts      = np.linspace(start_ts,start_ts+(protocol_tif_nframes[i]-1)*1/ops['fs'],num=protocol_tif_nframes[i])
        timestamps[startidx:endidx,0] = tempts
        
    #Verification of alignment:
    idx = np.append([0],np.cumsum(protocol_tif_nframes[:]).astype('int64')-1)
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