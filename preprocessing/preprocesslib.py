"""
Created on Mon Dec 12 11:05:43 2022
@author: USER
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from natsort import natsorted 
from datetime import datetime
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from ScanImageTiffReader import ScanImageTiffReader as imread

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
    behaviordata = behaviordata.iloc[::10, :].reset_index() #subsample data 10 times (to 100 Hz)

    return behaviordata

def proc_GR(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=0)
    
    return trialdata

def proc_RF(rawdatadir,sessiondata):
    sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')
    sesfolder       = Path(sesfolder)
    
    filenames       = os.listdir(sesfolder)
    
    log_file        = list(filter(lambda a: 'log' in a, filenames)) #find the trialdata file
    
    # log_file        = os.path.join(sesfolder,log_file)
    
    return sesfolder,log_file

def proc_behavior_vr(rawdatadir,animal_id,sessiondate,protocol):
    """ preprocess all the trial, stimulus and behavior data for one session """
    
    sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate,protocol,'Behavior')
    sesfolder       = Path(sesfolder)
    
    #Init output dataframes:
    trialdata       = pd.DataFrame()
    behaviordata    = pd.DataFrame()


    #Process behavioral data:

# ([pd.DataFrame([i], columns=['A']) for i in range(5)],
          # ignore_index=True)
    
    filenames       = os.listdir(sesfolder)
    
    harpdata_file   = list(filter(lambda a: 'harp' in a, filenames)) #find the harp files
    harpdata_file   = list(filter(lambda a: 'csv'  in a, harpdata_file)) #take the csv file, not the rawharp bin
    # harpdata        = pd.read_csv(os.path.join(sesfolder,harpdata_file[0]),skiprows=1).to_numpy()
    harpdata        = pd.read_csv(os.path.join(sesfolder,harpdata_file[0]),skiprows=0)
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    # trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=1).to_numpy()
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=1)
    
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),sep=",")
    
    trialdata       = pd.read_csv('W:/Users/Matthijs/Rawdata/NSH07429/2023_03_12/IM/Behavior/IM_NSH07429_trialdata_2023-03-12T15_14_35.csv')
    trialdata       = pd.read_csv('X:/RawData/NSH07422/2022_12_08/VR/Behavior/VR_NSH07422trialdata2022-12-08T16_18_09.csv')

    ## Start storing and processing the rawdata in the NWB session file:
    timestamps      = harpdata[:,1].astype(np.float64)
    
    behaviordata
    
    # ## Wheel voltage
    # time_series_with_timestamps = TimeSeries(
    # name            = "WheelVoltage",
    # description     = "Raw voltage from wheel rotary encoder",
    # data            = harpdata[:,0].astype(np.float64),
    # unit            = "V",
    # timestamps      = timestamps,
    # )
    # nwbfile.add_acquisition(time_series_with_timestamps)
    
    # ## Z position
    # time_series_with_timestamps = TimeSeries(
    # name            = "CorridorPosition",
    # description     = "z position along the corridor",
    # data            = harpdata[:,3].astype(np.float64),
    # unit            = "cm",
    # timestamps      = timestamps,
    # )
    # nwbfile.add_acquisition(time_series_with_timestamps)
        
    # ## Running speed
    # time_series_with_timestamps = TimeSeries(
    # name            = "RunningSpeed",
    # description     = "Speed of VR wheel rotation",
    # data            = harpdata[:,4].astype(np.float64),
    # unit            = "cm s-1",
    # timestamps      = timestamps,
    # )
    # nwbfile.add_acquisition(time_series_with_timestamps)
    
    # ## Wheel voltage
    # time_series_with_timestamps = TimeSeries(
    # name            = "TrialNumber",
    # description     = "During which trial number the other acquisition channels were sampled",
    # data            = harpdata[:,2].astype(np.int64),
    # unit            = "na",
    # timestamps      = timestamps,
    # )
    # nwbfile.add_acquisition(time_series_with_timestamps)
    
    ## Licks
    lickactivity    = np.diff(harpdata[:,5])
    lickactivity    = np.append(lickactivity,0)
    idx             = lickactivity==1
    print("%d licks" % idx.sum()) #Give output to check if reasonable
    
    # time_series = TimeSeries(
    #     name        = "Licks",
    #     data        = np.ones([idx.sum(),1]),
    #     timestamps  = timestamps[idx],
    #     description = "When luminance of tongue crossed a threshold at an ROI at the lick spout",
    #     unit        = "a.u.",
    # )
    
    # lick_events = BehavioralEvents(time_series=time_series, name="Licks")
    # behavior_module.add(lick_events)

    ## Rewards
    rewardactivity = np.diff(harpdata[:,6])
    rewardactivity = np.append(rewardactivity,0)
    idx = rewardactivity>0
    print("%d rewards" % idx.sum()) #Give output to check if reasonable
    
    # time_series = TimeSeries(
    #     name        = "Rewards",
    #     data        = np.ones([idx.sum(),1])*5,
    #     timestamps  = timestamps[idx],
    #     description = "Rewards delivered at lick spout",
    #     unit        = "uL",
    # )
    # reward_events = BehavioralEvents(time_series=time_series, name="Rewards")
    # behavior_module.add(reward_events)
    
    # ##Trial information
    # nwbfile.add_trial_column(name='trialnum', description='the number of the trial in this session')    # Add a column to the trial table.
    # nwbfile.add_trial_column(name='trialtype', description='G=go, N=nogo')
    # nwbfile.add_trial_column(name='rewardtrial', description='Whether licking this trial is rewarded')
    # nwbfile.add_trial_column(name='outcome', description='string describing outcome of trial HIT MISS FA CR')
    # nwbfile.add_trial_column(name='lickresponse', description='whether the animal licked in the reward zone')
    # nwbfile.add_trial_column(name='nlicks', description='number of licks within the reward zone')
    # nwbfile.add_trial_column(name='stimstart', description='Start of the stimulus in the corridor')
    # nwbfile.add_trial_column(name='stimstop', description='End of the stimulus in the corridor')
    # nwbfile.add_trial_column(name='rewardzonestart', description='Start of the response zone in the corridor')
    # nwbfile.add_trial_column(name='rewardzonestop', description='End of the response zone in the corridor')
    # nwbfile.add_trial_column(name='stimleft', description='the visual stimuli during the trial')
    # nwbfile.add_trial_column(name='stimright', description='the visual stimuli during the trial')

    # #Add trials to the trial table:
    # itrial=0 #for the first trial take time stamp from the start of the session
    # nwbfile.add_trial(start_time=harpdata[0,1],                stop_time=trialdata[itrial,2]+10, 
    #                      trialnum=trialdata[itrial,1],         trialtype=trialdata[itrial,3], 
    #                      rewardtrial=trialdata[itrial,4],      outcome=trialdata[itrial,0],
    #                      lickresponse=trialdata[itrial,8],     nlicks=trialdata[itrial,9],
    #                      stimstart=trialdata[itrial,5],        stimstop=trialdata[itrial,5]+30,
    #                      rewardzonestart=trialdata[itrial,6],  rewardzonestop=trialdata[itrial,7],
    #                      stimleft=trialdata[itrial,10],        stimright=trialdata[itrial,11])

    # for itrial in range(1,len(trialdata)):

    #     nwbfile.add_trial(start_time=trialdata[itrial-1,2],     stop_time=trialdata[itrial,2], 
    #                       trialnum=trialdata[itrial,1],         trialtype=trialdata[itrial,3], 
    #                       rewardtrial=trialdata[itrial,4],      outcome=trialdata[itrial,0],
    #                       lickresponse=trialdata[itrial,8],     nlicks=trialdata[itrial,9],
    #                       stimstart=trialdata[itrial,5],        stimstop=trialdata[itrial,5]+30,
    #                       rewardzonestart=trialdata[itrial,6],  rewardzonestop=trialdata[itrial,7],
    #                       stimleft=trialdata[itrial,10],        stimright=trialdata[itrial,11])
           
    return sessiondata, trialdata, behaviordata



def proc_imaging(sesfolder, sessiondata):
    """ integrate preprocessed calcium imaging data """
    
    suite2p_folder = os.path.join(sesfolder,"suite2p")
    
    plane_folders = natsorted([f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])

    # load ops of plane0:
    ops                = np.load(os.path.join(plane_folders[0], 'ops.npy'), allow_pickle=True).item()
    
    f = ops['filelist'][0]
    reader = imread(f)
    
#     # read meta
#     lines = reader.metadata().split('\n')
#     split_ix = lines.index('')

# for i in range(n_frames):
#     # the parser
#     t  = np.float32(reader.description(i).split('\n')[3].split(' = ')[1])
#     t_stamps[i] = t

# return t_stamps

# for line in meta_si:
#     if line.startswith(key):
#         return float(line.split(' = ')[1])
#     flyToTime = read_float_from_meta(meta_si,"SI.hScan2D.flytoTimePerScanfield")
#     linePeriod = read_float_from_meta(meta_si,"SI.hRoiManager.linePeriod")
    
#     # getting individual ROIsizes
#     nROIs = len(meta['RoiGroups']['imagingRoiGroup']['rois'])
#     xpx = []
#     ypx = []
#     for i in range(nROIs):
#         xpx_, ypx_ = meta['RoiGroups']['imagingRoiGroup']['rois'][i]['scanfields']['pixelResolutionXY']
#         xpx.append(xpx_)
#         ypx.append(ypx_)
        

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
    
    # triggerdata = np.empty([0,2])
    # for protocol in ['IM','GR','RF','SP']:
    #      ## Get trigger data to align timestamps:
    #      if os.path.exists(os.path.join(sesfolder,protocol,'Behavior')):
    #         filenames           = os.listdir(os.path.join(sesfolder,protocol,'Behavior'))
    #         triggerdata_file    = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
    #         triggerdata         = np.append(triggerdata,pd.read_csv(os.path.join(sesfolder,protocol,'Behavior',triggerdata_file[0]),skiprows=2).to_numpy())

    ## Get trigger data to align timestamps:
    filenames           = os.listdir(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior'))
    triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
    triggerdata       = pd.read_csv(os.path.join(sesfolder,sessiondata['protocol'][0],'Behavior',triggerdata_file[0]),skiprows=2).to_numpy()

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
    
    # for iplane,plane_folder in enumerate(plane_folders):
    #         print(5)
    
    
    iplane = 0
    plane_folder = plane_folders[0]
    for iplane,plane_folder in enumerate(plane_folders):
        ops                 = np.load(os.path.join(plane_folder, 'ops.npy'), allow_pickle=True).item()
        
        iscell              = np.load(os.path.join(plane_folder, 'iscell.npy'))
        stat                = np.load(os.path.join(plane_folder, 'stat.npy'), allow_pickle=True)

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
            celldata_plane['chan2_prob'][k] = stat[k]['chan2_prob']
            celldata_plane['radius'][k] = stat[k]['radius']
            celldata_plane['npix_soma'][k] = stat[k]['npix_soma']
            celldata_plane['npix'][k] = stat[k]['npix']
            celldata_plane['xloc'][k] = stat[k]['med'][0]
            celldata_plane['yloc'][k] = stat[k]['med'][1]
        
        celldata_plane['chan2_prob'] = stat[k]['med'][1]

        if iplane == 0:
            celldata = celldata_plane.copy()
        else:
            celldata = celldata.merge(celldata_plane)
        
        F                   = np.load(os.path.join(plane_folder, 'F.npy'), allow_pickle=True)
        Fneu                = np.load(os.path.join(plane_folder, 'Fneu.npy'), allow_pickle=True)
        spks                = np.load(os.path.join(plane_folder, 'spks.npy'), allow_pickle=True)

        # Construct dF/F:
        dF      = F - 0.7*Fneu
        
        dF      = calc_dF(dF, ops['baseline'], ops['win_baseline'], 
                                ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])
 
        calciumdata_plane   = pd.DataFrame()
        
        cell_ids = list(sessiondata['session_id'][0] + '_' + '%s' % iplane + '_' + '%s' % k for k in range(0,ncells_plane))
        calciumdata_plane = pd.DataFrame(F[:,protocol_frame_idx==1].transpose(), columns=cell_ids)

        calciumdata_plane = pd.DataFrame(np.random.rand(ncells_plane,50), columns=cell_ids)

        if iplane == 0:
            calciumdata = calciumdata_plane.copy()
        else:
            calciumdata = calciumdata.merge(calciumdata_plane)
            
        
    celldata["labeled"] = celldata["chan2_prob"] > 0.75


    
    return celldata,calciumdata

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