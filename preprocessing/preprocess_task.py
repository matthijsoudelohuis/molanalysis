import os
from pathlib import Path
# from preprocesslib.py import *
# import preprocesslib
import pandas as pd
import numpy as np
import tifffile

from natsort import natsorted 

from pynwb import NWBFile, TimeSeries, NWBHDF5IO
# from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.behavior import BehavioralEvents
from datetime import datetime
from dateutil import tz

# from pynwb.device import Device
from pynwb.ophys import OpticalChannel
from pynwb.ophys import TwoPhotonSeries
from pynwb.ophys import ImageSegmentation
from pynwb.ophys import RoiResponseSeries
from pynwb.ophys import Fluorescence
    
from pynwb.base import Images
from pynwb.image import GrayscaleImage

   
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter


rawdatadir      = "V:\\Rawdata\\"
rawdatadir      = "X:\\Rawdata\\"
procdatadir     = "V:\\Procdata\\"

animal_ids          = ['NSH07422'] #If empty than all animals in folder will be processed
# sessiondates        = ['2022_11_30',
#  '2022_12_1',
#  '2022_12_2',
#  '2022_12_5',
#  '2022_12_6',
#  '2022_12_7',
#  '2022_12_8'] #If empty than all animals in folder will be processed
sessiondates        = ['2022_12_9']

def proc_behavior(rawdatadir,animal_id,sessiondate,protocol):
    """ preprocess all the trial, stimulus and behavior data for one session """
    
    [y,m,d] = sessiondate.split('_')
    y = int(y)
    m = int(m)
    d = int(d)
    session_start_time = datetime(y, m, d, 2, 30, 3, tzinfo=tz.gettz("Europe/Lisbon"))
    
    nwbfile = NWBFile(
    session_description="MouseVirtualCorridor",  # required
    identifier          = animal_id + "_" + sessiondate,  # required
    session_start_time  = session_start_time,  # required
    # session_id        = "session_1234",  # optional
    experimenter        = "Matthijs Oude Lohuis",  # optional
    lab                 = "Petreanu Lab",  # optional
    institution         = "Champalimaud Research",  # optional
    )
    
    nwbfile.subject = Subject(
    subject_id      = animal_id,
    # age           = "P90D",
    # description   = "mouse 5",
    species         = "Mus musculus",
    # sex           = "M",
    )
    
    behavior_module = nwbfile.create_processing_module(
    name="behavior", description="processed behavioral data"
    )

    sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate,protocol)
    sesfolder       = Path(sesfolder)
    # os.chdir(folder)

    filenames = os.listdir(sesfolder)
    
    harpdata_file   = list(filter(lambda a: 'harp' in a, filenames)) #find the harp files
    harpdata_file   = list(filter(lambda a: 'csv'  in a, harpdata_file)) #take the csv file, not the rawharp bin
    harpdata        = pd.read_csv(os.path.join(sesfolder,harpdata_file[0]),skiprows=1).to_numpy()
    
    trialdata_file  = list(filter(lambda a: 'trialdata' in a, filenames)) #find the trialdata file
    trialdata       = pd.read_csv(os.path.join(sesfolder,trialdata_file[0]),skiprows=1).to_numpy()

    ## Start storing and processing the rawdata in the NWB session file:
    timestamps = harpdata[:,1].astype(np.float64)
    
    ## Wheel voltage
    time_series_with_timestamps = TimeSeries(
    name            = "WheelVoltage",
    description     = "Raw voltage from wheel rotary encoder",
    data            = harpdata[:,0].astype(np.float64),
    unit            = "V",
    timestamps      = timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
    
    ## Z position
    time_series_with_timestamps = TimeSeries(
    name            = "CorridorPosition",
    description     = "z position along the corridor",
    data            = harpdata[:,3].astype(np.float64),
    unit            = "cm",
    timestamps      = timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
        
    ## Running speed
    time_series_with_timestamps = TimeSeries(
    name            = "RunningSpeed",
    description     = "Speed of VR wheel rotation",
    data            = harpdata[:,4].astype(np.float64),
    unit            = "cm s-1",
    timestamps      = timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
    
    ## Wheel voltage
    time_series_with_timestamps = TimeSeries(
    name            = "TrialNumber",
    description     = "During which trial number the other acquisition channels were sampled",
    data            = harpdata[:,2].astype(np.int64),
    unit            = "na",
    timestamps      = timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
    
    ## Licks
    lickactivity    = np.diff(harpdata[:,5])
    lickactivity    = np.append(lickactivity,0)
    idx             = lickactivity==1
    print("%d licks" % idx.sum()) #Give output to check if reasonable
    
    time_series = TimeSeries(
        name        = "Licks",
        data        = np.ones([idx.sum(),1]),
        timestamps  = timestamps[idx],
        description = "When luminance of tongue crossed a threshold at an ROI at the lick spout",
        unit        = "a.u.",
    )
    
    lick_events = BehavioralEvents(time_series=time_series, name="Licks")
    behavior_module.add(lick_events)

    ## Rewards
    rewardactivity = np.diff(harpdata[:,6])
    rewardactivity = np.append(rewardactivity,0)
    idx = rewardactivity>0
    print("%d rewards" % idx.sum()) #Give output to check if reasonable
    
    time_series = TimeSeries(
        name        = "Rewards",
        data        = np.ones([idx.sum(),1])*5,
        timestamps  = timestamps[idx],
        description = "Rewards delivered at lick spout",
        unit        = "uL",
    )
    reward_events = BehavioralEvents(time_series=time_series, name="Rewards")
    behavior_module.add(reward_events)
    
    ##Trial information
    nwbfile.add_trial_column(name='trialnum', description='the number of the trial in this session')    # Add a column to the trial table.
    nwbfile.add_trial_column(name='trialtype', description='G=go, N=nogo')
    nwbfile.add_trial_column(name='rewardtrial', description='Whether licking this trial is rewarded')
    nwbfile.add_trial_column(name='outcome', description='string describing outcome of trial HIT MISS FA CR')
    nwbfile.add_trial_column(name='lickresponse', description='whether the animal licked in the reward zone')
    nwbfile.add_trial_column(name='nlicks', description='number of licks within the reward zone')
    nwbfile.add_trial_column(name='stimstart', description='Start of the stimulus in the corridor')
    nwbfile.add_trial_column(name='stimstop', description='End of the stimulus in the corridor')
    nwbfile.add_trial_column(name='rewardzonestart', description='Start of the response zone in the corridor')
    nwbfile.add_trial_column(name='rewardzonestop', description='End of the response zone in the corridor')
    nwbfile.add_trial_column(name='stimleft', description='the visual stimuli during the trial')
    nwbfile.add_trial_column(name='stimright', description='the visual stimuli during the trial')

    #Add trials to the trial table:
    itrial=0 #for the first trial take time stamp from the start of the session
    nwbfile.add_trial(start_time=harpdata[0,1],                stop_time=trialdata[itrial,2]+10, 
                         trialnum=trialdata[itrial,1],         trialtype=trialdata[itrial,3], 
                         rewardtrial=trialdata[itrial,4],      outcome=trialdata[itrial,0],
                         lickresponse=trialdata[itrial,8],     nlicks=trialdata[itrial,9],
                         stimstart=trialdata[itrial,5],        stimstop=trialdata[itrial,5]+30,
                         rewardzonestart=trialdata[itrial,6],  rewardzonestop=trialdata[itrial,7],
                         stimleft=trialdata[itrial,10],        stimright=trialdata[itrial,11])

    for itrial in range(1,len(trialdata)):

        nwbfile.add_trial(start_time=trialdata[itrial-1,2],     stop_time=trialdata[itrial,2], 
                          trialnum=trialdata[itrial,1],         trialtype=trialdata[itrial,3], 
                          rewardtrial=trialdata[itrial,4],      outcome=trialdata[itrial,0],
                          lickresponse=trialdata[itrial,8],     nlicks=trialdata[itrial,9],
                          stimstart=trialdata[itrial,5],        stimstop=trialdata[itrial,5]+30,
                          rewardzonestart=trialdata[itrial,6],  rewardzonestop=trialdata[itrial,7],
                          stimleft=trialdata[itrial,10],        stimright=trialdata[itrial,11])
           
    return nwbfile

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

def proc_imaging(sesfolder,nwbfile):
    """ integrate preprocessed calcium imaging data """
    ## Get imaging data from each ROI:
    imagingdir      = os.path.join(sesfolder,"Imaging")
    ROIdir          = os.path.join(imagingdir,"ROI_0")
    suite2p_folder = os.path.join(ROIdir,"suite2p")

    plane_folders = natsorted([ f.path for f in os.scandir(suite2p_folder) if f.is_dir() and f.name[:5]=='plane'])
    ops1                = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item() for f in plane_folders]

    ops                 = ops1[0]
    ops['save_path']    = 'X:/RawData/NSH07422/2022_12_9/VR/Imaging\\ROI_0\\suite2p\\plane0'
    ifi                 = 1 / ops['fs']
    
    ## Get trigger data to align timestamps:
    filenames       = os.listdir(sesfolder)
    triggerdata_file  = list(filter(lambda a: 'triggerdata' in a, filenames)) #find the trialdata file
    triggerdata       = pd.read_csv(os.path.join(sesfolder,triggerdata_file[0]),skiprows=2).to_numpy()

    ## Get the number of frames from all the tiff files:
    filenames       = os.listdir(ROIdir)
    nfiles          = len(filenames)
    nframesfile     = np.empty([nfiles,1])
    for i,x in enumerate(filenames):
        if x.endswith(".tif"):
            nframesfile[i,0] = len(tifffile.TiffFile(os.path.join(ROIdir,x)).pages)
            
    ## Get number of frames from the collected tiff page numbers:
    nchannels       = 2
    nframesfile     = (nframesfile/nchannels).astype('int64')
    nframes         = sum(nframesfile[:,0])

    #Check whether number of frames from tiff files matches deconvolved ca trace number of samples:
    print(nframes == ops['nframes'])

    ## Get trigger infomration:
    nTriggers = np.shape(triggerdata)[0]
    timestamps = np.empty([nframes,1])

    for i in range(nTriggers):
        startidx    = sum(nframesfile[0:i,0]) 
        endidx      = startidx + nframesfile[i,0]
        start_ts    = triggerdata[i,1]
        tempts      = np.linspace(start_ts,start_ts+(nframesfile[i,0]-1)*ifi,num=nframesfile[i,0])
        timestamps[startidx:endidx,0] = tempts

    #Verification of alignment:
    idx = np.append([0],np.cumsum(nframesfile[:,0]).astype('int64')-1)
    reconstr    = timestamps[idx,0]
    target      = triggerdata[:,1]
    diffvec     = reconstr[0:len(target)] - target
    h           = np.diff(timestamps[:,0])
    
    ########
    
    multiplane = False
    
    device = nwbfile.create_device(
        name='2p-ram Mesoscope', 
        description='large FOV two-photon microscope',
        manufacturer='Janelia / Thorlabs / Vidrio / MBFbiotech'
    )
    optical_channel = OpticalChannel(
        name='Laser', 
        description='IR laser', 
        emission_lambda=920.)
    
    imaging_plane = nwbfile.create_imaging_plane(
            name='ImagingPlane_ROI_0',
            optical_channel=optical_channel,
            imaging_rate=21.42,
            description='standard',
            device=device,
            excitation_lambda=600.,
            indicator='GCaMP6s',
            location='V1', #IMPORTANT!!! GET AREA
            grid_spacing=([2.0,2.0,30.0]), #if multiplane else [2.0,2.0]),
            grid_spacing_unit='microns'
        )
    
    #   link to external data
    image_series = TwoPhotonSeries(
        name='TwoPhotonSeries', 
        dimension=[ops['Ly'], ops['Lx']],
        external_file=(ops['filelist'] if 'filelist' in ops else ['']), 
        imaging_plane=imaging_plane,
        # starting_frame=[0],
        starting_frame=np.zeros(np.shape((ops['filelist'] if 'filelist' in ops else ['']))).astype('int64'),
        format='external', 
        starting_time=0.0, 
        rate=ops['fs'] * ops['nplanes']
    )
    nwbfile.add_acquisition(image_series)
        
    # processing
    img_seg = ImageSegmentation()
    ps = img_seg.create_plane_segmentation(
        name='PlaneSegmentation',
        description='suite2p output',
        imaging_plane=imaging_plane,
        reference_images=image_series
    )

    ophys_module = nwbfile.create_processing_module(
        name='ophys', 
        description='optical physiology processed data'
    )
    ophys_module.add(img_seg)
    
    file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
    traces = []
    ncells_all = 0
    Nfr = np.array([ops['nframes'] for ops in ops1]).max()

    for iplane, ops in enumerate(ops1):
        if iplane==0:
            iscell = np.load(os.path.join(ops['save_path'], 'iscell.npy'))
            for fstr in file_strs:
                traces.append(np.load(os.path.join(ops['save_path'], fstr)))
        else:
            iscell = np.append(iscell, np.load(os.path.join(ops['save_path'], 'iscell.npy')), axis=0)
            for i,fstr in enumerate(file_strs):
                trace = np.load(os.path.join(ops['save_path'], fstr))
                if trace.shape[1] < Nfr:
                    fcat    = np.zeros((trace.shape[0],Nfr-trace.shape[1]), 'float32')
                    trace   = np.concatenate((trace, fcat), axis=1)
                traces[i] = np.append(traces[i], trace, axis=0) 
        
        stat = np.load(os.path.join(ops['save_path'], 'stat.npy'), allow_pickle=True)
        ncells = len(stat)
        for n in range(ncells):
            if multiplane:
                pixel_mask = np.array([stat[n]['ypix'], stat[n]['xpix'], 
                                    iplane*np.ones(stat[n]['npix']), 
                                    stat[n]['lam']])
                ps.add_roi(voxel_mask=pixel_mask.T)
            else:
                pixel_mask = np.array([stat[n]['ypix'], stat[n]['xpix'], 
                                    stat[n]['lam']])
                ps.add_roi(pixel_mask=pixel_mask.T)
        ncells_all+=ncells

    ps.add_column('iscell', 'two columns - iscell & probcell', iscell)
        
    rt_region = ps.create_roi_table_region(
        # region=list(np.arange(0, ncells_all)),
        region=list(np.arange(0, ncells_all).transpose()),
        description='all ROIs'
    )

    # FLUORESCENCE (all are required)
    file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
    name_strs = ['Fluorescence', 'Neuropil', 'Deconvolved']
    
    # Construct dF/F:
    F       = traces[0]
    Fneu    = traces[1]
    dF      = F.copy() - 0.7*Fneu
    
    dF      = calc_dF(dF, ops['baseline'], ops['win_baseline'], 
                           ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])
    traces.append(dF)
    
    file_strs = ['F.npy', 'Fneu.npy', 'spks.npy','dF_F.npy']
    name_strs = ['Fluorescence', 'Neuropil', 'Deconvolved','dF_F']

    for i, (fstr,nstr) in enumerate(zip(file_strs, name_strs)):
        roi_resp_series = RoiResponseSeries(
            name=nstr,
            # data=np.transpose(traces[i]),
            # data=traces[i],
            data=np.transpose(traces[i]).astype('float64'),
            # rois=rt_region,
            rois=rt_region,
            unit='lumens',
            # rate=ops['fs'],
            # starting_time = timestamps[0,0],
            timestamps=np.transpose(timestamps[:,0])
            # timestamps=timestamps[:,0]
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
        ophys_module.add(fl)

    # BACKGROUNDS
    # (meanImg, Vcorr and max_proj are REQUIRED)
    bg_strs = ['meanImg', 'Vcorr', 'max_proj', 'meanImg_chan2']
    nplanes = ops['nplanes']
    for iplane in range(nplanes):
        images = Images('Backgrounds_%d'%iplane)
        for bstr in bg_strs:
            if bstr in ops:
                if bstr=='Vcorr' or bstr=='max_proj':
                    img = np.zeros((ops['Ly'], ops['Lx']), np.float32)
                    img[ops['yrange'][0]:ops['yrange'][-1], 
                        ops['xrange'][0]:ops['xrange'][-1]] = ops[bstr]
                else:
                    img = ops[bstr]
                images.add_image(GrayscaleImage(name=bstr, data=img))
            
        ophys_module.add(images)

    # nwbfile.add_acquisition(ophysnwbfile.acquisition['TwoPhotonSeries'])

    #nwbfile.create_imaging_plane(ophysnwbfile.imaging_planes['ImagingPlane'])
    
    # ophys_module.add(ophysnwbfile.processing['ophys']['ImageSegmentation'])
    # ophys_module.add(ophysnwbfile.processing['ophys']['Backgrounds_0'])
    # ophys_module.add(ophysnwbfile.processing['ophys']['Deconvolved'])
    # ophys_module.add(ophysnwbfile.processing['ophys']['Fluorescence'])
    # ophys_module.add(ophysnwbfile.processing['ophys']['Neuropil'])
    
    return nwbfile

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(rawdatadir)

for animal_id in animal_ids: #for each animal
    
    if len(sessiondates) == 0:
        sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

    for sessiondate in sessiondates: #for each of the sessions for this animal
        nwbfile         = proc_behavior(rawdatadir,animal_id,sessiondate,"VR") #main processing function
        sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate,"VR")

        if os.path.exists(os.path.join(sesfolder,"Imaging")):
            print('Detected imaging data\n')
            nwbfile         = proc_imaging(sesfolder,nwbfile) #main processing function for imaging data
        
        savefilename    = animal_id + "_" + sessiondate + "_VR.nwb" #define save file name
        outdir          = os.path.join(procdatadir,animal_id) #construct output save directory string

        if not os.path.exists(outdir): #check if output directory already exists, otherwise make
            os.mkdir(outdir)
            
        io = NWBHDF5IO(os.path.join(outdir,savefilename), mode="w") #save the NWB file
        io.write(nwbfile)
        io.close()


animal_id = animal_ids[0]
sessiondate = sessiondates[0]
protocol = "VR"

F           = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].data[:]
ts          = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].timestamps[:]

