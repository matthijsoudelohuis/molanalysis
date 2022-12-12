import os
from pathlib import Path
# from preprocesslib.py import *
# import preprocesslib
import pandas as pd
import numpy as np

from pynwb import NWBFile, TimeSeries, NWBHDF5IO
# from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.behavior import BehavioralEvents
from datetime import datetime
from dateutil import tz


rawdatadir      = "V:\\Rawdata\\"
procdatadir     = "V:\\Procdata\\"

animal_ids          = ['NSH07422'] #If empty than all animals in folder will be processed
sessiondates        = ['2022_11_30',
 '2022_12_1',
 '2022_12_2',
 '2022_12_5',
 '2022_12_6',
 '2022_12_7',
 '2022_12_8'] #If empty than all animals in folder will be processed

def proc_behavior(rawdatadir,animal_id,sessiondate,protocol):
    """ preprocess all the trial, stimulus and behavior data for one session """
    
    [y,m,d] = sessiondate.split('_')
    y = int(y)
    m = int(m)
    d = int(d)
    session_start_time = datetime(y, m, d, 2, 30, 3, tzinfo=tz.gettz("Europe/Lisbon"))
    
    nwbfile = NWBFile(
    session_description="MouseVirtualCorridor",  # required
    identifier=animal_id + "_" + sessiondate,  # required
    session_start_time=session_start_time,  # required
    session_id="session_1234",  # optional
    experimenter="Matthijs Oude Lohuis",  # optional
    lab="Petreanu Lab",  # optional
    institution="Champalimaud Research",  # optional
    )
    
    nwbfile.subject = Subject(
    subject_id=animal_id,
    # age="P90D",
    # description="mouse 5",
    species="Mus musculus",
    # sex="M",
    )
    
    behavior_module = nwbfile.create_processing_module(
    name="behavior", description="processed behavioral data"
    )

    sesfolder   = os.path.join(rawdatadir,animal_id,sessiondate,protocol)
    sesfolder      = Path(sesfolder)
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
    name="WheelVoltage",
    description="Raw voltage from wheel rotary encoder",
    data=harpdata[:,0].astype(np.float64),
    unit="V",
    timestamps=timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
    
    ## Z position
    time_series_with_timestamps = TimeSeries(
    name="CorridorPosition",
    description="z position along the corridor",
    data=harpdata[:,3].astype(np.float64),
    unit="cm",
    timestamps=timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
        
    ## Running speed
    time_series_with_timestamps = TimeSeries(
    name="RunningSpeed",
    description="Speed of VR wheel rotation",
    data=harpdata[:,4].astype(np.float64),
    unit="cm s-1",
    timestamps=timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
    
    ## Wheel voltage
    time_series_with_timestamps = TimeSeries(
    name="TrialNumber",
    description="During which trial number the other acquisition channels were sampled",
    data=harpdata[:,2].astype(np.int64),
    unit="na",
    timestamps=timestamps,
    )
    nwbfile.add_acquisition(time_series_with_timestamps)
    
    ## Licks
    lickactivity    = np.diff(harpdata[:,5])
    lickactivity    = np.append(lickactivity,0)
    idx             = lickactivity==1
    print("%d licks" % idx.sum()) #Give output to check if reasonable
    
    time_series = TimeSeries(
        name="Licks",
        data=np.ones([idx.sum(),1]),
        timestamps=timestamps[idx],
        description="When luminance of tongue crossed a threshold at an ROI at the lick spout",
        unit="a.u.",
    )
    
    lick_events = BehavioralEvents(time_series=time_series, name="Licks")
    behavior_module.add(lick_events)

    ## Rewards
    rewardactivity = np.diff(harpdata[:,6])
    rewardactivity = np.append(rewardactivity,0)
    idx = rewardactivity>0
    print("%d rewards" % idx.sum()) #Give output to check if reasonable
    
    time_series = TimeSeries(
        name="Rewards",
        data=np.ones([idx.sum(),1])*5,
        timestamps=timestamps[idx],
        description="Rewards delivered at lick spout",
        unit="uL",
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



## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(rawdatadir)

for animal_id in animal_ids: #for each animal
    
    if len(sessiondates) == 0:
        sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

    for sessiondate in sessiondates: #for each of the sessions for this animal
        nwbfile         = proc_behavior(rawdatadir,animal_id,sessiondate,"VR") #main processing function
        
        savefilename    = animal_id + "_" + sessiondate + "_VR.nwb" #define save file name
        outdir          = os.path.join(procdatadir,animal_id) #construct output save directory string

        if not os.path.exists(outdir): #check if output directory already exists, otherwise make
            os.mkdir(outdir)
            
        io = NWBHDF5IO(os.path.join(outdir,savefilename), mode="w") #save the NWB file
        io.write(nwbfile)
        io.close()



