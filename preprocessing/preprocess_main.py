"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

Main preprocessing function
Preprocesses behavioral data, task and trial data, facial video data, calcium imaging data etc.
"""

import os
os.chdir('e:\\Python\\molanalysis')
import numpy as np
from loaddata.get_data_folder import *
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))
from preprocessing.preprocesslib import *

# rawdatadir      = "G:\\RawData\\"

animal_ids          = [] #If empty than all animals in folder will be processed
date_filter         = []
# animal_ids          = ['LPE11495','LPE09665','LPE09830'] #If empty than all animals in folder will be processed
# date_filter        = ['2024_02_20','2024_02_21  ','2024_02_22','2024_02_23','2024_02_26','2024_02_27']
# date_filter        = ['2023_04_12']
# animal_ids          = ['LPE09665', 'LPE09830', 'LPE11495', 'LPE11998', 'LPE12013'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed

protocols           = ['GR','SP','IM','GN','RF']
# protocols           = ['GR']
# protocols           = ['DP','DM','DN']

processimagingflag  = True
savedataflag        = False

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids =  get_animals_protocol(protocols)
    # [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]

for animal_id in animal_ids: #for each animal

    rawdatadir = get_rawdata_drive(animal_id,protocols)

    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 
    
    if any(date_filter): #If dates specified, then process only those:
        sessiondates = [x for x in sessiondates if x in date_filter]

    for sessiondate in sessiondates: #for each of the sessions for this animal
        
        sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)
        
        for protocol in protocols: #for each of the protocols for this session
            
            if os.path.exists(os.path.join(sesfolder,protocol)):
                #set output saving dir:
                outdir          = os.path.join(get_local_drive(),'Procdata',protocol,animal_id,sessiondate) #construct output save directory string
        
                if not os.path.exists(outdir): #check if output directory already exists, otherwise make
                    os.makedirs(outdir)
                
                sessiondata         = proc_sessiondata(rawdatadir,animal_id,sessiondate,protocol)
                print(f'Processing {animal_id} - {sessiondate} - {protocol}')

                if protocol in ['DM','DP','DN']: #VR Task detection max, psy or noise
                    [sessiondata,trialdata,behaviordata] = proc_task(rawdatadir,sessiondata)
                else: #If not in a task then process behavior data in standard way, runspeed, etc.
                    behaviordata        = proc_behavior_passive(rawdatadir,sessiondata) #main processing function for harp data
        
                if 'GR' in protocol: # Grating Repetitions
                     sessiondata,trialdata  = proc_GR(rawdatadir,sessiondata)
        
                if 'GN' in protocol: # Grating Noise
                     sessiondata,trialdata  = proc_GN(rawdatadir,sessiondata)

                if 'IM' in protocol: #Natural Image Dataset
                    sessiondata,trialdata   = proc_IM(rawdatadir,sessiondata)
                
                videodata         = proc_videodata(rawdatadir,sessiondata,behaviordata)

                # give tStart and tEnd to sessiondata:
                if protocol in ['SP','RF']: #if no trials then base on behaviordata
                    sessiondata = add_session_bounds(sessiondata,behaviordata)
                else:
                    sessiondata = add_session_bounds(sessiondata,trialdata)

                # trim data to tStart and tEnd:
                behaviordata    = trim_session_bounds(sessiondata,behaviordata)
                videodata       = trim_session_bounds(sessiondata,videodata)
                
                if os.path.exists(os.path.join(sesfolder,"suite2p")) and processimagingflag:
                    print('Detected imaging data') #main processing function for imaging data:
                    [sessiondata,celldata,dFdata,deconvdata,Ftsdata,Fchan2data] = proc_imaging(sesfolder,sessiondata) 
                    celldata.to_csv(os.path.join(outdir,"celldata.csv"), sep=',')
                    Ftsdata.to_csv(os.path.join(outdir,"Ftsdata.csv"), sep=',')
                    Fchan2data.to_csv(os.path.join(outdir,"Fchan2data.csv"), sep=',')
                    if savedataflag:
                        print('\nSaving imaging data\n')
                        dFdata.to_csv(os.path.join(outdir,"dFdata.csv"), sep=',')
                        deconvdata.to_csv(os.path.join(outdir,"deconvdata.csv"), sep=',')
                
                # Save data:
                if savedataflag:
                    sessiondata.to_csv(os.path.join(outdir,"sessiondata.csv"), sep=',')
                    behaviordata.to_csv(os.path.join(outdir,"behaviordata.csv"), sep=',')
                    videodata.to_csv(os.path.join(outdir,"videodata.csv"), sep=',')
                    if protocol not in ['SP','RF']: #if trial based:
                        trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')

print(f'\n\nPreprocessing Completed')
