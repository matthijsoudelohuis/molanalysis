"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

Main preprocessing function
Preprocesses behavioral data, task and trial data, imaging data etc.
"""


import os, sys
import numpy as np
# os.chdir('E:\\Python\\molanalysis\\')
os.chdir('T:\\Python\\molanalysis\\')
# import suite2p
from preprocessing.preprocesslib import *

rawdatadir      = "I:\\RawData\\"
procdatadir     = "V:\\Procdata\\"

# rawdatadir      = "W:\\Users\\Matthijs\\Rawdata\\"
# procdatadir     = "E:\\Procdata\\"

# animal_ids          = [] #If empty than all animals in folder will be processed
date_filter         = []
animal_ids          = ['LPE11086'] #If empty than all animals in folder will be processed
# animal_ids          = ['LPE11622','LPE11495','LPE11623'] #If empty than all animals in folder will be processed
# date_filter        = ['2024_01_10']
# date_filter        = ['2024_02_20']
# date_filter        = ['2024_01_10']

protocols           = ['GR','SP','IM','GN']
# protocols           = ['GR']
# protocols           = ['DP','DM','DN']
# protocols           = ['DN']

processimagingflag  = True

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]

for animal_id in animal_ids: #for each animal

    sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 
    
    if any(date_filter): #If dates specified, then process only those:
        sessiondates = [x for x in sessiondates if x in date_filter]

    for sessiondate in sessiondates: #for each of the sessions for this animal
        
        sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)
        
        for protocol in protocols: #for each of the protocols for this session
            
            if os.path.exists(os.path.join(sesfolder,protocol)):
                #set output saving dir:
                outdir          = os.path.join(procdatadir,protocol,animal_id,sessiondate) #construct output save directory string
        
                if not os.path.exists(outdir): #check if output directory already exists, otherwise make
                    os.makedirs(outdir)
                
                sessiondata         = proc_sessiondata(rawdatadir,animal_id,sessiondate,protocol)
                print(f'Processing {animal_id} - {sessiondate} - {protocol}')

                if protocol in ['DM','DP','DN']: #VR Task detection max, psy or noise
                    [sessiondata,trialdata,behaviordata] = proc_task(rawdatadir,sessiondata)
                    trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
                    behaviordata.to_csv(os.path.join(outdir,"behaviordata.csv"), sep=',')
                else: #If not in a task then process behavior data in standard way, runspeed, etc.
                    behaviordata        = proc_behavior_passive(rawdatadir,sessiondata) #main processing function for harp data
                    behaviordata.to_csv(os.path.join(outdir,"behaviordata.csv"), sep=',')
        
                # if protocol == 'GR': # Grating Repetitions
                if 'GR' in protocol: # Grating Repetitions
                     trialdata = proc_GR(rawdatadir,sessiondata)
                     trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
        
                if protocol == 'GN': # Grating Noise
                     trialdata = proc_GN(rawdatadir,sessiondata)
                     trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
        
                if 'IM' in protocol: #Natural Image Dataset
                    trialdata = proc_IM(rawdatadir,sessiondata)
                    trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
                
                videodata         = proc_videodata(rawdatadir,sessiondata,behaviordata)
                videodata.to_csv(os.path.join(outdir,"videodata.csv"), sep=',')

                if os.path.exists(os.path.join(sesfolder,"suite2p")) and processimagingflag:
                    print('Detected imaging data')
                    [sessiondata,celldata,dFdata,deconvdata]         = proc_imaging(sesfolder,sessiondata) #main processing function for imaging data
                    print('\nSaving imaging data\n')
                    celldata.to_csv(os.path.join(outdir,"celldata.csv"), sep=',')
                    dFdata.to_csv(os.path.join(outdir,"dFdata.csv"), sep=',')
                    deconvdata.to_csv(os.path.join(outdir,"deconvdata.csv"), sep=',')
                
                #Save sessiondata:
                sessiondata.to_csv(os.path.join(outdir,"sessiondata.csv"), sep=',')

print(f'\n\nPreprocessing Completed')

