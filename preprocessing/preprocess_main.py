# -*- coding: utf-8 -*-
"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

Main preprocessing function
Preprocesses behavioral data, task and trial data, imaging data etc.
"""

import os, sys
import numpy as np
# sys.path.append('T:/Python/molanalysis/preprocessing')
os.chdir('E:\\Python\\molanalysis\\')

from preprocessing.preprocesslib import *

# rawdatadir      = "X:\\Rawdata\\"
# procdatadir     = "V:\\Procdata\\"

rawdatadir      = "W:\\Users\\Matthijs\\Rawdata\\"
procdatadir     = "E:\\Procdata\\"


# animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
# sessiondates        = ['2023_04_10']
animal_ids          = ['LPE09829'] #If empty than all animals in folder will be processed
# sessiondates        = ['2023_03_30']
sessiondates        = ['2023_03_29']

# protocols           = ['IM','GR','RF','SP']
# protocols           = ['RF']
protocols           = ['VR']

## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(rawdatadir)

for animal_id in animal_ids: #for each animal
    
    if len(sessiondates) == 0:
        sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

    for sessiondate in sessiondates: #for each of the sessions for this animal
        
        sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate)
        
        for protocol in protocols: #for each of the sessions for this animal
            
            if os.path.exists(os.path.join(sesfolder,protocol)):
                #set output saving dir:
                outdir          = os.path.join(procdatadir,protocol,animal_id,sessiondate) #construct output save directory string
        
                if not os.path.exists(outdir): #check if output directory already exists, otherwise make
                    os.makedirs(outdir)
                
                sessiondata         = proc_sessiondata(rawdatadir,animal_id,sessiondate,protocol)
        
                if protocol == 'VR':
                    [sessiondata,trialdata,behaviordata] = proc_task(rawdatadir,sessiondata)
                    trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
                    behaviordata.to_csv(os.path.join(outdir,"behaviordata.csv"), sep=',')
                else: 
                    behaviordata        = proc_behavior_passive(rawdatadir,sessiondata) #main processing function for harp data
                    behaviordata.to_csv(os.path.join(outdir,"behaviordata.csv"), sep=',')
        
                if protocol == 'GR':
                     trialdata = proc_GR(rawdatadir,sessiondata)
                     trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
        
                elif protocol == 'RF':
                    [grid_array,RF_timestamps] = proc_RF(rawdatadir,sessiondata)

                    # np.save(os.path.join(outdir,"trialdata.npy"),grid_array,RF_timestamps)
                    np.savez(os.path.join(outdir,"trialdata.npz"),x=grid_array,y=RF_timestamps)
                
                elif protocol == 'IM':
                    trialdata = proc_IM(rawdatadir,sessiondata)
                    trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
                
                
                videodata         = proc_videodata(rawdatadir,animal_id,sessiondate,protocol)
                
                sesfolder       = os.path.join(rawdatadir,sessiondata['animal_id'][0],sessiondata['sessiondate'][0],sessiondata['protocol'][0],'Behavior')

                filenames       = os.listdir(sesfolder)
    
                avi_file        = list(filter(lambda a: '.avi' in a, filenames)) #find the trialdata file
                csv_file        = list(filter(lambda a: 'cameracsv' in a, filenames)) #find the trialdata file

                csvdata       = pd.read_csv(os.path.join(sesfolder,csv_file[0]))
                nts = len(csvdata)
                ts = csvdata['Item2'].to_numpy()

                framerate = 30

                #Check that the number of frames is ballpark range of what it should be based on framerate and session duration:
                sesdur = behaviordata.loc[behaviordata.index[-1],'ts']  - behaviordata.loc[behaviordata.index[0],'ts'] 
                assert np.isclose(nts,sesdur * framerate,rtol=3)
                #Check that frame rate matches interframe interval:
                assert np.isclose(1/framerate,np.mean(np.diff(ts)),rtol=0.01)
                #Check that inter frame interval does not take on crazy values:
                assert ~np.any(np.logical_or(np.diff(ts)<0.01,np.diff(ts)>0.06))

                videodata = pd.DataFrame(data = ts, columns = 'timestamps')

                if os.path.exists(os.path.join(sesfolder,"suite2p")):
                    print('Detected imaging data\n')
                    [sessiondata,celldata,calciumdata]         = proc_imaging(sesfolder,sessiondata) #main processing function for imaging data
                    print('Saving imaging data\n')
                    celldata.to_csv(os.path.join(outdir,"celldata.csv"), sep=',')
                    calciumdata.to_csv(os.path.join(outdir,"calciumdata.csv"), sep=',')
                
                #Save sessiondata:
                sessiondata.to_csv(os.path.join(outdir,"sessiondata.csv"), sep=',')



# [celldata,calciumdata]         = proc_imaging(sesfolder,sessiondata) #main processing function for imaging data


# ## Loop over all selected animals and folders
# if len(animal_ids) == 0:
#     animal_ids = os.listdir(rawdatadir)

# for animal_id in animal_ids: #for each animal
    
#     if len(sessiondates) == 0:
#         sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

#     for sessiondate in sessiondates: #for each of the sessions for this animal
#         nwbfile         = proc_behavior(rawdatadir,animal_id,sessiondate,"VR") #main processing function
#         sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate,"VR")

#         if os.path.exists(os.path.join(sesfolder,"Imaging")):
#             print('Detected imaging data\n')
#             nwbfile         = proc_imaging(sesfolder,nwbfile) #main processing function for imaging data
        
#         savefilename    = animal_id + "_" + sessiondate + "_VR.nwb" #define save file name
#         outdir          = os.path.join(procdatadir,animal_id) #construct output save directory string

#         if not os.path.exists(outdir): #check if output directory already exists, otherwise make
#             os.mkdir(outdir)
            
#         io = NWBHDF5IO(os.path.join(outdir,savefilename), mode="w") #save the NWB file
#         io.write(nwbfile)
#         io.close()

# F           = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].data[:]
# ts          = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].timestamps[:]

