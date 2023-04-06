# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:51:50 2023

@author: USER
"""
import os, sys, shutil
sys.path.append('T:/Python/molanalysis/preprocessing')
from preprocesslib import *

rawdatadir      = "X:\\Rawdata\\"
procdatadir     = "V:\\Procdata\\"

animal_ids          = ['NSH07422'] #If empty than all animals in folder will be processed
animal_ids          = ['LPE09665'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_03_14']
# sessiondates        = ['2022_11_30',
# sessiondates        = ['2022_12_09']

# animal_id           = animal_ids[0]
# sessiondate         = sessiondates[0]

protocols           = ['IM','GR','RF','SP']


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
                sessiondata.to_csv(os.path.join(outdir,"sessiondata.csv"), sep=',')
        
                behaviordata        = proc_behavior_passive(rawdatadir,sessiondata) #main processing function
                behaviordata.to_csv(os.path.join(outdir,"behaviordata.csv"), sep=',')
        
                if protocol == 'GR':
                     trialdata = proc_GR(rawdatadir,sessiondata)
                     trialdata.to_csv(os.path.join(outdir,"trialdata.csv"), sep=',')
        
                elif protocol == 'RF':
                    [rf_folder,rf_file] = proc_RF(rawdatadir,sessiondata)
                    
                    shutil.copyfile(os.path.join(rf_folder,rf_file[0]),os.path.join(outdir,'RF_log.bin'))
        
                if os.path.exists(os.path.join(sesfolder,"suite2p")):
                    print('Detected imaging data\n')
                    # [celldata,calciumdata]         = proc_imaging(sesfolder,sessiondata) #main processing function for imaging data
                     # imagingdata.to_csv(os.path.join(outdir,"imagingdata.csv"), sep=',')




[celldata,calciumdata]         = proc_imaging(sesfolder,sessiondata) #main processing function for imaging data



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

