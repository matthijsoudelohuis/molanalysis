# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:53:12 2023

@author: USER
"""

import os, sys
# from pathlib import Path
# import pandas as pd
# import numpy as np

sys.path.append('T:/Python/molanalysis/preprocessing')
from preprocesslib import proc_sessiondata,proc_behavior,proc_imaging
# import preprocesslib 

rawdatadir      = "X:\\Rawdata\\"
procdatadir     = "V:\\Procdata\\"

animal_ids          = ['NSH07422'] #If empty than all animals in folder will be processed
# sessiondates        = ['2022_11_30',
#  '2022_12_8'] #If empty than all animals in folder will be processed
sessiondates        = ['2022_12_09']

animal_id       = animal_ids[0]
sessiondate     = sessiondates[0]

protocol        = 'VR'

sessiondata         = proc_sessiondata(rawdatadir,animal_id,sessiondate,protocol)

[sessiondata, trialdata, behaviordata]         = proc_behavior_passive(rawdatadir,sessiondata) #main processing function

[sessiondata, trialdata, behaviordata]         = proc_behavior(rawdatadir,animal_id,sessiondate,"VR") #main processing function


[imagingdata]         = proc_imaging(rawdatadir,animal_id,sessiondate,"VR") #main processing function



## Loop over all selected animals and folders
if len(animal_ids) == 0:
    animal_ids = os.listdir(rawdatadir)

for animal_id in animal_ids: #for each animal
    
    if len(sessiondates) == 0:
        sessiondates = os.listdir(os.path.join(rawdatadir,animal_id)) 

    for sessiondate in sessiondates: #for each of the sessions for this animal
        [sessiondata, trialdata, behaviordata]         = proc_behavior(rawdatadir,animal_id,sessiondate,"VR") #main processing function
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
nwbfile         = proc_behavior(rawdatadir,animal_id,sessiondate,"VR") #main processing function
sesfolder       = os.path.join(rawdatadir,animal_id,sessiondate,"VR")
nwbfile         = proc_imaging(sesfolder,nwbfile) #main processing function for imaging data

# F           = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].data[:]
# ts          = nwbfile.processing['ophys']['Fluorescence']['Fluorescence'].timestamps[:]

