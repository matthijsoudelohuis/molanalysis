# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:09:29 2023

@author: USER
"""

import os, shutil
import numpy as np
# import suite2p
from suite2p.io.binary import BinaryFile
from suite2p.extraction import extract
from utils.twoplib import get_meta

def init_ops(sesfolder):
    
    ops = np.load('T:/Python/ops_8planes.npy',allow_pickle='TRUE').item()
    
    ops['do_registration']      = True
    ops['roidetect']            = False #only do registration in this part
    
    protocols           = ['VR','IM','SP','RF','GR']
    
    db = {
        'data_path': [sesfolder],
        'save_path0': sesfolder,
        'look_one_level_down': True, # whether to look in ALL subfolders when searching for tiffs
    }
    #Find all protocols
    db['subfolders'] = [os.path.join(sesfolder,f,'Imaging') for f in os.listdir(db['data_path'][0]) if f in protocols]
        
    
    #identify number of planes in the session:
    firsttiff = [x for x in os.listdir(db['subfolders'][0]) if x.endswith(".tif")][0] #get first tif in first dir to read nplanes:
        
    # read metadata from tiff
    # metadata should be same for all if settings haven't changed during differernt protocols
    meta, meta_si   = get_meta(os.path.join(db['subfolders'][0],firsttiff))
    meta_dict       = dict() #convert to dictionary:
    for line in meta_si:
        meta_dict[line.split(' = ')[0]] = line.split(' = ')[1]
 
    nROIs = len(meta['RoiGroups']['imagingRoiGroup']['rois'])
    roi_area    = [meta['RoiGroups']['imagingRoiGroup']['rois'][i]['name'] for i in range(nROIs)]
    # roi_planes  = [len(meta['RoiGroups']['imagingRoiGroup']['rois'][i]['scanfields']) for i in range(nROIs)]
    
    # ops['nplanes'] = len(roi_area)
    ops['nplanes'] = 8

    ops['fs'] = float(meta_dict['SI.hRoiManager.scanFrameRate']) / ops['nplanes']
    
    return db, ops


def run_bleedthrough_corr(db,ops,coeff):

    
    #Write new binary file with corrected data per plane:
    for iplane in np.arange(ops['nplanes']):
        print('Correcting tdTomato bleedthrough for plane %s / %s' % (iplane+1,ops['nplanes']))
    
        file_chan1       = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data.bin')
        file_chan2       = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data_chan2.bin')
        file_chan1_corr   = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data_corr.bin')
        
        with BinaryFile(read_filename=file_chan1,write_filename=file_chan1_corr,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
        # with BinaryFile(filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(filename=file_chan2, Ly=512, Lx=512) as f2, BinaryFile(filename=file_chan1_corr,Ly=512, Lx=512,n_frames=f1.n_frames) as f3:
            
              for i in np.arange(f1.n_frames):
                  [ind,datagreen]      = f1.read(batch_size=1)
                  [ind,datared]        = f2.read(batch_size=1)
                  
                  datagreencorr = datagreen - coeff * datared

                  f1.write(data=datagreencorr)
                #   f3.write(data=datagreencorr)
            
    #move original to subdir and rename corrected to data.bin to be read by suite2p for detection:
    for iplane in np.arange(ops['nplanes']):
        planefolder = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane)
        file_chan1       = os.path.join(planefolder,'data.bin')
        file_chan1_corr   = os.path.join(planefolder,'data_corr.bin')
        
        os.mkdir(os.path.join(planefolder,'orig'))
    
        shutil.move(os.path.join(planefolder,file_chan1),os.path.join(planefolder,'orig'))
        
        os.rename(os.path.join(planefolder,file_chan1_corr), os.path.join(planefolder,file_chan1))
    
    ### Update mean images and add enhanced images:
    for iplane in np.arange(ops['nplanes']):
        print('Modifying mean images in ops file for plane %s / %s' % (iplane+1,ops['nplanes']))
        ops = np.load(os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'ops.npy'),allow_pickle='TRUE').item()
        # ops['reg_file']         = ops['reg_file'].replace('data','data_corr')
        
        # with BinaryFile(read_filename=os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data_corr.bin'),Ly=512, Lx=512) as f1:
        with BinaryFile(read_filename=os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data.bin'),Ly=512, Lx=512) as f1:
            ops['meanImg']      = f1.sampled_mean()
        
        ops                     = extract.enhanced_mean_image(ops)
        ops                     = extract.enhanced_mean_image_chan2(ops)
        np.save(os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'ops.npy'),ops)    
    
    return ops

