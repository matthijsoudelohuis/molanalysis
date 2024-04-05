# -*- coding: utf-8 -*-
"""
Suite2p functions that are necessary for automatic ROI detection pipeline of 
mesoscopic 2p Ca2+ imaging recordings
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

import os, shutil
import numpy as np
import suite2p
from suite2p.io.binary import BinaryFile
from suite2p.extraction import extract
from utils.twoplib import get_meta
import pandas as pd
from labeling.label_lib import bleedthrough_correction,load_tiffs_plane,estimate_correc_coeff
from ScanImageTiffReader import ScanImageTiffReader as imread


# set your options for running
def gen_ops():
    """
    Script to generate parameters for suite2p pipeline with mesoscope recordings in 8 planes
    Matthijs Oude Lohuis, Champalimaud 2023
    """
    ops = suite2p.default_ops() # populates ops with the default options

    ops['look_one_level_down']          = True

    ops['nplanes']          = 8
    ops['nchannels']        = 2
    ops['functional_chan']  = 1
    ops['tau']              = 0.7
    ops['fs']               = 42.857/8
    ops['save_mat']         = False
    ops['save_NWB']         = False
    ops['reg_tif']          = False
    ops['reg_tif_chan2']    = False
    ops['delete_bin']       = False
    ops['align_by_chan']    = 2
    ops['nimg_init']        = 500
    ops['batch_size']       = 500
    ops['nonrigid']         = True
    ops['block_size']       = [128,128]
    # ops['maxregshiftNR']    = 5
    ops['maxregshift']      = 0.15
    ops['1Preg']            = False

    ops['denoise']              = True
    ops['spatial_scale']        = 4
    ops['threshold_scaling']    = 0.5
    ops['max_overlap']          = 0.75
    ops['max_iterations']       = 50
    ops['high_pass']            = 100
    ops['spatial_hp_detect']    = 25
    ops['anatomical_only']      = 0
    ops['neuropil_extract']     = True
    ops['allow_overlap']        = True
    ops['soma_crop']            = True
    ops['win_baseline']         = 60
    ops['sig_baseline']         = 10
    ops['neucoeff']             = 0.7

    ops['do_registration']      = False
    ops['roidetect']            = False
    # np.save('T:/Python/ops_8planes.npy',ops)
    # np.save('E:/Python/ops_8planes.npy',ops)

    return ops


def init_ops(sesfolder):
    
    # ops = np.load('T:/Python/ops_8planes.npy',allow_pickle='TRUE').item()
    ops = gen_ops()

    ops['do_registration']      = True
    ops['roidetect']            = False #only do registration in this part
    
    protocols           = ['VR','IM','SP','RF','GR','GN','DN']
    # protocols           = ['GR1','IM1','SP1','GR2','IM2','SP2',
                        #    'GR3','IM3','SP3','GR4','IM4','SP4',]
    
    db = {
        'data_path': [sesfolder],
        'save_path0': sesfolder,
        'look_one_level_down': True, # whether to look in ALL subfolders when searching for tiffs
    }
    #Find all protocols
    db['subfolders'] = [os.path.join(sesfolder,f,'Imaging') for f in protocols if f in os.listdir(db['data_path'][0])]
    
    #identify number of planes in the session:
    firsttiff = [x for x in os.listdir(db['subfolders'][0]) if x.endswith(".tif")][0] #get first tif in first dir to read nplanes:
        
    # read metadata from tiff
    # metadata should be same for all if settings haven't changed during differernt protocols
    meta, meta_si   = get_meta(os.path.join(db['subfolders'][0],firsttiff))
    meta_dict       = dict() #convert to dictionary:
    for line in meta_si:
        meta_dict[line.split(' = ')[0]] = line.split(' = ')[1]
 
    ops['nROIs']        = len(meta['RoiGroups']['imagingRoiGroup']['rois'])
    ops['roi_names']    = [meta['RoiGroups']['imagingRoiGroup']['rois'][i]['name'] for i in range(ops['nROIs'])]
    
    ops['nplanes'] = 0
    for i in range(ops['nROIs']):
        # print(type(meta['RoiGroups']['imagingRoiGroup']['rois'][i]['scanfields']))
        if type(meta['RoiGroups']['imagingRoiGroup']['rois'][i]['scanfields'])==list:
            ops['nplanes'] += len(meta['RoiGroups']['imagingRoiGroup']['rois'][i]['scanfields'])
        elif type(meta['RoiGroups']['imagingRoiGroup']['rois'][i]['scanfields'])==dict:
            ops['nplanes'] += 1
        # ops['nplanes']      = 8

    ops['fs'] = float(meta_dict['SI.hRoiManager.scanFrameRate']) / ops['nplanes']
    
    return db, ops

def run_bleedthrough_corr(db,ops,coeff=None,gain1=0.6,gain2=0.4):

    #Write new binary file with corrected data per plane:
    for iplane in np.arange(ops['nplanes']):
        print('Correcting tdTomato bleedthrough for plane %s / %s' % (iplane+1,ops['nplanes']))
    
        file_chan1       = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data.bin')
        file_chan2       = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data_chan2.bin')
        file_chan1_corr   = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data_corr.bin')
        
        # if coeff is None:
        [data_green,data_red] = load_tiffs_plane(db['subfolders'][0],ops['nplanes'],iplane=iplane)
        coeff       = estimate_correc_coeff(data_green,data_red)

        # data_green_corr = bleedthrough_correction(data_green,data_red,coeff)

        # plot_corr_redgreen(data_green,data_red)
        # plot_bleedthrough_correction(np.mean(data_green,axis=0), np.mean(data_red,axis=0), np.mean(data_green_corr,axis=0))
        # plot_correction_images(greenchanim,redchanim)

        # with BinaryFile(read_filename=file_chan1,write_filename=file_chan1_corr,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
        with BinaryFile(filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(filename=file_chan2, Ly=512, Lx=512) as f2, BinaryFile(filename=file_chan1_corr,Ly=512, Lx=512,n_frames=f1.n_frames) as f3:
    
            #for i in np.arange(f1.n_frames):
            
            for i in np.arange(5):
                print(i)	
                f3.write
                datagreen       = f1.file[i]
                datared         = f2.file[i]

                [ind,datagreen]      = f1.read(batch_size=1)
                [ind,datared]        = f2.read(batch_size=1)

                datagreencorr = bleedthrough_correction(datagreen,datared,coeff,gain1,gain2)
                f3.file[i] = datagreencorr

                # f1.write(data=datagreencorr)
                f3.write(data=datagreencorr)
            f1.close()
            f2.close()
            f3.close()

    #delete original rename corrected to data.bin to be read by suite2p for detection:
    for iplane in np.arange(ops['nplanes']):
        planefolder = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane)
        file_chan1       = os.path.join(planefolder,'data.bin')
        file_chan1_corr   = os.path.join(planefolder,'data_corr.bin')
        
        # os.mkdir(os.path.join(planefolder,'orig'))
        # shutil.move(os.path.join(planefolder,file_chan1),os.path.join(planefolder,'orig'))
        os.remove(os.path.join(planefolder,file_chan1))

        os.rename(os.path.join(planefolder,file_chan1_corr), os.path.join(planefolder,file_chan1))

        # os.remove(os.path.join(planefolder,file_chan1_corr))


    # #move original to subdir and rename corrected to data.bin to be read by suite2p for detection:
    # for iplane in np.arange(ops['nplanes']):
    #     planefolder = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane)
    #     file_chan1       = os.path.join(planefolder,'data.bin')
    #     file_chan1_corr   = os.path.join(planefolder,'data_corr.bin')
        
    #     os.mkdir(os.path.join(planefolder,'orig'))
    
    #     shutil.move(os.path.join(planefolder,file_chan1),os.path.join(planefolder,'orig'))
        
    #     os.rename(os.path.join(planefolder,file_chan1_corr), os.path.join(planefolder,file_chan1))

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

def check_tiffs(direc):

    # for x in os.listdir(direc):
    for i,x in enumerate(os.listdir(direc)):
        if x.endswith(".tif"):
            # mROI_data, meta = split_mROIs(os.path.join(direc,x))
            # nROIs = len(mROI_data)
            print(x)
            fname = os.path.join(direc,x)
            reader = imread(str(fname)) # amazing - this librarty needs str
            reader.close()
# def get_bleedthrough_coeff(rawdatadir,animal_id,sessiondate):


#     sessions_overview_VISTA = pd.read_excel(os.path.join(rawdatadir,'VISTA_Sessions_Overview.xlsx'))
#     sessions_overview_VR    = pd.read_excel(os.path.join(rawdatadir,'VR_Sessions_Overview.xlsx'))

#     if np.any(np.logical_and(sessions_overview_VISTA["sessiondate"] == sessiondate,sessions_overview_VISTA["protocol"] == protocol)):
#         sessions_overview = sessions_overview_VISTA
#     elif np.any(np.logical_and(sessions_overview_VR["sessiondate"] == sessiondate,sessions_overview_VR["protocol"] == protocol)):
#         sessions_overview = sessions_overview_VR
#     else: 
#         print('Session not found in excel session overview')

#     # if protocol in ['IM','GR','RF','SP']:
#     #     sessions_overview = pd.read_excel(os.path.join(rawdatadir,'VISTA_Sessions_Overview.xlsx'))
#     # elif protocol in ['VR']: 
#     #     sessions_overview = pd.read_excel(os.path.join(rawdatadir,'VR_Sessions_Overview.xlsx'))

#     idx = np.where(np.logical_and(sessions_overview["sessiondate"] == sessiondate,sessions_overview["protocol"] == protocol))[0]
#     if np.any(idx):
#         sessiondata = pd.merge(sessiondata,sessions_overview.loc[idx]) #Copy all the data from the excel to sessiondata dataframe
#     else: 
#         print('Session not found in excel session overview')


#     return coeff