# -*- coding: utf-8 -*-
"""
This script run the suite2p analysis pipeline, but in separate steps. 
1) run suite2p registration using the tdtomato (red) channel
2) correct for the bleedthrough tdTomato signal to the green PMT
3) run suite2p calcium trace extraction
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

# TODO:
# learn right way of module and folders etc.

import os
from loaddata.get_data_folder import get_local_drive
os.chdir(os.path.join(get_local_drive(),'Python','molanalysis'))

import suite2p
from run_suite2p.mol_suite2p_funcs import init_ops, run_bleedthrough_corr
from preprocessing.locate_rf import locate_rf_session
from labeling.tdTom_labeling_cellpose import gen_red_images,proc_labeling_session

rawdatadir          = 'H:\\RawData\\'
animal_id           = 'LPE12013'
sessiondate         = '2024_05_07'

[db,ops] = init_ops(os.path.join(rawdatadir,animal_id,sessiondate))

ops['align_by_chan']    = 1 #1-indexed, 1=gcamp,2=tdtomato

##################    Run registration:  ############################
suite2p.run_s2p(ops=ops, db=db) 

################# tdTomato bleedthrough correction: ################
# gain1 = 0.6 # gain2 = 0.4

ops = run_bleedthrough_corr(db,ops) #if no gain parameters, autom fit bleedthrough coefficient

########################## ROI detection ###########################
ops['do_registration']      = False
ops['roidetect']            = True
# ops['nbinned']              = 2000

ops = suite2p.run_s2p(ops=ops, db=db)

gen_red_images(rawdatadir,animal_id,sessiondate)

######################## Receptive field localization  ##############
# Locate receptive field if RF protocol was run in this session: 
locate_rf_session(rawdatadir,animal_id,sessiondate)

# proc_labeling_session(rawdatadir,animal_id,sessiondate)

############################
# Debug / Verification code:

# for subf in db['subfolders']:
#     check_tiffs(subf)

# check_tiffs(db['subfolders'][1])

# Verify new images added to ops:
# import copy

# iplane = 1
# file_chan1_corr   = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data_corr.bin')
# file_chan2       = os.path.join(db['save_path0'],'suite2p','plane%s' % iplane,'data_chan2.bin')

# ops1 = np.load('X:\\RawData\\LPE09665\\2023_03_14\\suite2p\\plane1\\ops.npy',allow_pickle='TRUE').item()
# ops1_2 = copy.deepcopy(ops1)

# with BinaryFile(read_filename=file_chan1_corr,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2:
#     ops1_2['meanImg']               = f1.sampled_mean()
#     ops1_2['meanImg_chan2']         = f2.sampled_mean()

# ops1 = extract.enhanced_mean_image_chan2(ops1)

# ops1_2 = extract.enhanced_mean_image(ops1_2)
# ops1_2 = extract.enhanced_mean_image_chan2(ops1_2)

# fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4, figsize=(10,6.5))

# ax1.imshow(ops1['meanImg'],vmin=0, vmax=5000)
# ax1.set_axis_off()
# ax5.imshow(ops1_2['meanImg'],vmin=0, vmax=5000)
# ax5.set_axis_off()

# ax2.imshow(ops1['meanImgE'],vmin=0, vmax=1)
# ax2.set_axis_off()
# ax6.imshow(ops1_2['meanImgE'],vmin=0, vmax=1)
# ax6.set_axis_off()

# ax3.imshow(ops1['meanImg_chan2'],vmin=0, vmax=5000)
# ax3.set_axis_off()
# ax7.imshow(ops1_2['meanImg_chan2'],vmin=0, vmax=5000)
# ax7.set_axis_off()

# ax4.imshow(ops1['meanImgE_chan2'],vmin=0, vmax=1)
# ax4.set_axis_off()
# ax8.imshow(ops1_2['meanImgE_chan2'],vmin=0, vmax=1)
# ax8.set_axis_off()
