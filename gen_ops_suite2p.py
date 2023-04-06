# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:55:09 2023

@author: USER
"""
import numpy as np
import suite2p

# set your options for running
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
ops['roidetect']        = False
ops['maxregshiftNR']    = 5
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

np.save('T:/Python/ops_8planes.npy',ops)