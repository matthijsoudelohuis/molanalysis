# -*- coding: utf-8 -*-
"""
This script run the suite2p analysis pipeline, but in separate steps. 
1) run suite2p registration using the tdtomato (red) channel
2) correct for the bleedthrough tdTomato signal to the green PMT
3) run suite2p calcium trace extraction
Matthijs Oude Lohuis, 2023, Champalimaud Center

# VISTA 1: LPE09829, NSH07422
# VISTA 2: LPE09830, 
"""

# TODO:
# auto detect folder for molanalysis
# auto detect raw data folder
# automatically taking right correction of PMT signals
# take right nplanes in init_ops 
# learn right way of module and folders etc.

import os
os.chdir('e:\\Python\\molanalysis\\')

import suite2p
from mol_suite2p_funcs import init_ops, run_bleedthrough_corr

rawdatadir          ='G:\\RawData\\'
animal_ids          = ['LPE09830'] #If empty than all animals in folder will be processed
sessiondates        = ['2023_04_11']

[db,ops] = init_ops(os.path.join(rawdatadir,animal_ids[0],sessiondates[0]))

###################################################################
## Run registration:
suite2p.run_s2p(ops=ops, db=db)

###################################################################
## tdTomato bleedthrough correction:

coeff = 1.54 #for 0.6 and 0.4 combination of PMT gains
# coeff = 0.068 #for 0.6 and 0.6 combination of PMT gains

ops = run_bleedthrough_corr(db,ops,coeff)

###################################################################
## ROI detection: 
ops['do_registration']      = False
ops['roidetect']            = True

# ops['nbinned'] = 1000 #standard 5000

ops = suite2p.run_s2p(ops=ops, db=db)

############################
# Debug / Verification code:

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

    
# ###################################################################

# import matplotlib.pyplot as plt

# data_green      = np.empty([0,512,512])
# data_red        = np.empty([0,512,512])

# # with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan2, Ly=512, Lx=512) as f2, BinaryFile(read_filename=file_chan1_corr, Ly=512, Lx=512) as fout:
# with BinaryFile(read_filename=file_chan1,Ly=512, Lx=512) as f1, BinaryFile(read_filename=file_chan1_corr, Ly=512, Lx=512) as f2, BinaryFile(read_filename=file_chan1_corr, Ly=512, Lx=512) as fout:
#      for i in np.arange(100):
#          [ind,datagreen]      = f1.read(batch_size=1)
#          [ind,datared]        = f2.read(batch_size=1)
         
#          data_green = np.append(data_green, datagreen,axis=0)
#          data_red = np.append(data_red,datared,axis=0)


# ## Show 
# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,6.5))

# greenchanim = np.average(data_green,axis=0)
# ax1.imshow(greenchanim,vmin=0, vmax=5000)
# ax1.set_title('Chan 1')
# ax1.set_axis_off()

# redchanim = np.average(data_red,axis=0)
# ax2.imshow(redchanim,vmin=0, vmax=3000)
# ax2.set_title('Chan 2')
# ax2.set_axis_off()

# greenchan = greenchanim.reshape(1,512*512)[0]
# redchan = redchanim.reshape(1,512*512)[0]

# ax3.scatter(redchan,greenchan,0.02)
# # ax3.scatter(data_green.flatten(),data_red.flatten(),0.02)
# ax3.set_xlabel('Chan 2')
# ax3.set_ylabel('Chan 1')

# # Fit linear regression via least squares with numpy.polyfit
# b, a = np.polyfit(redchan, greenchan, deg=1)

# xseq = np.linspace(-15000, 32000, num=32000)
# # Plot regression line
# ax3.plot(xseq, a + b * xseq, color="k", lw=1.5);

# ax3.set_xlim([-2000,20000])
# ax3.set_ylim([-2000,20000])
# ax3.plot(xseq, coeff * xseq, color="k", lw=1.5);

# txt1 = "Coefficient is %1.4f" % b

# ax3.text(2500,1000,txt1, fontsize=9)

# #Correction:
# # data_green_corr = data_green - coeff * data_red
# temp = np.repeat(np.average(data_red,axis=0)[np.newaxis,:, :], np.shape(data_green)[0], axis=0)
# data_green_corr = data_green - coeff * temp

# greenchanim = np.average(data_green_corr,axis=0)
# ax4.imshow(greenchanim,vmin=-200, vmax=6000)
# ax4.set_title('Chan 1')
# ax4.set_axis_off()

# redchanim = np.average(data_red,axis=0)
# ax5.imshow(redchanim,vmin=-200, vmax=6000)
# ax5.set_title('Chan 2')
# ax5.set_axis_off()

# greenchan = greenchanim.reshape(1,512*512)[0]
# redchan = redchanim.reshape(1,512*512)[0]

# ax6.scatter(redchan,greenchan,0.02)
# ax6.set_xlabel('Chan 2')
# ax6.set_ylabel('Chan 1')