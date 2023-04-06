# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:25:22 2023

@author: USER
"""

directory = 'C:\\TempData\\LPE09665\\2023_03_14\\GR\\\Imaging_tdTomcorr\\'

directory = 'X:\\RawData\LPE09665\\2023_03_14\\'

from ScanImageTiffReader import ScanImageTiffReader as imread
import tifffile
import os 

coeff = 1.54

def list_tifs(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            filepath = root + os.sep + name
            if filepath.endswith(".tif"):
                r.append(os.path.join(root, name))
    return r

r = list_tifs(directory)

for f in r:
     if os.path.isfile(f):
        print(f)
        reader  = imread(f)
        Data    = []
        Data    = reader.data()
        reader.close()
        Data[0::2,:,:] = Data[0::2,:,:] - coeff * Data[1::2,:,:]
        
        outpath = f.replace(".tif", "_corr.tif")
        # outpath = f.replace("Imaging", "Imaging_corr")
        with open(outpath,'wb') as fH:
            tifffile.imwrite(fH,Data.astype('int16'), bigtiff=True)
            fH.close()
            # del(Data)
            # print(fH)

# # iterate over files in that directory
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         print(f)
#         reader = imread(f)
#         Data = reader.data()
#         # data_green = np.append(data_green, Data[0::2,:,:],axis=0)
#         # data_red = np.append(data_red, Data[1::2,:,:],axis=0)
        
#         # Data2 = Data
#         Data[0::2,:,:] = Data[0::2,:,:] - coeff * Data[1::2,:,:]
        
#         outpath = f.replace(".tif", "_corr.tif")
#         with open(outpath,'wb') as fH:
#             tifffile.imwrite(fH,Data.astype('int16'), bigtiff=True)
        
