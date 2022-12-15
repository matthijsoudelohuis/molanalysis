# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 08:50:40 2022

@author: USER
"""

import pandas as pd

from scipy.io import loadmat
annots = loadmat('annotation_0001.mat')
annots = loadmat('X:/RawData/VR/NSH07422/2022_12_09/config.cfg')
print(annots)

annots = loadmat('X:/RawData/VR/NSH07422/2022_12_09/window.roi')


import json

f = open('X:/RawData/VR/NSH07422/2022_12_09/VR/NSH07429_VR_00001_00001_meta.json',)

json.load('X:/RawData/VR/NSH07422/2022_12_09/window.roi')
h = json.load(f)


# # zip provides us with both the x and y in a tuple.
# newData = list(zip(con_list[0], con_list[1]))
# columns = ['obj_contour_x', 'obj_contour_y']
# df = pd.DataFrame(newData, columns=columns)
