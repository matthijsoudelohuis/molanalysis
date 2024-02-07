import os
import numpy as np
from PIL import Image

def im_log(image):
    # Apply log transformation method 
    c = 255 / np.log(1 + np.max(image)) 
    log_image = c * (np.log(image + 1)) 
    return log_image

def im_sqrt(image):
    # Apply sqrt transformation 
    return image ** 0.5

def im_norm(I,min=0,max=100):
    mn = np.percentile(I,min)
    mx = np.percentile(I,max)
    mx -= mn
    I = ((I - mn)/mx) * 255
    I[I<0] =0
    I[I>255] = 255
    return I

def im_norm8(I,min=0,max=100):
    I = im_norm(I,min=min,max=max)
    # Specify the data type so that float value will be converted to int 
    return I.astype(np.uint8)
