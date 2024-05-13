
import os, sys
import numpy as np
import win32api
import shutil

drive_id        = 3
sourcedrive     = "I:\\"
backupdrive     = "F:\\"

filetypes_to_backup = np.array([
'jpg',
'png',
'npy'])

# perform check before overwriting stuff:
sourcedrive_name = win32api.GetVolumeInformation(sourcedrive)[0]
backupdrive_name = win32api.GetVolumeInformation(backupdrive)[0]
assert(sourcedrive_name=='VISTA %d' % drive_id), 'wrong drive id or name with sourcedrive'
# assert(backupdrive_name=='VISTA %d BACKUP' % drive_id), 'wrong drive id or name with sourcedrive'

for path, subdirs, files in os.walk(sourcedrive):
    for name in files:
        # print(os.path.join(path, name))

        if np.isin(name[-3:],filetypes_to_backup) and not 'RECYCLE' in path:

        # if np.isin(name,list_to_backup) and not 'RECYCLE' in path:
            print(os.path.join(path, name))
            backuppath = path.replace(sourcedrive,backupdrive)
            if not os.path.exists(backuppath):
                os.makedirs(backuppath)
            shutil.copyfile(os.path.join(path, name), os.path.join(backuppath, name))

