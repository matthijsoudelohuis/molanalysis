import os
import numpy as np
from operator import itemgetter

# figure out the paths depending on which machine you are on
def get_data_folder():
    user = os.environ['USERDOMAIN']

    if user == 'MATTHIJSOUDELOH':
        DATA_FOLDER = 'V:/Procdata'

    elif user == 'PCMatthijs':
        DATA_FOLDER = 'E:/Procdata'

    elif user == 'MOL-MACHINE':
        DATA_FOLDER = 'C:/Procdata'

    elif user == 'KEREMSARIKAYA-P':
        DATA_FOLDER = 'D:/Procdata'

    return DATA_FOLDER

def get_rawdata_drive(animal_id,protocols):
    if isinstance(animal_id, str):
        animal_id =  [animal_id]
    if isinstance(protocols, str):
        protocols =  [protocols]

    if np.isin(protocols[0], ['IM','GR','GN','SP']):
        drives       = {'LPE09665' : 'H:',
                    'LPE09830' : 'H:',
                    'LPE11495' : 'H:',
                    'LPE11998' : 'H:',
                    'LPE12013' : 'H:',
                    'LPE10884' : 'I:',
                    'LPE10885' : 'I:',
                    'LPE11622' : 'I:',
                    'LPE10883' : 'J:',
                    'LPE11086' : 'J:',
                    'LPE10919' : 'K:',
                    'LPE12223' : 'K:',
                    'LPE12385' : 'K:'
                    }
    elif np.isin(protocols[0], ['VR','DM','DN','DP']):
        drives       = {'LPE09667' : 'L:',
                    'LPE09829' : 'L:',
                    'LPE11081' : 'L:',
                    'LPE11086' : 'L:',
                    'LPE11495' : 'L:',
                    'LPE10884' : 'L:',
                    'LPE11623' : 'L:',
                    'LPE11622' : 'L:',
                    'LPE11997' : 'L:',
                    'LPE11998' : 'M:',
                    'LPE12013' : 'M:',
                    'LPE12223' : 'M:',
                    'LPE12385' : 'M:'
                    }
    return itemgetter(*animal_id)(drives) + '\\RawData\\'

# person = {"name": "Alice", "age": 25, "occupation": "Engineer"}

# name_and_occupation = itemgetter("name")
# print(name_and_occupation(person))

# drives       = {'LPE09665' : 'G:',
#                 'LPE09830' : 'G:',
#                 'LPE11495' : 'G:',
#                 'LPE09885' : 'H:'}
# itemgetter(*animal_id)
def get_local_drive():
    user = os.environ['USERDOMAIN']

    if user == 'MATTHIJSOUDELOH':
        LOCAL_DRIVE = 'T:/'

    elif user == 'PCMatthijs':
        LOCAL_DRIVE = 'E:/'

    elif user == 'MOL-MACHINE':
        LOCAL_DRIVE = 'C:/'

    elif user == 'KEREMSARIKAYA-P':
        LOCAL_DRIVE = 'D:/'

    return LOCAL_DRIVE

def get_drive_name():

    ## drives:
    drivelist = {'VISTA 1': 'G:',
    'VISTA 2': 'H:',
    'VISTA 3': 'I:',
    'VISTA 4': 'J:',
    'VISTA 5': 'K:',
    'VISTA 6': 'L:',
    'VISTA 7': 'M:',
    'VISTA 8': 'N:',
    'VISTA 1 BACKUP': 'O:',
    'VISTA 2 BACKUP': 'P:',
    'VISTA 3 BACKUP': 'Q:',
    'VISTA 4 BACKUP': 'R:',
    'VISTA 5 BACKUP': 'S:',
    'VISTA 6 BACKUP': 'T:',
    'VISTA 7 BACKUP': 'U:',
    'VISTA 8 BACKUP': 'V:',
    }
    return drivelist