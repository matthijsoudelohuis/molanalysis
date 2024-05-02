import os
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

def get_rawdata_drive(animal_id):
    if isinstance(animal_id, str):
        animal_id =  [animal_id]
    print('Make specific for active and passive protocols:\n')
    drives       = {'LPE09665' : 'G:',
                    'LPE09830' : 'G:',
                    'LPE11495' : 'G:',
                    'LPE10884' : 'H:',
                    'LPE10885' : 'H:',
                    'LPE11622' : 'H:',
                    'LPE10883' : 'I:',
                    'LPE11086' : 'I:',
                    'LPE10919' : 'J:'
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