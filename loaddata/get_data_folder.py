import os
# figure out the paths depending on which machine you are on
def get_data_folder():
    user = os.environ['USERDOMAIN']

    if user == 'MATTHIJSOUDELOH':
        DATA_FOLDER = 'V:/Procdata'

    elif user == 'PCMatthijs':
        DATA_FOLDER = 'E:/Procdata'

    elif user == 'MOL-MACHINE':
        DATA_FOLDER = 'C:/Procdata'

    return DATA_FOLDER

def get_local_drive():
    user = os.environ['USERDOMAIN']

    if user == 'MATTHIJSOUDELOH':
        LOCAL_DRIVE = 'T:/'

    elif user == 'PCMatthijs':
        LOCAL_DRIVE = 'E:/'

    elif user == 'MOL-MACHINE':
        LOCAL_DRIVE = 'C:/'

    return LOCAL_DRIVE