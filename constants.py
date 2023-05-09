"""
This script load some constant values necessary for analysis into workspace
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""
import os

# --- PATHS --------------------------------------------------------------------
# figure out the paths depending on which machine you are on

def get_data_folder():
    user = os.environ['USERDOMAIN']

    if user == 'MATTHIJSOUDELOH':
        DATA_FOLDER = 'V:/Procdata'

    elif user == 'PCMatthijs':
        DATA_FOLDER = 'E:/Procdata'

    return DATA_FOLDER
