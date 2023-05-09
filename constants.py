import os

# --- PATHS --------------------------------------------------------------------
# figure out the paths depending on which machine you are on
user = os.environ['USERDOMAIN']

if user == 'MATTHIJSOUDELOH':
    DATA_FOLDER = 'V:/Procdata'

elif user == 'MOL':
    DATA_FOLDER = 'V:/Procdata'


# --- SESSIONS -----------------------------------------------------------------
# list all sessions for different purposes:

vista_sessions = {'LPE09665' : ['2023_03_14'],
                     'LPE09830' : ['2023_04_10']}

