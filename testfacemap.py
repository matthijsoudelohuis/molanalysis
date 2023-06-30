from facemap import process
from glob import glob

sesfolder = 'W:\\Users\\Matthijs\\Rawdata\\LPE09829\\2023_03_29\\VR\\Behavior'

# sesfolder = "W:\\Users\\Matthijs\\Rawdata\\NSH07422\\2023_03_13\\SP\\Behavior"

session_folders = [sesfolder]

for indexSession, folder in enumerate(session_folders):
    video_files = glob(folder+"/*.avi") # replace .ext with one of ['*.mj2','*.mp4','*.mkv','*.avi','*.mpeg','*.mpg','*.asf']
    if video_files:
        process.run(filenames=[video_files],sbin=10,motSVD=True,movSVD=True)
    else: 
        print("Could not find video files in directory")

    # if SVDs of ROIs is required, use 'save ROIs' from GUI and use the following command
    # process.run(video_files, proc="/path_to_saved_rois")

