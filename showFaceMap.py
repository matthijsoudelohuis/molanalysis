import numpy as np
from matplotlib import pyplot as plt

facemapfile = "W:\\Users\\Matthijs\\Rawdata\\LPE09829\\2023_03_29\\VR\\Behavior\\VR_LPE09829_camera_2023-03-29T15_32_29_proc.npy"

proc = np.load(facemapfile,allow_pickle=True).item()

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(proc['avgframe_reshape'])
plt.title("average frame")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(proc['avgmotion_reshape'], vmin=-10, vmax=10)
plt.title("average motion")
plt.axis("off")
plt.show()

plt.figure(figsize=(15, 8))
for i in range(15):
    ax = plt.subplot(3, 5, i + 1)
    ax.imshow(proc['motMask_reshape'][0][:, :, i] / proc['motMask_reshape'][0][:, :, i].std(), vmin=-2, vmax=2)
    ax.axis("off")
plt.show()

