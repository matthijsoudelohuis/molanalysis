import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from utils.explorefigs import plot_PCA_gratings
from loaddata.session import Session
import pandas as pd


savedir = 'D:\\OneDrive\\PostDoc\\Figures\\Neural - Gratings\\GainModel\\'



nNeurons        = 1000
nTrials         = 3200

noise_level     = 0.3
gain_level      = 10
offset_level    = 0.5

noris           = 16

oris            = np.linspace(0,360,noris+1)[:-1]
locs            = np.random.rand(nNeurons) * np.pi * 2  # circular mean
kappas          = np.random.rand(nNeurons) * 2  # concentration

ori_trials = np.random.choice(oris,nTrials)

R = np.empty((nNeurons,nTrials))
for iN in range(nNeurons):
    R[iN,:] = vonmises.pdf(np.deg2rad(ori_trials), loc=locs[iN], kappa=kappas[iN])

# plt.figure()
# plt.imshow(R)

# plt.scatter(ori_trials,R[23,:])

gain_trials = np.random.rand(nTrials)
gain_weights = np.random.randn(nNeurons) * gain_level

G = 1 + np.outer(gain_weights,gain_trials) 

offset_trials = np.random.rand(nTrials)
offset_weights = np.random.randn(nNeurons) * offset_level

O = np.outer(offset_weights,offset_trials) 

N = np.random.randn(nNeurons,nTrials) * noise_level

Full = R * G + O + N

model_ses = Session()
model_ses.respmat = Full
model_ses.trialdata = pd.DataFrame()
model_ses.trialdata['Orientation'] = ori_trials
model_ses.respmat_runspeed = gain_trials

fig = plot_PCA_gratings(model_ses)

fig.savefig(os.path.join(savedir,'AffineModel_Gain%1.2f_O%1.2f_noise%1.2f_N%d_K%d' % (gain_level,offset_level,noise_level,nNeurons,nTrials) + '.png'), format = 'png')
