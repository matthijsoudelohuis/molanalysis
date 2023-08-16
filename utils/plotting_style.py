import pandas as pd
import seaborn as sns
# from matplotlib.lines import Line2D
from operator import itemgetter

desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 14)

################################################################
## Series of function that spit out lists of colors for different combinations of 
## areas, protocols, mice, stimuli, etc. 

def get_clr_areas(areas):
    palette       = {'V1'  : sns.xkcd_rgb['bright blue'],
                    'PM' : sns.xkcd_rgb['coral'],
                    'AL' : sns.xkcd_rgb['emerald'],
                    'RSP' : sns.xkcd_rgb['maroon']}
    return itemgetter(*areas)(palette)

def get_clr_protocols(protocols):
    palette       = {'GR': sns.xkcd_rgb['bright blue'],
                    'SP' : sns.xkcd_rgb['coral'],
                    'RF' : sns.xkcd_rgb['emerald'],
                    'VR' : sns.xkcd_rgb['emerald'],
                    'IM' : sns.xkcd_rgb['maroon']}
    return itemgetter(*protocols)(palette)

def get_clr_stimuli_vr(stimuli):
    stims           = ['A','B','C','D']
    clrs            = sns.color_palette('husl', 4)
    palette         = {stims[i]: clrs[i] for i in range(len(stims))}
    return itemgetter(*stimuli)(palette)

def get_clr_stimuli_vr_norm(stimuli):
    stims           = [0,1,2,3]
    clrs            = sns.color_palette('husl', 4)
    palette         = {stims[i]: clrs[i] for i in range(len(stims))}
    return itemgetter(*stimuli)(palette)

def get_clr_stimuli_vr(stimuli):
    stims           = ['A','B','C','D']
    clrs            = sns.color_palette('husl', 4)
    palette         = {stims[i]: clrs[i] for i in range(len(stims))}
    return itemgetter(*stimuli)(palette)

def get_clr_blocks(stimuli):
    clrs            = ['#ff8274','#74f1ff']
    return clrs

# decoding_inputs = ['rate', 'phase', 'energy', 'rate+lfp', 'rate+energy']
# decoding_inputs_control = ['shuffled_rate', 'rate+shuffled_phase']

# AREAS = ['V1', 'PPC', 'CG1']
# LAYERS = ['SG', 'G', 'IG']

# def get_markers_and_labels(palette_dict, labels_dict=None):
#     markers, labels = [], []
#     for key in palette_dict.keys():
#         if labels_dict is not None:
#             labels.append(labels_dict[key])
#         else:
#             labels.append(key)
#         markers.append(
#             Line2D([0], [0], marker='o', color=palette_dict[key], lw=0))
#     return markers, labels


# palette = sns.color_palette('colorblind')

# decoding_inputs_colormap = {'rate' : palette[0],
#                             'phase' : palette[1],
#                             'energy' : palette[2],
#                             'rate+phase' : palette[4],
#                             'rate+energy' : palette[6],
#                             'shuffled_rate' : palette[7],
#                             'rate+shuffled_phase' : palette[8]}

# shuffle_grey = sns.xkcd_rgb['grey']

# small_decoding_panel_size = [5, 3]

# small_square_side = 3

# small_decoding_panel_size_with_legend = [7, 3]

# palette2 = sns.color_palette('husl', n_colors=7)
# animals_colormap = {'2003' : palette2[0],
#                     '2004' : palette2[1],
#                     '2009' : palette2[2],
#                     '2010' : palette2[3],
#                     '2011' : palette2[4],
#                     '2012' : palette2[5],
#                     '2013' : palette2[6]}



# trial_type_strings = {'X' : 'Visual',
#                       'Y' : 'Audio'}

# area_palette = {'V1'  : sns.xkcd_rgb['bright blue'],
#                 'PPC' : sns.xkcd_rgb['coral'],
#                 'CG1' : sns.xkcd_rgb['emerald']}

# area_cmap = {'V1' : sns.light_palette(area_palette['V1'], as_cmap=True),
#              'PPC': sns.light_palette(area_palette['PPC'], as_cmap=True),
#              'CG1': sns.light_palette(area_palette['CG1'], as_cmap=True)}


# stimulus_palette_all = {(1, 2, 3, 4) : sns.xkcd_rgb['grey'],
#                         (1, 2) : sns.xkcd_rgb['indian red'],
#                         (3, 4) : sns.xkcd_rgb['orange yellow'],
#                         1      : sns.xkcd_rgb['indian red'],
#                         2      : sns.xkcd_rgb['indian red'],
#                         3      : sns.xkcd_rgb['orange yellow'],
#                         4      : sns.xkcd_rgb['orange yellow']}

# # -----------------------------------------------------------------------------

# reward_palette = {0      : sns.xkcd_rgb['bright red'],
#                   1      : sns.xkcd_rgb['mid green']}

# lick_palette = {0      : sns.xkcd_rgb['grey'],
#                 1      : sns.xkcd_rgb['mid green']}

# modality_palette = {'X' : sns.xkcd_rgb['light green'],
#                     'Y' : sns.xkcd_rgb['light red'],
#                     'C' : sns.xkcd_rgb['sea blue'],
#                     'P' : sns.xkcd_rgb['grey']}


# stimulus_palette = {1 : sns.xkcd_rgb['grey'],
#                     2 : sns.xkcd_rgb['indian red'],
#                     3 : sns.xkcd_rgb['orange yellow']}


# stimulus_change_palette = {1      : sns.xkcd_rgb['grey'],
#                            2      : sns.xkcd_rgb['lightish blue'],
#                            3      : sns.xkcd_rgb['dark teal']}

# lick_side_palette =     {'n'      : sns.xkcd_rgb['grey'],
#                          'L'      : sns.xkcd_rgb['lightish blue'],
#                          'R'      : sns.xkcd_rgb['dark teal']}

# targets_palette = {'rew'     : sns.xkcd_rgb['mid green'],
#                    'lick'    : sns.xkcd_rgb['grey'],
#                    'side'    : sns.xkcd_rgb['lightish blue'],
#                    'mod'     : sns.xkcd_rgb['light red'],
#                    'stim'    : sns.xkcd_rgb['orange yellow'],
#                    'stim_ch' : sns.xkcd_rgb['dark teal'],
#                    'X_stim'  : sns.xkcd_rgb['orange yellow'],
#                    'Y_stim'  : sns.xkcd_rgb['indian red'],
#                    'X_stim_ch': sns.xkcd_rgb['dark teal'],
#                    'Y_stim_ch': sns.xkcd_rgb['sea blue']}



# targets_palette_matthijs = {'correctResponse'           : sns.xkcd_rgb['mid green'],
#                             'Lick'                      : sns.xkcd_rgb['grey'],
#                             'trialType'                 : sns.xkcd_rgb['light red'],
#                             'visualOriPostChangeNorm'   : sns.xkcd_rgb['orange yellow'],
#                             'audioFreqPostChangeNorm'   : sns.xkcd_rgb['indian red'],
#                             'visualOriChangeNorm'       : sns.xkcd_rgb['dark teal'],
#                             'audioFreqChangeNorm'       : sns.xkcd_rgb['sea blue']}


# lick_palette = {0 : sns.xkcd_rgb['grey'],
#                 1 : sns.xkcd_rgb['dark teal']}


# layer_palette = {'SG' : sns.xkcd_rgb['periwinkle blue'],
#                  'G'  : sns.xkcd_rgb['orchid'],
#                  'IG' : sns.xkcd_rgb['coral']}

# # -----------------------------------------------------------------------------

# reward_labels = {0 : 'No reward',
#                  1 : 'Reward'}

# lick_labels = {0 : 'No lick',
#                1 : 'Lick'}

# modality_labels= {'X' : 'Visual',
#                   'Y' : 'Auditory',
#                   'P' : 'Probe',
#                   'C' : 'Conflict'}

# stimulus_labels = {1 : 'No stimulus',
#                    2 : 'Stimulus 1',
#                    3 : 'Stimulus 2'}

# stimulus_change_labels = {1      : 'No change',
#                           2      : 'Small change',
#                           3      : 'Big change'}

# lick_side_labels     = {'n'      : 'No lick',
#                         'L'      : 'Left lick',
#                         'R'      : 'Right lick'}

# targets_labels = {'rew'     : 'Reward',
#                   'lick'    : 'Lick',
#                   'side'    : 'Lick side',
#                   'mod'     : 'Modality',
#                   'stim'    : 'Stimulus identity',
#                   'stim_ch' : 'Stimulus change',
#                   'X_stim'  : 'Orientation',
#                   'Y_stim'  : 'Frequency',
#                   'X_stim_ch' : 'Orientation change',
#                   'Y_stim_ch' : 'Frequency change'}


# targets_labels_matthijs = {'correctResponse'           : 'Reward',
#                            'Lick'                      : 'Lick',
#                            'trialType'                 : 'Modality',
#                            'visualOriPostChangeNorm'   : 'Orientation',
#                            'visualOriChangeNorm'       : 'Orientation change',
#                            'audioFreqPostChangeNorm'   : 'Frequency',
#                            'audioFreqChangeNorm'       : 'Frequency change'}

# lick_labels = {0 : 'No lick',
#                1 : 'Lick'}

# layer_labels = {'SG' : 'SG',
#                 'G'  : 'G',
#                 'IG' : 'IG'}

# unit_score_labels = {'auc' : 'AUC score',
#                      'feature_importance' : 'Feature importance'}


# factor_metric_labels = {'kurtosis' : 'Kurtosis',
#                         'perc_nonzero_units' : 'Units with\nnonzero factor (%)',
#                         'peak_time' : 'Peak time',
#                         'lambda' : 'Factor size'}