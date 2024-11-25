import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import copy

from scipy.stats import ttest_ind,ttest_1samp
from scipy.stats import wilcoxon


#     # #######    #    #     #    ######  #######  #####  ######  
##   ## #         # #   ##    #    #     # #       #     # #     # 
# # # # #        #   #  # #   #    #     # #       #       #     # 
#  #  # #####   #     # #  #  #    ######  #####    #####  ######  
#     # #       ####### #   # #    #   #   #             # #       
#     # #       #     # #    ##    #    #  #       #     # #       
#     # ####### #     # #     #    #     # #######  #####  #       

def mean_resp_image(ses):
    nNeurons = np.shape(ses.respmat)[0]
    imageids = np.unique(ses.trialdata['ImageNumber'])
    respmean = np.empty((nNeurons,len(imageids)))
    for im,imid in enumerate(imageids):
        respmean[:,im] = np.mean(ses.respmat[:,ses.trialdata['ImageNumber']==imid],axis=1)
    return respmean,imageids

def mean_resp_gr(ses,trialfilter=None):

    data        = copy.deepcopy(ses.respmat)
    trial_ori   = ses.trialdata['Orientation']

    if trialfilter is not None:
        data        = data[:,trialfilter]
        trial_ori   = trial_ori[trialfilter]

    # get signal correlations:
    [N,K]           = np.shape(data) #get dimensions of response matrix

    oris            = np.sort(trial_ori.unique())
    ori_counts      = ses.trialdata.groupby(['Orientation'])['Orientation'].count().to_numpy()
    assert(len(ori_counts) == 16 or len(ori_counts) == 8)
    resp_meanori    = np.empty([N,len(oris)])

    for i,ori in enumerate(oris):
        resp_meanori[:,i] = np.nanmean(data[:,trial_ori==ori],axis=1)
    
    respmat_res                     = data.copy()

    ## Compute residuals:
    for ori in oris:
        ori_idx     = np.where(trial_ori==ori)[0]
        temp        = np.mean(respmat_res[:,ori_idx],axis=1)
        respmat_res[:,ori_idx] = respmat_res[:,ori_idx] - np.repeat(temp[:, np.newaxis], len(ori_idx), axis=1)

    return resp_meanori,respmat_res

def mean_resp_gn(ses,trialfilter=None):

    data        = copy.deepcopy(ses.respmat)
    trial_ori   = ses.trialdata['centerOrientation']
    trial_spd   = ses.trialdata['centerSpeed']

    if trialfilter is not None:
        data        = data[:,trialfilter]
        trial_ori   = trial_ori[trialfilter]
        trial_spd   = trial_spd[trialfilter]

    # get signal correlations:
    [N,K]           = np.shape(data) #get dimensions of response matrix

    oris            = np.sort(pd.Series.unique(trial_ori)).astype('int')
    speeds          = np.sort(pd.Series.unique(trial_spd)).astype('int')
    noris           = len(oris) 
    nspeeds         = len(speeds)

    ### Mean response per condition:
    resp_mean       = np.empty([N,noris,nspeeds])

    for iO,ori in enumerate(oris):
        for iS,speed in enumerate(speeds):
            
            idx_trial = np.logical_and(trial_ori==ori,trial_spd==speed)
            resp_mean[:,iO,iS] = np.nanmean(data[:,idx_trial],axis=1)

    ## Compute residual response:
    respmat_res = data.copy()
    for iO,ori in enumerate(oris):
        for iS,speed in enumerate(speeds):
            
            idx_trial = np.logical_and(trial_ori==ori,trial_spd==speed)
            tempmean = np.nanmean(data[:,idx_trial],axis=1)
            respmat_res[:,idx_trial] -= tempmean[:,np.newaxis]

    return resp_mean,respmat_res

 #####  ######     #    ####### ### #     #  #####     ####### #     # #     # ### #     #  #####  
#     # #     #   # #      #     #  ##    # #     #       #    #     # ##    #  #  ##    # #     # 
#       #     #  #   #     #     #  # #   # #             #    #     # # #   #  #  # #   # #       
#  #### ######  #     #    #     #  #  #  # #  ####       #    #     # #  #  #  #  #  #  # #  #### 
#     # #   #   #######    #     #  #   # # #     #       #    #     # #   # #  #  #   # # #     # 
#     # #    #  #     #    #     #  #    ## #     #       #    #     # #    ##  #  #    ## #     # 
 #####  #     # #     #    #    ### #     #  #####        #     #####  #     # ### #     #  #####  

def get_pref_orispeed(resp_mean,oris,speeds,asindex=True):
    
    #Find preferred orientation and speed for each cell
    pref_ori,pref_speed = np.unravel_index(np.argmax(resp_mean.reshape([np.shape(resp_mean)[0],-1]),axis=1), (len(oris), len(speeds)))
    
    if not asindex:
        pref_ori = oris[pref_ori] #find preferred orientation for each
        pref_speed = speeds[pref_speed] #find preferred orientation for each 
        
    return pref_ori,pref_speed

def comp_grating_responsive(sessions,pthr = 0.001):
    
    for ises in tqdm(range(len(sessions)),desc= 'Identifying significant responsive neurons for each session'):
        if sessions[ises].sessiondata['protocol'][0]=='GR':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(sessions[ises].trialdata['Orientation'].unique())
            sessions[ises].celldata['vis_resp'] = False
            
            for iN in range(N):
                for iOri,Ori in enumerate(oris):
                    # pval = ttest_1samp(sessions[ises].respmat[iN,sessions[ises].trialdata['Orientation']==Ori],popmean=0)[1]
                    pval = wilcoxon(sessions[ises].respmat[iN,sessions[ises].trialdata['Orientation']==Ori])[1]
                    if pval < pthr:
                        sessions[ises].celldata.loc[iN,'vis_resp'] = True

        elif sessions[ises].sessiondata['protocol'][0]=='GN':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(pd.Series.unique(sessions[ises].trialdata['centerOrientation']))
            speeds                          = np.sort(pd.Series.unique(sessions[ises].trialdata['centerSpeed']))
            sessions[ises].celldata['vis_resp'] = False
            
            for iN in range(N):
                for iOri,Ori in enumerate(oris):
                    for iS,Speed in enumerate(speeds):
                        idx_trial = np.logical_and(sessions[ises].trialdata['centerOrientation']==Ori,sessions[ises].trialdata['centerSpeed']==Speed)
                        # pval = ttest_1samp(sessions[ises].respmat[iN,idx_trial],popmean=0)[1]
                        pval = wilcoxon(sessions[ises].respmat[iN,idx_trial])[1]
                        if pval < pthr:
                            sessions[ises].celldata.loc[iN,'vis_resp'] = True

    return sessions

def compute_prefori(response_matrix,conditions_vector):
    """
    Compute preferred orientation for multiple neurons across trials
    Parameters:
    - response_matrix: 2D array or list where each row corresponds to responses of a single neuron across trials
    - conditions: 1D array or list with the condition for each trial (e.g. orientation)
    
    Returns:
    - preferred orientation for each neuron
    """

    # Convert response_matrix and orientations_vector to numpy arrays
    response_matrix         = np.array(response_matrix)
    conditions_vector       = np.array(conditions_vector)

    conditions              = np.sort(np.unique(conditions_vector))
    C                       = len(conditions)

    # Ensure the dimensions match
    if np.shape(response_matrix)[1] != len(conditions_vector):
        raise ValueError("Number of trials in response_matrix should match the length of orientations_vector.")

    [N,K]           = np.shape(response_matrix) #get dimensions of response matrix

    resp_mean       = np.empty((N,C))

    for iC,cond in enumerate(conditions):
        tempmean                            = np.nanmean(response_matrix[:,conditions_vector==cond],axis=1)
        resp_mean[:,iC]                     = tempmean
    
    pref_cond             = conditions[np.argmax(resp_mean,axis=1)]

    return pref_cond

def compute_tuning(response_matrix,conditions_vector,tuning_metric='OSI'):

    """
    Compute Orientation Selectivity Index (OSI) for multiple neurons across trials

    Parameters:
    - response_matrix: 2D array or list where each row corresponds to responses of a single neuron across trials
    - conditions: 1D array or list with the condition for each trial (e.g. orientation)
    
    Returns:
    - metric values: e.g. List of Orientation Selectivity Indices for each neuron
    """

    # Convert response_matrix and orientations_vector to numpy arrays
    response_matrix         = np.array(response_matrix)
    conditions_vector       = np.array(conditions_vector)

    conditions              = np.sort(np.unique(conditions_vector))
    C                       = len(conditions)

    # Ensure the dimensions match
    if np.shape(response_matrix)[1] != len(conditions_vector):
        raise ValueError("Number of trials in response_matrix should match the length of orientations_vector.")

    [N,K]           = np.shape(response_matrix) #get dimensions of response matrix

    resp_mean       = np.empty((N,C))
    resp_res        = response_matrix.copy()

    for iC,cond in enumerate(conditions):
        tempmean                            = np.nanmean(response_matrix[:,conditions_vector==cond],axis=1)
        resp_mean[:,iC]                     = tempmean
        resp_res[:,conditions_vector==cond] -= tempmean[:,np.newaxis]

    if tuning_metric=='OSI':
        tuning_values = compute_OSI(resp_mean)
    elif tuning_metric=='DSI':
        tuning_values = compute_DSI(resp_mean)
    elif tuning_metric=='gOSI':
        tuning_values = compute_gOSI(resp_mean)
    elif tuning_metric=='tuning_var':
        tuning_values = compute_tuning_var(response_matrix,resp_res)
    else: 
        print('unknown tuning metric requested')

    return tuning_values

def compute_gOSI(response_matrix):
    """
    Compute Global Orientation Selectivity Index (gOSI) for multiple neurons across trials

    Parameters:
    - response_matrix: 2D array or list where each row corresponds to the average
        responses of a single neuron across trials of the same condition (e.g. orientation)

    Returns:
    - gOSI_values: List of Global Orientation Selectivity Indices for each neuron
    """

    # Convert response_matrix and orientations_vector to numpy arrays
    response_matrix     = np.array(response_matrix)

    # Min-max normalize each row independently
    response_matrix = (response_matrix - np.min(response_matrix, axis=1, keepdims=True)) / (np.max(response_matrix, axis=1, keepdims=True) - np.min(response_matrix, axis=1, keepdims=True))

    # Initialize a list to store gOSI values for each neuron
    gOSI_values = []

    # Iterate over each neuron
    for neuron_responses in response_matrix:
        # Compute the vector components (real and imaginary parts)
        vector_components = neuron_responses * np.exp(2j * np.deg2rad(np.arange(0, 360, 360 / len(neuron_responses))))

        # Sum the vector components
        vector_sum = np.sum(vector_components)

        # Calculate the gOSI
        gOSI = np.abs(vector_sum) / np.sum(neuron_responses)

        gOSI_values.append(gOSI)

    return gOSI_values

def compute_OSI(response_matrix):
    """
    Compute Orientation Selectivity Index (OSI) for multiple neurons

    Parameters:
    - response_matrix: 2D array or list where each row corresponds to responses of a single neuron to different orientations

    Returns:
    - OSI_values: List of Orientation Selectivity Indices for each neuron
    """

    # Convert response_matrix to a numpy array
    response_matrix = np.array(response_matrix)

     # Min-max normalize each row independently
    response_matrix = (response_matrix - np.min(response_matrix, axis=1, keepdims=True)) / (np.max(response_matrix, axis=1, keepdims=True) - np.min(response_matrix, axis=1, keepdims=True))

    # Initialize a list to store OSI values for each neuron
    OSI_values = []

    # Iterate over each neuron
    for neuron_responses in response_matrix:
        # Find the preferred orientation (angle with maximum response)
        pref_orientation_index = np.argmax(neuron_responses)
        pref_orientation_response = neuron_responses[pref_orientation_index]

        # Find the orthogonal orientation (angle 90 degrees away from preferred)
        orthogonal_orientation_index = (pref_orientation_index + len(neuron_responses) // 2) % len(neuron_responses)
        orthogonal_orientation_response = neuron_responses[orthogonal_orientation_index]

        # Compute OSI for the current neuron
        if pref_orientation_response == 0:
            # Handle the case where the response to the preferred orientation is zero
            OSI = 0.0
        else:
            OSI = (pref_orientation_response - orthogonal_orientation_response) / pref_orientation_response

        OSI_values.append(OSI)

    return OSI_values

def compute_DSI(response_matrix):
    """
    Compute Direction Selectivity Index (DSI) for multiple neurons

    Parameters:
    - response_matrix: 2D array or list where each row corresponds to responses of a single neuron to different orientations

    Returns:
    - DSI_values: List of Direction Selectivity Indices for each neuron
    """

    # Convert response_matrix to a numpy array
    response_matrix = np.array(response_matrix)

    # Initialize a list to store DSI values for each neuron
    DSI_values = []

    # Iterate over each neuron
    for neuron_responses in response_matrix:
        # Find the preferred direction (angle with maximum response)
        pref_direction_index = np.argmax(neuron_responses)
        pref_direction_response = neuron_responses[pref_direction_index]

        # Find the opposite direction (angle 180 degrees away from preferred)
        opposite_direction_index = (pref_direction_index + len(neuron_responses) // 2) % len(neuron_responses)
        opposite_direction_response = neuron_responses[opposite_direction_index]

        # Compute DSI for the current neuron
        if pref_direction_response == 0:
            # Handle the case where the response to the preferred direction is zero
            DSI = 0.0
        else:
            DSI = (pref_direction_response - opposite_direction_response) / pref_direction_response

        DSI_values.append(DSI)

    return DSI_values

def compute_tuning_var(resp_mat,resp_res):
    """
    Compute variance explained by conditions for multiple single trial responses

    Parameters:
    - resp_mat: responses across all trials for a number of neurons
    - resp_res: residuals to different conditions

    Returns:
    - Tuning Variance: 0-1 (1: all variance across trials is explained by conditions)
    """
    assert np.shape(resp_mat) == np.shape(resp_res), "shape mismatch"
    tuning_var = 1 - np.var(resp_res,axis=1) / np.var(resp_mat,axis=1)

    return tuning_var

def compute_tuning_wrapper(sessions):
    """
    Wrapper function to compute several tuning metrics for GR and GN protocols
    Currently computes OSI, DSI, gOSI, and Tuning Variance, plus preferred orientation for GR
    For GN tuning variance and preferred orientation and speed
    """
    for ises in tqdm(range(len(sessions)),desc= 'Computing tuning metrics: '):
        if sessions[ises].sessiondata['protocol'].isin(['GR'])[0]:
            sessions[ises].celldata['OSI'] = compute_tuning(sessions[ises].respmat,
                                                        sessions[ises].trialdata['Orientation'],
                                                        tuning_metric='OSI')
            sessions[ises].celldata['gOSI'] = compute_tuning(sessions[ises].respmat,
                                                            sessions[ises].trialdata['Orientation'],
                                                            tuning_metric='gOSI')
            sessions[ises].celldata['tuning_var'] = compute_tuning(sessions[ises].respmat,
                                                            sessions[ises].trialdata['Orientation'],
                                                            tuning_metric='tuning_var')
            sessions[ises].celldata['pref_ori'] = compute_prefori(sessions[ises].respmat,
                                                            sessions[ises].trialdata['Orientation'])
            sessions[ises].celldata['DSI'] = compute_tuning(sessions[ises].respmat,
                                                    sessions[ises].trialdata['Orientation'],
                                                    tuning_metric='DSI')

        elif sessions[ises].sessiondata['protocol'].isin(['GN'])[0]:

            resp_mean,resp_res      = mean_resp_gn(sessions[ises])
            sessions[ises].celldata['tuning_var'] = compute_tuning_var(resp_mat=sessions[ises].respmat,resp_res=resp_res)
            oris, speeds    = [np.unique(sessions[ises].trialdata[col]).astype('int') for col in ('centerOrientation', 'centerSpeed')]
            sessions[ises].celldata['pref_ori'],sessions[ises].celldata['pref_speed'] = get_pref_orispeed(resp_mean,oris,speeds)


    return sessions


#     #    #    ####### #     # ######     #    #          ### #     #    #     #####  #######  #####  
##    #   # #      #    #     # #     #   # #   #           #  ##   ##   # #   #     # #       #     # 
# #   #  #   #     #    #     # #     #  #   #  #           #  # # # #  #   #  #       #       #       
#  #  # #     #    #    #     # ######  #     # #           #  #  #  # #     # #  #### #####    #####  
#   # # #######    #    #     # #   #   ####### #           #  #     # ####### #     # #             # 
#    ## #     #    #    #     # #    #  #     # #           #  #     # #     # #     # #       #     # 
#     # #     #    #     #####  #     # #     # #######    ### #     # #     #  #####  #######  #####  

def compute_tuning_SNR(ses):
    """
    #From stringer et al. 2019
    To compute the tuning-related SNR (Fig. 1f), we first estimated the signal variance of each neuron 
    as the covariance of its response to all stimuli across two repeats
    # The noise variance was defined as the difference between the within-repeat variance (reflecting both signal and noise)
    # and this signal variance estimate, and the SNR was defined as their ratio. The SNR estimate is positive when a neuron 
    # has responses to stimuli above its noise baseline; note that as is an unbiased estimate, it can take negative values 
    # when the true signal variance is zero.
    """
    nNeurons = np.shape(ses.respmat)[0]

    if 'repetition' not in ses.trialdata:
        ses.trialdata['repetition'] = np.empty(np.shape(ses.trialdata)[0])
        for iT in range(len(ses.trialdata)):
            ses.trialdata.loc[iT,'repetition'] = np.sum(ses.trialdata['ImageNumber'][:iT] == ses.trialdata['ImageNumber'][iT])
    
    # Compute the covariance between the first and the second presentation of each image
    cov_signal = np.zeros(nNeurons)
    for iN in range(nNeurons):
        resp1 = ses.respmat[iN,ses.trialdata['ImageNumber'][ses.trialdata['repetition']==0].index[np.argsort(ses.trialdata['ImageNumber'][ses.trialdata['repetition']==0])]]
        resp2 = ses.respmat[iN,ses.trialdata['ImageNumber'][ses.trialdata['repetition']==1].index[np.argsort(ses.trialdata['ImageNumber'][ses.trialdata['repetition']==1])]]
        cov_signal[iN] = np.cov(resp1,resp2)[0,1]
        
        # cov_signal[iN] = np.cov(ses.respmat[iN,ses.trialdata['ImageNumber'][ses.trialdata['repetition']==0].index],
        #                   ses.respmat[iN,ses.trialdata['ImageNumber'][ses.trialdata['repetition']==1].index])[0,1]
    cov_noise = np.var(ses.respmat,axis=1) - cov_signal
    SNR = cov_signal / cov_noise

    return SNR

def compute_splithalf_reliability(ses):
    """
    #From Tong et al. 2023
    Spearman-Brown corrected correlation coefficient across half-splits of repeated presentation

    """
    nNeurons = np.shape(ses.respmat)[0]

    if 'repetition' not in ses.trialdata:
        ses.trialdata['repetition'] = np.empty(np.shape(ses.trialdata)[0])
        for iT in range(len(ses.trialdata)):
            ses.trialdata.loc[iT,'repetition'] = np.sum(ses.trialdata['ImageNumber'][:iT] == ses.trialdata['ImageNumber'][iT])
    
    # Compute the covariance between the first and the second presentation of each image
    corr = np.zeros(nNeurons)
    for iN in range(nNeurons):
        resp1 = ses.respmat[iN,ses.trialdata['ImageNumber'][ses.trialdata['repetition']==0].index[np.argsort(ses.trialdata['ImageNumber'][ses.trialdata['repetition']==0])]]
        resp2 = ses.respmat[iN,ses.trialdata['ImageNumber'][ses.trialdata['repetition']==1].index[np.argsort(ses.trialdata['ImageNumber'][ses.trialdata['repetition']==1])]]
     
        corr[iN] = np.corrcoef(resp1,resp2)[0,1]
    
    rel = (2 * corr) / (1 + corr)

    return corr,rel


def compute_sparseness(responses):
    """Computes the sparseness of average neuronal responses to natural images. 
    Input is a 2D numpy array where axis=0 are the different neurons 
    and axis=1 are the responses across the different natural images.
    Returns a 1D numpy array with the sparseness for each neuron."""
    mean_response = np.mean(responses,axis=1)
    mean_square_response = np.mean(np.square(responses),axis=1)
    sparseness = (mean_response ** 2) / mean_square_response
    return sparseness

def compute_selectivity_index(responses):
    """Computes the selectivity index of average neuronal responses to natural images. 
    Input is a 2D numpy array where axis=0 are the different neurons 
    and axis=1 are the responses across the different natural images.
    Returns a 1D numpy array with the selectivity index for each neuron."""
    max_responses = np.max(responses,axis=1)
    mean_responses = np.mean(responses,axis=1)
    selectivity_index = (max_responses - mean_responses) / (max_responses + mean_responses)
    return selectivity_index

def compute_fano_factor(responses):
    """Computes the Fano Factor of average neuronal responses to natural images. 
    Input is a 2D numpy array where axis=0 are the different neurons 
    and axis=1 are the responses across the different natural images.
    Returns a 1D numpy array with the Fano Factor for each neuron."""
    mean_response = np.mean(responses,axis=1)
    variance = np.var(responses,axis=1)
    fano_factor = variance / mean_response
    return fano_factor

def compute_gini_coefficient(responses):
    """Computes the Gini coefficient of average neuronal responses to natural images. 
    Input is a 2D numpy array where axis=0 are the different neurons 
    and axis=1 are the responses across the different natural images.
    Returns a 1D numpy array with the Gini coefficient for each neuron."""
    nNeurons = responses.shape[0]
    gini_coefficient = np.zeros(nNeurons)
    for iN in range(nNeurons):
        sorted_responses = np.sort(responses[iN,:])
        N = len(sorted_responses)
        cumulative_responses = np.cumsum(sorted_responses)
        gini_numerator = np.sum((2 * np.arange(1, N + 1) - N - 1) * sorted_responses)
        gini_denominator = N * np.sum(sorted_responses)
        gini_coefficient[iN] = gini_numerator / gini_denominator
    return gini_coefficient


