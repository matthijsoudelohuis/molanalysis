import numpy as np


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

def compute_tuning_var(resp_mat,resp_res):
    """
    Compute variance explained by conditions for multiple single trial responses

    Parameters:
    - resp_mat: responses across all trials for a number of neurons
    - resp_res: residuals to different conditions

    Returns:
    - Tuning Variance: 0-1 (1: all variance across trials is explained by conditions)
    """
    tuning_var = 1 - np.var(resp_res,axis=1) / np.var(resp_mat,axis=1)

    return tuning_var

# # Example usage:
# orientation_responses = [[10.0, 9.0, 2.0, 8.0, 4.0, 1.0],
#                           [1.0, 5.0, 2.0, 8.0, 4.0, 1.0]]  # Replace with the actual responses to different orientations

# orientations            = [0,45,90,180,210,270]

# OSI = compute_OSI(orientation_responses)
# print("Orientation Selectivity Index (OSI):", OSI)

# gOSI = compute_gOSI(orientation_responses)
# print("Global Orientation Selectivity Index (gOSI):", gOSI)

# compute_tuning(orientation_responses,orientations,tuning_metric='OSI')
