import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def compute_power_spectra(fs,data):
    """
    Compute the power spectrum of the dF/F signal for each neuron in a session.
    Parameters
    ----------
    fs : float
        sampling frequency in Hz
    data : pandas DataFrame
        The dF/F data to compute the power spectrum from. The columns are the
        different neurons and the rows are the time bins.
    Returns
    -------
    freqs : numpy array
        The frequencies at which the power spectrum was computed.
    power_spectra : numpy array
        The power spectrum averaged across all neurons
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    T = data.shape[0]
    power_spectra = []
    for neuron in range(data.shape[1]):
        fft_vals = np.fft.fft(data[:,neuron])
        power_spectrum = np.abs(fft_vals)**2
        power_spectra.append(power_spectrum)
    power_spectra = np.array(power_spectra)
    avg_power_spectrum = np.mean(power_spectra, axis=0)
    freqs = np.fft.fftfreq(T, d=1/fs)
    freqs_out = freqs[freqs >= 0]
    psd = avg_power_spectrum[freqs >= 0]
    return freqs_out,psd


def my_highpass_filter(data, cutoff, fs, order=2): # Define a high-pass filter function
    """
    data : pandas DataFrame
    The dF/F data to compute the power spectrum from. The columns are the
    different neurons and the rows are the time bins.
    """
    if isinstance(data, pd.DataFrame):
        columns = data.columns
        index = data.index
        data = data.to_numpy()
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)  # High-pass filter
    y = filtfilt(b, a, data, axis=0)  # Apply the filter with zero-phase distortion
    if columns is not None and index is not None:
        y = pd.DataFrame(y, columns=columns, index=index) # Reinstantiate pd DataFrame with same columns and indices
    
    return y
