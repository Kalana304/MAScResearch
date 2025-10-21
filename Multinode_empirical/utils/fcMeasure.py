################################################################################################################
#
# Author            : Kalana Abeywardena
# Affiliation       : University of Toronto, Canada 
# Date of creation  : 01/11/2023
#
# This script provides the helper functions to compute the fucntional connectivity either in time or freq domain.
#
#################################################################################################################

import dcor
import numpy as np
import scipy.signal as ss

def dcor_connectivity(sensors, data):
    """ 
        This function computes pair-wise distance correlation between time-series data from different nodes in
        brain graph (ref: https://en.wikipedia.org/wiki/Distance_correlation).

        Parameters:
        -----------
            sensors (int): No. of nodes in the graph
            data (ndarray): Time series data from each node

        Returns:
        --------
            connectivity_matrix (ndarray): functional connectome based on distance corr
            connectivity_vector (ndarray): upper triangle elements 
    """
    # Predefining connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)

    for n in range(sensors):
        for m in range(sensors):
            connectivity_matrix[n, m] = dcor.distance_correlation(data[n, :], data[m, :], method='naive')
    
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 

    return connectivity_matrix, connectivity_vector

def filteration(data, f_min, f_max, fs):
    """
    Performing band pass filteration for synchrony-based measures for different freq. bands.
    
    Parameters
    ----------
        data (ndarray): Time series data 
        f_min (float): Low pass frequency of band pass filter given in hertz 
        f_max (float): High pass frequency of band pass filter given in hertz
        fs (float): Sampling frequency of data given in hertz

    Returns 
    -------
        filtered data (ndarray): Filtered time series data
    """
   
    # Filter design
    sos = ss.butter(N=10,Wn=[f_min,f_max],btype='bandpass', analog=False,output='sos',fs=fs)

    return ss.sosfilt(sos, data)

# synchornization analysis - spectral analysis

class SpectrumSynch(object):

    def __init__(self, time_series, tstart, tend, fmin, fmax, fs, window_size, step_window) -> None:
        """ 
            This class implements weighted phase locked value-based function synchronization measured paire-wise. 
            Phase information is deived using Wavelet transformation, and are weighted based on their normalized amplitudes 
            to avoid unneccessary synchronization due to DC levels. 

            Parameters:
            -----------
                time_series (ndarray): simulated time-series data
                tstart (int)         : start time index of time-series data
                tend (int)           : end time index of time series data
                fmin (float)         : low-pass frequency of band pass filter given in hertz
                fmax (float)         : high-pass frequency of band pass filter given in hertz
                fs (float)           : sampling frequency of data given in hertz
                window_size (int)    : sliding window size
                step_window (int)    : shift of each window
        """
        nTrials, _, _ = time_series.shape
        
        self.ue = time_series[:, 0::2, tstart : tend]
        self.nodes = self.ue.shape[1]
        self.t1 = tstart
        self.t2 = tend
        self.fs = fs
        self.Minfreq = fmin
        self.Maxfreq = fmax
        self.nTrials = nTrials
        self.window_size = window_size
        self.step_window = step_window
        
        return

    def WaveletCalc(self, ue_series, center_freq=12.):
        """ 
            This metho computes the Wavelt transformation of a given time series data from 
            excitatory subpopulation. 

            Parameters:
            -----------
                ue_series (ndarray): time series data of a single node
                center_freq (float): center frequency in hertz
            
            Returns:
            --------
                freq_arr (ndarray): frequency array 
                wtmatr (ndarray): Wavelet matrix 
        """
        freq_arr = np.linspace(1, self.fs / 2, 1000)
        widths = center_freq * self.fs / (2 * freq_arr * np.pi)

        wtmatr = ss.cwt(ue_series, ss.morlet2, widths, w=center_freq)
        wtmatr = np.flipud(wtmatr)

        return freq_arr, wtmatr

    def SpectBand(self, ue_series):
        """ 
            This method extracts wavelet coefficients of a given frequency band and computes
            phase series and normalized amplitude series for correlation computation.

            Parameters:
            -----------
                ue_series (ndarray): time series of a single node
            
            Returns:
            --------
                psi_series (ndarray): Phase series of the given frequency range
                prop_series (ndarray): Normalized amplitude series of the given frequency range
        """

        freq_arr, spect_ue = self.WaveletCalc(ue_series)        # computes the wavelet matrix and corresponding frequency array
        
        fmax_ind = np.argmin(abs(freq_arr - self.Maxfreq))      # find matrix index corresponding to maximum frequency
        fmin_ind = np.argmin(abs(freq_arr - self.Minfreq))      # find matric index corresponding to minimum frequency

        spect_band = np.flipud(spect_ue)[fmin_ind : fmax_ind, :]    # get the wavelet coefficients of the spectral range 
        
        spect_argmax = np.argmax(np.abs(spect_band), axis=0)        # identify the dominating wavelet coefficients 
        spect_series = np.array([spect_band[x, t] for t, x in enumerate(spect_argmax)])     

        psi_series = np.angle(spect_series)         # extract the phase information
        amp_series = np.abs(spect_series)           # extract the amplitude information

        maxamp_series = np.max(abs(spect_ue), axis=0)   # get the maximum amplitude
        prop_series =  amp_series / maxamp_series       # normalize the amplitude series

        nanVal = np.argwhere(np.isnan(prop_series))     # identify any not a number values
        if len(nanVal) != 0:
            prop_series[nanVal] = 0                     # those with nan values are replaced with 0

        return psi_series, prop_series
    
    def modphasesynch(self, psi_1, psi_2, prop_1, prop_2):
        """ 
            This method implements weighted Phase Locking Value (PLV), that weights the phase time series with 
            relative wavelet strength using the normalized amplitude wavelet series. This zeros out phase synchrony 
            for flat activities in simulated time series. 

            Parameters:
            -----------
                psi_1 (ndarray): phase series of node 1
                psi_2 (ndarray): phase series of node 2
                prop_1 (ndarray): normalized amplitude series of node 1
                prop_2 (ndarray): normalized amplitude series of node 2
            
            Returns:
            --------
                mean_plc_synch (float): mean PLV value between two nodes
        """
        plv_synch_local = []

        psi_1_exp = np.exp(1j * psi_1)
        psi_2_exp = np.exp(1j * psi_2)

        for t in range(0, self.t2 - self.t1 - self.window_size, self.step_window):
            phase_diff = psi_1_exp[t : t + self.window_size] / psi_2_exp[t : t + self.window_size]  # get the phase difference
            weight_ = prop_1[t : t + self.window_size] * prop_2[t : t + self.window_size]           # computes the overall weight
            plv_synch_local.append(np.mean(abs(weight_ * phase_diff)))                              # weighted plv synchronizaition for each time window
        
        plv_synch_local = np.array(plv_synch_local)
        return np.mean(plv_synch_local)

    def calc_synchronization(self):

        # Predefining connectivity matrix
        connectivity_matrix = np.zeros([self.nTrials, self.nodes, self.nodes],dtype=float)
        
        for q in range(self.nTrials):
            for n in range(self.nodes):
                psi_1, prop_1 = self.SpectBand(self.ue[q, n, :])
                for m in range(n, self.nodes):
                    psi_2, prop_2 = self.SpectBand(self.ue[q, m, :])
                    connectivity_matrix[q, n, m] = self.modphasesynch(psi_1=psi_1, psi_2=psi_2, prop_1=prop_1, prop_2=prop_2)
        
        connectivity_matrix = np.mean(connectivity_matrix, axis=0)
        connectivity_matrix = connectivity_matrix + connectivity_matrix.T

        return connectivity_matrix
    
def plv_connectivity(sensors,data):
    """
    Computing PLV connectivity (ref. https://sapienlabs.org/lab-talk/eeg-connectivity-using-phase-lag-index/). Unlike SpectrumSynch()
    this implements the PLV from its usual definition based on Hilbert transformation.
    
    Parameters
    ----------
        sensors (int) : No of sensors used for capturing EEG
        data (ndarray): time series data
    
    Returns
    -------
        connectivity_matrix (ndarray): PLV connectivity matrix
        connectivity_vector (ndarray): PLV connectivity vector

    """
    # Predefining connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)
    
    # Computing hilbert transform
    data_points = data.shape[-1]
    data_hilbert = np.imag(ss.hilbert(data))
    phase = np.arctan(data_hilbert/data)
    
    # Computing connectivity matrix 
    for i in range(sensors):
        for k in range(sensors):
            connectivity_matrix[i,k] = np.abs(np.sum(np.exp(1j*(phase[i,:]-phase[k,:]))))/data_points
            
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 
      
    # returning connectivity matrix and vector
    return connectivity_matrix, connectivity_vector
            
def pli_connectivity(sensors,data):
    """
    Computing Phase Lag Index (PLI) connectivity (ref:https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.20346)
    
    Parameters
    ----------
        sensors (int): No of nodes in the brain graph
        data (ndarray): time series data 

    Returns
    -------
        connectivity_matrix (ndarray): PLI connectivity matrix
        connectivity_vector (ndarray): PLI connectivity vector

    """
    # Predefining connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)
    
    # Computing hilbert transform
    data_points = data.shape[-1]
    data_hilbert = np.imag(ss.hilbert(data))
    phase = np.arctan(data_hilbert/data)
    
    # Computing connectivity matrix
    for i in range(sensors):
        for k in range(sensors):
            connectivity_matrix[i,k] = np.abs(np.sum(np.sign(phase[i,:]-phase[k,:])))/data_points
    
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 
    
    # returning connectivity matrix and vector
    return connectivity_matrix, connectivity_vector


def ccf_connectivity(sensors,data):
    """
    Computing Cross Correlation (ref:https://en.wikipedia.org/wiki/Cross-correlation).
    
    Parameters
    ----------
        sensors (int): No of nodes in the brain graph
        data (ndarray): time series data 

    Returns
    -------
        connectivity_matrix (ndarray): CCF connectivity matrix
        connectivity_vector (ndarray): CCF connectivity vector

    """
    # Predefining connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)
    
    # Computing cross correlation
    for i in range(sensors):
        for k in range(sensors):
            temp = np.corrcoef(data[i,:],data[k,:])
            connectivity_matrix[i,k] = temp[0][1]
    
    # Computing connectvity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 
    
    # Returning connectivity matrix and connectivity vector
    return connectivity_matrix, connectivity_vector


def coh_connectivity(sensors,data,f_min,f_max,fs):
    """
    Computing Coherence

    Parameters
    ----------
        sensors (int): No of nodes in the brain graph
        data (ndarray): time series data 
        f_min (float): Low pass frequency of band pass filter given in hertz
        f_max (float): High pass frequency of band pass filter given in hertz
        fs (float): Sampling frequency of data given in hertz
    

    Returns
    -------
        connectivity_matrix (ndarray): COH connectivity matrix
        connectivity_vector (ndarray): COH connectivity vector
    """

    # Predefinig connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)
    
    # Computing coherence 
    for i in range(sensors):
        for k in range(sensors):
            f, Cxy = ss.coherence(data[i,:],data[k,:],fs = fs)
            connectivity_matrix[i,k] = np.mean(Cxy[np.where((f>=f_min) & (f<=f_max))])
    
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 
    
    # returning connectivity matrix and/or vector
    return connectivity_matrix, connectivity_vector

def icoh_connectivity(sensors,data,f_min,f_max,fs):
    """
    Computing imaginary Coherence
    
    Parameters
    ----------
        sensors (int): No of nodes in the brain graph
        data (ndarray): time series data 
        f_min (float): Low pass frequency of band pass filter given in hertz
        f_max (float): High pass frequency of band pass filter given in hertz
        fs (float): Sampling frequency of data given in hertz
    

    Returns
    -------
        connectivity_matrix (ndarray): ICOH connectivity matrix
        connectivity_vector (ndarray): ICOH connectivity vector

    """ 
    # Predefinig connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)
    
    # Computing imaginary coherence 
    for i in range(sensors):
        _, Pxx = ss.welch(data[i,:],fs=fs) 
        for k in range(sensors):
            _, Pyy = ss.welch(data[k,:],fs=fs) 
            f, Pxy = ss.csd(data[i,:],data[k,:],fs=fs)
            icoh = np.imag(Pxy)/(np.sqrt(Pxx*Pyy))
            connectivity_matrix[i,k] = np.mean(icoh[np.where((f>=f_min) & (f<=f_max))])
    
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 
    
    # returning connectivity matrix and/or vector
    return connectivity_matrix, connectivity_vector

            
    