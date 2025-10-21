## Importing necassary libraries
import dcor
import numpy as np
import scipy.signal as ss

def dcor_connectivity(sensors, data):
    # Predefining connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)

    for n in range(sensors):
        for m in range(sensors):
            connectivity_matrix[n, m] = dcor.distance_correlation(data[n, :], data[m, :], method='naive')
    
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 

    return connectivity_matrix, connectivity_vector

def filteration(data,f_min,f_max,fs):
    """
    Performing band pass filteration
    
    Parameters
    ----------
    data : Array of float
        DESCRIPTION. EEG data
    f_min : float
        DESCRIPTION. Low pass frequency of band pass filter given in hertz
    f_max : float
        DESCRIPTION. High pass frequency of band pass filter given in hertz
    fs : float
        DESCRIPTION. Sampling frequency of data given in hertz

    Returns 
    -------
    TYPE: Array of float
        DESCRIPTION. Filtered EEG data

    """
    # print("Filteration in process.....")
    
    # Filter design
    sos = ss.butter(N=10,Wn=[f_min,f_max],btype='bandpass',
                    analog=False,output='sos',fs=fs)

    # Returning filtered data
    # print("Filteration done!")
    return ss.sosfilt(sos,data)

# synchornization analysis - spectral analysis

class SpectrumSynch(object):
    def __init__(self, time_series, tstart, tend, fmin, fmax, fs, window_size, step_window) -> None:
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
        freq_arr = np.linspace(1, self.fs / 2, 1000)
        widths = center_freq * self.fs / (2 * freq_arr * np.pi)

        wtmatr = ss.cwt(ue_series, ss.morlet2, widths, w=center_freq)
        wtmatr = np.flipud(wtmatr)

        return freq_arr, wtmatr

    def SpectBand(self, ue_series):
        freq_arr, spect_ue = self.WaveletCalc(ue_series)
        
        fmax_ind = np.argmin(abs(freq_arr - self.Maxfreq))
        fmin_ind = np.argmin(abs(freq_arr - self.Minfreq))

        spect_band = np.flipud(spect_ue)[fmin_ind : fmax_ind, :]
        
        spect_argmax = np.argmax(np.abs(spect_band), axis=0)
        spect_series = np.array([spect_band[x, t] for t, x in enumerate(spect_argmax)])

        psi_series = np.angle(spect_series)
        amp_series = np.abs(spect_series)

        maxamp_series = np.max(abs(spect_ue), axis=0)
        prop_series =  amp_series / maxamp_series 

        nanVal = np.argwhere(np.isnan(prop_series))
        if len(nanVal) != 0:
            prop_series[nanVal] = 0

        return psi_series, prop_series
    
    def modphasesynch(self, psi_1, psi_2, prop_1, prop_2):
        plv_synch_local = []

        psi_1_exp = np.exp(1j * psi_1)
        psi_2_exp = np.exp(1j * psi_2)

        for t in range(0, self.t2 - self.t1 - self.window_size, self.step_window):
            phase_diff = psi_1_exp[t : t + self.window_size] / psi_2_exp[t : t + self.window_size]
            weight_ = prop_1[t : t + self.window_size] * prop_2[t : t + self.window_size]
            plv_synch_local.append(np.mean(abs(weight_ * phase_diff)))
        
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
    Computing PLV connectivity
    
    Parameters
    ----------
    sensors : INT
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data
    
    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. PLV connectivity matrix
    connectivity_vector : Vector of flaot 
        DESCRIPTION. PLV connectivity vector

    """

    
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
    Computing PLI connectivity
    
    Parameters
    ----------
    sensors : INT
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data

    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. PLI connectivity matrix
    connectivity_vector : Vector of flaot 
        DESCRIPTION. PLI connectivity vector

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
    Computing Cross Correlation
    
    Parameters
    ----------
    sensors : INT
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data

    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. CCF connectivity matrix
    connectivity_vector : Vector of float 
        DESCRIPTION. CCF connectivity vector

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
    sensors : INT
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data
    f_min : float
        DESCRIPTION. Low pass frequency of band pass filter given in hertz
    f_max : TYPE: float
        DESCRIPTION. High pass frequency of band pass filter given in hertz
    fs : TYPE: float
        DESCRIPTION. Sampling frequency of data given in hertz
    
    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. COH connectivity matrix
    connectivity_vector : Vector of float 
        DESCRIPTION. COH connectivity vector

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
    sensors : INT
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data
    f_min : float
        DESCRIPTION. Low pass frequency of band pass filter given in hertz
    f_max : TYPE: float
        DESCRIPTION. High pass frequency of band pass filter given in hertz
    fs : TYPE: float
        DESCRIPTION. Sampling frequency of data given in hertz
    
    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. ICOH connectivity matrix
    connectivity_vector : Vector of float 
        DESCRIPTION. ICOH connectivity vector

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

            
    