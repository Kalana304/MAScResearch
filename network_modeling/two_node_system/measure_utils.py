import random
import numpy as np
from copy import deepcopy

from scipy import signal
from sklearn.feature_selection import mutual_info_regression as MI

# Stability evaluation

def spectrum_analysis(eig_vals):
    n_eigvals = len(eig_vals)
    spec_rad = np.zeros(n_eigvals)
    spec_center = np.zeros(n_eigvals, dtype=np.complex_)
    eig_real = np.zeros((n_eigvals, 4))
    eig_img = np.zeros((n_eigvals, 4))

    for ilambda, eigval in enumerate(eig_vals):
        spec_center[ilambda] = np.mean(eigval)
        spec_rad[ilambda] = abs(eigval).max()
        eig_real[ilambda, :] = eigval.real
        eig_img[ilambda, :] = eigval.imag

    if n_eigvals > 1:
        _max_rad_i = np.argmax(spec_rad)
        return spec_rad[_max_rad_i], spec_center[_max_rad_i], eig_real[_max_rad_i, :], eig_img[_max_rad_i, :]

    elif n_eigvals == 1:
        return spec_rad[0], spec_center[0], eig_real[0, :], eig_img[0, :]
    else:
        return np.inf, 0, np.inf, 0

def CoeffVariance(time_series, tstart, tend):
    """
        This function calculates the coefficient of variance computed based on following.
        [Hellyer, Peter J., et al.]
        "Local inhibitory plasticity tunes macroscopic brain dynamics and allows the emergence of functional brain networks." NeuroImage 124 (2016): 85-95.

    """
    nTrials, _, _ = time_series.shape

    CV = np.zeros(nTrials)

    for q in range(nTrials):
        ue1_arr = time_series[q, 0, tstart : tend]
        ue2_arr = time_series[q, 2, tstart : tend]

        CV[q] = (np.std(ue1_arr) / np.mean(ue1_arr) + np.std(ue2_arr) / np.mean(ue2_arr)) / 2
    
    return CV

def LyapunovStability(time_series, tstart, tend):
    nTrials, _, _ = time_series.shape

    LE = np.zeros(nTrials)

    for q in range(nTrials):
        diff = np.abs(np.diff(time_series[q, 0, tstart : tend]))
        LE[q] = np.mean(np.log(diff))

    return LE

# Synchronization - time series analysis

class AvalanchSynch(object):
    """
        This class calculates the Avalanche-type event synchronization based on the standarized
        time series and a set zlim threshold. 
    """
    def __init__(self, time_series, thresh1, thresh2, tstart, tend) -> None:
        """
            parameters:
                time_series (np.darray): N x 4 x T numpy array of node dynamics
                thresh1 (float): z value as the threshold for node 1
                thresh2 (float): z value as the threshold for node 2
                tstart (int): starting t index
                tend (int): end t index
        """
        nTrials, _, _ = time_series.shape

        self.ue1 = time_series[:, 0, tstart : tend]
        self.ue2 = time_series[:, 2, tstart : tend]
        self.zlim1 = thresh1
        self.zlim2 = thresh2 
        self.t1 = tstart
        self.t2 = tend
        self.nTrials = nTrials
        return

    def standarize_series(self, unstd_series):
        std_ = np.std(unstd_series)
        mean_ = np.mean(unstd_series)

        if round(std_ / mean_, 2) == 0:
            norm_series = unstd_series - mean_
        
        else:
            norm_series = (unstd_series - mean_) / std_

        return norm_series
    
    def event_detection(self, ue_series, zlimit):
        t_max_events = []
        t_min_events = []

        for t in range(0, self.t2 - self.t1):
            if ue_series[t] <= -zlimit:
                t_min_events.append(t)

            if ue_series[t] >= zlimit:
                t_max_events.append(t)

        return np.array(t_max_events), np.array(t_min_events)

    def QValue_calc(self, tij_1, tji_2):
        Q = 0
        M1 = len(tij_1)
        M2 = len(tji_2)

        if M1 != 0 and M2 != 0:
            delta_tij = np.diff(tij_1)
            delta_tji = np.diff(tji_2)

            if M1 > 1 and M2 > 1:
                l_thresh = min(np.min(delta_tij), np.min(delta_tji)) / 2
            elif M1 > 1 and M2 == 1:
                l_thresh = np.min(delta_tij) / 2
            elif M1 == 1 and M2 > 1:
                l_thresh = np.min(delta_tji) / 2
            else:
                l_thresh = 0.1      

            J_12 = np.zeros((M1, M2))
            J_21 = np.zeros((M2, M1))

            # synchronization scoring mechanism
            for m_i, t_i in enumerate(tij_1):
                for m_j, t_j in enumerate(tji_2):

                    if t_i == t_j:
                        J_12[m_i, m_j] = 0.5

                    elif 0 < t_i - t_j <= l_thresh:
                        J_12[m_i, m_j] = 1.0
            
            for m_j, t_j in enumerate(tji_2):
                for m_i, t_i in enumerate(tij_1):

                    if t_j == t_i:
                        J_21[m_j, m_i] = 0.5

                    elif 0 < t_j - t_i <= l_thresh:
                        J_21[m_j, m_i] = 1.0

            c_12 = sum(J_12.flatten())
            c_21 = sum(J_21.flatten())

            Q = (c_12 + c_21) / np.sqrt(M1 * M2)
        
        return Q

    def remove_spurious(self, tseries):
        trueEvent_mask = np.where(np.diff(tseries) > 1)[0]
        tseries = tseries[trueEvent_mask]

        return tseries

    def _synchronAvg(self):
        Qavg = 0

        ue1_arr = self.standarize_series(np.mean(self.ue1, axis=0))
        ue2_arr = self.standarize_series(np.mean(self.ue2, axis=0))

        # Finding the extreme events
        t_ue1_max, t_ue1_min = self.event_detection(ue1_arr, self.zlim1)
        t_ue2_max, t_ue2_min = self.event_detection(ue2_arr, self.zlim2)

        # measure synch for maximum events
        t_ue1_max = self.remove_spurious(t_ue1_max)
        t_ue2_max = self.remove_spurious(t_ue2_max)
        Qmax = self.QValue_calc(t_ue1_max, t_ue2_max)

        # measure synch for minimum events
        t_ue1_min = self.remove_spurious(t_ue1_min)
        t_ue2_min = self.remove_spurious(t_ue2_min)
        Qmin = self.QValue_calc(t_ue1_min, t_ue2_min)

        nEvents = len(t_ue1_max) + len(t_ue2_max) + len(t_ue1_min) + len(t_ue2_min)
        nMaxEvents = len(t_ue1_max) + len(t_ue2_max)
        nMinEvents = len(t_ue1_min) + len(t_ue2_min)
        
        if nEvents != 0:
            Qavg = (nMaxEvents / nEvents) * Qmax + (nMinEvents / nEvents) * Qmin
        
        return Qavg

    def calc_synchronization(self):
        Qfinal = np.zeros(self.nTrials)

        for q in range(self.nTrials):
            ue1_arr = self.standarize_series(self.ue1[q, :])
            ue2_arr = self.standarize_series(self.ue2[q, :])

            # Finding the extreme events
            t_ue1_max, t_ue1_min = self.event_detection(ue1_arr, self.zlim1)
            t_ue2_max, t_ue2_min = self.event_detection(ue2_arr, self.zlim2)

            # measure synch for maximum events
            t_ue1_max = self.remove_spurious(t_ue1_max)
            t_ue2_max = self.remove_spurious(t_ue2_max)
            Qmax = self.QValue_calc(t_ue1_max, t_ue2_max)

            # measure synch for minimum events
            t_ue1_min = self.remove_spurious(t_ue1_min)
            t_ue2_min = self.remove_spurious(t_ue2_min)
            Qmin = self.QValue_calc(t_ue1_min, t_ue2_min)

            nEvents = len(t_ue1_max) + len(t_ue2_max) + len(t_ue1_min) + len(t_ue2_min)
            nMaxEvents = len(t_ue1_max) + len(t_ue2_max)
            nMinEvents = len(t_ue1_min) + len(t_ue2_min)
            
            if nEvents != 0:
                Qfinal[q] = (nMaxEvents / nEvents) * Qmax + (nMinEvents / nEvents) * Qmin
        
        if self.calc_avg:
            Qavg = self._synchronAvg()
            return Qfinal, Qavg
        
        else:
            return Qfinal

class LocalMaxSynch(object):
    """
        This calculates the local maxima synchronization between two time series.
        [Source] 
        Kreuz, T., Mormann, F., Andrzejak, R. G., Kraskov, A., Lehnertz, K., & Grassberger, P. (2007). 
        Measuring synchronization in coupled model systems: A comparison of different approaches. Physica D: Nonlinear Phenomena, 225(1), 29-42.

    """
    def __init__(self, time_series, nSamples, tstart, tend) -> None:
        nTrials, _, _ = time_series.shape
        self.ue1 = time_series[:, 0, tstart : tend]
        self.ue2 = time_series[:, 2, tstart : tend]
        self.t1 = tstart
        self.t2 = tend
        self.nTrials = nTrials
        self.nSamples = nSamples

        return

    def event_detection(self, ue_series):
        t_events = []
        for t in range(2, self.t2 - self.t1 - 3):
            if (ue_series[t - 1] < ue_series[t]) and (ue_series[t + 1] < ue_series[t]) and \
                (ue_series[t - 2] < ue_series[t - 1]) and (ue_series[t + 2] < ue_series[t + 1]):
                t_events.append(t)
        
        return np.array(t_events) 

    def calcDelay(self, tij_1, tji_2, M1, M2):
        tij_delay = max(tji_2[-1] - tij_1[0], 0)
        tji_delay = max(tij_1[-1] - tji_2[0], 0)
 
        for i in range(M1):
            for j in range(M2):
                if 0 < tji_2[j] - tij_1[i] < tij_delay and tij_1[i] != tji_2[j]:
                    tij_delay = tji_2[j] - tij_1[i]
                    break

        for i in range(M2):
            for j in range(M1):
                if 0 < tij_1[j] - tji_2[i] < tji_delay and tij_1[j] != tji_2[i]:
                    tji_delay = tij_1[j] - tji_2[i]      
        
        tau = min(tij_delay, tji_delay) / 2
        return tau

    def QValue_calc(self, tij_1, tji_2):
        Q = 0

        M1 = len(tij_1)
        M2 = len(tji_2)

        self.l_thresh = 0.0

        if M1 != 0 and M2 != 0:
            self.l_thresh = self.calcDelay(tij_1, tji_2, M1, M2)
            # delta_tij = np.diff(tij_1)
            # delta_tji = np.diff(tji_2)

            # if M1 > 1 and M2 > 1:
            #     l_thresh = min(np.min(delta_tij), np.min(delta_tji)) / 2
            # elif M1 > 1 and M2 == 1:
            #     l_thresh = np.min(delta_tij) / 2
            # elif M1 == 1 and M2 > 1:
            #     l_thresh = np.min(delta_tji) / 2
            # else:
            #     l_thresh = 0.1      

            J_12 = np.zeros((M1, M2))
            J_21 = np.zeros((M2, M1))

            # synchronization scoring mechanism
            for m_i, t_i in enumerate(tij_1):
                for m_j, t_j in enumerate(tji_2):

                    if t_i == t_j:
                        J_12[m_i, m_j] = 0.5

                    elif 0 < t_i - t_j <= self.l_thresh:
                        J_12[m_i, m_j] = 1.0
            
            for m_j, t_j in enumerate(tji_2):
                for m_i, t_i in enumerate(tij_1):

                    if t_j == t_i:
                        J_21[m_j, m_i] = 0.5

                    elif 0 < t_j - t_i <= self.l_thresh:
                        J_21[m_j, m_i] = 1.0

            c_12 = sum(J_12.flatten())
            c_21 = sum(J_21.flatten())

            Q = (c_12 + c_21) / np.sqrt(M1 * M2)

        return Q

    def calc_surrogate(self, tij_1, tji_2):
        Qs = 0
        tji_perm = deepcopy(tji_2)

        for _ in range(self.nSamples):
            random.shuffle(tji_perm)
            Qs += (1 / self.nSamples) * self.QValue_calc(tij_1, tji_perm)
        
        return Qs

    def QRemap(self, Q, Qs):
        if Q > Qs:
            return (Q - Qs) / (1 - Qs)
        elif Q < Qs:
            return (Q - Qs) / Qs 
        else:
            return 0

    def calc_synchronization(self):
        Qfinal = np.zeros(self.nTrials)

        for q in range(self.nTrials):
            ue1_arr = self.ue1[q, :]
            ue2_arr = self.ue2[q, :]
            
            # Finding the extreme events
            t_ue1 = self.event_detection(ue1_arr)
            t_ue2 = self.event_detection(ue2_arr)

            # measure synch for local maximum events
            Qorig = self.QValue_calc(t_ue1, t_ue2)

            # measure surrogate Q value
            Qs = self.calc_surrogate(t_ue1, t_ue2)

            # readjust Q value
            Qfinal[q] = self.QRemap(Qorig, Qs)

        return Qfinal

def variation_adjust(time_series, nTrials, tstart, tend):
    for q in range(nTrials):
        for nPop in range(4):
            sig_var = np.std(time_series[q, nPop, tstart : tend])
            sig_mean = np.mean(time_series[q, nPop, tstart : tend])
            if round(sig_var / sig_mean, 3) == 0:
                time_series[q, nPop, tstart : tend] = time_series[q, nPop, tstart : tend] - sig_mean
    return time_series

def kuramoto_order_calc(time_series, tstart, tend, window_size = 200, step_window = 50):
    nTrials, _, _ = time_series.shape
    # time_series = variation_adjust(time_series, nTrials, tstart, tend)
    num_windows = (tend - tstart - window_size) // step_window
    kuramoto_order = np.zeros(nTrials)

    # ------- #
    # kuramoto_window = []

    # ue1_avg = np.mean(time_series[:, 0, : ], axis=0)
    # ui1_avg = np.mean(time_series[:, 1, : ], axis=0)
    # ue2_avg = np.mean(time_series[:, 2, : ], axis=0)
    # ui2_avg = np.mean(time_series[:, 3, : ], axis=0)
    
    
    # for t in range(tstart, tend - window_size, step_window):
    #     ue = ue1_avg[t : t + window_size] + ue2_avg[t : t + window_size]
    #     ui = ui1_avg[t : t + window_size] + ui2_avg[t : t + window_size]
    #     real_kur = ue - np.mean(ue)
    #     img_kur = ui - np.mean(ui)

    #     kuramoto_window.append(np.mean(np.abs(real_kur + 1j * img_kur)))
    
    # kuramoto_window = np.array(kuramoto_window)
    # # avg_kuramoto = np.mean(kuramoto_window)
    # # kuramoto_window = (kuramoto_window - avg_kuramoto) / avg_kuramoto

    # kuramoto_order = np.sum(kuramoto_window) / num_windows
    
    # ------- #

    for q in range(nTrials):
        trial_kuramoto = []

        time_series[q, 0, tstart : ] = time_series[q, 0, tstart : ] - np.mean(time_series[q, 0, tstart : ])
        time_series[q, 1, tstart : ] = time_series[q, 1, tstart : ] - np.mean(time_series[q, 1, tstart : ])
        time_series[q, 2, tstart : ] = time_series[q, 2, tstart : ] - np.mean(time_series[q, 2, tstart : ])
        time_series[q, 3, tstart : ] = time_series[q, 3, tstart : ] - np.mean(time_series[q, 3, tstart : ])

        for t in range(tstart, tend - window_size, step_window):
            ue = time_series[q, 0, t : t + window_size] + time_series[q, 2, t : t + window_size]
            ui = time_series[q, 1, t : t + window_size] + time_series[q, 3, t : t + window_size]
            real_kur = ue - np.mean(ue)
            img_kur = ui - np.mean(ui)

            trial_kuramoto.append(np.mean(np.abs(real_kur + 1j * img_kur)))
        
        trial_kuramoto = np.array(trial_kuramoto)
        # avg_kuramoto = np.mean(trial_kuramoto)
        # norm_kuramoto = (trial_kuramoto - avg_kuramoto) / avg_kuramoto

        # kuramotoMask = np.where(np.abs(norm_kuramoto) <= 1)
        # norm_kuramoto = norm_kuramoto[kuramotoMask]

        kuramoto_order[q] = np.sum(trial_kuramoto) / num_windows


    
    return kuramoto_order

# Synchornization - spectrum time series

# class SpectrumSynch(object):
#     def __init__(self, time_series, tstart, tend, fmin, fmax, fs, window_size, step_window) -> None:
#         nTrials, _, _ = time_series.shape
#         time_series = variation_adjust(time_series, nTrials, tstart, tend)
        
#         self.ue1 = time_series[:, 0, tstart : tend]
#         self.ue2 = time_series[:, 2, tstart : tend]
#         self.t1 = tstart
#         self.t2 = tend
#         self.fs = fs
#         self.Minfreq = fmin
#         self.Maxfreq = fmax
#         self.nTrials = nTrials
#         self.window_size = window_size
#         self.step_window = step_window
        
#         return

#     def WaveletCalc(self, ue_series, center_freq=12.):
#         freq_arr = np.linspace(1, self.fs / 2, 1000)
#         widths = center_freq * self.fs / (2 * freq_arr * np.pi)

#         wtmatr = signal.cwt(ue_series, signal.morlet2, widths, w=center_freq)
#         wtmatr = np.flipud(wtmatr)

#         return freq_arr, wtmatr

#     def SpectBand(self, ue_series):
#         freq_arr, spect_ue = self.WaveletCalc(ue_series)
        
#         fmax_ind = np.argmin(abs(freq_arr - self.Maxfreq))
#         fmin_ind = np.argmin(abs(freq_arr - self.Minfreq))

#         spect_band = np.flipud(spect_ue)[fmin_ind : fmax_ind, :]
        
#         spect_argmax = np.argmax(np.abs(spect_band), axis=0)
#         spect_series = np.array([spect_band[x, t] for t, x in enumerate(spect_argmax)])

#         psi_series = np.angle(spect_series)
#         amp_series = np.abs(spect_series)

#         return psi_series, amp_series
    
#     def GlobalSynch(self, psi_1, psi_2):
#         # Sliding window-based Global Synchronization
#         globsync_order = []

#         for t in range(0, self.t2 - self.t1 - self.window_size, self.step_window):
#             phase_sum = np.exp(1j * psi_1[t : t + self.window_size]) + np.exp(1j * psi_2[t : t + self.window_size])
#             phase_sum = phase_sum / 2 
#             globsync_order.append(np.mean(np.abs(phase_sum)))
        
#         globsync_order = np.array(globsync_order)
        
#         return np.mean(globsync_order)

#     def SVDSynch(self, psi_1, psi_2):
#         svd_synch_local = []

#         psi_1_exp = np.exp(1j * psi_1)
#         psi_2_exp = np.exp(1j * psi_2)
        
#         for t in range(0, self.t2 - self.t1 - self.window_size, self.step_window):
#             psi_mat = np.array([psi_1[t : t + self.window_size], psi_2[t : t + self.window_size]])
            
#             # SVD decomposition
#             u_vect, sing_val, _ = np.linalg.svd(psi_mat, full_matrices=False)
#             uSigma_ = u_vect * sing_val
#             svdMeasure_ = min(abs(uSigma_[0, 0] / uSigma_[0, 1]), abs(uSigma_[0, 1] / uSigma_[0, 0]))

#             if round(uSigma_[0, 0], 3) == 0 and round(uSigma_[0, 1], 3) == 0:
#                 svdMeasure_ = 0.0

#             svd_synch_local.append(svdMeasure_)

#         svd_synch_local = np.array(svd_synch_local)
        
#         return np.mean(svd_synch_local)


#     def MIPhase(self, psi_1, psi_2, n_neighbors = 4):
#         return MI(psi_1.reshape(-1, 1), psi_2, n_neighbors = n_neighbors, discrete_features=False)[0] 

#     def MIAmplitude(self, amp_1, amp_2, n_neighbors = 4):
#         return MI(amp_1.reshape(-1, 1), amp_2, n_neighbors = n_neighbors, discrete_features=False)[0]
    
#     def calc_synchronization(self):
#         GlobSync_avg = np.zeros(self.nTrials)
#         SVDSynch_avg = np.zeros(self.nTrials)
#         MI_psi = np.zeros(self.nTrials)
#         MI_abs = np.zeros(self.nTrials)

#         for q in range(self.nTrials):
#             psi_1, amp_1 = self.SpectBand(self.ue1[q, :])
#             psi_2, amp_2 = self.SpectBand(self.ue2[q, :])

#             GlobSync_avg[q] = self.GlobalSynch(psi_1=psi_1, psi_2=psi_2)
#             SVDSynch_avg[q] = self.SVDSynch(psi_1=psi_1, psi_2=psi_2)
#             MI_psi[q] = self.MIPhase(psi_1=psi_1, psi_2=psi_2)
#             MI_abs[q] = self.MIAmplitude(amp_1=amp_1, amp_2=amp_2)
        
#         return GlobSync_avg, SVDSynch_avg, MI_psi, MI_abs

# Test MI

class SpectrumSynch(object):
    def __init__(self, time_series, tstart, tend, fmin, fmax, fs, window_size, step_window) -> None:
        nTrials, _, _ = time_series.shape
        # time_series = variation_adjust(time_series, nTrials, tstart, tend)
        
        self.ue1 = time_series[:, 0, tstart : tend]
        self.ue2 = time_series[:, 2, tstart : tend]
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

        wtmatr = signal.cwt(ue_series, signal.morlet2, widths, w=center_freq)
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

        return psi_series, amp_series, prop_series
    
    def GlobalSynch(self, psi_1, psi_2, prop_1, prop_2):
        # Sliding window-based Global Synchronization
        globsync_order = []

        for t in range(0, self.t2 - self.t1 - self.window_size, self.step_window):
            phase_sum = np.mean(prop_1[t : t + self.window_size]) * np.exp(1j * psi_1[t : t + self.window_size]) +\
                        np.mean(prop_2[t : t + self.window_size]) * np.exp(1j * psi_2[t : t + self.window_size])
            phase_sum = phase_sum / 2 
            globsync_order.append(np.mean(np.abs(phase_sum)))
        
        globsync_order = np.array(globsync_order)
        
        return np.mean(globsync_order)

    def SVDSynch(self, psi_1, psi_2):
        svd_synch_local = []
        
        for t in range(0, self.t2 - self.t1 - self.window_size, self.step_window):
            psi_mat = np.array([psi_1[t : t + self.window_size], psi_2[t : t + self.window_size]])
            
            # SVD decomposition
            u_vect, sing_val, _ = np.linalg.svd(psi_mat, full_matrices=False)
            uSigma_ = u_vect * sing_val
            svdMeasure_ = min(abs(uSigma_[0, 0] / uSigma_[0, 1]), abs(uSigma_[0, 1] / uSigma_[0, 0]))

            if round(uSigma_[0, 0], 3) == 0 and round(uSigma_[0, 1], 3) == 0:
                svdMeasure_ = 0.0

            svd_synch_local.append(svdMeasure_)

        svd_synch_local = np.array(svd_synch_local)
        
        return np.mean(svd_synch_local)


    def MIPhase(self, psi_1, psi_2, prop_1, prop_2, n_neighbors = 4):
        return np.mean(prop_1) * np.mean(prop_2) * MI(psi_1.reshape(-1, 1), psi_2, n_neighbors = n_neighbors, discrete_features=False)[0] 

    def MIAmplitude(self, amp_1, amp_2, prop_1, prop_2, n_neighbors = 4):
        return np.mean(prop_1) * np.mean(prop_2) * MI(amp_1.reshape(-1, 1), amp_2, n_neighbors = n_neighbors, discrete_features=False)[0]
    
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
        GlobSync_avg = np.zeros(self.nTrials)
        SVDSynch_avg = np.zeros(self.nTrials)
        MI_psi = np.zeros(self.nTrials)
        MI_abs = np.zeros(self.nTrials)
        PLV_avg = np.zeros(self.nTrials)

        for q in range(self.nTrials):
            psi_1, amp_1, prop_1 = self.SpectBand(self.ue1[q, :])
            psi_2, amp_2, prop_2 = self.SpectBand(self.ue2[q, :])

            GlobSync_avg[q] = self.GlobalSynch(psi_1=psi_1, psi_2=psi_2, prop_1=prop_1, prop_2=prop_2)
            SVDSynch_avg[q] = self.SVDSynch(psi_1=psi_1, psi_2=psi_2)
            MI_psi[q] = self.MIPhase(psi_1=psi_1, psi_2=psi_2, prop_1=prop_1, prop_2=prop_2)
            MI_abs[q] = self.MIAmplitude(amp_1=amp_1, amp_2=amp_2, prop_1=prop_1, prop_2=prop_2)
            PLV_avg[q] = self.modphasesynch(psi_1=psi_1, psi_2=psi_2, prop_1=prop_1, prop_2=prop_2)
        
        return GlobSync_avg, SVDSynch_avg, MI_psi, MI_abs, PLV_avg



