from math import log, log10
from numpy.core.fromnumeric import argmax
import pyargus
from pyargus.directionEstimation import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.io import wavfile
from scipy.signal.filter_design import normalize

'''
SNR method 1 - we got signal and the noise.
Analyze first second of audio signal, when there is no sound to estimate noise level
Analyze all audio signal for signal+noise level
estimate snr ratio

SNR method 2 - we just make special snr from audio (assuing, that it's clean signal),
and we add some amount of noise to create specific snr.
'''

def signalPower(x):
    return np.average(x**2) #modified parseval's theory

def add_noise_to_signal(rec_signal, noise, wanted_snr):
    signal_power = signalPower(rec_signal)
    noise_power = signalPower(noise)
    pn = signal_power / (10**(wanted_snr/10))

    # normalizing noisy signal to get unity power then multiplying it by the new power to achieve the required SNR
    noised_signal = (noise / np.sqrt(noise_power)) * np.sqrt(pn);
    noised_signal_power = signalPower(noised_signal)
    SNR = 10 * log10(signal_power / noised_signal_power) #just test

    return rec_signal + noised_signal, SNR #return noised original signal and computed snr

def prepare_signal(microphones=8):
    rec_signal = []

    for i in range(1, microphones + 1):
        audiofile_samplerate, audiofile_data_original = wavfile.read("./generated_audio/mic_" + str(i) + ".wav")
        rec_signal.append(audiofile_data_original)
        #plt.subplot(M, 1, i)
        #plt.xlim([-50,50])
        #plt.ylim([-100,100])
        #plt.plot(rec_signal[i - 1])
        #print(str(signal_to_noise_db(audiofile_data_original)) + " dB")
        

    #SCALING SIGNAL TO -1; 1
    rec_signal = rec_signal / np.max(rec_signal)
    plt.subplot(3,1,1)
    plt.title("Original signal")
    plt.plot(rec_signal[0])
    
    noise = np.random.normal(0,np.sqrt(10**-2),(microphones,len(rec_signal[0]))) #create noise
    rec_signal_noised, received_snr = add_noise_to_signal(rec_signal, noise, 25)

    plt.subplot(3,1,3)
    plt.title("Signal with added noise. SNR: " + str(received_snr) + " dB")
    plt.plot(rec_signal_noised[0])
    #plt.show()

    return rec_signal_noised


# ---------------------------------------------------------------------------------------------------------


def compute_doa_via_MUSIC(signal, M=8, r=0.034, realdoa = 34):
    """
     Description:
     ------------
         Benchmark - MUSIC algorithm

    Parameters:
    -----------          
        M      : (int) Number of microphones. Default value: 8
        r      : (float) Radius of UCA, Default value: 0.038
        realdoa: (float) Real DOA of readed wav
    """
    plt.figure(2)

    rec_signal_transposed = np.asarray(signal).T

    ## R matrix calculation
    R = pyargus.directionEstimation.corr_matrix_estimate(rec_signal_transposed, imp="mem_eff")
    
    # Generate scanning vectors
    num_of_microphones = M
    radius_of_uca = r 
    incident_angles= np.arange(0, 360, 1)
    #pyargus.directionEstimation.gen_uca_scanning_vectors()
    uca_scanning_vectors = pyargus.directionEstimation.gen_uca_scanning_vectors(num_of_microphones, radius_of_uca, incident_angles)
      
    # DOA estimation           
    MUSIC = pyargus.directionEstimation.DOA_MUSIC(R, uca_scanning_vectors, signal_dimension = M - 2)
    #DOA_plot(MUSIC, incident_angles, log_scale_min = -50)
    plt.subplot(1,1,1)
    plt.title("MUSIC")
    plt.plot(realdoa, max(MUSIC), "r*")
    plt.text(1,1,"MUSIC: Real DOA: " + str(realdoa) + " Estimated DOA: " + str(argmax(MUSIC)))
    plt.plot(abs(MUSIC))
    #plt.show()


# ---------------------------------------------------------------------------------------------------------

#cross correlation (and gcc-phat)
def xcorr_freq(s1,s2):
    pad1 = np.zeros(len(s1))
    pad2 = np.zeros(len(s2))
    s1 = np.hstack([s1,pad1])
    s2 = np.hstack([pad2,s2])
    f_s1 = fft(s1)
    f_s2 = fft(s2)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = abs(f_s)
    denom[denom < 1e-6] = 1e-6
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation. if commented - xcorr, else phat
    return np.abs(ifft(f_s))[1:]

def compute_doa_via_GCC_PHAT(signal, microphones):
    # This approach is created into given way:
    # We are estimating for every pair of microphone the GCC-PHAT correlation
    # After that, we're searching for the pair, which peak amplitude is the most in the middle
    # Last step is computing difference from middle to peak and write that into array
    #rec signal is an array [number_of_channels][sample_size]
    plt.figure(3)

    detection_angle = 360 / microphones
    corr_peaks = []
    for i in range (0, microphones - 1):
        # plt.subplot(microphones, 1, i + 1)
        corr = xcorr_freq(signal[i], signal[i+1])
        corr_peaks.append(np.argmax(corr))
        # plt.plot(corr)
        # doa.append()

    
    corr = xcorr_freq(signal[microphones - 1], signal[0])
    corr_peaks.append(np.argmax(corr))
    length_of_signal_in_samples = len(signal[0])

    #test
    corr = xcorr_freq(signal[1], signal[5])
    corr_peaks.append(np.argmax(corr))
    corr = xcorr_freq(signal[2], signal[4])
    corr_peaks.append(np.argmax(corr))
    #endtest
    return 0

    #OTHERS TEST
    # audiofile_samplerate, a = wavfile.read("./generated_audio/mic_1.wav")
    # audiofile_samplerate2, b = wavfile.read("./generated_audio/mic_2.wav")
    # plt.title("GCC PHAT")
    # plt.plot(xcorr_freq(a,b))
    #plt.show()
    #OTHERS TEST




# ---------------------------------------------------------------------------------------------------------


def main():
    microphones = 8
    radius = 0.034
    realdoa = 34

    signal = prepare_signal(microphones)
    compute_doa_via_MUSIC(signal, microphones, radius, realdoa)
    compute_doa_via_GCC_PHAT(signal, microphones)
    plt.show()

if __name__ == "__main__":
    main()