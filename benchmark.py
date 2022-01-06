from math import inf, sqrt, log10, cos
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import angle
import pyargus
from pyargus.directionEstimation import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.io import wavfile
from scipy.signal.filter_design import normalize

def read_info_file(path):
    f = open(path, 'r')
    mics = int(f.readline())
    matrix_radius = float(f.readline())
    doa = int(f.readline())

    f.close()
    return mics, matrix_radius, doa


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


def prepare_signal(microphones=8, noise_dbm = 100, reverbation = 0):
    rec_signal = []

    for i in range(1, microphones + 1):
        audiofile_samplerate, audiofile_data_original = wavfile.read("./generated_audio/bed/0a7c2a8d_nohash_0/mic_" + str(i) + ".wav")
        rec_signal.append(audiofile_data_original)
        
    #SCALING SIGNAL TO -1; 1
    rec_signal = rec_signal / np.max(rec_signal)
    plt.subplot(3,1,1)
    plt.title("Original signal")
    plt.plot(rec_signal[0])
    
    # TODO add types of noises.
    noise = np.random.normal(0,np.sqrt(10**-2),(microphones,len(rec_signal[0])))        #create noise
    rec_signal_noised, received_snr = add_noise_to_signal(rec_signal, noise, noise_dbm)

    plt.subplot(3,1,3)
    plt.title("Signal with added noise. SNR: " + str(received_snr) + " dB")
    plt.plot(rec_signal_noised[0])
    #plt.show()

    return rec_signal_noised, audiofile_samplerate


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
    plt.title("MUSIC")
    plt.plot(realdoa, max(MUSIC), "r*")
    first_peak = argmax(MUSIC)
    if first_peak > 180:
        first_peak -= 180
    plt.text(0, 0, "MUSIC: Real DOA: " + str(realdoa) + " Estimated DOA: " + str(first_peak) + " and/or " + str(first_peak + 180) + "\n")
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

def compute_doa_via_GCC_PHAT(signal, microphones, radius, signal_samplerate):
    # This approach is created into given way:
    # We are estimating for every pair of microphone the GCC-PHAT correlation
    # After that, we're searching for the pair, which peak amplitude is the most in the middle
    # Last step is computing difference from middle to peak and write that into array
    #rec signal is an array [number_of_channels][sample_size]

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

    print(corr_peaks)
    
    signal_length = len(signal[0])
    current_best_peak_index = 0
    for i in range (0, len(corr_peaks)): 
        if abs(corr_peaks[i] - signal_length) < abs(corr_peaks[current_best_peak_index] - signal_length):
            current_best_peak_index = i
    
    first_best_peak = current_best_peak_index
    second_best_peak = (current_best_peak_index + microphones // 2) % microphones
    print("current_best_peeak index pair: " + str(first_best_peak) + " and " + str(second_best_peak))
    best_peaks_correlation_peak = np.argmax(xcorr_freq(signal[first_best_peak], signal[second_best_peak]))
    print("correlation between part: " + str(best_peaks_correlation_peak))
    microphone_sound_arrival_two_pairs_corr_difference = best_peaks_correlation_peak - len(signal[0])
    microphone_sound_arrival_pair = first_best_peak if microphone_sound_arrival_two_pairs_corr_difference > 0 else second_best_peak
    print("Assumed microphone pair nr: " + str(microphone_sound_arrival_pair) + " (readed from 0), angle: <" + \
                                            str(microphone_sound_arrival_pair * 360 / microphones) + "; " + \
                                            str((microphone_sound_arrival_pair + 1) * 360 / microphones) + ")")
    #microphone_sound_arrival_pair_corr_difference = len(signal[0]) - np.argmax(xcorr_freq(signal[microphone_sound_arrival_pair], \
    #                                                                                      signal[microphone_sound_arrival_pair + 1]))

    estimated_angle = (((microphone_sound_arrival_pair * 360 / microphones) + ((microphone_sound_arrival_pair + 1) * 360 / microphones)) / 2)
    text_to_print = "Estimated angle based on radius, samplerate and gccphat.\n There can be big errors if some of the values are ridiculous.\n Angle between: " + \
           str((microphone_sound_arrival_pair * 360 / microphones)) + " and " + str((microphone_sound_arrival_pair + 1) * 360 / microphones) + \
               " center: " + str(estimated_angle) + "\n"
    print(text_to_print)
    

    plt.title("GCC PHAT")
    plt.plot(xcorr_freq(signal[microphone_sound_arrival_pair], signal[microphone_sound_arrival_pair + 1]))
    plt.text(0, 0, text_to_print)

def compute_doa_via_neural_network(signal, microphones, radius):
    #TODO
    pass
# ---------------------------------------------------------------------------------------------------------

def main():
    microphones, radius, realdoa = read_info_file("./generated_audio/bed/0a7c2a8d_nohash_0/info.txt")
    signal, signal_samplerate = prepare_signal(microphones, 0)
    plt.figure(2)
    plt.subplot(3, 1, 1)
    compute_doa_via_MUSIC(signal, microphones, radius, realdoa)
    plt.subplot(3, 1, 2)
    compute_doa_via_GCC_PHAT(signal, microphones, radius, signal_samplerate)
    plt.subplot(3, 1, 3)
    compute_doa_via_neural_network(signal, microphones, radius)
    plt.show(block=True)

if __name__ == "__main__":
    main()







    #copy
    # def compute_doa_via_GCC_PHAT(signal, microphones, radius, signal_samplerate):
    # # This approach is created into given way:
    # # We are estimating for every pair of microphone the GCC-PHAT correlation
    # # After that, we're searching for the pair, which peak amplitude is the most in the middle
    # # Last step is computing difference from middle to peak and write that into array
    # #rec signal is an array [number_of_channels][sample_size]
    # plt.figure(3)

    # detection_angle = 360 / microphones
    # corr_peaks = []
    # for i in range (0, microphones - 1):
    #     # plt.subplot(microphones, 1, i + 1)
    #     corr = xcorr_freq(signal[i], signal[i+1])
    #     corr_peaks.append(np.argmax(corr))
    #     # plt.plot(corr)
    #     # doa.append()

    
    # corr = xcorr_freq(signal[microphones - 1], signal[0])
    # corr_peaks.append(np.argmax(corr))
    # length_of_signal_in_samples = len(signal[0])

    # print(corr_peaks)
    
    # signal_length = len(signal[0])
    # current_best_peak_index = 0
    # for i in range (0, len(corr_peaks)): 
    #     if abs(corr_peaks[i] - signal_length) < abs(corr_peaks[current_best_peak_index] - signal_length):
    #         current_best_peak_index = i
    
    # first_best_peak = current_best_peak_index
    # second_best_peak = (current_best_peak_index + microphones // 2) % microphones
    # print("current_best_peeak index pair: " + str(first_best_peak) + " and " + str(second_best_peak))
    # best_peaks_correlation_peak = np.argmax(xcorr_freq(signal[first_best_peak], signal[second_best_peak]))
    # print("correlation between part: " + str(best_peaks_correlation_peak))
    # microphone_sound_arrival_two_pairs_corr_difference = best_peaks_correlation_peak - len(signal[0])
    # microphone_sound_arrival_pair = first_best_peak if microphone_sound_arrival_two_pairs_corr_difference > 0 else second_best_peak
    # print("Assumed microphone pair nr: " + str(microphone_sound_arrival_pair) + " (readed from 0), angle: <" + \
    #                                         str(microphone_sound_arrival_pair * 360 / microphones) + "; " + \
    #                                         str((microphone_sound_arrival_pair + 1) * 360 / microphones) + ")")
    # #microphone_sound_arrival_pair_corr_difference = len(signal[0]) - np.argmax(xcorr_freq(signal[microphone_sound_arrival_pair], \
    # #                                                                                      signal[microphone_sound_arrival_pair + 1]))

    # estimated_angle = (((microphone_sound_arrival_pair * 360 / microphones) + ((microphone_sound_arrival_pair + 1) * 360 / microphones)) / 2)
    # print("Estimated angle based on radius, samplerate and gccphat. There can be big errors if some of the values are ridiculous. Estimated angle: " + \
    #        str(estimated_angle))
    
    # plt.title("GCC PHAT")
    # plt.plot(xcorr_freq(signal[microphone_sound_arrival_pair], signal[microphone_sound_arrival_pair + 1]))
    # plt.text(0, 0, "Estimated angle based on radius, samplerate and gccphat.\nThere can be big errors if some of the values are ridiculous.\n" + \
    #                "Estimated angle: " + str(estimated_angle))




    # def compute_doa_via_GCC_PHAT(signal, microphones, radius, signal_samplerate):
    # # This approach is created into given way:
    # # We are estimating for every pair of microphone the GCC-PHAT correlation
    # # After that, we're searching for the pair, which peak amplitude is the most in the middle
    # # Last step is computing difference from middle to peak and write that into array
    # #rec signal is an array [number_of_channels][sample_size]
    # plt.figure(3)

    # detection_angle = 360 / microphones
    # corr_peaks = []
    # for i in range (0, microphones - 1):
    #     # plt.subplot(microphones, 1, i + 1)
    #     corr = xcorr_freq(signal[i], signal[i+1])
    #     corr_peaks.append(np.argmax(corr))
    #     # plt.plot(corr)
    #     # doa.append()

    
    # corr = xcorr_freq(signal[microphones - 1], signal[0])
    # corr_peaks.append(np.argmax(corr))

    # print(corr_peaks)
    
    # signal_length = len(signal[0])
    # current_best_peak_index = 0
    # for i in range (0, len(corr_peaks)): 
    #     if abs(corr_peaks[i] - signal_length) < abs(corr_peaks[current_best_peak_index] - signal_length):
    #         current_best_peak_index = i
    
    # print("current_best_peak index pair: " + str(current_best_peak_index))

    # estimated_angle = "from " + str(current_best_peak_index * 360 / microphones) + " to " + str((current_best_peak_index + 1) * 360 / microphones)
    # print("Estimated angle based on radius, samplerate and gccphat. Estimated angle: " + str(estimated_angle))
    
    # estimate = "closer to first angle" if corr_peaks[current_best_peak_index] > signal_length else "closer to second angle"

    # plt.title("GCC PHAT")
    # plt.plot(xcorr_freq(signal[current_best_peak_index], signal[current_best_peak_index + 1]))
    # plt.text(0, 0, "Estimated angle based on radius, samplerate and gccphat.\n" + \
    #                "Estimated angle: " + str(estimated_angle) + " probably " + estimate + "\n")