from math import inf, sqrt, log10, cos, acos
from tokenize import Double
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import angle
import pyargus
from pyargus.directionEstimation import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.io import wavfile
from scipy.signal.filter_design import normalize
import statistics as stat
import neural_network_dataset_prepare as nndp
import neural_network as nn
from tqdm import tqdm
import os 


SHOW_MATLAB_VERBOSE_PLOTS = False

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
and we add some amount of noise to create specific snr. #Edit: This is used later.
'''

def signalPower(x):
    return np.average(x**2) #modified parseval's theory

def add_noise_to_signal(rec_signal, noise, wanted_snr):
    signal_power = signalPower(rec_signal)
    noise_power = signalPower(noise)
    pn = signal_power / (10**(wanted_snr/10))

    # normalizing noisy signal to get unity power then multiplying it by the new power to achieve the required SNR
    noised_signal = (noise / np.sqrt(noise_power)) * np.sqrt(pn);

    return rec_signal + noised_signal


def prepare_signal(signal_path, microphones=8, noise_dbm = 0, reverbation = 0):
    rec_signal = []

    for i in range(1, microphones + 1):
        audiofile_samplerate, audiofile_data_original = wavfile.read(signal_path + "/mic_" + str(i) + ".wav")
        rec_signal.append(audiofile_data_original)
        
    #SCALING SIGNAL TO -1; 1
    rec_signal = rec_signal / np.max(rec_signal)
    
    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.subplot(3,1,1)
        plt.title("Original signal")
        plt.plot(rec_signal[0])
    
    # TODO add types of noises.
    noise = np.random.normal(0, np.sqrt(10**-2), (microphones, len(rec_signal[0])))
    rec_signal_noised = add_noise_to_signal(rec_signal, noise, noise_dbm)

    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.subplot(3,1,3)
        plt.title("Signal with added noise. SNR: " + str(noise_dbm) + " dB")
        plt.plot(rec_signal_noised[0])

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
    
    uca_scanning_vectors = pyargus.directionEstimation.gen_uca_scanning_vectors(num_of_microphones, radius_of_uca, incident_angles)
      
    # DOA estimation           
    MUSIC = pyargus.directionEstimation.DOA_MUSIC(R, uca_scanning_vectors, signal_dimension = 1, angle_resolution = 1)#M - 3)

    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.title("MUSIC")
        plt.plot(realdoa, max(abs(MUSIC)), "r*")

    first_peak = argmax(MUSIC)
    if first_peak > 180:
        first_peak -= 180

    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.plot(abs(MUSIC))
        ax = plt.gca()
        xl, xr = ax.get_xlim()
        yd, yh = ax.get_ylim()
        plt.text(xl, yh, "MUSIC: Real DOA: " + str(realdoa) + " Estimated DOA: " + str(first_peak) + " and/or " + str(first_peak + 180) + "\n")
        # print("MUSIC: Real DOA: " + str(realdoa) + " Estimated DOA: " + str(first_peak) + " and/or " + str(first_peak + 180) + "\n")

    estimated_doa = first_peak
    estimated_sec_doa = first_peak + 180
    return estimated_doa, estimated_sec_doa

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
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation. if commented - xcorr, else phat (weight scale funct in GCC)
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
        corr = xcorr_freq(signal[i], signal[i+1])
        corr_peaks.append(np.argmax(corr))

    
    corr = xcorr_freq(signal[microphones - 1], signal[0])
    corr_peaks.append(np.argmax(corr))
    length_of_signal_in_samples = len(signal[0])

    # print(corr_peaks)
    
    signal_length = len(signal[0])
    current_best_peak_index = 0
    for i in range (0, len(corr_peaks)): 
        if abs(corr_peaks[i] - signal_length) < abs(corr_peaks[current_best_peak_index] - signal_length):
            current_best_peak_index = i
    
    first_best_peak = current_best_peak_index
    second_best_peak = (current_best_peak_index + microphones // 2) % microphones
    # print("current_best_peak index pair: " + str(first_best_peak) + " and " + str(second_best_peak))
    best_peaks_correlation_peak = np.argmax(xcorr_freq(signal[first_best_peak], signal[second_best_peak]))
    # print("correlation between part: " + str(best_peaks_correlation_peak))
    microphone_sound_arrival_two_pairs_corr_difference = best_peaks_correlation_peak - len(signal[0])
    microphone_sound_arrival_pair = first_best_peak if microphone_sound_arrival_two_pairs_corr_difference > 0 else second_best_peak
    # print("Assumed microphone pair nr: " + str(microphone_sound_arrival_pair) + " (numerated from 0), angle: <" + \
    #                                        str(microphone_sound_arrival_pair * 360 / microphones) + "; " + \
    #                                        str((microphone_sound_arrival_pair + 1) * 360 / microphones) + ")")

    #Angle may be better determined by computing correct angle based on V of sound: acos(v*samples_diff / distance_between_pair)
    estimated_angle = (((microphone_sound_arrival_pair * 360 / microphones) + ((microphone_sound_arrival_pair + 1) * 360 / microphones)) / 2)
    text_to_print = "Estimated angle based on radius, samplerate and gccphat.\n There can be big errors if some of the values are ridiculous.\n Angle between: " + \
           str((microphone_sound_arrival_pair * 360 / microphones)) + " and " + str((microphone_sound_arrival_pair + 1) * 360 / microphones) + \
               " center: " + str(estimated_angle) + "\n"
    #  print(text_to_print)
    
    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.title("GCC PHAT")
        plt.plot(xcorr_freq(signal[microphone_sound_arrival_pair], signal[microphone_sound_arrival_pair + 1]))
        ax = plt.gca()
        xl, xr = ax.get_xlim()
        yd, yh = ax.get_ylim()
        plt.text(xl, yh, text_to_print)

    return estimated_angle

# ---------------------------------------------------------------------------------------------------------

def neural_network_get_doa_from_file(model, signal_filepath):    
    nndp_class = nndp.nn_database()
    nndp_dataset = nndp.Dataset()
    operation_successful, microphones, matrix_radius, doa, reverb = nndp_class.get_info_from_dataset_part_info_file(signal_filepath)
    nndp_dataset = nndp_class.load_dataset_part(signal_filepath)

    if operation_successful is False or nndp_dataset is None:
        # Failure in computing dataset in neural network operations
        return None, None

    predicted_doas = []
    for i in range (0, 100):
        to_predict = nndp_dataset.batches[i, 0:8, 0:441, 0]
        to_predict = to_predict.reshape(1,8,441,1)
        model_predicted_raw = model.predict(to_predict)
        predicted_doa = np.round((model_predicted_raw * 180) + 180, decimals=0).astype(np.int32)
        predicted_doas.append(predicted_doa)

    true_doa = nndp_dataset.target

    return predicted_doas, true_doa

def compute_doa_via_Neural_Network(model, filepath):
    predicted_doas, true_doa = neural_network_get_doa_from_file(model, filepath)

    if predicted_doas is None and true_doa is None:
        return None, None
    
    predicted_doas_list = []
    for i in range (0, 100):
        predicted_doas_list.append(predicted_doas[i][0][0])

    if SHOW_MATLAB_VERBOSE_PLOTS:
        n = plt.hist(predicted_doas_list, bins=360, range=[0, 359], density=True)

    hp_doa = stat.mode(predicted_doas_list)     #dominanta
    mean_doa = stat.mean(predicted_doas_list)

    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.title("Neural Network")
        ax = plt.gca()
        xl, xr = ax.get_xlim()
        yd, yh = ax.get_ylim()
        plt.text(xl, yh, "Highest probability doa: " + str(hp_doa) + ", doa mean: " + str(mean_doa))
        plt.show(block=True)
    
    estimated_doa = predicted_doas_list[0]
    estimated_doa_from_all = hp_doa
    return estimated_doa, estimated_doa_from_all

# ---------------------------------------------------------------------------------------------------------

def compute_doa_from_all_algorithms(filepath, snr_dbm, neural_network_model):
    microphones, radius, realdoa = read_info_file(filepath + "/info.txt")
    signal, signal_samplerate = prepare_signal(filepath, microphones, snr_dbm)

    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.figure(2)
        plt.subplot(3, 1, 1)

    music_estimated_doa, music_estimated_sec_doa = compute_doa_via_MUSIC(signal, microphones, radius, realdoa)
    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.subplot(3, 1, 2)

    gcc_phat_estimated_angle = compute_doa_via_GCC_PHAT(signal, microphones, radius, signal_samplerate)
    if SHOW_MATLAB_VERBOSE_PLOTS:
        plt.subplot(3, 1, 3)

    nn_estimated_doa, nn_estimated_doa_from_all = compute_doa_via_Neural_Network(neural_network_model, filepath)
    return realdoa, music_estimated_doa, music_estimated_sec_doa, gcc_phat_estimated_angle, nn_estimated_doa, nn_estimated_doa_from_all

def main():
    model = nn.create_keras_model()
    checkpoint_filepath = 'tmp/checkpoint'
    model.load_weights(checkpoint_filepath)

    grand_filepath = "./generated_audio/yes/"
    filepath_database_list = os.listdir(grand_filepath)
    skipped = 0

    for current_snr in tqdm([-20, -10, 0, 10, 20], desc="Total benchmark process"):                  
        dictionary_result_data = {"Dirname":[],"SNR":[],"Real DOA":[],"GCC_PHAT":[],"MUSIC DOA 1":[],"MUSIC DOA 2":[],"Neural_network":[],"Neural_network_hist":[]}
       
        for next_folder_path in tqdm(filepath_database_list, desc="Estimating DOA's"):
            next_test_dir = grand_filepath + next_folder_path

            if next_folder_path[1] == 'p':  #skip if pickled database
                continue

            snr = current_snr

            real_doa, music_doa, music_sec_doa, gcc_phat_doa, nn_doa, nn_doa_from_all = compute_doa_from_all_algorithms(next_test_dir, snr, model)
            if nn_doa is None or nn_doa_from_all is None:
                skipped = skipped + 1
                continue

            dictionary_result_data["Dirname"].append(next_folder_path)
            dictionary_result_data["SNR"].append(snr)
            dictionary_result_data["Real DOA"].append(real_doa)
            dictionary_result_data["GCC_PHAT"].append(gcc_phat_doa)
            dictionary_result_data["MUSIC DOA 1"].append(music_doa)
            dictionary_result_data["MUSIC DOA 2"].append(music_sec_doa)
            dictionary_result_data["Neural_network"].append(nn_doa)
            dictionary_result_data["Neural_network_hist"].append(nn_doa_from_all)

            excel_df = pd.DataFrame(data = dictionary_result_data, index=None)
            excel_df.to_excel("./benchmark_results_snr_" + str(snr) + ".xlsx", "Results")
            print("Total skipped so far: " + str(skipped))

if __name__ == "__main__":
    main()