from math import inf, sqrt, log10, cos
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import angle
from pyargus.directionEstimation import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.io import wavfile
from scipy.signal.filter_design import normalize
from tqdm import tqdm
import os
import shutil

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
    #noised_signal_power = signalPower(noised_signal)
    #SNR = 10 * log10(signal_power / noised_signal_power) #just test

    return rec_signal + noised_signal, #SNR #return noised original signal and computed snr #edit-without computed snr.


def prepare_signal(microphones=8, noise_dbm = 0, reverbation = 0, main_path = ""):
    rec_signal = []

    for i in range(1, microphones + 1):
        audiofile_samplerate, audiofile_data_original = wavfile.read(main_path + "/mic_" + str(i) + ".wav")
        rec_signal.append(audiofile_data_original)
        
    #SCALING SIGNAL TO -1; 1
    rec_signal = rec_signal / np.max(rec_signal)
    #plt.subplot(3,1,1)
    #plt.title("Original signal")
    #plt.plot(rec_signal[0])
    
    # TODO add types of noises.
    noise = np.random.normal(0, np.sqrt(10**-2), (microphones, len(rec_signal[0])))
    rec_signal_noised = add_noise_to_signal(rec_signal, noise, noise_dbm)
    rec_signal_noised = rec_signal_noised[0]

    #plt.subplot(3,1,3)
    #plt.title("Signal with added noise. SNR: " + str(noise_dbm) + " dB")
    #plt.plot(rec_signal_noised[0])
    #plt.show()

    return rec_signal_noised, audiofile_samplerate


# ---------------------------------------------------------------------------------------------------------

def add_noise_to_generated_audios(snr_value):
    target_snr = snr_value
    generated_audio_path = "generated_audio"
    output_path = "X:/generated_audio_" + str(target_snr) + "SNR"
    next_path_to_load = [f for f in os.listdir(generated_audio_path) if os.path.isdir(os.path.join(generated_audio_path, f))]

    for nxt_path in tqdm(next_path_to_load, desc="Next audio signal"):
        next_audioset_to_load = [f for f in os.listdir(generated_audio_path + '/' + nxt_path) if os.path.isdir(os.path.join(generated_audio_path + '/' + nxt_path, f))]
        for nxt_file in tqdm(next_audioset_to_load, desc="Next dataset"):
            path_to_fileset = generated_audio_path + '/' + nxt_path + '/' + nxt_file + '/'
            try:
                microphones, radius, realdoa = read_info_file(path_to_fileset + 'info.txt')
            except:
                print("Skipped " + str(path_to_fileset) + " (error)")
                continue
             signal_table, signal_samplerate = prepare_signal(microphones, target_snr, main_path=path_to_fileset)
             for signal_table_num in range (0, len(signal_table)):
                 if not os.path.isdir(output_path + '/' + path_to_fileset):
                     os.makedirs(output_path + '/' + path_to_fileset)

                 wavfile.write(output_path + '/' + path_to_fileset + 'mic_' + str(signal_table_num + 1) + '.wav', signal_samplerate, signal_table[signal_table_num])
            #Copy also info.txt file
            shutil.copy2(path_to_fileset + 'info.txt', output_path + '/' + path_to_fileset + 'info.txt')
            
def main():
    add_noise_to_generated_audios(20)
    add_noise_to_generated_audios(10)
    add_noise_to_generated_audios(0)

if __name__ == "__main__": 
    main()