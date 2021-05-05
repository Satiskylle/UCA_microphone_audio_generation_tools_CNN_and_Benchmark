import os
import math
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal

#--------------------------------------------------------------------------
# defines (constants)
MINIMUM_MICROPHONES_IN_ARRAY = 8


#--------------------------------------------------------------------------
# constants

folder_with_audio_to_read_from = "../Database/tensorflow_recognition_challenge/train/audio/bed"
generated_audio_path = "./generated_audio"

#--------------------------------------------------------------------------
# static functions

'''
    @brief Resamples audio to particular sampling rate
    @param [in] input_path - path to input .wav file
    @param [in] output_path - path to output .wav file
    @param [in] final_sampling_rate - sampling rate to which function converts (@param input_path) .wav audio.
'''
def resample_specific_audio(input_path, output_path, final_sampling_rate):
    # Read file
    audiofile_samplerate, audiofile_data_original = wavfile.read(input_path)

    # Resample file to (@param [in] final_sampling_rate).
    audiofile_number_of_samples = round(len(audiofile_data_original) * float(final_sampling_rate) / audiofile_samplerate)
    audiofile_data_resampled = signal.resample(audiofile_data_original, audiofile_number_of_samples).astype(np.int16)
    wavfile.write(output_path, final_sampling_rate, audiofile_data_resampled)

'''
    @brief Creates folder for storaging generated audio files
'''
def create_generated_audio_folder():
    try:
        os.mkdir(generated_audio_path)
    except OSError as error:
        print("Folder: generated_audio already created")

#--------------------------------------------------------------------------

def degrees_to_radians(angle):
    return angle * (math.pi / 180)

def check_parameters_validity(mic_num, matrix_radius, doa_angle):
    if mic_num < MINIMUM_MICROPHONES_IN_ARRAY:
        print(mic_num + " microphones exceeds assert (max " + MINIMUM_MICROPHONES_IN_ARRAY + ")")
        return False
    
    if matrix_radius < 0:
        print("Matrix radius is lower than 0 (matrix radius: " + matrix_radius + ")")
        return False

    if doa_angle < 0 or doa_angle >= 360:
        print("Set DOA angle in range <0:360)")
        return False

'''
    @brief Calculates length difference between two specific microphones for specific angle of flat-wave
    @note  difference is computed for angles <0:360) basing on ideal circled-microphone-matrix.
    @param [in] mic_num - number of microphones in matrix
    @param [in] matrix_radius - radius of matrix (length between center of matrix and microphone)
    @param [in] first mic - first microphone. Must be less than @param mic_num
    @param [in] sec_mic - second microphone. Must be less than @param mic_num
    @param [in] wave_angle - angle specified in degrees in range <0:360)
'''
def calculate_audio_length_arrival_shift_between_two_microphones_for_specific_doa(mic_num, matrix_radius, first_mic, sec_mic, wave_angle):
    if (check_parameters_validity(mic_num, matrix_radius, doa_angle) == False):
        return -1

    alpha = 360 / mic_num
    gamma = (180 - alpha) / 2
    print("alpha: " + str(alpha) + " gamma: " + str(gamma))
    cos_alpha = math.cos(degrees_to_radians(alpha))
    cos_fi = math.cos(degrees_to_radians(wave_angle))
    sin_fi = math.sin(degrees_to_radians(wave_angle))
    cos_gamma = math.cos(degrees_to_radians(gamma))
    sin_gamma = math.sin(degrees_to_radians(gamma))
    #print("cos_alpha:" + str(cos_alpha) + " cos_fi:" + str(cos_fi) + " sin_fi:" + str(sin_fi) + " cos_gamma:" + str(cos_gamma) + " sin_gamma:" + str(sin_gamma))
    length_between_two_microphones = math.sqrt(2 * matrix_radius * (1 - cos_alpha))
    print("length between two microphones: " + str(length_between_two_microphones))
    microphone_audio_arrival_length_shift = length_between_two_microphones * (cos_gamma * cos_fi - sin_gamma * sin_fi)
    
    if abs(microphone_audio_arrival_length_shift) < 1e-10:
        microphone_audio_arrival_length_shift = 0
    
    print("microphone audio arrival shift length: " + str(microphone_audio_arrival_length_shift))
    return microphone_audio_arrival_length_shift


#--------------------------------------------------------------------------
# API functions

def main():
    #create_generated_audio_folder()
    ## os.chdir(folder_with_audio_to_read_from)
    #print("Current directory: " + os.getcwd())
    #resample_specific_audio("../Database/tensorflow_recognition_challenge/train/audio/bed/00f0204f_nohash_0.wav", generated_audio_path + "/test.wav", 44100)
    #resample_specific_audio(generated_audio_path + "/test.wav", generated_audio_path + "/test_resampled_again.wav", 16000)
    calculate_audio_length_arrival_shift_between_two_microphones_for_specific_doa(8, 1, 90 + (90 - 67.5))


    
if __name__ == '__main__':
    main()