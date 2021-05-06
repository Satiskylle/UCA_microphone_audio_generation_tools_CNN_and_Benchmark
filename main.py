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

class constants:
    log_verbose_mode = 1

    folder_with_audio_to_read_from = "../Database/tensorflow_recognition_challenge/train/audio/bed"
    generated_audio_path = "./generated_audio"

    speed_of_sound_in_air = 343

constants = constants

class log_type:
    LOG_CRITICAL = 0
    LOG_INFO = 1
    LOG_DEBUG = 2

#--------------------------------------------------------------------------
# static functions

'''
    @brief Wrapper to print logs with verbose leveling
    @note  0 - only critical logs, 1 - info logs, 2 - all logs
    @param [in] - datastring to print
    @param [in] - log type. (critical, info, debug) ref @class log_type
'''
def LOG(datastring, log_type):
    if (log_type < 0 or log_type > 2):
        print("LOG incorrect parameter.")

    if (log_type <= constants.log_verbose_mode):
        print(datastring)
    
    return

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

def check_parameters_validity(mic_num, matrix_radius, sec_mic, wave_angle):
    if mic_num < MINIMUM_MICROPHONES_IN_ARRAY:
        LOG(str(mic_num) + " microphones exceeds assert (min " + str(MINIMUM_MICROPHONES_IN_ARRAY) + ")", log_type.LOG_CRITICAL)
        return False
    
    if sec_mic < 1 or sec_mic > mic_num:
        LOG("Second microphone parameter is incorrect (second microphone: " + str(sec_mic) + ")", log_type.LOG_CRITICAL)
        return False

    if matrix_radius < 0:
        LOG("Matrix radius is lower than 0 (matrix radius: " + str(matrix_radius) + ")", log_type.LOG_CRITICAL)
        return False

    if wave_angle < 0 or wave_angle >= 360:
        LOG("Set DOA angle in range <0:360)", log_type.LOG_CRITICAL)
        return False

'''
    @brief Calculates length difference between first microphone and selected one for specific angle of flat-wave
    @note  difference is computed for angles <0:360) basing on ideal circled-microphone-matrix.
    @param [in] mic_num - number of microphones in matrix
    @param [in] matrix_radius - radius of matrix (length between center of matrix and microphone)
    @param [in] sec_mic - second microphone. Must be less than @param mic_num
    @param [in] wave_angle - angle specified in degrees in range <0:360)
    @return microphones length shift. Value lesser than 0 indicates, that sound firsly arrived to second microphone.
'''
def calculate_audio_length_arrival_shift_between_two_microphones_for_specific_doa(mic_num, matrix_radius, sec_mic, wave_angle):
    if (check_parameters_validity(mic_num, matrix_radius, sec_mic, wave_angle) == False):
        return 0.0

    alpha = (sec_mic - 1) * (360 / mic_num)
    gamma = (180 - alpha) / 2
    LOG("alpha: " + str(alpha) + " gamma: " + str(gamma), log_type.LOG_DEBUG)
    cos_alpha = math.cos(degrees_to_radians(alpha))
    cos_fi = math.cos(degrees_to_radians(wave_angle))
    sin_fi = math.sin(degrees_to_radians(wave_angle))
    cos_gamma = math.cos(degrees_to_radians(gamma))
    sin_gamma = math.sin(degrees_to_radians(gamma))
    #print("cos_alpha:" + str(cos_alpha) + " cos_fi:" + str(cos_fi) + " sin_fi:" + str(sin_fi) + " cos_gamma:" + str(cos_gamma) + " sin_gamma:" + str(sin_gamma))
    length_between_two_microphones = math.sqrt(2 * matrix_radius * (1 - cos_alpha))
    LOG("length between two microphones: " + str(length_between_two_microphones), log_type.LOG_DEBUG)
    microphone_audio_arrival_length_shift = length_between_two_microphones * (cos_gamma * cos_fi - sin_gamma * sin_fi)
    
    if abs(microphone_audio_arrival_length_shift) < 1e-10:
        microphone_audio_arrival_length_shift = 0
    
    LOG("microphone audio arrival shift length: " + str(microphone_audio_arrival_length_shift), log_type.LOG_DEBUG)
    return microphone_audio_arrival_length_shift

'''
    @brief Computes shifting difference [in distance] to shifting difference [in audio samples].
    @param [in] length - length difference value
    @param [in] samplerate - samplerate
    @return computed sample-length-difference (shift in number of samples).
'''
def calculate_sample_shift_from_length_shift(length, samplerate):
    time = length / constants.speed_of_sound_in_air
    shift_in_samples = samplerate * time
    LOG("Shifting (in samples) for length: " + str(length) + " and samplerate: " + str(samplerate) + " is " + str(shift_in_samples), log_type.LOG_DEBUG)
    return shift_in_samples

'''
    @brief Calculates length difference (in samples) for all microphones.
    @note  for (at least) one microphone, there will be 0-length. It means, that sound arrived to this mic as first.
    @param [in] mic_num - number of microphones in matrix
    @param [in] matrix_radius - radius of matrix (length between center of matrix and microphone)
    @param [in] wave_angle - angle of incoming audio source, specified in degrees in range <0:360)
    @param [in] wave_samplerate - samplerate of audio wave, used to compute shift in number of samples
    @return array with computed audio length shifts for all microphones. Rounded to integer values.
'''
def calculate_shift_for_all_microphones(mic_num, matrix_radius, wave_angle, wave_samplerate):
    mic_array = np.array([])
    for second_mic in range(1, mic_num + 1):
        mic_array = np.append(mic_array, calculate_audio_length_arrival_shift_between_two_microphones_for_specific_doa(mic_num, matrix_radius, second_mic, wave_angle))
    
    mic_array = mic_array + abs(mic_array.min())
    LOG("Microphones arrival shifts:\n" + np.array_str(mic_array, precision=5), log_type.LOG_DEBUG)

    sampleshift_array = np.array([], dtype=int)
    for i in mic_array:
        sampleshift_array = np.append(sampleshift_array, round(calculate_sample_shift_from_length_shift(i, wave_samplerate)))

    LOG("Computed shift in samples for each microphone:\n" + np.array_str(sampleshift_array), log_type.LOG_INFO)
    return


#--------------------------------------------------------------------------
# API functions

def generate_shifted_audio_files():
    #generate...
    return

def main():
    #create_generated_audio_folder()
    ## os.chdir(folder_with_audio_to_read_from)
    #print("Current directory: " + os.getcwd())
    #resample_specific_audio("../Database/tensorflow_recognition_challenge/train/audio/bed/00f0204f_nohash_0.wav", generated_audio_path + "/test.wav", 44100)
    #resample_specific_audio(generated_audio_path + "/test.wav", generated_audio_path + "/test_resampled_again.wav", 16000)

##TESTS
    #8 mics, 68mm diameter, waveangle 0*, samplerate 44100
    calculate_shift_for_all_microphones(8, 0.034, 0, 44100)
    
if __name__ == '__main__':
    main()