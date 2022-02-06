import os
import math
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
from pysndfx import AudioEffectsChain
import sys
from tqdm import tqdm
import random

#TODO: Change doxygen comments into python one

#--------------------------------------------------------------------------
# defines (constants)
MINIMUM_MICROPHONES_IN_ARRAY = 8 #not tested for other than 8

#--------------------------------------------------------------------------
# constants

class constants:
    log_verbose_mode = 0

    folder_with_audio_to_read_from = "./Database/tensorflow_recognition_challenge/train/audio/bed"
    generated_audio_path = "/generated_audio"
    temporal_shifting_samplerate = 192000

    speed_of_sound_in_air = 343

# constants = constants

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
    @brief Creates folder for storaging generated audio files
'''
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as error:
        LOG("Folder: \"generated_audio\" already created", log_type.LOG_INFO)

#--------------------------------------------------------------------------

'''
    @brief Converts angle from degrees into radians
'''
def degrees_to_radians(angle):
    return angle * (math.pi / 180)

'''
    @brief Checks matrix input parameters validity
'''
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
    @return microphones length shift. Value lesser than 0 indicates, that sound firstly arrived to second microphone.
'''
def calculate_audio_length_arrival_shift_between_two_microphones_for_specific_doa(mic_num, matrix_radius, sec_mic, wave_angle):
    if (check_parameters_validity(mic_num, matrix_radius, sec_mic, wave_angle) == False):
        sys.exit()

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
    return sampleshift_array

class audio_functions:
    '''
        @brief Edits audio length to specific value
        @param [in] path - path to .wav file
        @param [in] length - wanted length of .wav file in s
    '''
    def edit_audio_length(path, wav_length=1):
        try: 
            audio_samplerate, audio_data = wavfile.read(path)
            if (len(audio_data) != audio_samplerate * wav_length):
                number_of_samples_missing = audio_samplerate * wav_length - len(audio_data)
                if (number_of_samples_missing > 0):
                    audio_edited = np.append(audio_data, np.zeros(number_of_samples_missing).astype(np.int16))
                    wavfile.write(path, audio_samplerate, audio_edited)

            return True
        except:
            return False

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

        # Save resampled file and return samplerate
        wavfile.write(output_path, final_sampling_rate, audiofile_data_resampled)
        LOG("Audio " + str(input_path) + " resampled from " + str(audiofile_samplerate) + " to " + str(final_sampling_rate) + ".", log_type.LOG_INFO)
        return audiofile_samplerate
        
    def shift_audio_file(wav_file_to_shift, shift_in_samples, output_wav_file):
        """
        Truncates beginning of the WAV file for x samples and adds zeroes to the end.

        Parameters
        ----------
        wav_file_to_shift : string
            Input wav file path.
        shift_in_samples : int
            Number of samples to shift.
        output_wav_file : string
            Output wav file path.
        """
        audiofile_samplerate, audiofile_data_original = wavfile.read(wav_file_to_shift)
        audiofile_data_original = audiofile_data_original.astype(np.int16)
        shifted_audio = audiofile_data_original[shift_in_samples:]
        shifted_audio = np.append(shifted_audio, np.zeros(shift_in_samples).astype(np.int16))

        #zeros_shift_array = np.zeros(shift_in_samples).astype(np.int16)
        #shifted_audio = np.append(zeros_shift_array, audiofile_data_original) 
        #                       #  TODO: This shouldn't be adding, but deleting some samples.
        #                       #  may be crucial, when we'll have to add some noise to audio.
        #                       #  What about end of file (i.ex. it is 30samples less)
        wavfile.write(output_wav_file, audiofile_samplerate, shifted_audio)
        return

    def merge_two_wav_audio_files(first_wav_path, second_wav_path, shift_second_wav=0): #TODO: not tested yet, logs may be added
        """
        Merges two audio files (WAV) into one. TODO:NOT TESTED

        Returns merged data array.

        Parameters
        ----------
        first_wav_path : string
            Path to first WAV file.
        second_wav_path : string
            Path to second WAV file.
        shift_second_wav : int, optional
            Shift second WAV file for 'x' samples.

        Note
        ----
        This function's purpose is to add noise to original audio file.

        If second wav is longer, file is truncated into length of first_wav_path file.
        """
        first_wav_samplerate, first_wav_data = wavfile.read(first_wav_path)
        second_wav_samplerate, second_wav_data = wavfile.read(second_wav_path)

        if (first_wav_samplerate != second_wav_samplerate): #TODO: not tested
            LOG("Samplerate differs! changing samplerate of second wav.", log_type.LOG_CRITICAL)
            num_of_samples = round(len(second_wav_samplerate) * float(first_wav_samplerate) / second_wav_samplerate)
            second_wav_data = signal.resample(second_wav_data, num_of_samples).astype(np.int16)
            second_wav_samplerate = first_wav_samplerate

        first_wav_data = first_wav_data.np.astype(np.int16)
        second_wav_data = second_wav_data.np.astype(np.int16)
        if shift_second_wav > 0:
            zeros_to_append_on_beginning = np.zeros(shift_second_wav).astype(np.int16)
            np.append(zeros_to_append_on_beginning, second_wav_data)
            second_wav_data = zeros_to_append_on_beginning
        if shift_second_wav < 0:
            second_wav_data = second_wav_data[-shift_second_wav:]
            np.append(second_wav_data, np.zeros(shift_second_wav).astype(np.int16))

        if len(first_wav_data) < len(second_wav_data):
            second_wav_data = second_wav_data[:len(first_wav_data)]

        return first_wav_data + second_wav_data

#--------------------------------------------------------------------------
# API functions

'''
    Prepares audio signal for apply'ing reverb. It is depracated and no longer supported (TODO: delete?)
'''
def prepare_audio_signal_for_reverb(path_to_file, path_to_output):
    fx = (AudioEffectsChain().reverb())
    fx(path_to_file, path_to_output)

'''
    Syntheses/Generates audio files for full matrix based on one audio file.
    Computes shift for all microphones based on number of microphones, matrix radius and audiowave arrival angle.

    Parameters
    ----------
    mic_num : int
        Number of microphones in output matrix.
    matrix_radius : float
        Radius of circular array [m].
    audiowave_angle : int
        Audiowave arrival angle (degrees from 0 to 359) 0 is north, like azimute.
    path_to_file : string
        Path to file [.wav] from which another microphones signals in matrix are computed from.
    path_to_output : string
        Path to output folder.
    final_audios_samplerate : int
        Final samplerate of all audio files. if set to 0 - Function resamples to original.

    Notes
    ---
    Function resamples (upsamples) audio to 192000, computes shifts, shifts, and then again resamples into @final_audios_samplerate
'''
def generate_shifted_audio_files(mic_num, matrix_radius, audiowave_angle, path_to_file, path_to_output, final_audios_samplerate):
    LOG("Current directory: " + os.getcwd(), log_type.LOG_DEBUG)

    #Calculate array of shift to apply for all microphones
    shifting_array = calculate_shift_for_all_microphones(mic_num, matrix_radius, audiowave_angle, constants.temporal_shifting_samplerate)

    #Resample original audio and store as temporary file "resampled.temp"
    original_audio_samplerate = audio_functions.resample_specific_audio(path_to_file, "." + path_to_output + "/resampled.temp", constants.temporal_shifting_samplerate)
    if (final_audios_samplerate == 0): #If final_audio_samplerate set to zero, return to original sampling
        final_audios_samplerate = original_audio_samplerate
    
    #Shift for all microphones
    for i in range(len(shifting_array)):
        audio_functions.shift_audio_file("." + path_to_output + "/resampled.temp", shifting_array[i], "." + path_to_output + "/resampled_" + str(i + 1) +".temp")
        audio_functions.resample_specific_audio("." + path_to_output + "/resampled_" + str(i + 1) + ".temp", "." + path_to_output + "/mic_" + str(i + 1) + ".wav", final_audios_samplerate)
        os.remove("." + path_to_output + "/resampled_" + str(i + 1) + ".temp")
    
    #Delete temporary file "resampled.temp"
    os.remove("." + path_to_output + "/resampled.temp")
    return
    
'''
    Generates info file for synthesed mics.
    Creates file with data:
    (int)   'microphone numbers'
    (float) 'circular matrix radius'
    (int)   'audiowave angle of arrival'
    (bool)  'audio with reverb'
'''
def generate_info_file(mic_num, matrix_radius, audiowave_angle, reverb, path_to_output):
    f = open(path_to_output + "/info.txt", "w")
    f.write(str(mic_num) + "\n")
    f.write(str(matrix_radius) + "\n")
    f.write(str(audiowave_angle) + "\n")
    f.write(str(reverb) + "\n")
    f.close()

'''
    Generates N audio files, for each microphone.

    Note
    ----
    This function's purpose is to add noise to original audio file.

    If second wav is longer, file is truncated into length of first_wav_path file.
'''
def main():
    #8 mics, 68mm diameter, ?(random)-angle of arrival, ?random reverb
    microphones = 8
    uca_radius = 0.034
    final_samplerate = 44100

    all_database_folders = os.listdir("./Database/tensorflow_recognition_challenge/train/audio/")
    all_database_list = []
    for i in tqdm(all_database_folders, "Reading all files to compute"):
        database_list = os.listdir("./Database/tensorflow_recognition_challenge/train/audio/" + i)
        for a in database_list:
            all_database_list.append(str(i) + "/" + a)

    for i in tqdm(all_database_list, "Generating all audios"):
        input_file = "./Database/tensorflow_recognition_challenge/train/audio/" + i
        reverb = 0 #random.randrange(0, 2, 1) #0 or 1     ##Set always as false. Reverb is unsupported for now
        arrival_angle = random.randint(0, 359)
        outputfolder = i.split(".")
        folder_name = outputfolder[0].split("/")
        outputfolder = outputfolder[0]

        create_folder("generated_audio/" + folder_name[0])
        create_folder("generated_audio/" + outputfolder)

        #This function is used to fix data when it's length is not 1-second. (it's going to be changed)
        audio_functions.edit_audio_length(input_file)

        # reverb is depracated / no longer used. It's too complex to implement
        if (reverb):
            prepare_audio_signal_for_reverb(input_file, "." + constants.generated_audio_path + "/" + outputfolder + "/reverbed.wav")
            generate_shifted_audio_files(microphones, uca_radius, arrival_angle, "." + constants.generated_audio_path + "/" + outputfolder + "/reverbed.wav",
                                         constants.generated_audio_path + "/" + outputfolder, final_samplerate)
        else:
            generate_shifted_audio_files(microphones, uca_radius, arrival_angle, input_file,
                                         constants.generated_audio_path + "/" + outputfolder, final_samplerate)
    
        generate_info_file(microphones, uca_radius, arrival_angle, reverb, "." + constants.generated_audio_path + "/" + outputfolder)


if __name__ == '__main__':
    main()