from ctypes import sizeof
import os
import numpy as np
from scipy.io import wavfile
import pickle

from tqdm import tqdm

VERBOSE = False

class Dataset:
  quasi_mic_samples_img = []              #this should be full 44100x8 image
  batches = np.array([],dtype=np.int32)   #this should be divided quasi_mic_samples_img -> i.ex. 100x441x8
                                          #now, it is 441samples_ch_0, 441samples_ch_1, 441samples_ch_2...
                                          #   ...next 441_samples_ch_0, 441_samples_ch1,
  target = np.array([],dtype=np.int32)

# Neural network class database
class nn_database:

  def __check_dataset_corruption(samplerate, data, error_description):
    if (samplerate != len(data)):
      #data is corrupted, only 1second samples are allowed for input.
      if VERBOSE:
        print(error_description + " Reason: too short")

      return True

    if (samplerate != 44100):
      #data is corrupted, only 44100 sps files are allowed for input.
      if VERBOSE:
        print(error_description + " Reason: bad sampling")

      return True
    
    return False


  def __get_info_from_dataset_part_info_file(filepath):
    #Check if info file exists
    if (os.path.isfile(filepath + "/info.txt") == False):
      if VERBOSE:
        print("Info file do not exist. Ommiting + " + filepath)
      return False, None, None, None, None

    f = open(filepath + "/info.txt", 'r')
    mics = int(f.readline())
    matrix_radius = float(f.readline())
    doa = int(f.readline()) // (360//mics)
    reverb = bool(int(f.readline()))
    f.close()
    return True, mics, matrix_radius, doa, reverb


  def __load_dataset_part(filepath_to_open):
    success, mics, matrix_radius, doa, reverb = nn_database.__get_info_from_dataset_part_info_file(filepath_to_open)
    if (success == False):
      return

    # reverb files are not supported for now
    if (reverb == True):
      return

    class Dataset_part:
      quasi_mic_samples_img = []  #this should be full 44100x8 image
      batches = []                #this should be divided quasi_mic_samples_img -> i.ex. 100x441x8
                                  #now, it is 441samples_ch_0, 441samples_ch_1, 441samples_ch_2...
                                  #   ...next 441_samples_ch_0, 441_samples_ch1,
      target = []

    dataset = Dataset_part()

    for k in range(0, mics):
      samplerate, data = wavfile.read(filepath_to_open + "/mic_" + str(k + 1) + ".wav")
      if (k == 0): #check only once - if any of the files is corrupted, first is also.
        error_desc = filepath_to_open + "/mic_" + str(k + 1) + ".wav"
        if nn_database.__check_dataset_corruption(samplerate, data, error_desc):
          return

      dataset.quasi_mic_samples_img.append(data)

    dataset.target.append(doa) #add doa if data is not corrupted
    for k in range(0, 100):
      for i in range (0, mics):
        dataset.batches.append(dataset.quasi_mic_samples_img[i][441 * k: 441 * k + 441]) #then, it will be 100x441x8 !

      #EN: NOW, batches are (in series):
      #       1(first_batch): CH0 - 441 samples,      2(next_batch): CH0 - 441 samples,
      #                       CH1 - 441 samples,                     CH1 - 441 samples,  
      #                             ...                                  ...
      #                       CH7 - 441 samples,                     CH7 - 441 samples,
      #
      #PL: Ok, jest 100 kawałków które maja 8kanałow po kolei posiadajace po 441 probek
    dataset.batches = np.asarray(dataset.batches) #, dtype = np.int32) #make batches an array #should be np.int16
    dataset.batches = dataset.batches.reshape(100, mics, 441, 1)    #, order='F')

    # If the size of every mic is not 44100, there is exception. Dtype is not np.int16 but 'O'.
    return dataset


  def __load_dataset(dataset_path):
    audio_paths_to_load = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    dataset_total = Dataset()
    for audio in tqdm(audio_paths_to_load, desc="Generating neural network dataset", unit=" file structs", leave=False):
      dataset_part = nn_database.__load_dataset_part(dataset_path + audio)
      if (dataset_part is None):
        continue

      dataset_total.batches = np.append(dataset_total.batches, dataset_part.batches)
      dataset_total.target = np.append(dataset_total.target, dataset_part.target)

    return dataset_total


  def __dump_dataset(object_to_dump, dump_filename = ".dumped_dataset"):
    with open(dump_filename, 'wb') as file:
      pickle.dump(object_to_dump, file)
      file.close()

  #----------------------------------------------------------------------

  def receive_dumped_object(dump_filename = ".dumped_dataset"):
    with open(dump_filename, 'rb') as file:
      object_dumped = pickle.load(file)
      file.close()

    return object_dumped


  def create_database_file(generated_audio_files, dataset_pickle_file):
    dataset_total = nn_database.__load_dataset(generated_audio_files)
    nn_database.__dump_dataset(dataset_total, dataset_pickle_file)


import os
def main():
  #".superfast_database" - contains ~16 elements    #EXIST
  #".fast_database" - containst ~100 elements       #DO NOT EXIST
  #".quite_database" - containst ~425 elements      #EXIST
  #".normal_database" - containst ~10000 elements   #EXIST

  #This is sample
  #dataset_total = nn_database.load_dataset("generated_audio/test_delete_me_3/")
  #nn_database.dump_dataset(dataset_total, ".normal_database")
  dirs = os.listdir("generated_audio/")
  for i in tqdm(dirs, desc="Total neural network dataset processed"):
    nn_database.create_database_file("generated_audio/" + i + "/", "generated_audio/" + i + "/.pickled_database")


if __name__ == "__main__":
  main()