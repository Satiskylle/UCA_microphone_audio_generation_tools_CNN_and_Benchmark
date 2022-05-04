from ctypes import sizeof
import os
import numpy as np
from scipy.io import wavfile
import pickle
import math
from sklearn import neural_network

from tqdm import tqdm

VERBOSE = False

class Dataset:
  quasi_mic_samples_img = []              #this should be full 44100x8 image
  batches = np.array([],dtype=np.int32)   #this should be divided quasi_mic_samples_img -> i.ex. 100x441x8
                                          #now, it is 441samples_ch_0, 441samples_ch_1, 441samples_ch_2...
                                          #   ...next 441_samples_ch_0, 441_samples_ch1,
  target = np.array([],dtype=np.int32)
  target_float = np.array([],dtype=np.float)

# Neural network class database
class nn_database:
  def __init__(self):
    pass

  def __check_dataset_corruption(self, samplerate, data, error_description):
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

  def __get_info_from_dataset_part_info_file(self, filepath):
    #Check if info file exists
    if (os.path.isfile(filepath + "/info.txt") == False):
      if VERBOSE:
        print("Info file do not exist. Ommiting + " + filepath)
      return False, None, None, None, None

    f = open(filepath + "/info.txt", 'r')
    mics = int(f.readline())
    matrix_radius = float(f.readline())
    doa = int(f.readline()) #// (360//mics) //READ FULL, not only sector of the circle.
    reverb = bool(int(f.readline()))
    f.close()
    return True, mics, matrix_radius, doa, reverb

  def __load_dataset_part(self, filepath_to_open):
    success, mics, matrix_radius, doa, reverb = self.__get_info_from_dataset_part_info_file(filepath_to_open)
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
        if self.__check_dataset_corruption(samplerate, data, error_desc):
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

  def __number_of_dirs(self, path):
    return len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

  def __load_dataset_from_dir(self, dataset_path, start_from=0, dataset_size_to_load=-1):
    audio_paths_to_load = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    if (dataset_size_to_load == -1):
      dataset_size_to_load = len(audio_paths_to_load)

    audio_paths_to_load = audio_paths_to_load[start_from:start_from + dataset_size_to_load]

    if (len(audio_paths_to_load) == 0):
      return

    dataset_total = Dataset()
    for audio in tqdm(audio_paths_to_load, desc="Generating neural network dataset", unit=" file structs", leave=False):
      dataset_part = self.__load_dataset_part(dataset_path + audio)
      if (dataset_part is None):
        continue

      dataset_total.batches = np.append(dataset_total.batches, dataset_part.batches)
      dataset_total.target = np.append(dataset_total.target, dataset_part.target)

    return dataset_total

  def __dump_dataset(self, object_to_dump, dump_filename = ".dumped_dataset"):
    if (str(dump_filename).find('/')):
      create_dir = str(dump_filename).rpartition('/')
      try:
        os.makedirs(create_dir[0])
      except FileExistsError:
        pass
    
    with open(dump_filename, 'wb') as file:
      pickle.dump(object_to_dump, file)
      file.close()

  def receive_dumped_object(self, dump_filename = ".dumped_dataset"):
    with open(dump_filename, 'rb') as file:
      object_dumped = pickle.load(file)
      file.close()

    return object_dumped

  '''
  Creates database files.
  If not specified, creates one big database.
  '''
  def create_databases(self, generated_audio_files, dataset_pickle_filename, size_of_databases=-1):
    num_of_iters = math.ceil(self.__number_of_dirs(generated_audio_files) / size_of_databases)
    if (num_of_iters < 0):
      num_of_iters = 1

    for next_database in tqdm(range(0, num_of_iters), desc="Datasets in directory", unit=" datasets", leave=False):
      dataset_total = self.__load_dataset_from_dir(generated_audio_files, start_from=size_of_databases*next_database, dataset_size_to_load=size_of_databases)
      if (dataset_total == None):
        return

      self.__dump_dataset(dataset_total, dataset_pickle_filename + "_" + str(next_database))

  #----------------------------------------------------------------------

def main():

  #This is sample
  #dataset_total = nn_database.load_dataset("generated_audio/test_delete_me_3/")
  #nn_database.dump_dataset(dataset_total, ".normal_database")

  neural_network_database = nn_database()
  dirs = os.listdir("generated_audio/")
  for i in tqdm(dirs, desc="Total neural network dataset processed", unit=" datasets"):
    neural_network_database.create_databases(generated_audio_files="generated_audio/" + i + "/",
                                              dataset_pickle_filename="X:/generated_audio_5_SNR/" + i + "/.pickled_database",
                                              size_of_databases=500)


if __name__ == "__main__":
  main()