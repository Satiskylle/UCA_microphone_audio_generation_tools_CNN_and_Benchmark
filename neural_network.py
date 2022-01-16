from ctypes import sizeof
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tqdm import tqdm #Import progressbar


# import sklearn
# from sklearn import datasets
# from sklearn import svm
# from sklearn import metrics

class Dataset:
  quasi_mic_samples_img = []              #this should be full 44100x8 image
  batches = np.array([],dtype=np.int32)   #this should be divided quasi_mic_samples_img -> i.ex. 100x441x8
                                          #now, it is 441samples_ch_0, 441samples_ch_1, 441samples_ch_2...
                                          #   ...next 441_samples_ch_0, 441_samples_ch1,
  target = np.array([],dtype=np.int32)

def check_dataset_corruption(samplerate, data, error_description):
  if (samplerate != len(data)):
    #data is corrupted, only 1second samples are allowed for input.
    print(error_description + " Reason: too short")
    return True
  if (samplerate != 44100):
    #data is corrupted, only 44100 sps files are allowed for input.
    print(error_description + " Reason: bad sampling")
    return True
  
  return False

def get_info_from_dataset_part_info_file(filepath):
  #Check if info file exists
  if (os.path.isfile(filepath + "/info.txt") == False):
    print("Info file do not exist. Ommiting + " + filepath)
    return False

  f = open(filepath + "/info.txt", 'r')
  mics = int(f.readline())
  matrix_radius = float(f.readline())
  doa = int(f.readline()) // (360//mics)
  reverb = bool(int(f.readline()))
  f.close()
  return True, mics, matrix_radius, doa, reverb

def load_dataset_part(filepath_to_open):
  success, mics, matrix_radius, doa, reverb = get_info_from_dataset_part_info_file(filepath_to_open)
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
      if check_dataset_corruption(samplerate, data, error_desc):
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

def load_dataset(dataset_path):
  audio_paths_to_load = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

  dataset_total = Dataset()
  for audio in tqdm(audio_paths_to_load):
    dataset_part = load_dataset_part(dataset_path + audio)
    if (dataset_part is None):
      continue

    dataset_total.batches = np.append(dataset_total.batches, dataset_part.batches)
    dataset_total.target = np.append(dataset_total.target, dataset_part.target)

  return dataset_total

def show_results(history, epochs):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show(block=True)



def main():
  dataset_total = load_dataset("generated_audio/test_delete_me/")

  model = keras.Sequential()
  
  #batch 441, 100 rows, 8 cols, 1 channel, so there is 100 samples with 8 channel each,  
  model.add(layers.Conv2D(160, (8, 1), padding='valid', activation='relu', input_shape=(8, 441, 1))) 
  model.add(layers.MaxPooling2D(pool_size=(1,4)))
  model.add(layers.Conv2D(80, 1, padding='valid', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(1,4)))
  model.add(layers.Conv2D(20, 1, padding='valid', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(1,4)))
  model.add(layers.Conv2D(8, 1, padding='valid', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(1,2)))
  model.summary()
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(32, activation='sigmoid'))
  model.add(layers.Dense(8))
  model.summary()

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


  total_computed_audios = int(dataset_total.batches.size/441/8/1)
  x_dataset = dataset_total.batches.reshape(total_computed_audios, 8, 441, 1) #reshaped data #was 100 initialy for 1 computed audio

  y_dataset = []
  for i in range(0, len(dataset_total.target)):
    for k in range (0, total_computed_audios // len(dataset_total.target)):
      y_dataset.append(dataset_total.target[i]) #every 441 samples per 100data there is change of targetDOA

  y_dataset = np.asarray(y_dataset)
  epochs = 100
  #history = model.fit(x = x_train, y = y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size = 100, verbose = 1)
  history = model.fit(x = x_dataset, y = y_dataset, validation_split = 0.2, epochs=epochs, batch_size = 100, verbose = 1)

  show_results(history, epochs)

if __name__ == "__main__":
  main()