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

def load_dataset(filepath_to_open):
  f = open(filepath_to_open + "/info.txt", 'r')
  mics = int(f.readline())
  matrix_radius = float(f.readline())
  doa = int(f.readline()) // (360//mics)
  reverb = bool(int(f.readline()))
  f.close()

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
    if (k == 0): #check only once
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
  dataset.batches = np.asarray(dataset.batches) #, dtype = np.int32) #make batches an array
  dataset.batches = dataset.batches.reshape(100, mics, 441, 1)    #, order='F')

  #if the size of everry mic is not 44100, there is exception. Dtype is not np.int16 but 'O'. This bug must be fixed.
  return dataset

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

  path = "generated_audio/test_delete_me/"
  audio_paths_to_load = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

  dataset_total = Dataset()
  for audio in tqdm(audio_paths_to_load):
    dataset = load_dataset(path + audio)
    if (dataset is None):
      continue

    dataset_total.batches = np.append(dataset_total.batches, dataset.batches)
    dataset_total.target = np.append(dataset_total.target, dataset.target)

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


  # x_train = (np.transpose(dataset.data))
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




  #oldest:
    # x = dataset.data  #here load all samples from microphones
  # y = dataset.target #number of angle from which sound has been gotten. classify that to 0-7 (where 0 is 0* to 45* for 8 mics)

  # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
  
  # clf = svm.SVC() #gamma = 0.001 C= 100 -- test. gamma automaticaly??
  # clf.fit(x_train, y_train)
  # y_test_pred = clf.predict(x_test)

  # acc = metrics.accuracy_score(y_test, y_test_pred)
  # print("Prediction: ", str(acc))