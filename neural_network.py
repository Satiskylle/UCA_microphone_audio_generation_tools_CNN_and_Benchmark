from ctypes import sizeof
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# import sklearn
# from sklearn import datasets
# from sklearn import svm
# from sklearn import metrics

def load_dataset():
  f = open("generated_audio/bed/0a7c2a8d_nohash_0/info.txt", 'r')
  mics = int(f.readline())
  matrix_radius = float(f.readline())
  doa = int(f.readline()) // (360//mics)
  f.close()

  class Dataset:
    quasi_mic_samples_img = []  #this should be full 44100x8 image
    batches = []                #this should be divided quasi_mic_samples_img -> i.ex. 100x441x8
                                #now, it is 441samples_ch_0, 441samples_ch_1, 441samples_ch_2...
                                #   ...next 441_samples_ch_0, 441_samples_ch1,
    target = []

  dataset = Dataset()
  for k in range(0, mics):
    samplerate, data = wavfile.read("./generated_audio/bed/0a7c2a8d_nohash_0/mic_" + str(k + 1) + ".wav")
    dataset.quasi_mic_samples_img.append(data)
    dataset.target.append(doa)

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
  dataset.batches = np.asarray(dataset.batches) #make batches an array
  dataset.batches = dataset.batches.reshape(100, 8, 441, 1)#, order='F')
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
  dataset = load_dataset()
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
  model.add(layers.Dense(32, activation='relu'))
  model.add(layers.Dense(8))
  

  model.summary()

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


  # x_train = (np.transpose(dataset.data))
  x_dataset = dataset.batches.reshape(100, 8, 441, 1) #reshaped data
  x_train = x_dataset[:-20] # od całosci do przed10 ostatniego
  #x_test = x_dataset[-20:-10]  # 10 ostatnich

  y_dataset = np.full((100, 1), 4)
  y_train = y_dataset[:-20] # wszystkie oprócz 10 ostatnich
  #y_test = y_dataset[-20:]  # 10 ostatnich

  x_val = x_dataset[-20:] #for now..
  y_val = y_dataset[-20:]
  # Reserve X samples for validation
  #  x_val = x_train[-1000:]
  #  y_val = y_train[-1000:]
  #  x_train = x_train[:-1000]
  #  y_train = y_train[:-1000]

  epochs = 20
  history = model.fit(x = x_train, y = y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size = 100, verbose = 1)

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