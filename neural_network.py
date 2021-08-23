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
    data = []
    target = []

  dataset = Dataset()
  for i in range(0, mics):
    samplerate, data = wavfile.read("./generated_audio/bed/0a7c2a8d_nohash_0/mic_" + str(i + 1) + ".wav")
    dataset.target.append(doa)
    dataset.data.append(data)

  dataset.data = np.asarray(dataset.data)
  #This is loading only one file. That should be treated as "image"?
  return dataset

def show_results(history, epochs):
  acc = history.history['accuracy']
  #val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  #val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  #plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  #plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show(block=True)


def main():
  dataset = load_dataset()
  model = Sequential([
  layers.Conv2D(16, 8, padding='same', activation='relu', input_shape=(8, 44100, 1)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 8, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 8, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(8)
  ])

  model.summary()

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  epochs = 3
  x=dataset.data
  # x = [[1,2],[1,2],[1,3],[2,3],[3,3],[4,4],[5,5],[6,6]]
  # x = np.asarray(x)
  d = totestowyreshape_wyglada_ze_dziala = x.reshape(1, 8, 44100, 1)


  history = model.fit(x=d, y=np.asarray([3]), validation_data=None, epochs=epochs)

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