from neural_network_dataset_prepare import Dataset, nn_database as nn

from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


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
  #nn.create_database_file("generated_audio/test_delete_me_2/", ".test_database")
  dataset_total = Dataset()
  dataset_total = nn.receive_dumped_object(".normal_database")
  

  model = keras.Sequential()
  
  #batch 441, 100 rows, 8 cols, 1 channel, so there is 100 samples with 8 channel each,  
  model.add(layers.Conv2D(8, (4, 1), padding='valid', activation='relu', input_shape=(8, 441, 1))) #160
  model.add(layers.MaxPooling2D(pool_size=(1,8)))
  model.add(layers.Conv2D(4, (2, 1), padding='valid', activation='relu'))  #80
  model.add(layers.MaxPooling2D(pool_size=(1,4)))
  model.add(layers.Conv2D(2, 1, padding='valid', activation='relu'))       #20
  model.add(layers.MaxPooling2D(pool_size=(1,4)))
  model.add(layers.Conv2D(8, 1, padding='valid', activation='relu'))        #8
  model.add(layers.MaxPooling2D(pool_size=(1,2)))
  model.summary()
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  # model.add(layers.Dropout(0.2))
  model.add(layers.Dense(1024, activation='relu'))    #was 1024
  # model.add(layers.Dropout(0.2))
  model.add(layers.Dense(512, activation='sigmoid'))  #was 512
  # model.add(layers.Dropout(0.2))
  model.add(layers.Dense(8, activation='sigmoid'))
  model.summary()

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),#'adam', LR=0.001 default
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


  total_computed_audios = int(dataset_total.batches.size/441/8/1)
  x_dataset = dataset_total.batches.reshape(total_computed_audios, 8, 441, 1) #reshaped data #was 100 initialy for 1 computed audio

  y_dataset = []
  for i in range(0, len(dataset_total.target)):
    for k in range (0, total_computed_audios // len(dataset_total.target)):
      y_dataset.append(dataset_total.target[i]) #every 441 samples per 100data there is change of targetDOA

  y_dataset = np.asarray(y_dataset)
  epochs = 15
  #model.load_weights(checkpoint_filepath) #load weights for further training

  checkpoint_filepath = 'tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                  save_weights_only=True,
                                                                  monitor='val_accuracy',
                                                                  mode='max',
                                                                  save_best_only=True)
  #history = model.fit(x = x_train, y = y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size = 100, verbose = 1)
  history = model.fit(x = x_dataset, y = y_dataset, validation_split = 0.25, epochs=epochs, batch_size = 100, verbose = 1, callbacks=[model_checkpoint_callback])

  show_results(history, epochs)

  #load best weights before prediction
  model.load_weights(checkpoint_filepath)

  #predict first and last - change it
  to_predict = x_dataset[0, 0:8, 0:441, 0]
  to_predict = to_predict.reshape(1, 8, 441, 1)
  prediction = model.predict(to_predict)
  print("Is" + str(prediction) + " should be " + str(y_dataset[0]))

  to_predict = x_dataset[-1, 0:8, 0:441, 0]
  to_predict = to_predict.reshape(1,8,441,1)
  prediction = model.predict(to_predict)
  print("Is" + str(prediction) + " should be " + str(y_dataset[-1]))

if __name__ == "__main__":
  main()