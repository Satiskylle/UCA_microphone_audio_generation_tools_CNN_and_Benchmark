from sklearn.metrics import plot_confusion_matrix
from neural_network_dataset_prepare import Dataset, nn_database as nn

from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import os

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
  plt.show(block=False)

def create_keras_model():
  model = keras.Sequential()
  
  #batch 441, 100 rows, 8 cols, 1 channel, so there is 100 samples with 8 channel each, 
  model.add(layers.Conv2D(16, (2, 1), padding='valid', activation='relu', input_shape=(8, 441, 1)))
  model.add(layers.Conv2D(8, (2, 1), padding='valid', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(1,4)))
  model.add(layers.Conv2D(16, 1, padding='valid', activation='relu'))
  model.add(layers.Conv2D(16, 1, padding='valid', activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(1,2)))
  model.add(layers.Conv2D(16, 1, padding='valid', activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.15))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.15))
  model.add(layers.Dense(512, activation='sigmoid'))
  model.add(layers.Dropout(0.15))
  model.add(layers.Dense(1, activation='tanh'))
  model.summary()

  model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),#'adam', LR=0.001 default
                loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
                metrics=['accuracy'])

  return model

def get_data_from_dataset(dump_filename_path):
  dataset_total = Dataset()
  dataset_total = nn.receive_dumped_object(dataset_total, dump_filename = dump_filename_path)
  total_computed_audios = int(dataset_total.batches.size/441/8/1)
  x_dataset = dataset_total.batches.reshape(total_computed_audios, 8, 441, 1) #reshaped data #was 100 initialy for 1 computed audio

  for i in range(0, len(dataset_total.target)):
    dataset_total.target_float = np.append(dataset_total.target_float, ((dataset_total.target[i].astype(np.float) - 180) / 180))

  y_dataset = []
  for i in range(0, len(dataset_total.target_float)):
    for k in range (0, total_computed_audios // len(dataset_total.target_float)):
      y_dataset.append(dataset_total.target_float[i]) #every 441 samples per 100data there is change of targetDOA

  y_dataset = np.asarray(y_dataset)

  return x_dataset, y_dataset
  
def train_network(network_model, epochs, model_checkpoint_callback, dataset_path):
  x_dataset, y_dataset = get_data_from_dataset(dataset_path)
  history = network_model.fit(x = x_dataset, y = y_dataset, validation_split = 0.25, epochs=epochs, batch_size = 32, verbose = 1, callbacks=[model_checkpoint_callback])
  return history

def main():
  model = create_keras_model()
  epochs = 10
  checkpoint_filepath = 'tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                  save_weights_only=True,
                                                                  monitor='val_loss',
                                                                  mode='auto',
                                                                  save_best_only=True)

  model.load_weights(checkpoint_filepath) #load weights for further training
  
  path_containing_dataset = "X:/generated_audio_0SNR_pickle/"
  audio_paths_which_can_be_loaded = [f for f in os.listdir(path_containing_dataset) if os.path.isdir(os.path.join(path_containing_dataset, f))]
  for next_pickle_folder in range (29, 30):
    pickle_path = path_containing_dataset + audio_paths_which_can_be_loaded[next_pickle_folder]
    pickles = [f for f in os.listdir(pickle_path) if not os.path.isdir(os.path.join(pickle_path, f))]
    for next_pickle in pickles:
      next_pickle_path = pickle_path + '/' + next_pickle
      print("Running pickle " + next_pickle_path)
      history = train_network(model, epochs, model_checkpoint_callback, next_pickle_path)
      show_results(history, epochs)

  #history = train_network(model, epochs, model_checkpoint_callback, "X:/generated_audio_0_SNR/cat/.pickled_database_1")

  # Below, there are simple testing functions to determine neural network behaviour. For benchmarking tests, use benchmark.py script.
  #load best weights before prediction
  model.load_weights(checkpoint_filepath)

  prediction_dataset = Dataset()
  prediction_dataset = nn.receive_dumped_object(prediction_dataset, dump_filename="X:/generated_audio_test/.pickled_database_val_0")
  total_computed_predictions_audios = int(prediction_dataset.batches.size/441/8/1)
  validation_dataset = prediction_dataset.batches.reshape(total_computed_predictions_audios, 8, 441, 1)
  
  predicted_labels = []
  true_labels = []

  for audio_batch_to_predict in tqdm(range(0, total_computed_predictions_audios//100), desc="Predicting values via model."):
    to_predict = validation_dataset[audio_batch_to_predict*100, 0:8, 0:441, 0] #check every 100'th value
    to_predict = to_predict.reshape(1,8,441,1)
    model_predicted_raw = model.predict(to_predict)
    prediction = np.round((model_predicted_raw * 180) + 180, decimals=0).astype(np.int32) #for float values
    predicted_labels.append(prediction[0][0])
    true_labels.append(prediction_dataset.target[audio_batch_to_predict])

  plt.figure(2)
  print("Confusion matrix")
  cf_matrix = confusion_matrix(y_pred=predicted_labels, y_true=true_labels)
  ax = sns.heatmap(cf_matrix, annot=False, cmap='Blues')
  ax.set_title('Confusion Matrix\n')
  ax.set_xlabel('\nPredicted Values')
  ax.set_ylabel('Actual Values ')

  ## Ticket labels - List must be in alphabetical order
  heatmap_labels = []
  list_of_ticks = []
  for i in range (0, 359):
    heatmap_labels.append(str(i))
    list_of_ticks.append(i)

  ax.set_xticks(list_of_ticks)
  ax.set_yticks(list_of_ticks)
  ax.xaxis.set_ticklabels(heatmap_labels)
  ax.yaxis.set_ticklabels(heatmap_labels)

  ## Display the visualization of the Confusion Matrix.
  plt.show(block=True)

if __name__ == "__main__":
  main()