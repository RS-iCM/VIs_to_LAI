###################################################################
### DNN regression model to simulate LAI using VIs data for wheat
### Input: Rice VIs and LAI data "Wheat_LAI_n_VIs.csv"
### Output: Pickle model "wheat_NN.h5"
### Last modified: June 20, 2025
###################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('data/Wheat_LAI_n_VIs.csv')  # Rice
data = df[['DOY', 'MTVI1', 'NDVI', 'OSAVI', 'RDVI']].to_numpy()
LAI = df[['LAI']].to_numpy()

# Produce a scale of the data
scaler = preprocessing.StandardScaler()

# change data
data_standardized = scaler.fit_transform(data)

# Splitting the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(
data_standardized, LAI, test_size=0.30, random_state=0)

# Creating the model
model = models.Sequential()

# Adding layers to the model with the relu activation function
model.add(layers.Dense(units=10,
                         activation="relu",
                         input_shape=(features_train.shape[1],)))

# Adding more layers to the model
model.add(layers.Dense(units=100, activation="relu"))
model.add(layers.Dense(units=500, activation="relu"))
model.add(layers.Dense(units=1000, activation="relu"))
model.add(layers.Dense(units=500, activation="relu"))
model.add(layers.Dense(units=100, activation="relu"))
model.add(layers.Dense(units=10, activation="relu"))

# Adding the dropout layer to prevent overfitting
model.add(keras.layers.Dropout(0.15))

# Adding the output layer with an activation function
model.add(layers.Dense(units=1))

# Completing the model setup
rmse = tf.keras.metrics.RootMeanSquaredError()
model.compile(loss="mse",
                optimizer="RMSprop",
                #metrics=["accuracy"])
                metrics=["mae", "mse", rmse])

checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Make a callback to save the model weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Show the progress of training by printing a dot (.) at the end of each epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 300
    
# Training the model
history = model.fit(features_train, # Features vector
                      target_train, # Target vector
                      epochs=EPOCHS, # Epochs number
                      verbose=0, # No output during training
                      batch_size=50, # Batch size
                      validation_data=(features_test, target_test), # Test data
                      callbacks=[cp_callback])  # Transfer the callback to save the model weights
                      #callbacks=[PrintDot()]) 

#"""# Save the training and validation loss records
training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

# Produce a count object for the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Root Mean Squared Error
training_rmse = history.history["root_mean_squared_error"]
test_rmse = history.history["val_root_mean_squared_error"]
plt.plot(epoch_count, training_rmse, "r--")
plt.plot(epoch_count, test_rmse,"b-")

# plot
plt.legend(["Training RMSE", "Test RMSE"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
#plt.ylim((0,1))
plt.show(); #"""

# Calling `save('my_model')`
model.save('models/wheat_NN.h5')
