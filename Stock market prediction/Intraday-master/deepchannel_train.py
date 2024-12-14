import os
import numpy
import time
from datetime import datetime, timedelta 
import random
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
warnings.filterwarnings("ignore")


def preprocess(df, fill_method = "mean", columns = None):
  df = df.rename({"SIZE": "VOL"}, axis = 1)
  df["DATETIME"] = ((df["DATE"].astype(str)) + " " + (df["TIME"].astype(str))).astype('datetime64[ns]')
  df["DATETIME_FLOAT"] = df["DATETIME"].astype("int64") / 1e9
  if columns is None:
        columns = df.columns  
    
  for column in columns:
      if df[column].isnull().any(): 
        if fill_method == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif fill_method == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif fill_method == 'mode':
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
        elif fill_method == 'ffill':
            df[column].fillna(method='ffill', inplace=True)
        elif fill_method == 'bfill':
            df[column].fillna(method='bfill', inplace=True)
        else:
            df[column].fillna(fill_method, inplace=True) 
  return df


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.001
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# Data loading and Initializations
batch_size = 256

df = pd.read_csv('D:/DEEP-CHANNEL/Stocks.csv', header = None)
df_volume = df[4]
df = df.drop([0, 4], axis = 1)
df = df.drop([0], axis = 0)

dataset = df.values.astype('float64')
timep = dataset[:, 0] 
maxer = np.amax(dataset[:, 2])
maxeri = maxer.astype('int')
maxchannels = maxeri
idataset = dataset[:, 1].astype(int) 

# Scaling
features = dataset[:, :-1]  
company_labels = dataset[:, -1]
scaler = MinMaxScaler(feature_range = (0,1)) 
dataset = scaler.fit_transform(features)
dataset = np.hstack([dataset, company_labels.reshape(-1, 1)])


# Train and test set split and reshaping
train_size = int(len(dataset) * 0.80)
modder = math.floor(train_size/batch_size)
train_size = int(modder*batch_size)
test_size = int(len(dataset) - train_size)
modder = math.floor(test_size/batch_size)
test_size = int(modder*batch_size)

print(f'training set = {train_size}')
print(f'test set = {test_size}')
print(f'total length = {test_size + train_size}')

x_train = dataset[:, 1] 
y_train = idataset[:] 

x_train = x_train.reshape((len(x_train), 1))
y_train = y_train.reshape((len(y_train), 1))
print(len(x_train), len(y_train))

# Random Over Sampler
ros = RandomOverSampler(random_state=42)
X_res, Y_res = ros.fit_resample(x_train, y_train)

yy_res = Y_res.reshape((len(Y_res), 1))
xx_res, yy_res = shuffle(X_res, yy_res)

# Reshaping and setting training parameters
trainy_size = int(len(xx_res) * 0.80)
modder = math.floor(trainy_size/batch_size)
trainy_size = int(modder*batch_size)
testy_size = int(len(xx_res) - trainy_size)
modder = math.floor(testy_size/batch_size)
testy_size = int(modder*batch_size)

print('training set= ', trainy_size)
print('test set =', testy_size)
print('total length', testy_size+trainy_size)

new_shape = (batch_size, -1, 3, 1, 1)  

in_train, in_test = xx_res[0:trainy_size,0], xx_res[trainy_size:trainy_size+testy_size, 0]
target_train, target_test = yy_res[0:trainy_size,:], yy_res[trainy_size:trainy_size+testy_size, :]
print("IN-TRAIN: ", in_train[0:5])
print("IN-TEST", in_test[0:5])

in_train = in_train.reshape(new_shape)
in_test = in_test.reshape(new_shape)
target_train = target_train.reshape(new_shape)  
target_test = target_test.reshape(new_shape)  

if len(in_train) % batch_size != 0:
    raise ValueError("Data length is not divisible by batch size!")

# print("Shapes of in_train, target_train, in_test, target_test: ", in_train.shape, target_train.shape, in_test.shape, target_test.shape)

# Model architecture
in_train = in_train[:, :10, :, :, :]  # Selecting only the first 10 timesteps
in_test = in_test[:, :10, :, :, :]

target_train = target_train[:, :10, :, :, :]  
target_test = target_test[:, :10, :, :, :]

timestep = 10
input_dim = 3 

model_input = Input(shape=(10, 3, 1, 1)) 

newmodel = Sequential()

newmodel.add(ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu',  input_shape=(10, 3, 1, 1), return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(ConvLSTM2D(filters=32, kernel_size=(1, 1), activation='relu', return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(ConvLSTM2D(filters=16, kernel_size=(1, 1), activation='relu', return_sequences=True)) 
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(Flatten()) 
newmodel.add(Dense(10*3*1*1, activation='sigmoid'))
newmodel.add(Reshape((10,3,1,1)))

print(len(in_train), len(target_train))

newmodel.compile(loss="mean_squared_error",  
                 optimizer="adam",   
                 metrics=["mse", Precision(), Recall()])  

newmodel.summary()

lrate = LearningRateScheduler(step_decay)
epochers = 2

for i, layer in enumerate(newmodel.layers):
    print(f"Layer {i}: {layer.name}, Output shape: {newmodel.output_shape}")

history = newmodel.fit(x=in_train, y=target_train, epochs=epochers, batch_size=256, callbacks=[lrate])

predict = newmodel.predict(in_test, batch_size=batch_size)
print("predict - shape: ", len(predict))
predict_reshape = predict.reshape(-1, predict.shape[-1])

timep = timep[:256]
timep = [datetime.fromtimestamp(ms / 1000.0) for ms in timep]

plt.figure(figsize=(10, 5))
plt.plot(timep[-200:], predict_reshape[-200:], color="orange")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

plt.xlabel('Timestamp')
plt.ylabel('Predictions')
plt.title('Normalized Predictions vs Time')
plt.savefig("Stock_predictions.png")