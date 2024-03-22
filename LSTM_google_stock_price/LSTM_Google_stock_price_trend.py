import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# PART 1
# Data Import
dataset_train = pd.read_csv("Data_sets/google_stock_prices/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values    # Upper Bound excluded

# Feature Scaling
# Normalisation  Xnorm = (X-min(X))/(max(X)-min(X))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping from 2 to 3 dimensions. 1 outside [] - indicator
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# PART 2
# Building LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
regressor = Sequential()

# Adding Layers
# 1 Layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))
# 2 Layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
# 3 Layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
# 4 Layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))
# Output
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 3
# Setting real stock price
dataset_test = pd.read_csv("Data_sets/google_stock_prices/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Predicting Stock Price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)





print(real_stock_price)
