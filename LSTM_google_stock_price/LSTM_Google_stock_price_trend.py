import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from keras.callbacks import EarlyStopping


# PART 1
# Data Import
dataset_train = pd.read_csv("Data_sets/KO/KO_feb_24_2004_to_mar_23_2004.csv")
training_set = dataset_train.iloc[:, 1:2].values    # Upper Bound excluded

# Feature Scaling
# Normalisation  Xnorm = (X-min(X))/(max(X)-min(X))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 time steps and 1 output
X_train = []
y_train = []
# To update correct ranges of the test file strings amount use -1 from the list
for i in range(60, 5035):
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
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,         # Number of epochs with no improvement
    verbose=1,
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored metric
)
regressor.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

# Part 3
# Setting real stock price
dataset_test = pd.read_csv("Data_sets/KO/KO_feb_23_24_to_mar_23_24.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Predicting Stock Price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising
# plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()


# Using Plotly to make it compatible on different IDEs
trace1 = go.Scatter(
    x=list(range(len(real_stock_price))),
    y=real_stock_price.flatten(),
    mode='lines',
    name='Real Stock Price'
)
trace2 = go.Scatter(
    x=list(range(len(predicted_stock_price))),
    y=predicted_stock_price.flatten(),
    mode='lines',
    name='Predicted Stock Price'
)
# Layout
layout = go.Layout(
    title='Google Stock Price Prediction',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Stock Price')
)
# Figure
fig = go.Figure(data=[trace1, trace2], layout=layout)
# Show figure in a browser
fig.write_html('temp-plot.html', auto_open=True)

print(predicted_stock_price)
