import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from keras.callbacks import EarlyStopping

# Data Set location
dataset_train_location = "Data_sets/Stock_predictions_for_April2024/Financial_sector/JPM/JPM_20years.csv"
# Variables
time_steps = 180
n_days_in_the_future = 20
epochs_number = 30
batch_size_number = 32
# Variables for Fitting
monitor_type = 'val_loss'
patience_number = 10
verbose_number = 1
restore_best_weights = True
# Variables Compiler
optimizer_type = 'adam'
loss_type = 'mean_squared_error'

# Data Import
dataset_train = pd.read_csv(dataset_train_location)
training_set = dataset_train[['Open']].values    # Upper Bound excluded

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(time_steps, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-time_steps:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping from 2 to 3 dimensions. 1 outside [] - indicator
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
regressor = Sequential()

# Adding Layers
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=40))
regressor.add(Dropout(rate=0.2))
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer=optimizer_type, loss=loss_type)

# Fitting the RNN to the training set
early_stopping = EarlyStopping(
    monitor=monitor_type,
    # Number of epochs with no improvement
    patience=patience_number,
    verbose=verbose_number,
    # Restores model weights from the epoch with the best value of the monitored metric
    restore_best_weights=restore_best_weights)
regressor.fit(X_train, y_train, epochs=epochs_number, batch_size=batch_size_number, callbacks=[early_stopping])

last_n_days = training_set_scaled[-time_steps:].reshape(1, time_steps, 1)
future_predictions_scaled = []
for _ in range(n_days_in_the_future):  # Predict N days into the future
    next_day_price_scaled = regressor.predict(last_n_days)
    future_predictions_scaled.append(next_day_price_scaled[0, 0])
    next_day_price_reshaped = next_day_price_scaled.reshape(1, 1, 1)
    last_n_days = np.append(last_n_days[:, 1:, :], next_day_price_reshaped, axis=1)

future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)

future_predictions = sc.inverse_transform(future_predictions_scaled)
future_time_axis = list(range(training_set.shape[0]))

# Visualization with Plotly for Future Predictions
trace_future = go.Scatter(
    x=future_time_axis,
    y=future_predictions.flatten(),
    mode='lines',
    name='Future Predicted Stock Price'
)

# Setting up the layout for future predictions
layout_future = go.Layout(
    title='Future Stock Price Prediction',
    xaxis=dict(title='Day'),
    yaxis=dict(title='Stock Price')
)

# Constructing the figure for future predictions
fig_future = go.Figure(data=[trace_future], layout=layout_future)

# Displaying the figure in the browser for future predictions
fig_future.write_html('future_prediction_plot.html', auto_open=True)

# Create CSV with predicted prices
predicted_stock_price_df = pd.DataFrame(future_predictions, columns=['Predicted Price'])
predicted_stock_price_df.to_csv('predicted_stock_prices.csv', index=False)