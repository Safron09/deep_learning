import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping


data_file_path = 'Data_sets/appl_stock/AAPL.csv'
dataset = pd.read_csv(data_file_path)
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Next Close'] = dataset['Close'].shift(-1)
# Dropping the last row
dataset.dropna(inplace=True)
# Last known price and date from the dataset
last_known_price = dataset['Close'].iloc[-1]
last_known_date = dataset['Date'].iloc[-1]

X = dataset[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
y = dataset['Next Close'].values

# Normalizing the features
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l1=1e-4, l2=1e-3),
               activity_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
model.add(Dropout(0.2))
model.add(LSTM(units=30, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l1=1e-4, l2=1e-3),
               activity_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer=Adam(learning_rate=0.001 ), loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
model.fit(X_train, y_train, epochs=100, batch_size=32)

loss = model.evaluate(X_test, y_test)

last_input = X_scaled[-1].reshape((1, X_scaled.shape[1], X_scaled.shape[2]))
next_day_prediction_scaled = model.predict(last_input)
next_day_prediction = scaler_y.inverse_transform(next_day_prediction_scaled)

predicted_date = last_known_date + pd.Timedelta(days=1)
predicted_date_str = predicted_date.strftime('%Y-%m-%d')
last_known_date_str = last_known_date.strftime('%Y-%m-%d')

if next_day_prediction[0, 0] > last_known_price:
    print(f"1. Price will go up to {next_day_prediction[0, 0]:.2f} from {last_known_date_str} to {predicted_date_str}, "
          f"compared to the last known price of {last_known_price:.2f} in the dataset.")
else:
    print(f"2. Price will go down to {next_day_prediction[0, 0]:.2f} from {last_known_date_str} to {predicted_date_str},"
          f" compared to the last known price of {last_known_price:.2f} in the dataset.")