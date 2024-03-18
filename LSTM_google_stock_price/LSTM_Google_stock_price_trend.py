import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Import
dataset_train = pd.read_csv("Data_sets/google_stock_prices/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values    # Upper Bound excluded

# Feature Scaling
# Normalisation  Xnorm = (X-min(X))/(max(X)-min(X))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)


print(training_set_scaled)