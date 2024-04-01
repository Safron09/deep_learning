import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Variables
data_set_train = 'Data_sets/SOM_Data/Credit_Card_Applications.csv'

# Import Data
dataset = pd.read_csv(data_set_train)
X = dataset.iloc[:, :-1].valuse  # Took all columns but last
y = dataset.iloc[-1].valuse  # Only Last column

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

