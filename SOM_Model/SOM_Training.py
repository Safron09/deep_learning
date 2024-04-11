import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to 'Qt5Agg' or another interactive backend
import matplotlib.pyplot as plt


# Variables
data_set_train = 'Data_sets/SOM_Data/Credit_Card_Applications.csv'

# Import Data
dataset = pd.read_csv(data_set_train)
X = dataset.iloc[:, :-1].values  # Took all columns but last
y = dataset.iloc[:, -1].values  # Only Last column

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=10)

# Visualizing SOM results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)

# Finding Fraud
mappings = som.win_map(X)
fraud = np.concatenate((mappings[(8, 5)], mappings[(6, 7)]), axis=0) # Axis 0 - vertical, axis 1 - horizontal
fraud = sc.inverse_transform(fraud)
print(fraud)
show()