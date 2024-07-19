import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sklearn

# Data set Variables for prediction.
# In Geography France [1, 0, 0], Germany [0, 0, 1], Spain [0, 1, 0]
geography_1 = 0  # 0
geography_2 = 1   # 1
geography_3 = 0  # 0
credit_score = 0  # 0
# Male as 1, Female as 0
gender = 1    # 1
age = 60    # 60
tenure = 0     # 0
balance = 100000  # 0
num_of_products = 3    # 1
has_cr_card = 1         # 1
is_active_member = 1    # 0
salary = 100000         # 200000

# dataset = pd.read_csv('C:\MY_Coding_projects\Projects\DEEP_Learning\Deep_learning\Data_sets\Churn_Modelling.csv')
# Code Below helps to make this project portable and does not require Absolut Path
script_dir = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_dir, '../..', 'Data_sets', 'Churn_Modelling.csv')
dataset = pd.read_csv(data_file_path)

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Gender as binary
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encoding Geography as binary
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting Data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing ANN, adding layers
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training SET
ann.fit(X_train, y_train, batch_size=32, epochs=20)

prediction = ann.predict(sc.transform([[geography_1, geography_2, geography_3, credit_score, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, salary]])) > 0.5

age_str = str(age)
balance_str = str(balance)
salary_str = str(salary)

message = "Will person with this parameters leave the Bank: Age=" + age_str + ", Balance=" + balance_str + ", Salary=" + salary_str
print(message)

if prediction[0][0]:
    print("Yes, the person is likely to leave the bank.")
else:
    print("No, the person is not likely to leave the bank.")
