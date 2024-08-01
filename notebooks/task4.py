import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import cardiac_ml_tools as cmt
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
)
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR = "intracardiac_dataset"  # This should be the path to the intracardiac_dataset
for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(os.path.join(DIR, x))  # Use os.path.join to correctly form the path

# Generates a list of ECG/Vm pairs
file_pairs = cmt.read_data_dirs(data_dirs, verbose=1)
print('Number of file pairs: {}'.format(len(file_pairs)))
ecg_data_array = np.load('ecg_data_list.npy')
act_time_array = np.load('act_time_array.npy')
vm_data_array = np.load('vm_data_list.npy')
print("Done with loading npy files")
print(len(vm_data_array))
print(len(vm_data_array[0]))
print(vm_data_array[0].shape[1])
X = ecg_data_array.reshape(16117, -1)
y = vm_data_array.reshape(16117, -1)
print(len(y))
print(len(y[0]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Done with splitting")
'''
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
print("Length of Y-Test",len(y_test))
'''
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Dense layer NN
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_train.shape[1]))  # Output layer with the same number of features as y

# Compile the model
optimizer = Adam(learning_rate=0.05)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict
y_pred = model.predict(X_test)
print("Model Performance:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("Mean Absolute error:", mean_absolute_error(y_test, y_pred) )

y_pred = y_pred.reshape(len(y_pred), 500, 75)
y_test = y_test.reshape(len(y_test), 500, 75)

# Example of file pair
case = 213
num_timesteps = 500

# Vm plot
row = 7
column = 10
plt.figure(figsize=(18, 9))
print('Case {} : {}'.format(case, file_pairs[case][0]))
print('Y_Pred Length:', y_pred[case].shape[1]) # 3224 needs to be 75
for count, i in enumerate(range(y_pred[case].shape[1])):
    plt.subplot(8, 10, count + 1)
    plt.plot(y_test[case][0:num_timesteps,i], color='blue', label='Actual')
    plt.plot(y_pred[case][0:num_timesteps,i], color='orange', label='Prediction')
    plt.title(f'i = {i}')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # plt.xlabel('msec')
    # plt.ylabel('mV')
plt.tight_layout()
plt.show()

# close
plt.close()
print("Y_Test:", y_test[case])
print("Y_Pred:", y_pred[case])