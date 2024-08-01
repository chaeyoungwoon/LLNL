import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import cardiac_ml_tools as cardiac_ml_tools


data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR = "intracardiac_dataset"  # This should be the path to the intracardiac_dataset
for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(os.path.join(DIR, x))  # Use os.path.join to correctly form the path
file_pairs = cardiac_ml_tools.read_data_dirs(data_dirs, verbose=1)
print('Number of file pairs: {}'.format(len(file_pairs)))
# example of file pair
print("Example of file pair:")
print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))

# Example of file pair
case = 213

# ECG plot
row = 3
column = 4
num_timesteps = 500
plt.figure(figsize=(10, 7))
titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
reorder = {1:1,2:5,3:9,4:2,5:6,6:10,7:3,8:7,9:11,10:4,11:8,12:12} # reorder the leads to standard 12-lead ECG display
print('Case {} : {}'.format(case, file_pairs[case][0]))
pECGData = np.load(file_pairs[case][0])
pECGData = cardiac_ml_tools.get_standard_leads(pECGData)

# create a figure with 12 subplots
for i in range(pECGData.shape[1]):
    plt.subplot(row, column, reorder[i + 1])
    plt.plot(pECGData[0:num_timesteps,i],'r')
    plt.title(titles[i])
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xlabel('msec')
    plt.ylabel('mV')
plt.tight_layout()
plt.show()

# close
plt.close()

# make a plot with the "pECGData" -> "ActTime"
case = 213
print('Case {} : {}'.format(case, file_pairs[case][0]))
pECGData = np.load(file_pairs[case][0])
pECGData = cardiac_ml_tools.get_standard_leads(pECGData)
print('Case {} : {}'.format(case, file_pairs[case][0]))
VmData = np.load(file_pairs[case][1])
ActTime = cardiac_ml_tools.get_activation_time(VmData)

# plot in row the tensors pECGData and ActTime with an arrow pointing to the activation time
row = 1
column = 3
plt.figure(figsize=(20, 5))
plt.subplot(row, column, 1)

# plot pECGData transposed
plt.imshow(pECGData.T, cmap='jet', interpolation='nearest', aspect='auto')
plt.title('pECGData')
plt.subplot(row, column, 2)

# print an arrow
plt.text(0.5, 0.5, '-------->', horizontalalignment='center', verticalalignment='center', fontsize=20)
plt.axis('off')
plt.subplot(row, column, 3)

# plot ActTime
plt.imshow(ActTime, cmap='jet', interpolation='nearest', aspect='auto')

# not xticks
plt.xticks([])
plt.title('ActTime')
plt.show()
plt.close()

# Import image module
from IPython.display import Image

# Get the image
Image(url="figures/banner.png")
import os
print('The working directory is :{}'.format(os.getcwd()))
import sys
print("Python is in %s" % sys.executable)
print("Python version is %s.%s.%s" % sys.version_info[:3])
import glob
import re, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data_dirs(dirs_names, verbose = 0):
    file_pairs = []
    for dir in dirs_names:
        all_files = sorted(glob.glob(dir + '/*.npy'))
        files_Vm=[]
        files_pECG=[]
        if verbose > 0:
            print('Reading files...',end='')
        for file in all_files:
            if 'VmData' in file:
                files_Vm.append(file)
            if 'pECGData' in file:
                files_pECG.append(file)
        if verbose > 0:
            print(' done.')
        if verbose > 0:
            print('len(files_pECG) : {}'.format(len(files_pECG)))
            print('len(files_Vm) : {}'.format(len(files_Vm)))
        for i in range(len(files_pECG)):
            VmName =  files_Vm[i]
            VmName = VmName.replace('VmData', '')
            pECGName =  files_pECG[i]
            pECGName = pECGName.replace('pECGData', '')
            if pECGName == VmName :
                file_pairs.append([files_pECG[i], files_Vm[i]])
            else:
                print('Automatic sorted not matching, looking for pairs ...',end='')
                for j in range(len(files_Vm)):
                    VmName =  files_Vm[j]
                    VmName = VmName.replace('VmData', '')
                    if pECGName == VmName :
                        file_pairs.append([files_pECG[i], files_Vm[j]])
                print('done.')
    return file_pairs
    data_dirs = []
regex = r'data_hearts_dd_0p2*'

for x in os.listdir('./'):
    if re.match(regex, x):
        data_dirs.append(x)

file_pairs = read_data_dirs(data_dirs)
len(file_pairs)
row = 3; column = 4
scaling_ecg = "none"
plt.figure(figsize=(10, 7))
titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
row = 3; column = 4
reorder = {1:1,2:5,3:9,4:2,5:6,6:10,7:3,8:7,9:11,10:4,11:8,12:12}
index = np.random.choice(range(len(file_pairs)),1)
index = [213]

for case in index:
    print('Case {} : {}'.format(case, file_pairs[case][0]))
    pECGData = np.load(file_pairs[case][0])
    VmData = np.load(file_pairs[case][1])
    dataECG = pECGData               # dataECG  : RA LA LL RL V1 V2 V3 V4 V5 V6
    ecg12aux = np.zeros((dataECG.shape[0], 12))    # ecg12aux : i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6
    WilsonLead = 0.33333333 * (dataECG[:, 0] + dataECG[:, 1] + dataECG[:, 2])
    # Lead I: LA - RA
    ecg12aux[:, 0] = dataECG[:, 1] - dataECG[:, 0]
    # Lead II: LL - RA
    ecg12aux[:, 1] = dataECG[:, 2] - dataECG[:, 0]
    # Lead III: LL - LA
    ecg12aux[:, 2] = dataECG[:, 2] - dataECG[:, 1]
    # Lead aVR: 3/2 (RA - Vw)
    ecg12aux[:, 3] = 1.5 * (dataECG[:, 0] - WilsonLead)
    # Lead aVL: 3/2 (LA - Vw)
    ecg12aux[:, 4] = 1.5 * (dataECG[:, 1] - WilsonLead)
    # Lead aVF: 3/2 (LL - Vw)
    ecg12aux[:, 5] = 1.5 * (dataECG[:, 2] - WilsonLead)
    # Lead V1: V1 - Vw
    ecg12aux[:, 6] = dataECG[:, 4] - WilsonLead
    # Lead V2: V2 - Vw
    ecg12aux[:, 7] = dataECG[:, 5] - WilsonLead
    # Lead V3: V3 - Vw
    ecg12aux[:, 8] = dataECG[:, 6] - WilsonLead
    # Lead V4: V4 - Vw
    ecg12aux[:, 9] = dataECG[:, 7] - WilsonLead
    # Lead V5: V5 - Vw
    ecg12aux[:, 10] = dataECG[:, 8] - WilsonLead
    # Lead V6: V6 - Vw
    ecg12aux[:, 11] = dataECG[:, 9] - WilsonLead
    pECGData = ecg12aux
    for i in range(pECGData.shape[1]):
        plt.subplot(row, column, reorder[i + 1])
        plt.plot(pECGData[0:500, i], 'r')
        plt.title(titles[i])
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        # plt.xlabel('msec')
        # plt.ylabel('mV')
    plt.tight_layout()
    plt.show()

num_timesteps = 500
plt.figure(figsize=(18, 9))
row = 7; column = 10
index = np.random.choice(range(len(file_pairs)),1)
index = [213]

for case in index:
    VmData = np.load(file_pairs[case][1])
    randomIndex = range(VmData.shape[1])
    for count, i in enumerate(randomIndex):
        plt.subplot(8, 10, count + 1)
        plt.plot(VmData[0:num_timesteps,i])
        plt.title(f'i = {i}')
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.tight_layout()
plt.show()

import numpy
import matplotlib.pyplot as plt
pECG = np.load('intracardiac_dataset/data_hearts_dd_0p2/pECGData_hearts_dd_0p2_volunteer.v10_pattern.0.npy')
Vm = np.load('intracardiac_dataset/data_hearts_dd_0p2/VmData_hearts_dd_0p2_volunteer.v10_pattern.0.npy')
plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
plt.plot(pECG)
plt.subplot(1,2,2)
plt.plot(Vm)
plt.show()

# Prepare the data
ecg_data_list = []
act_time_list = []
for i, (ecg_file, vm_file) in enumerate(file_pairs):
    try:
        # Load ECG data and get standard leads
        pECGData = np.load(ecg_file)
        pECGData = cardiac_ml_tools.get_standard_leads(pECGData)

        # Load Vm data and get activation time
        VmData = np.load(vm_file)
        ActTime = cardiac_ml_tools.get_activation_time(VmData)

        # print(ActTime)
        # Flatten the activation time array to match the shape of the ECG data
        ActTime_flat = ActTime.flatten()

        # print(ActTime_flat)
        ecg_data_list.append(pECGData)
        act_time_list.append(ActTime_flat)
    except Exception as e:
        print(f"Error processing case {i}: {e}")
ecg_data_array = np.array(ecg_data_list)
act_time_array = np.array(act_time_list)

# print(ecg_data_array.shape)
# print(act_time_array.shape)
# Example of file pair
case = 213
plt.figure(figsize=(1, 10))
print('Case {} : {}'.format(case, file_pairs[case][0]))
VmData = np.load(file_pairs[case][1])
ActTime = cardiac_ml_tools.get_activation_time(VmData)

# plot the Activation Time array for train
plt.imshow(ActTime, cmap='jet', interpolation='nearest', aspect='auto')
plt.title('Activation Time')
plt.colorbar()
plt.grid(visible=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()

# not xticks
plt.xticks([])
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

X = ecg_data_array.reshape(16117, -1)
# print(X)
# print(X.shape)

y = act_time_array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Train a neural network
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(len(y_pred))
print(y_pred.shape)

mse = mean_squared_error(y_test, y_pred)

print("Number of predictions:", len(y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))
print("Root mean squared error:", root_mean_squared_error(y_test, y_pred))
print ("Mean absolute error:", mean_absolute_error(y_test, y_pred))

"""

# plot of the increase of mean absolute error with prediction error
from matplotlib import pyplot

"""
# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, color='red', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()


case = 213
plt.figure(figsize=(1, 10))
print('Case {} : {}'.format(case, file_pairs[case][0]))
VmData = np.load(file_pairs[case][1])
ActTime = cardiac_ml_tools.get_activation_time(VmData)

# plot the Activation Time array for prediction
plt.imshow(ActTime, cmap='jet', interpolation='nearest', aspect='auto')
plt.title('Activation Time')
plt.colorbar()
plt.grid(visible=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()

# not xticks
plt.xticks([])
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

#######
#######
#######
# Visualizing the accuracy of our data 
# Scatter plot with line of best fit for mean absolute error and distance of accuracy measured in ms
# Define variables for future use
errors = y_test - y_pred
y_test_flat = y_test.flatten()
errors_flat = errors.flatten()

# Calculate line of best fit
m, b = np.polyfit(y_test_flat, errors_flat, 1)

# Calculate the predicted errors from the line of best fit
# By multiplying y test by m, it will scale the values by the slope of the line
# Add b to shift up to match
predicted_errors = m * y_test_flat + b

# Make the scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(y_test.flatten(), errors.flatten(), color='blue', label='Errors')

# Plot
m, b = np.polyfit(y_test_flat, errors_flat, 1)
plt.plot(y_test_flat, predicted_errors, color='red', label='Line of Best Fit')

# Var to get total number of points 
totalNumber = len(y_test_flat)

# Label
plt.xlabel('Actual Values in ms')
plt.ylabel('Prediction Errors in ms')
plt.title(f'Prediction Errors vs. Actual Values\n(Total Number: {totalNumber})')
plt.legend()
plt.show()

#
#
# Second scatter plot that will get points within 1ms
# This is to estimate our accuracy
# Calculate errors
errors = y_test - y_pred

# This is a bound that gets points within 1ms of line of best fit (using mean abs errors)
accuracyScore = 1.0

# Filter points within the accuracyScore (1ms)
is_accurate = np.abs(errors_flat - predicted_errors) <= accuracyScore
y_test_filtered = y_test_flat[is_accurate]
errors_filtered = errors_flat[is_accurate]

# Var to get total number of points in the bounded plot
is_accurate = len(y_test_filtered)

# Make the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_filtered, errors_filtered, color='blue', label='Accurate Errors')

# Plot
plt.plot(y_test_flat, predicted_errors, color='red', label='Line of Best Fit')

# Label
plt.xlabel('Actual Values in ms')
plt.ylabel('Prediction Errors in ms')
plt.title(f'Prediction Errors vs. Actual Values\n(Number of Points within 1ms: {is_accurate})')
plt.legend()
plt.show()

#
#
# Third scatter plot that will get points outside of 1ms
# This is to estimate our inaccuracy
# Calculate errors
errors = y_test - y_pred

# This is a bound that gets points within 1ms of line of best fit (using mean abs errors)
accuracyScore = 1.0

# Filter points within the accuracyScore (1ms)
isnt_accurate = np.abs(errors_flat - predicted_errors) > accuracyScore
y_test_filtered = y_test_flat[isnt_accurate]
errors_filtered = errors_flat[isnt_accurate]

# Var to get total number of points in the bounded plot
isnt_accurate = len(y_test_filtered)

# Make the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_filtered, errors_filtered, color='blue', label='Errors')

# Plot
plt.plot(y_test_flat, predicted_errors, color='red', label='Line of Best Fit')

# Label
plt.xlabel('Actual Values in ms')
plt.ylabel('Prediction Errors in ms')
plt.title(f'Prediction Errors vs. Actual Values\n(Number of Points greater than 1ms: {isnt_accurate})')
plt.legend()
plt.show()

#99.3% accurate at 0.2 training data




#############################################
# TASK 4
