### Additional weight fitting plots and Figure 4.3 b,c,d

from os.path import abspath
import sys
sys.path.append(abspath(''))
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import Implementation.network_model as nm
from scipy import integrate
from joblib import Parallel, delayed
from shutil import copyfile
from scipy.optimize import curve_fit

#%% Load data

dir_file = "data_files/17May2021_14:46:16" # change data file name here

activity_all = [[],[],[],[]]
weights_all = [[],[],[],[]]
 
for test_index in range(4):
    for model_index in range(3):
        file_name_activity = dir_file + '/test'+str(test_index+1)+'_model'+str(model_index+1)+'_epsilon0.05_activity.npy'
        file_name_weights = dir_file + '/test'+str(test_index+1)+'_model'+str(model_index+1)+'_epsilon0.05_weights.npy'
        activity_all[test_index].append(np.load(file_name_activity))
        weights_all[test_index].append(np.load(file_name_weights))
        
#%% Calculate Sum of two weights
weight_step_array = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]      
for test_index in range(4):
    for model_index in range(3):   
        weight_steps = np.transpose(weights_all[test_index][model_index][:,:,:],(1,2,0))
        weight_step_array[test_index][model_index] = weight_steps


w2_all = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]      
for test_index in range(4):
    for model_index in range(3):   
        weight_steps = np.transpose(weights_all[test_index][model_index][:,:,:],(1,2,0))
        w2 = (weight_step_array[test_index][model_index][0][0]+weight_step_array[test_index][model_index][1][1])/2
        w2_all[test_index][model_index] = w2

#%% Weight exponential Plot for all 4 tests

def func(t,m,T,c):
  return m*(1-np.exp(-t/T)) + c

fit_param_m = [[],[],[],[]]
fit_param_T = [[],[],[],[]]
fit_param_n = [[],[],[],[]]

for test_index in range(4):
    print("Test " + str(test_index+1))
    for model_index in range(3):  
        
        x = np.linspace(0, 90000, 190001)
        if test_index == 3:
            x = np.linspace(0, 190000, 390001)
        y = w2_all[test_index][model_index][10000:]

        popt, pcov = curve_fit(func, x, y)
        fit_param_m[test_index].append(popt[0])
        fit_param_T[test_index].append(popt[1])
        fit_param_n[test_index].append(popt[2])
        plt.plot(x, func(x, *popt), label='Model '+str(model_index+1)+', fit: m=%5.3f, T=%5.3f, n=%5.3f' % tuple(popt))
        
    plt.xlabel("time (tau)")
    plt.ylabel("weight average")
    plt.legend()
    plt.show()
        
#%% Weight exponential Plot for all 3 models

def func(t,m,T,c):
  return m*(1-np.exp(-t/T)) + c

for model_index in range(3):  
    print("Model " + str(model_index+1))
    for test_index in range(4):
        x = np.linspace(0, 90000, 190001)
        if test_index == 3:
            x = np.linspace(0, 190000, 390001)
        y = w2_all[test_index][model_index][10000:]#-w2_all[test_index][model_index][10000]

        popt, pcov = curve_fit(func, x, y)
        plt.plot(x, func(x, *popt), label='Test '+str(test_index+1)+', fit: m=%5.3f, T=%5.3f, n=%5.3f' % tuple(popt))
        
    plt.xlabel("time (tau)")
    plt.ylabel("weight average")
    plt.legend()
    plt.show()
    
#%% Bar Plot for all fitting parameters m_end, T, m_start
# Figure 4.3 b, c, d

data_all = [np.array(fit_param_m).T, np.array(fit_param_T).T, np.array(fit_param_n).T]
ylabel_all = [r'parameter $m_{end}$',r'parameter $T$',r'parameter $m_{start}$']

for i in range(3):
    data = data_all[i]
    
    # set width of bars
    barWidth = 0.25
     
    # Set position of bar on X axis
    r1 = np.arange(len(data[0]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
     
    # Make the plot
    fig, ax = plt.subplots(figsize = (8,6))
    
    plt.bar(r1, data[0], color='#7f6d5f', width=barWidth, edgecolor='white', label='Model 1')
    plt.bar(r2, data[1], color='#557f2d', width=barWidth, edgecolor='white', label='Model 2')
    plt.bar(r3, data[2], color='#2d7f5e', width=barWidth, edgecolor='white', label='Model 3')
     
    # Add xticks on the middle of the group bars
    size_text = 20
    plt.xlabel('Test', size = size_text)
    plt.ylabel(ylabel_all[i], size = size_text)
    plt.xticks([r + barWidth for r in range(len(data[0]))], ['1', '2','3','4'], size = size_text)
    plt.yticks(size = size_text)
     
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Create legend & Show graphic
    plt.legend(fontsize = size_text)
    plt.tight_layout()
    fig.savefig('figures/res3'+str(i)+'leg.pdf')
    plt.show()
