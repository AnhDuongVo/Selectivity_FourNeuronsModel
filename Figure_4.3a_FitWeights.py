### Figure 4.3 a

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

weights_all = [[],[],[],[]]
 
for test_index in range(4):
    for model_index in range(3):
        file_name_weights = dir_file + '/test'+str(test_index+1)+'_model'+str(model_index+1)+'_epsilon0.05_weights.npy'
        weights_all[test_index].append(np.load(file_name_weights))

            
#%% Calculate steps of weights
weight_step_array = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]      
for test_index in range(4):
    for model_index in range(3):   
        weight_steps = np.transpose(weights_all[test_index][model_index][:,:,:],(1,2,0))[:,:,500::2000]
        weight_step_array[test_index][model_index] = weight_steps

#%% Exemplary Weight exponential Plot
# Figure 4.3 a

def func(x,a,t):
  return a*(np.exp(-t/x))

for test_index in [0]:
    for model_index in [0]:  
        for i in [0]:
            for j in [0]:
                x = np.arange(0,100000,1000)[1:-2]   # changed boundary conditions to avoid division by 0
                y = weight_step_array[0][model_index][i][j][3:]-weight_step_array[0][model_index][i][j][3]
                
                popt, pcov = curve_fit(func, x, y)
                
                fig, ax = plt.subplots(figsize = (8,6))
                plt.scatter(x, y, s = 0.3, label = "Data")
                plt.plot(x, func(x, *popt), 'r-', label="Exponential fit")
                size_text = 20
                plt.xlabel(r"time ($\tau$)", size = size_text)
                plt.ylabel("weight", size = size_text)
                plt.legend(fontsize = size_text)
                plt.yticks(size = size_text)
                plt.xticks(size = size_text)
     
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.tight_layout()
                plt.show()   
                fig.savefig("figures/res3fit.pdf")
