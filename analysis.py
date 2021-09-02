#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:17:33 2021

@author: anhduongvo
"""

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

#%%

dir_file = "data_files/17May2021_14:46:16"

activity_all = [[],[],[],[]]
weights_all = [[],[],[],[]]
 
for test_index in range(4):
    for model_index in range(3):
        file_name_activity = dir_file + '/test'+str(test_index+1)+'_model'+str(model_index+1)+'_epsilon0.05_activity.npy'
        file_name_weights = dir_file + '/test'+str(test_index+1)+'_model'+str(model_index+1)+'_epsilon0.05_weights.npy'
        activity_all[test_index].append(np.load(file_name_activity))
        weights_all[test_index].append(np.load(file_name_weights))
        
#%% Output Selectivity
output_sel_start_all = [[],[],[],[]] 
output_sel_end_all = [[],[],[],[]]       
for test_index in range(4):
    for model_index in range(3):     
        end = activity_all[test_index][model_index].shape[0]
        end = int(end - end%1000 -755)
        r1,r2 = activity_all[test_index][model_index][1245][0], activity_all[test_index][model_index][1245][1]
        output_sel_start_all[test_index].append(abs(r1-r2)/(r1+r2))
        r1,r2 = activity_all[test_index][model_index][end][0], activity_all[test_index][model_index][end][1]
        output_sel_end_all[test_index].append(abs(r1-r2)/(r1+r2))

#%% Weight growth
weight_start_all = [[],[],[],[]] 
weight_end_all = [[],[],[],[]]       
for test_index in range(4):
    for model_index in range(3):     
        weights = weights_all[test_index][model_index] 
        weights_start_sorted = [weights[0,0,0],weights[0,1,0],weights[0,0,1],weights[0,1,1]]
        weights_start_sorted.sort()
        r1 = (weights_start_sorted[0]+weights_start_sorted[1])/2
        r2 = (weights_start_sorted[2]+weights_start_sorted[3])/2
        weight_start_all[test_index].append(abs(r1-r2)/(r1+r2))
        
        weights_start_sorted = [weights[-1,0,0],weights[-1,1,0],weights[-1,0,1],weights[-1,1,1]]
        weights_start_sorted.sort()
        r1 = (weights_start_sorted[0]+weights_start_sorted[1])/2
        r2 = (weights_start_sorted[2]+weights_start_sorted[3])/2
        weight_end_all[test_index].append(abs(r1-r2)/(r1+r2))
        
#%% Output Selectivity
ls_hs_all_r1 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]      
for test_index in range(4):
    for model_index in range(3):   
        act_array = np.array(activity_all[test_index][model_index]).T
        for t in range(25,len(act_array[1]),2000):
            print(act_array[1][t])
            if act_array[1][t] <= 0.00001:
                ls_hs_all_r1[test_index][model_index] = t
                print("ok")
                break
            
ls_hs_all_r2 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]      
for test_index in range(4):
    for model_index in range(3):   
        act_array = np.array(activity_all[test_index][model_index]).T
        for t in range(1025,len(act_array[0]),2000):
            print(act_array[0][t])
            if act_array[0][t] <= 0.00001:
                ls_hs_all_r2[test_index][model_index] = t
                print("ok")
                break
            
#%% Weight explonential
weight_step_array = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]      
for test_index in range(4):
    for model_index in range(3):   
        weight_steps = np.transpose(weights_all[test_index][model_index][:,:,:],(1,2,0))[:,:,500::2000]
        weight_step_array[test_index][model_index] = weight_steps
        print(test_index, model_index)
        #for i in range(2):
         #   for j in range(2):
        for [i,j] in [[0,1],[1,0]]: 
            plt.plot(range(100), weight_step_array[test_index][model_index][i][j])
        plt.show()

#%% Weight exponential Plot

def func(x,a,t):
  return a*(np.exp(-t/x))

for test_index in range(4):
    for model_index in range(3):  
        for i in range(2):
            for j in range(2):
                print(test_index, model_index, i, j)
                x = np.arange(0,100000,1000)[1:-2]   # changed boundary conditions to avoid division by 0
                y = weight_step_array[0][model_index][i][j][3:]-weight_step_array[0][model_index][i][j][3]
                
                popt, pcov = curve_fit(func, x, y)
                print("Fitting data",popt)
                
                plt.figure()
                plt.scatter(x, y, s = 0.3, label = "data")
                plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
                plt.xlabel("time (tau)")
                plt.ylabel("weight")
                plt.legend()
                plt.show()
  
#%% output
print("")
print("output sel start")
print(output_sel_start_all)
print("")
print("output sel end")
print(output_sel_end_all)
print("")
print("weight sel start")
print(weight_start_all)
print("")
print("weight sel end")
print(weight_end_all)  
print("")
print("LS to HS r1")
print(ls_hs_all_r1)  
print("LS to HS r2")
print(ls_hs_all_r2)    

#%% Weight exponential Plot

def func(x,a,t):
  return a*(np.exp(-t/x))

for test_index in [0]:
    for model_index in [0]:  
        for i in [0]:
            for j in [0]:
                print(test_index, model_index, i, j)
                x = np.arange(0,100000,1000)[1:-2]   # changed boundary conditions to avoid division by 0
                y = weight_step_array[0][model_index][i][j][3:]-weight_step_array[0][model_index][i][j][3]
                
                popt, pcov = curve_fit(func, x, y)
                print("Fitting data",popt)
                
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
