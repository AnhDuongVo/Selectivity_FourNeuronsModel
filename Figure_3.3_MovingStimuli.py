### Figure 3.3 a,b
# Evolution of activity in case of Moving Stimuli
# Bar plot of output selectivity in case of Moving Stimuli

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
test_number = 4
model_number = 3
dir_file = "data_files/27May2021_09:44:48" # Change file name here

activity_all = [[],[],[],[]]
 
for test_index in range(test_number):
    for model_index in range(3):
        file_name_activity = dir_file + '/test'+str(test_index+1)+'_model'+str(model_index+1)+'_epsilon0.1_activity.npy'
        file_name_weights = dir_file + '/test'+str(test_index+1)+'_model'+str(model_index+1)+'_epsilon0.1_weights.npy'
        activity_all[test_index].append(np.load(file_name_activity)[50:2000])
        
#%% Calculate mean of activity
r1_all = [[],[],[],[]]   
r2_all = [[],[],[],[]]        
for test_index in range(test_number):
    for model_index in range(3):   
        stim_area = np.concatenate([activity_all[test_index][model_index].T[:,1::4], activity_all[test_index][model_index].T[:,2::4]],axis=1)
        r1, r2 = np.mean(stim_area[0]), np.mean(stim_area[1]) 
        r1_all[test_index].append(r1)
        r2_all[test_index].append(r2)

#%% Evolution of activity 
# Figure 3.3 a
        
for test_index in [0]:
    for model_index in [0]:  
        fig, ax = plt.subplots(figsize = (8,2.8))
        x_axis = np.arange(0,int(activity_all[test_index][model_index].shape[0]),1)
        time_steps = 30
        plt.plot(x_axis[:time_steps],activity_all[test_index][model_index].T[0][:time_steps],label=r"$r_1$")
        plt.plot(x_axis[:time_steps],activity_all[test_index][model_index].T[1][:time_steps],label=r"$r_2$")
        stimulus_colors =["lightsteelblue","none","none","mistyrose"]
        presentation_times = [1,1,1,1]
        counter=0
        
        size_text = 20
        plt.xlabel(r"time ($\tau$)", size = size_text)
        plt.ylabel("activity", size = size_text)
        plt.yticks(size = size_text)
        plt.xticks(size = size_text)
        
        while counter<int(activity_all[test_index][model_index][:time_steps].shape[0]-4):        
            for i_input in range(4):  
                #if counter+presentation_times[i_input]>int(activity_all[test_index][model_index].shape[0])-5: continue
                plt.axvspan(x_axis[counter], x_axis[counter+presentation_times[i_input]],
                            facecolor=stimulus_colors[i_input], alpha=0.5)
                counter+=presentation_times[i_input] 
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.show()   
        fig.savefig("figures/moving_stimuli_leftwards.pdf")

output_sel_neuron = [[],[],[],[]]

for test_index in range(test_number):
    for model_index in range(3):          
        r1 = r1_all[test_index][model_index]
        r2 = r2_all[test_index][model_index]
        output_sel_neuron[test_index].append(abs(r1-r2)/(r1+r2))

#%% Bar plot of output selectivity of Test 1 and 2
# Figure 3.3 b

data = np.array(output_sel_neuron).T[:,:2]

# set width of bars
barWidth = 0.25
fig, ax = plt.subplots(figsize = (4,6))
# Set position of bar on X axis
r1 = np.arange(len(data[0]))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, data[0], color='#7f6d5f', width=barWidth, edgecolor='white', label='Model 1')
plt.bar(r2, data[1], color='#557f2d', width=barWidth, edgecolor='white', label='Model 2')
plt.bar(r3, data[2], color='#2d7f5e', width=barWidth, edgecolor='white', label='Model 3')
 
# Add xticks on the middle of the group bars
size_text = 20
plt.xlabel('Test', size = size_text)
plt.ylabel('output selectivity', size = size_text)
plt.xticks([r + barWidth for r in range(len(data[0]))], ['1', '2'],size = size_text)
plt.yticks(size = size_text)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(fontsize = 15, bbox_to_anchor=(0.38,1))
plt.tight_layout()
plt.show()   
fig.savefig("figures/res2barplot.pdf") # Change folder and name of file here