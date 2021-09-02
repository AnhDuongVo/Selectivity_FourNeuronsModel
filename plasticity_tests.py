#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


def generate_SymmetricConnectivity(a,b,x: 'either connection c, d or e',model: 'number of model'):
    """Defines Jacobi matrix"""
    if model==1: 
        J = np.asarray([[0,x,0,-a], [x,0,-a,0], [b,0,0,0], [0,b,0,0]])
    elif model==2: 
        J = np.asarray([[0,0,0,-a], [0,0,-a,0], [b,0,0,-x], [0,b,-x,0]])
    else:
        J = np.asarray([[0,0,-x,-a], [0,0,-a,-x], [b,0,0,0], [0,b,0,0]])
    return J


# In[5]:


def generate_input(number_steps: 'number of steps', 
                   presentation_times: 'presentation time of single stimuli, in steps',
                   input_bias: 'input patterns for single stimuli'):
    """
    Function to generate time-varying input based on presentation times and biases
    """
    assert len(input_bias)==len(presentation_times)
    number_stimuli=len(input_bias)
    number_units = input_bias[0].shape[0]    
    inputs_all = np.empty((0,number_units), int)   
    counter=0
    while counter<number_steps:        
        for i_input in range(number_stimuli):            
            inputs_all= np.append(inputs_all, 
                            np.tile(input_bias[i_input].reshape(1, number_units),
                                    (presentation_times[i_input], 1)),
                                 axis=0)            
            counter=len(inputs_all)
    inputs_all=inputs_all[:number_steps]
    return inputs_all


# In[6]:


def plot_threshold(activity_all, dt, timescale):
    """Returns a list with thresholds over time calculated with activity"""
    thresholds_all = []
    for i in range(1,len(activity_all)):
        if i%10==0:
            activity = activity_all[:i]
            number_timepoints=len(activity)
            prefactors_inside_integral_pre=np.arange(-1*number_timepoints+1,1)*dt
            prefactors_inside_integral=np.exp(prefactors_inside_integral_pre*timescale)
            thresholds=timescale*integrate.simps(y=prefactors_inside_integral[:,None]*activity**2,x=None, dx=dt, axis=0)
            thresholds_all.append(thresholds)
    return thresholds_all


# In[18]:


def plot_figure(Sn,A,B,presentation_times,number_stimuli,tau_threshold_i,tau_learn_i, number_steps_before_learning,file_name):
    
    plt.style.use('default')
    figsize = (25,6*5)
    plt.rcParams.update({'font.size': 28})
    stimulus_colors =["lightsteelblue", "none", "mistyrose","none"]
    
    fig, axs = plt.subplots(5, figsize=figsize)
    
    fig.patch.set_facecolor('white')
    time_sec=np.arange(Sn.tsteps+1)*Sn.delta_t
    time_tau=time_sec/Sn.tau
    
    # Figure: activity of neuron 1 and 2
    axs[0].plot(time_tau, np.asarray(A)[:,:2])
    counter=0
    while counter<(Sn.tsteps-presentation_times[0]):        
        for i_input in range(number_stimuli):  
            if counter+presentation_times[i_input]>Sn.tsteps: continue
            axs[0].axvspan(time_tau[counter], time_tau[counter+presentation_times[i_input]],
                        facecolor=stimulus_colors[i_input], alpha=0.5)
            counter+=presentation_times[i_input] 
    axs[0].set_xlabel(r'time ($\tau$)')
    axs[0].set_ylabel("activity (a.u.)")
    axs[0].axvline(x=number_steps_before_learning,color='black')
    
    # Figure: activity of all neurons
    labels=["$r_1$", "$r_2$", "$r_3$", "$r_4$"]
    for i in range(4):
        axs[1].plot(time_tau, np.asarray(A)[:,i], label=labels[i])
    counter=0
    while counter<(Sn.tsteps-presentation_times[0]):        
        for i_input in range(number_stimuli):  
            if counter+presentation_times[i_input]>Sn.tsteps: continue
            axs[1].axvspan(time_tau[counter], time_tau[counter+presentation_times[i_input]],
                        facecolor=stimulus_colors[i_input], alpha=0.5)
            counter+=presentation_times[i_input]   
    axs[1].set_xlabel(r'time ($\tau$)')
    axs[1].set_ylabel("activity (a.u.)")
    axs[1].legend()
    axs[1].axvline(x=number_steps_before_learning,color='black')

    # Figure: weights
    weights=np.asarray(B)
    axs[2].plot(time_tau,weights[:,0,0], label=r"$w_{U1S1}$", color="mediumslateblue")
    axs[2].plot(time_tau,weights[:,1,0], label=r"$w_{U2S1}$", color="fuchsia")
    axs[2].plot(time_tau,weights[:,0,1], label=r"$w_{U1S2}$", color="limegreen")
    axs[2].plot(time_tau,weights[:,1,1], label=r"$w_{U2S2}$", color="crimson")
    counter=0
    while counter<(Sn.tsteps-presentation_times[0]):        
        for i_input in range(number_stimuli):  
            if counter+presentation_times[i_input]>Sn.tsteps: continue
            axs[2].axvspan(time_tau[counter], time_tau[counter+presentation_times[i_input]],
                        facecolor=stimulus_colors[i_input], alpha=0.5)
            counter+=presentation_times[i_input] 
    axs[2].set_xlabel(r'time ($\tau$)')
    axs[2].set_ylabel("weights")
    axs[2].legend()
    axs[2].axvline(x=number_steps_before_learning,color='black')

    # Figure: plot only small weights
    
    weights_all = [weights[:,0,0],weights[:,1,0],weights[:,0,1],weights[:,1,1]]
    weights_label = [r"$w_{U1S1}$",r"$w_{U2S1}$",r"$w_{U1S2}$",r"$w_{U2S2}$"]
    weights_colors = ["mediumslateblue","fuchsia","limegreen","crimson"]
    weights_index_sorted = []
    counter = 0
    for i in weights_all:
        weights_index_sorted.append([i[-1],counter])
        counter += 1
    weights_index_sorted.sort()
    i1 = weights_index_sorted[0][1]
    i2 = weights_index_sorted[1][1]
    
    axs[3].plot(time_tau,weights_all[i1], label=weights_label[i1], color=weights_colors[i1])
    axs[3].plot(time_tau,weights_all[i2], label=weights_label[i2], color=weights_colors[i2])
    counter=0
    while counter<(Sn.tsteps-presentation_times[0]):        
        for i_input in range(number_stimuli):  
            if counter+presentation_times[i_input]>Sn.tsteps: continue
            axs[3].axvspan(time_tau[counter], time_tau[counter+presentation_times[i_input]],
                        facecolor=stimulus_colors[i_input], alpha=0.5)
            counter+=presentation_times[i_input] 
    axs[3].set_xlabel(r'time ($\tau$)')
    axs[3].set_ylabel("weights")
    #minimum = min(np.concatenate((weights[:,0,0],weights[:,1,0],weights[:,0,1],weights[:,1,1])))
    #axs[3].set_ylim(minimum-0.01,minimum+0.01)
    axs[3].legend()
    axs[3].axvline(x=number_steps_before_learning,color='black')

    # Figure: thresholds
    thresholds_data = plot_threshold(np.asarray(A), 0.01, 1./tau_learn_i)
    time_tau2 = []
    for i in range(len(time_tau[:-1])):
        if i%10==0:
            time_tau2.append(time_tau[i])
    for i in range(4):
        axs[4].plot(time_tau2,np.asarray(thresholds_data).T[i], linewidth = 0.5, label = "r"+str(i+1))
    counter=0
    while counter<(Sn.tsteps-presentation_times[0]):        
        for i_input in range(number_stimuli):  
            if counter+presentation_times[i_input]>Sn.tsteps: continue
            axs[4].axvspan(time_tau[counter], time_tau[counter+presentation_times[i_input]],
                        facecolor=stimulus_colors[i_input], alpha=0.5)
            counter+=presentation_times[i_input] 
    axs[4].set_xlabel(r'time ($\tau$)')
    axs[4].set_ylabel("threshold")
    axs[4].legend()
    axs[4].axvline(x=number_steps_before_learning,color='black')
    
    fig.savefig(file_name)
    
#%%
def generate_W_project_initial(test,epsilon):
    if test == 1 or test == 3:
        W = np.asarray([[1,0],
                        [0,1],
                        [0,0],
                        [0,0]])
    elif test == 2 or test == 4:
        W = np.asarray([[1+epsilon,1],
                        [1,1+epsilon],
                        [0,0],
                        [0,0]])
    return(W)


# In[20]:

input_eps = 0.1

network_params = {
'mode': 'short_range',        
'nonlinearity_rule'	: 'rectification',
'gamma': 1,
'delta_t' : 0.05,        
'a_HS': [0.4,0.3,1.8],
'b_HS': [2.8,2.3,1.5],
'x_HS': [0.1,0.3,1.1],
'a_LS': [0.1,0.1,0.1],
'b_LS': [1.1,0.1,0.1],
'x_LS': [0.1,0.1,0.1],
'model': 1,
'w_initial' :1,
'update_function': 'version_normal',
'learning_rule': 'BCM',
'delta_t_learn_i': 10,
'nonlinearity_rule':'supralinear',
'integrator':'runge_kutta',  #runge kutta for greater accuracy
'tau': 0.1,  #intrinsic timescale of neurons
'tau_learn': 32,#tau_learn_i,  #timescale of learning rule to adapt weights
'tau_threshold': 100, #tau_threshold_i, #timescale over wich to calculate expected value --> thresholds 
'Ttau': 30000,        #how long to simulate        
'delta_t_learn': 10,
'number_steps_before_learning': 3500*2,
'equilibrium_diff': None,
'W_structure': np.asarray([[1,1],
                          [1,1],
                          [0,0],
                          [0,0]]),
'presentation_times': [1000,500,1000,500],
'input_bias': [np.asarray([1,1-input_eps]), 
               np.asarray([0,0]),
               np.asarray([1-input_eps,1]),
               np.asarray([0,0])],
'W_project_initial_epsilon': 0.05
}


# In[14]:


def run_pipeline(network_params,model,test,W_project_initial_epsilon,input_eps,tau_learn,tau_threshold,dir_name,save_data): #tau_learn_i,tau_threshold_i,
    start_time = time.time()

    #################### network  ####################
    network_params['model'] = model
    network_params['W_project_initial_epsilon'] = W_project_initial_epsilon
    network_params['input_bias'] = [np.asarray([1,1-input_eps]), 
                                   np.asarray([0,0]),
                                   np.asarray([1-input_eps,1]),
                                   np.asarray([0,0])]
    network_params['tau_learn'] = tau_learn
    network_params['tau_threshold'] = tau_threshold
    
    
    if test == 1 or test == 2:
        network_params['W_rec'] = generate_SymmetricConnectivity(a=network_params['a_HS'][model-1],
                                   b=network_params['b_HS'][model-1],
                                   x=network_params['x_HS'][model-1],
                                    model=model)
    else:
        network_params['W_rec'] = generate_SymmetricConnectivity(a=network_params['a_LS'][model-1],
                                   b=network_params['b_LS'][model-1],
                                   x=network_params['x_LS'][model-1],
                                    model=model)

    
    network_params['W_project_initial'] = \
        generate_W_project_initial(test,network_params['W_project_initial_epsilon'])

    Sn=nm.SimpleNetwork(params=network_params) 
    
    

    #################### Input  ####################
    inputs=generate_input(Sn.tsteps,
                          network_params['presentation_times'], 
                          network_params['input_bias'])

    #################### Simulation  ####################
    A, B=Sn.run(inputs, np.random.rand(4))
    
    varied_params_string = "test" + str(test) + "_model" + str(model) + "_input_eps" + \
                    str(input_eps) + "_Weps" + str(W_project_initial_epsilon) + "_tauthr" + str(tau_threshold)
                    #str(input_eps) + "_taulearn" + str(tau_learn) + "_tauthr" + str(tau_threshold) \
    if save_data == 1:
        with open(dir_name + "/" + varied_params_string+'_activity.npy', 'wb') as f:
            np.save(f, np.array(A))
        f.close()
        with open(dir_name + "/" + varied_params_string+'_weights.npy', 'wb') as f:
            np.save(f, np.array(B))
        f.close()
    
    file_name = dir_name + "/" + varied_params_string + ".pdf"
    plot_figure(Sn,A,B,network_params['presentation_times'],
                len(network_params['input_bias']),
                network_params['tau_threshold'],
                network_params['tau_learn'],
                network_params['number_steps_before_learning']/2,
                file_name)
    
    print(time.time()-start_time)


# In[19]:
data_file = "data_files"
if not os.path.exists(data_file):
    os.makedirs(data_file)
    
save_data = 1
    
par = [[[0.4, 0.5, 2.9], [2.6, 1.9, 2.6], [0.1, 0.1, 2.7],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 0.9, 2.9]]
       ] 
"""par = [[[0.4, 0.5, 2.9], [2.6, 1.9, 2.6], [0.1, 0.1, 2.7],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 0.9, 2.9]],
       [[0.4, 0.5, 2.9], [2.6, 1.9, 1.9], [0.1, 0.1, 2.2],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 1.8, 2.9]],
       [[0.4, 0.5, 2.9], [2.6, 1.9, 0.7], [0.1, 0.1, 1.3],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 1.9, 2.9]],
       [[0.4, 0.5, 2.8], [2.6, 1.9, 2.3], [0.1, 0.1, 2.2],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 2.0, 2.9]],
       [[0.4, 0.5, 2.8], [2.6, 1.9, 2.1], [0.1, 0.1, 2.5],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 2.1, 2.9]],
       [[0.4, 0.5, 2.8], [2.6, 1.9, 1.2], [0.1, 0.1, 1.8],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 2.2, 2.9]],
       [[0.4, 0.5, 2.7], [2.6, 1.9, 2.7], [0.1, 0.1, 2.5],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 2.3, 2.9]],
       [[0.4, 0.5, 2.7], [2.6, 1.9, 1.5], [0.1, 0.1, 2.2],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 2.4, 2.9]],
       [[0.4, 0.5, 2.7], [2.6, 1.9, 1.3], [0.1, 0.1, 2.1],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 2.5, 2.9]],
       [[0.4, 0.5, 2.7], [2.6, 1.9, 0.6], [0.1, 0.1, 1.2],[0.1, 0.1, 0.1], [0.1, 2.2, 0.1], [0.1, 2.6, 2.9]]
       ] """
number = 0
for [network_params['a_HS'],network_params['b_HS'],network_params['x_HS'],network_params['a_LS'],network_params['b_LS'],network_params['x_LS']] in par:   
    dir_name = data_file+"/"+time.strftime("%d%b%Y_%H:%M:%S")+"_"+str(number)
    
    number += 1
    os.makedirs(dir_name)
    copyfile('plasticity_tests.py', dir_name+"/Parameters.py")
    Parallel(n_jobs=2)(delayed(run_pipeline)(network_params,model,test,W_project_initial_epsilon,input_eps,tau_learn,tau_threshold,dir_name,save_data) 
                        for test in [4]
                       for model in [1] 
                        for W_project_initial_epsilon in [0.05]
                        for input_eps in [1.0]
                        for tau_learn in [100]
                        for tau_threshold in [64])
    
    
    

#%%
file_names = os.listdir(dir_name)
#%%

activity_names = []
for f in file_names:
    if f[-12:] == "activity.npy":
        activity_names.append(f)
activity_names.sort()

weights_names = []
for f in file_names:
    if f[-11:] == "weights.npy":
        weights_names.append(f)
weights_names.sort()

#%%

def weight_output_sel(weights,time_point):
    weights_start_sorted = [weights[time_point,0,0],weights[time_point,1,0],weights[time_point,0,1],weights[time_point,1,1]]
    weights_start_sorted.sort()
    r1 = (weights_start_sorted[0]+weights_start_sorted[1])/2
    r2 = (weights_start_sorted[2]+weights_start_sorted[3])/2
    return(abs(r1-r2)/(r1+r2))

def activity_output_sel(activity,time_point):
    r1,r2 = activity[time_point][0], activity[time_point][1]
    return(abs(r1-r2)/(r1+r2))

if len(activity_names) <= 15:
    print("weird")
    tsteps = network_params['Ttau']*2
    
    plt.style.use('default')
    figsize = (25,6*len(activity_names))
    plt.rcParams.update({'font.size': 28})
    stimulus_colors =["lightsteelblue", "none", "mistyrose","none"]
    plt.tight_layout()
    
    
    fig, axs = plt.subplots(len(activity_names), figsize=figsize)
    
    fig.patch.set_facecolor('white')
    time_sec=np.arange(tsteps+1)*network_params['delta_t']
    time_tau=time_sec/network_params['tau']
    
    for i in range(len(activity_names)):
        activity = np.load(dir_name+"/"+activity_names[i])
        
        act_start = round(activity_output_sel(activity,int(network_params['presentation_times'][0]/2)),3)
        
        end = activity.shape[0]
        var = sum(network_params['presentation_times'])-network_params['presentation_times'][0]/2
        end = int(end - end%sum(network_params['presentation_times'])-var)
        
        act_end = round(activity_output_sel(activity,end),3)
        
        weights = np.load(dir_name+"/"+weights_names[i])
        print(weights[0,:,:])
        wei_start = round(weight_output_sel(weights,0),3)
        wei_end = round(weight_output_sel(weights,-1),3)
        
        # Figure: activity of neuron 1 and 2
        axs[i].plot(time_tau, activity[:,:2])
        counter=0
        while counter<(tsteps-network_params['presentation_times'][1]):        
            for i_input in range(4):  
                if counter+network_params['presentation_times'][i_input]>tsteps: continue
                axs[i].axvspan(time_tau[counter], time_tau[counter+network_params['presentation_times'][i_input]],
                            facecolor=stimulus_colors[i_input], alpha=0.5)
                counter+=network_params['presentation_times'][i_input] 
        axs[i].set_ylabel("activity (a.u.)")
        output_sel_titel = "aStart"+str(act_start)+"  aEnd"+str(act_end)+"  wStart"+str(wei_start)+"  wEnd"+str(wei_end)
        axs[i].set_title(activity_names[i][:-13]+"\n"+output_sel_titel)
        axs[i].axvline(x=network_params['number_steps_before_learning']/2,color='black')
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    
    fig.tight_layout()
    axs[-1].set_xlabel(r'time ($\tau$)')

    fig.savefig(dir_name+"/activity_all.pdf")
 
else:

    for model in ["1","2","3"]:
        for test in ["1","2","3","4"]:
            print("hi")
            activity_names = []
            for f in file_names:
                if f[-12:] == "activity.npy" and f[11] == model and f[4] == test:
                    activity_names.append(f)
            activity_names.sort()
            if len(activity_names) == 0:
                continue
            
            weights_names = []
            for f in file_names:
                if f[-11:] == "weights.npy" and f[11] == model and f[4] == test:
                    weights_names.append(f)
            weights_names.sort()

            
            tsteps = network_params['Ttau']*2
            
            plt.style.use('default')
            figsize = (25,6*len(activity_names))
            plt.rcParams.update({'font.size': 28})
            stimulus_colors =["lightsteelblue", "none", "mistyrose","none"]
            plt.tight_layout()
            
            
            fig, axs = plt.subplots(len(activity_names), figsize=figsize)
            
            fig.patch.set_facecolor('white')
            time_sec=np.arange(tsteps+1)*network_params['delta_t']
            time_tau=time_sec/network_params['tau']
            
            for i in range(len(activity_names)):
                activity = np.load(dir_name+"/"+activity_names[i])
                
                act_start = round(activity_output_sel(activity,int(network_params['presentation_times'][0]/2)),3)
                
                end = activity.shape[0]
                var = sum(network_params['presentation_times'])-network_params['presentation_times'][0]/2
                end = int(end - end%sum(network_params['presentation_times'])-var)
                
                act_end = round(activity_output_sel(activity,end),3)
                
                weights = np.load(dir_name+"/"+weights_names[i])
                wei_start = round(weight_output_sel(weights,0),3)
                wei_end = round(weight_output_sel(weights,-1),3)
                print(activity_names[i])
                print(weights_names[i])
                print(weights[0,:,:])
                

                # Figure: activity of neuron 1 and 2
                axs[i].plot(time_tau, activity[:,:2])
                counter=0
                while counter<(tsteps-network_params['presentation_times'][1]):        
                    for i_input in range(4):  
                        if counter+network_params['presentation_times'][i_input]>tsteps: continue
                        axs[i].axvspan(time_tau[counter], time_tau[counter+network_params['presentation_times'][i_input]],
                                    facecolor=stimulus_colors[i_input], alpha=0.5)
                        counter+=network_params['presentation_times'][i_input] 
                axs[i].set_ylabel("activity (a.u.)")
                output_sel_titel = "aStart"+str(act_start)+"  aEnd"+str(act_end)+"  wStart"+str(wei_start)+"  wEnd"+str(wei_end)
                axs[i].set_title(activity_names[i][:-13]+"\n"+output_sel_titel)
                axs[i].axvline(x=network_params['number_steps_before_learning']/2,color='black')
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
            fig.tight_layout()
            axs[-1].set_xlabel(r'time ($\tau$)')
    
            fig.savefig(dir_name+"/test"+test+"_model"+model+".pdf")
            
    

