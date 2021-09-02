from os.path import abspath, sep, pardir
import sys
sys.path.append(abspath('') + sep + pardir + sep )
import numpy as np
import time
import Implementation.tools as snt
import Implementation.integration_methods as im

class SimpleNetwork:
    def __init__(self,
                params=None):
        self.W_rec=params['W_rec']
        self.W_input=params['W_project_initial']
        if params['W_structure'] is not None:
            self.W_structure=params['W_structure']
        else:
            self.W_structure=np.ones((W_rec.shape[0], W_project.shape[-1]))
        self.delta_t = params['delta_t']
        self.tau=params['tau']
        self.tau_learn=params['tau_learn']
        self.tau_threshold=params['tau_threshold']
        self.tsteps=int((params['Ttau']*params['tau'])/params['delta_t'])
        self.number_steps_before_learning=params['number_steps_before_learning']
        self.equilibrium_diff=params['equilibrium_diff']
        self.number_timepoints_plasticity=int(-1*(self.tau_threshold/self.delta_t)*np.log(0.1))
        
        self.nonlinearity_rule = params['nonlinearity_rule']
        self.integrator = params['integrator']   
        self.inputs = None        
        self.delta_t_learn=params['delta_t_learn']
        self.learningrule=params['learning_rule'] 
        self.update_function=params['update_function'] 
        self.gamma=params['gamma']
        self.start_activity = 0.
        self._init_nonlinearity()
        self._init_update_function()
        self._init_learningrule()
        self._init_integrator()

        
    def _init_nonlinearity(self):
        if self.nonlinearity_rule=='supralinear':
            self.np_nonlinearity = snt.supralinear(self.gamma)
            
        elif self.nonlinearity_rule == 'sigmoid':
            self.t_nonlinearity = snt.nl_sigmoid
        elif self.nonlinearity_rule == 'tanh':
            self.t_nonlinearity = snt.nl_tanh
            
    def _init_update_function(self):
        if self.update_function=='version_normal':
            self.update_act = im.update_network
            
    def _init_learningrule(self):
        if self.learningrule=='none':
            self.learningrule_function = im.nonlearning_weights  
        if self.learningrule=='BCM_rule_test':
            self.learningrule_function = BCM_rule_test
        if self.learningrule=='BCM':
            self.learningrule_function = im.BCM_rule_sliding_th
            
    def _init_integrator(self):        
        if self.integrator == 'runge_kutta':
            self.integrator_function = im.runge_kutta_explicit    
        elif self.integrator == 'forward_euler':
            self.integrator_function = im.forward_euler
        else:
            raise Exception("Unknown integrator ({0})".format(self.integrator))
    
    def _check_input(self, inputs):
        Ntotal = self.tsteps
        assert inputs.shape[-1]==self.W_input.shape[-1]
        if len(inputs.shape)==1:            
            inputs_time=np.tile(inputs, (Ntotal,1))
        if len(inputs.shape)==2:
            assert inputs.shape[0]==Ntotal
            inputs_time=inputs
        return inputs_time
            
    def run(self, inputs, start_activity):
        Ntotal = self.tsteps
        all_act=[]
        all_act.append(start_activity)
        all_weights=[]
        all_weights.append(self.W_input)
        all_thresholds=[]
        
        inputs_time = self._check_input(inputs)
        
        if self.equilibrium_diff != None:
            # perturbation during equilibrium
        
            equilibrium_counter = 0 # indicator for equilibrium
        
            for step in range(Ntotal):
                new_act=self.integrator_function(self.update_act,  #intergation method
                                  all_act[-1],  #general parameters     
                                  delta_t=self.delta_t,
                                  tau=self.tau, w_rec=self.W_rec , w_input=all_weights[-1], #kwargs
                                  Input=inputs_time[step],            
                                  nonlinearity=self.np_nonlinearity, )
                
                # check whether in equilibrium or not
                if np.abs(np.sum(all_act[-1]-new_act)) <= 0.0001:
                    equilibrium_counter += 1
                    if equilibrium_counter == 50: # if 50 times small changes --> equilibrium
                        new_act += self.equilibrium_diff
                else:
                    equilibrium_counter = 0
                
                # append new activity to use fosr learning
                all_act.append(new_act) 
                
                if step<self.number_steps_before_learning or not(step%self.delta_t_learn==0): # added not(step%100...)
                    new_weights = all_weights[-1]
                else:
                    new_weights = self.integrator_function(self.learningrule_function,   #learning rule
                                             all_weights[-1], #general parameters 
                                             delta_t=self.delta_t, #kwargs
                                             tau=self.tau, tau_learn=self.tau_learn, 
                                             tau_threshold=self.tau_threshold,
                                             w_rec=self.W_rec , w_input=self.W_input,
                                             w_struct_mask=self.W_structure,
                                             Input=inputs_time[step], 
                                             prev_act=all_act[-self.number_timepoints_plasticity:], 
                                             nonlinearity=self.np_nonlinearity,)
                    #all_thresholds.append(new_thresholds)
                
                all_weights.append(new_weights)
                
        else:
            for step in range(Ntotal):
                new_act=self.integrator_function(self.update_act,  #intergation method
                                  all_act[-1],  #general parameters     
                                  delta_t=self.delta_t,
                                  tau=self.tau, w_rec=self.W_rec , w_input=all_weights[-1], #kwargs
                                  Input=inputs_time[step],            
                                  nonlinearity=self.np_nonlinearity, )
                
                all_act.append(new_act) # append new activity to use for learning
                
                if step<self.number_steps_before_learning or not(step%self.delta_t_learn==0): # added not(step%100...)
                    new_weights = all_weights[-1]
                else:
                    new_weights = self.integrator_function(self.learningrule_function,   #learning rule
                                             all_weights[-1], #general parameters 
                                             delta_t=self.delta_t, #kwargs
                                             tau=self.tau, tau_learn=self.tau_learn, 
                                             tau_threshold=self.tau_threshold,
                                             w_rec=self.W_rec , w_input=self.W_input,
                                             w_struct_mask=self.W_structure,
                                             Input=inputs_time[step], 
                                             prev_act=all_act[-self.number_timepoints_plasticity:], 
                                             nonlinearity=self.np_nonlinearity,)
                    #all_thresholds.append(new_thresholds)
                
                all_weights.append(new_weights)
                #print(new_weights)
                #print(new_act)
                
        return all_act, all_weights
            
        
        
        
    