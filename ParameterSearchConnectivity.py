### Parameter Search for connectivities
# where initial output selectivity is approx. the same across all models

from os.path import abspath
import sys
sys.path.append(abspath(''))
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import integrate
from joblib import Parallel, delayed

#%% Functions for output selectivity and conditions of different selectivity
# states for all models

def g1(K, M):
    """Calculate Output Selectivity for Model 1 given connectivity terms"""
    result = abs((1+M*K-M-K)/(1-K**2))/((1+M*K+M+K)/(1-K**2))
    return(result)

def g2(l, d, M):
    """Calculate Output Selectivity for Model 2 given connectivity terms"""
    result = abs((d*l-1+M*(l-d)-(l-d+M*(d*l-1)))/(l**2-1))/((d*l-1+M*(l-d)+l-d+M*(d*l-1))/(l**2-1))
    return(result)


def g3(z, k, M):
    """Calculate Output Selectivity for Model 3 given connectivity terms"""
    result = abs((k - M*z-(M*k-z))/(k**2-z**2))/((k - M*z+M*k-z)/(k**2-z**2)) 
    return(result)

def comparable_connectivity(M, model, needed_state): 
    """ Calculate all possible connectivities given needed selectivity states
    Input: 
        M (Float)- Input Modulator
        model (either 1,2 or 3) - Model name
        needed_states ([[bool,bool,bool,bool,bool],...]) - List of needed
            selectivity states [<LS>,<HS Left>,<HS Right>,<SHS Left>,<SHS Right>]
            where activeness is indicated by bool
    Returns:
        [states, a, b, x, output selectivity, model] where x is either c,d,e
        and a, b, c, d, e are defined connectivities in our models
    """
    
    candidates = []
    if model == 1:
        for a in np.arange(0.1,3,0.1):
            for b in np.arange(0.1,3,0.1):
                for c in np.arange(0.1,3,0.1):
                    k = c-a*b
                    #if abs(g2(l, d, M)-output_sel) <= 0.00001:
                    states = []
                    bool1 = (abs(k)<1 and 1+k*M>=0 and k+M>=0) or (abs(k)>1 and M<=-k and 1+k*M<=0)
                    if bool1:
                        states.append(1)
                    else:
                        states.append(0)
                    if M<=-k and k<0: 
                        states.append(1)
                    else:
                        states.append(0)
                    if M>=-1/k and k<0:
                        states.append(1)
                    else:
                        states.append(0)
                    states.append(0)
                    states.append(0)
                    if states in needed_state:
                        new_candidate = [states, a,b,c,g1(k, M), model]
                        #print(new_candidate)
                        candidates.append(new_candidate)
    
    if model == 2:
        for a in np.arange(0.1,3,0.1):
            for b in np.arange(0.1,3,0.1):
                for d in np.arange(0.1,3,0.1):
                    l = a*b+d
                    #if abs(g2(l, d, M)-output_sel) <= 0.00001:
                    states = []
                    bool1 = abs(l)>1 and l > 1/M and l > M and l > (M+d)/(1+M*d) and l > (1+M*d)/(d+M)
                    bool2 = abs(l)<1 and l < 1/M and l < M and l < (M+d)/(1+M*d) and l < (1+M*d)/(d+M)
                    if bool1 or bool2: # LS
                        states.append(1)
                    else:
                        states.append(0)
                    if M+d<=l: # left HS
                        states.append(1)
                    else:
                        states.append(0)
                    if 1<=M*(l-d): # right HS
                        states.append(1)
                    else:
                        states.append(0)
                    if M+d>l and M<l+2*d: # left SHS
                        states.append(1)
                    else:
                        states.append(0)
                    if 1>M*(l-d) and 1<M*l: # right SHS
                        states.append(1)
                    else:
                        states.append(0)
                    if states in needed_state:
                        new_candidate = [states, a,b,d,g2(l, d, M), model]
                        #print(new_candidate)
                        candidates.append(new_candidate)
                        
    if model == 3:
        for a in np.arange(0.1,3,0.1):
            for b in np.arange(0.1,3,0.1):
                for e in np.arange(0.1,3,0.1):
                    z = a*b 
                    k = b*e+1
                    #if abs(g3(l, d, M)-output_sel) <= 0.00001:
                    bool1 = k**2>z**2 and k>M*z and M*k>z
                    bool2 = k**2<z**2 and k<M*z and M*k<z  
                    states = []
                    if bool1 or bool2:
                        states.append(1)
                    else:
                        states.append(0)
                    if M*k<=z:
                        states.append(1)
                    else:
                        states.append(0)
                    if k<=M*z:
                        states.append(1)
                    else:
                        states.append(0)
                    states.append(0)
                    states.append(0)
                    if states in needed_state:
                        new_candidate = [states, a,b,e,g3(z, k, M), model]
                        #print(new_candidate)
                        candidates.append(new_candidate)
    return(candidates)

#%% Define the wanted states in the list needed_states
# needed states is a list of lists that represent the needed states
# where each entry is 0 (selectivity state deactive) or 1 (selec. state active)
# needed_states = [[<LS>,<HS Left>,<HS Right>,<SHS Left>,<SHS Right>],...]

needed_states = []
for s0 in [0,1]:
  for s1 in [1,0]:
      for s2 in [1,0]:
          for s3 in [0,1]:
              for s4 in [0,1]:
                  state = [s0,s1,s2,s3,s4]
                  if sum(state[1:3]) >= 1:
                      needed_states.append(state)

#%% Define input modulator M and get all possible connectivities 
M=1.1           

conn_model1 = comparable_connectivity(M, 1,needed_states)
conn_model2 = comparable_connectivity(M, 2,needed_states)
conn_model3 = comparable_connectivity(M, 3,needed_states)

#%% Generate the Output 
# Print out the connectivities with the smallest differenct in initial output
# selectivity across models

output_liste = []
conn_liste = []
minimum = 10000000
for o1 in conn_model1:
  for o2 in conn_model2:
    for o3 in conn_model3:
      candidate_min = np.sum(abs(np.diff(np.array([o1[4],o2[4],o3[4],o1[4]]))))
      if candidate_min <= minimum:
        minimum = candidate_min
        print("")
        print(np.sum(abs(np.diff(np.array([o1[4],o2[4],o3[4],o1[4]])))))
        print(o1)
        print(o2)
        print(o3)
        a_ = [o1[1],o2[1],o3[1]]
        b_ = [o1[2],o2[2],o3[2]]
        x_ = [o1[3],o2[3],o3[3]]
        a = [round(num, 3) for num in a_]
        b = [round(num, 3) for num in b_]
        x = [round(num, 3) for num in x_]
        print([a,b,x])
        
        





