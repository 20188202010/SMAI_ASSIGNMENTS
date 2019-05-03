#!/usr/bin/env python
# coding: utf-8

# # Question 2
# Given a DNA sequence and a state path sequence, find the probability that the given DNA sequence is generated from the given path sequence only <br>
# Input DNA sequence:  “CTTCATGTGAAAGCAGACGTAAGTCA” <br>
# Input State path sequence:  “EEEEEEEEEEEEEEEEEE5IIIIIII$" <br>
# Output:  State path probability for above state path [Log probability required i.e.log(p)]

# <img src="../input_data/ques.png">

# In[1]:


states          = ['start','E', '5', 'I','$']
observations    = ['A', 'C', 'G', 'T']
observation_seq = "CTTCATGTGAAAGCAGACGTAAGTCA"
len_obs         = len(observation_seq)
state_seq       = "EEEEEEEEEEEEEEEEEE5IIIIIII$"
len_states      = len(state_seq)
transition_prob = {'start': {'E': 1.0},
                   'E'    : {'E': 0.9, '5': 0.1},
                   '5'    : {'I': 1.0},
                   'I'    : {'I': 0.9, '$': 0.1}
                  }
emission_prob    = {'E': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
                    '5': {'A': 0.05, 'G': 0.95},
                    'I': {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4}
                   }


# In[2]:


# print type(emission_prob), type(emission_prob['E']), type(emission_prob['E']['A'])
# print len_obs, len_states


# In[3]:


total_pr = 1
for obs_idx in range(0, len_obs):
    obs_symbol = observation_seq[obs_idx]
    curr_state = state_seq[obs_idx]
    next_state = state_seq[obs_idx+1]
    trans_pr = transition_prob[curr_state][next_state]
    emiss_pr = emission_prob[curr_state][obs_symbol]
    
    combined_pr = trans_pr * emiss_pr
    total_pr *= combined_pr
    


# In[4]:


from numpy import log


# In[5]:


print "Required Prob: p: ",total_pr, "log(p): ", log(total_pr) 

