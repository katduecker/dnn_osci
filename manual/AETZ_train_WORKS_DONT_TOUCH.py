# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:05:42 2022

@author: DueckerK
"""
from os import chdir
from itertools import combinations

import torch
import numpy as np
import matplotlib.pyplot as plt

from statistics import mode
# change working directory
WD = r'C:\Users\dueckerk\Desktop\Projects\AET NN\networks'
chdir(WD)
# import AET model
import AETZ_model1 as nn


data_train,output_train = nn.mkstim(noise_=False) 

label_train = ['A', 'E', 'T', 'Z']

N_HIDNODE = 16
dims = [(int(data_train.shape[1]/2),int(data_train.shape[2]/2)),N_HIDNODE,4]

N_EPO = 50
sig_param = [2,2.5]

# check orthogonality after training
ortho_idx = torch.zeros(2,)
idx = np.array((0,5,9,-1))
label_idx = np.arange(5)
inp_combi = list(combinations(idx,2))           # possible combinations of two letters

plt.rcParams.update({'font.size': 22})

ETA_ = 0.15

train_param = [ETA_,N_EPO]

ortho_idx = np.zeros((2,))
#sparse_param = [[0,0],[0.001, 0.5],[0.001, 0.9]]

model_param = [[],[]]

sparse_param = [[0.00, 0.01]]               # [beta, p]
for _im, sp in enumerate(sparse_param):
    model = nn.netw(data_train,dims,train_param,sig_param,sp,lossfun=nn.cross_entr)
    loss_hist = model.train_nn(data_train, output_train, mini_size = int(data_train.shape[0]/10),update_bias=False).detach()
    
    model_param[_im] = model.params
    fig, axs = plt.subplots()
    axs.plot(loss_hist)
    axs.set_ylabel('loss')
    axs.set_xlabel('epoch')
        
        
    # activations
    fig, axs = plt.subplots(2,2)
    axs = axs.ravel()
    hid_acti = np.zeros((dims[1],len(axs)))
    for i,ax in enumerate(axs):
        hid_acti[:,i] = model.forw_conv(data_train[idx[i]])[1].detach().numpy()
        ax.bar(np.arange(dims[1]),hid_acti[:,i])
        ax.set_title(label_train[np.argmax(output_train[idx[i]]).numpy()])
    fig.tight_layout(pad=1)
         
     # orthogonality
    dot_all = torch.zeros(len(inp_combi),)
    for _ii,_ic in enumerate(inp_combi):
        # dot product close to 0 -> activations are orthogonal
        dot_all[_ii] = torch.dot(model.forw_conv(data_train[_ic[0]])[1], model.forw_conv(data_train[_ic[1]])[1]).detach()
    
    ortho_idx[_im] = dot_all.sum()

####

fig, axs = plt.subplots(2,2)
axs = axs.ravel()
for i,ax in enumerate(axs):
    ax.hist(hid_acti[:,i])
    
# forward dynamics 

dyn_params = [0.01,0.1,2,0.05,0,0]   # [tau_h, tau_r, r_scale factor, T, h start, R start]
timevec = np.linspace(0,1,1000)
alpha_params = [0,0]

H_t, R_t, O_t, Z_t = model.forw_dyn(data_train[0], dyn_params, timevec, alpha_params)

fig,ax = plt.subplots(3,1)
ax[2].plot(O_t.T.detach().numpy())
ax[2].legend(('A(t)','E(t)','T(t)','Z(t)'),loc='lower right')
ax[2].set_title('output')

ax[1].plot(R_t.T)
ax[1].set_title('relaxation')
ax[0].plot(H_t.T)
ax[0].set_title('hidden acti')
fig.tight_layout()



# does H_t go to correct values?

# these settings work for all without alpha
alpha_params = [10,2]
dyn_params = [0.01,0.1,2,0.05,0,0]   # [tau_h, tau_r, r_scale factor, T, h start, R start]


timevec = np.linspace(0,0.25,250)

max_H = torch.zeros((4,dims[1]))
all_H = torch.zeros((4,dims[1]))
all_Z = torch.zeros((4,dims[1]))
all_Ht = torch.zeros((4,dims[1],len(timevec)+1))

for c,i in enumerate(idx):
    # difference between actual value and dynamic end-point
    H_t,_,O_t,Z_t = model.forw_dyn(data_train[i], dyn_params, timevec, alpha_params)
    Z,H,_ = model.forw_conv(data_train[i])
    
    # find peak 
    max_H_t = torch.max(H_t,1)[1]
    
    # H_t at peak
    max_H[c] = H_t[:,mode(np.sort(max_H_t))]
    all_H[c] = H
    all_Z[c] = Z
    all_Ht[c] = H_t
    #ax[c].hist(diff_H[c].detach().numpy())
    

for l in range(4):
    fig, ax = plt.subplots(3,1)
    
    rH = np.round(all_H[l].detach().numpy(),1)
    ax[0].plot(timevec,all_Ht[l][rH==1,1:].T);
    ax[0].plot(timevec,alpha_params[1]*np.sin(2*np.pi*timevec*alpha_params[0]) + alpha_params[1], 'k' )
    ax[0].set_title('hidden nodes with acti 1')
    ax[0].set_ylim((0, 1))
    
    idxR = np.array(rH!=1) & np.array(rH>0)
    ax[1].plot(timevec,all_Ht[l][np.array(rH!=1) & np.array(rH>0),1:].T);
    ax[1].plot(timevec,np.tile(rH[idxR].reshape(-1,1),(len(timevec),)).T)
    ax[1].set_title('hidden nodes with 0 < acti < 1')
    ax[1].set_ylim((0, 1))
    
    ax[2].plot(timevec,all_Ht[l][rH==0,1:].T);
    ax[2].set_title('hidden nodes with acti=0')
    ax[2].set_ylim((0, 1))
    
    fig.tight_layout()

## dynamics all

fig,ax = plt.subplots(2,2)
ax = ax.ravel()

for c,i in enumerate(idx):
    _,_,O_t,_ = model.forw_dyn(data_train[i], dyn_params, timevec, alpha_params)
    ax[c].plot(O_t.T)
    ax[c].set_title(label_train[c])

fig.tight_layout()



# competing inputs 
alpha_params = [10,1]
dyn_params = [0.01,0.1,4,0.01,0,1]   # [tau_h, tau_r, r_scale factor, T, h start, R start]



idx = np.array((0,5,9,13))

label_idx = np.arange(4)
inp_combi = list(combinations(idx,2))           # possible combinations


fig,ax = plt.subplots(3,2)
ax = ax.ravel()
plt.setp(ax,yticks=np.arange(0,1.5,0.5))
for i,comp_inp in enumerate(inp_combi):
    input_ = data_train[comp_inp[0]] *1.1+ data_train[comp_inp[1]]*0.9
    _,_,O_t,_ = model.forw_dyn(input_, dyn_params, timevec, alpha_params)
    
    ax[i].plot(O_t.T)
    ax[i].set_ylim((0,1))
    fi = [np.where(idx == comp_inp[0])[0][0], np.where(idx == comp_inp[1])[0][0]]
    ax[i].set_title(label_train[fi[0]] + ' + ' + label_train[fi[1]])
fig.legend((label_train))
fig.tight_layout()



fig,ax = plt.subplots(3,2)
ax = ax.ravel()
plt.setp(ax,yticks=np.arange(0,1.5,0.5))
for i,comp_inp in enumerate(inp_combi):
    input_ = data_train[comp_inp[0]] *0.9+ data_train[comp_inp[1]]*1.1
    _,_,O_t,_ = model.forw_dyn(input_, dyn_params, timevec, alpha_params)
    
    ax[i].plot(O_t.T)
    ax[i].set_ylim((0,1))
    fi = [np.where(idx == comp_inp[0])[0][0], np.where(idx == comp_inp[1])[0][0]]
    ax[i].set_title(label_train[fi[1]] + ' + ' + label_train[fi[0]])
fig.legend((label_train))
fig.tight_layout()




# What is the reason for th network only recognizing one stimulus?
# number of activations that don't reach 1?



# 
idx = np.arange(0,16,5)
inp_combi = list(combinations(idx,2))           # possible combinations

ed_ = torch.zeros((len(inp_combi)))
angle_ = torch.zeros((len(inp_combi)))

for i,c in enumerate(inp_combi):
    
    I1 = data_train[c[0]]
    I2 = data_train[c[1]]
    I3 = data_train[c[0]]+data_train[c[1]]
    
    H1 = model.forw_conv(I1)[1]
    H2 = model.forw_conv(I2)[1]
    H3 = model.forw_conv(I3)[1]


    H1_2 = H1 + H2

    ed_[i] = torch.mean((H3-H1_2)**2)
    
    num_ = torch.matmul(H3,H1_2)
    denom_ = torch.linalg.vector_norm(H3)*torch.linalg.vector_norm(H1_2)
    
    angle_[i] = (torch.acos(num_/denom_).cpu().detach()*180)/torch.pi



# euclidean distance between linear comb and combined inputs = very small!
# angle ~12 - 17 degr
