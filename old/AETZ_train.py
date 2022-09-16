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

# import AET model
import AET_2_sparse as nn

# change working directory
WD = r'C:\Users\dueckerk\Desktop\Projects\AET NN\networks'
chdir(WD)

data_train,output_train = nn.mkstim(noise_=False)

label_train = ['A', 'E', 'T', 'Z']
         
dims = [(int(data_train.shape[1]/2),int(data_train.shape[2]/2)),64,4]

N_EPO = 50
sig_param = [1,0]

# check orthogonality after training
ortho_idx = torch.zeros(2,)
idx = np.array((0,5,9,-1))
label_idx = np.arange(4)
inp_combi = list(combinations(idx,2))           # possible combinations of two letters

plt.rcParams.update({'font.size': 22})

ETA_ = 0.15
PDROP_ = 1

train_param = [ETA_,N_EPO,PDROP_]

ortho_idx = np.zeros((2,))
#sparse_param = [[0,0],[0.001, 0.5],[0.001, 0.9]]

model_param = [[],[]]

sparse_param = [[0.05, 1e-5]]               # [beta, p]
for _im, sp in enumerate(sparse_param):
    model = nn.netw(dims,train_param,sig_param,sp,lossfun=nn.cross_entr)
    loss_hist = model.train_nn(data_train, output_train, mini_size = int(data_train.shape[0]/10),update_bias=True).detach()
    
    model_param[_im] = model.params
    fig, axs = plt.subplots()
    axs.plot(loss_hist)
    axs.set_ylabel('loss')
    axs.set_xlabel('epoch')
        
        
    # activations
    fig, axs = plt.subplots(2,2)
    axs = axs.ravel()
    for i,ax in enumerate(axs):
        hid_acti = model.forw_conv(data_train[idx[i]])[1]
        ax.bar(np.arange(dims[1]),hid_acti.detach().numpy())
        ax.set_title(label_train[np.argmax(output_train[idx[i]]).numpy()])
    fig.tight_layout(pad=1)
         
     # orthogonality
    dot_all = torch.zeros(len(inp_combi),)
    for _ii,_ic in enumerate(inp_combi):
        # dot product close to 0 -> activations are orthogonal
        dot_all[_ii] = torch.dot(model.forw_conv(data_train[_ic[0]])[1], model.forw_conv(data_train[_ic[1]])[1]).detach()
    
    ortho_idx[_im] = dot_all.sum()

####


# forward dynamics 

dyn_params = [0.007,0.07,3,0.025,0,0]   # [tau_h, tau_r, r_scale factor, T, h start, R start]
timevec = np.linspace(0,1,1000)
alpha_params = [0,0]

H_t,R_t,O_t = model.forw_dyn(data_train[0], dyn_params, timevec, alpha_params)

fig,ax = plt.subplots(3,1)
ax[2].plot(O_t.T.detach().numpy())
ax[2].legend(('A(t)','E(t)','T(t)','Z(t)'),loc='lower right')
ax[2].set_title('output')

ax[1].plot(R_t.T)
ax[1].set_title('relaxation')
ax[0].plot(H_t.T)
ax[0].set_title('hidden acti')
fig.tight_layout()



fig,ax = plt.subplots(2,2)
ax = ax.ravel()

for c,i in enumerate(idx):
    _,_,O_t = model.forw_dyn(data_train[i], dyn_params, timevec, alpha_params)
    ax[c].plot(O_t.T)
    ax[c].set_title(label_train[c])

fig.tight_layout()

# competing inputs
dyn_params = [0.005,0.05,3,0.05,0,0]   # [tau_h, tau_r, r_scale factor, T, h start, R start]

idx = np.array((0,5,9,-1))

label_idx = np.arange(4)
inp_combi = list(combinations(idx,2))           # possible combinations


fig,ax = plt.subplots(3,2)
ax = ax.ravel()
plt.setp(ax,yticks=np.arange(0,1.5,0.5))
for i,comp_inp in enumerate(inp_combi):
    input_ = data_train[comp_inp[0]] *1.1+ data_train[comp_inp[1]]*0.9
    _,_,O_t = model.forw_dyn(input_, dyn_params, timevec, alpha_params)
    
    ax[i].plot(O_t.T)
    ax[i].set_ylim((0,1))
    fi = [np.where(idx == comp_inp[0])[0][0], np.where(idx == comp_inp[1])[0][0]]
    ax[i].set_title(label_train[fi[0]] + ' + ' + label_train[fi[1]])
fig.legend((label_train))
fig.tight_layout()


## ALPHA

#dyn_params = [0.015,0.2,3,0.1,0,0]
dyn_params = [0.01,0.07,3,0.05,0,0]

timevec = np.linspace(0,1,1000)
alpha_params = [10,1]

H_t,R_t,O_t = model.forw_dyn(data_train[0], dyn_params, timevec, alpha_params)

fig,ax = plt.subplots(3,1)
ax[2].plot(O_t.T.detach().numpy())
ax[2].plot(0.5*np.sin(2*np.pi*alpha_params[0]*timevec)+0.5,color='black',linestyle='-.')
ax[2].set_title('output')

ax[1].plot(R_t.T)
ax[1].set_title('relaxation')
ax[0].plot(H_t.T)
ax[0].set_title('hidden acti')
fig.legend(('A(t)','E(t)','T(t)','Z(t)',r'$\alpha$(t)'),loc='lower right')

fig.tight_layout()


# all inputs
fig,ax = plt.subplots(2,2)
ax = ax.ravel()

for c,i in enumerate(idx):
    _,_,O_t = model.forw_dyn(data_train[i], dyn_params, timevec, alpha_params)
    ax[c].plot(O_t.T)
    ax[c].set_title(label_train[c])
    ax[c].plot(0.5*np.sin(2*np.pi*alpha_params[0]*timevec)+0.5,color='black',linestyle='-.')


fig.tight_layout()



# competition and alpha
alpha_params = [10,3]
dyn_params = [0.005,0.07,6,0.05,0,0.5]   # [tau_h, tau_r, r_scale factor, T, h start, R start]


fig,ax = plt.subplots(3,2)
ax = ax.ravel()
plt.setp(ax,yticks=np.arange(0,1.5,0.5))
for i,comp_inp in enumerate(inp_combi):
    input_ = data_train[comp_inp[0]]*1.1 + data_train[comp_inp[1]]*0.9
    _,_,O_t = model.forw_dyn(input_, dyn_params, timevec, alpha_params)
    
    ax[i].plot(O_t.T)
    ax[i].plot(0.5*np.sin(2*np.pi*alpha_params[0]*timevec)+0.5,color='black',linestyle='-.')

    ax[i].set_ylim((0,1))
    fi = [np.where(idx == comp_inp[0])[0][0], np.where(idx == comp_inp[1])[0][0]]
    ax[i].set_title(label_train[fi[0]] + ' att + ' + label_train[fi[1]] + ' unatt')
fig.legend((label_train))
fig.tight_layout()


fig,ax = plt.subplots(3,2)
ax = ax.ravel()
plt.setp(ax,yticks=np.arange(0,1.5,0.5))
for i,comp_inp in enumerate(inp_combi):
    input_ = data_train[comp_inp[0]]*0.9 + data_train[comp_inp[1]]*1.1
    _,_,O_t = model.forw_dyn(input_, dyn_params, timevec, alpha_params)
    
    ax[i].plot(O_t.T)
    ax[i].plot(0.5*np.sin(2*np.pi*alpha_params[0]*timevec)+0.5,color='black',linestyle='-.')

    ax[i].set_ylim((0,1))
    fi = [np.where(idx == comp_inp[0])[0][0], np.where(idx == comp_inp[1])[0][0]]
    ax[i].set_title(label_train[fi[0]] + ' unatt + ' + label_train[fi[1]] + ' att')
fig.legend((label_train))
fig.tight_layout()

