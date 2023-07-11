# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:35:28 2023

@author: DueckerK
"""

import torch
from torch import nn
import numpy as np


import os

os.chdir(r'Z:\AET NN\MNIST Pytorch')
import aet_net
import aet_dyn
from itertools import combinations

import matplotlib.pyplot as plt


# import learned parameters
import pickle
with open('optuna_aet_trial.pkl','rb') as fp:
    optuna_aet_result = pickle.load(fp)
    print('optuna result')
    print(optuna_aet_result)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
# model parameters
nn_dim_ = [28,68,3]   # [quadrant size, number of hidden nodes, number of output nodes]
eta_ = optuna_aet_result[1]['ETA_']           # learning rate
mini_sz_ = 1          # mini batch size (1 = use SGD)
num_epo_ = 80

beta_ = optuna_aet_result[1]['BETA_']
p_ = optuna_aet_result[1]['P_']
kl_reg_ = [beta_,p_]#[0,0.001] # sparsity constraint parameters (not used for manual model)
sig_param = [optuna_aet_result[1]['slope_sig'], 0] # sigmoid slope and shift in x direction

# loss function & final layer activation (for binary crossentropy use sigmoid)
lossfun = [nn.MSELoss(), nn.Softmax(dim=-1)]

params = nn_dim_,eta_,mini_sz_,num_epo_,kl_reg_,sig_param

# initialize model and weights
model = aet_net.net(params,lossfun)
model = aet_net.init_params(model,weight_init='uni')
optimizer = torch.optim.SGD(model.parameters(),lr=eta_)

model.to(DEVICE)
loss_hist = model.train(optimizer,noise=False,print_loss=False)


plt.rcParams["figure.figsize"] = (10,5)
plt.plot(np.arange(model.num_ep),loss_hist.cpu().detach().numpy())
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# plot hidden activations
x_train, y_train = aet_net.aet_stim.mkstim()
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)

# subset of inputs
idx = np.array((0,5,10))#,-1))
data_sub = x_train[idx]

label = ['A','E','T']

plt.rcParams["figure.figsize"] = (15,5)

fig, axs = plt.subplots(1,3)
axs = axs.ravel()

for i,ax in enumerate(axs):
    Z,H,O = model.forw_conv(data_sub[i])
    
    ax.bar(np.arange(model.dims[1]),H.cpu().detach().numpy())
    ax.set_title(label[i])
    
# dynamics single input
alpha_params = [10,.5]
dyn_params = [0.01,0.05,3,0.1,0,0]   # [tau_h, tau_r, r_scale factor, T, h start, R start]
_io = 0

timevec = np.linspace(0,0.6,600)

Z_t,H_t, R_t, O_t  = aet_dyn.euler_dyn(model,x_train[0], dyn_params, timevec, alpha_params,DEVICE,inp_on=_io)

fig,ax = plt.subplots(3,1)
ax[2].plot(O_t.T.cpu().detach().numpy())
ax[2].legend(('A(t)','E(t)','T(t)','Z(t)'),loc='lower right')
ax[2].set_title('output')

ax[1].plot(R_t.T.cpu().detach().numpy())
ax[1].set_title('relaxation')
ax[0].plot(H_t.T.cpu().detach().numpy())
ax[0].set_title('hidden acti')

if alpha_params[1]:
    alpha_inh = 0.5*np.sin(2*np.pi*timevec*10)+0.5
    ax[2].plot(alpha_inh,'k',linewidth=0.5,linestyle='-.')
    ax[0].plot(alpha_inh,'k',linewidth=1,linestyle='-.')
fig.tight_layout()

# dynamics competing inputs
inp_combi = list(combinations(idx,2))           # possible combinations

#for aa in np.arange(0.1,1,0.1):
alpha_params = [10,1]
dyn_params = [0.01,0.03,6.5,0.01,0,0.5]    # [tau_h, tau_r, r_scale factor, T, h start, R start]
_io = 0
timevec = np.linspace(0,0.6,600)
fig,ax = plt.subplots(2,3,sharex=True,sharey=True)
#ax = ax.ravel()
plt.setp(ax,yticks=np.arange(0,1.5,0.5))

ax[0][0].set_ylabel('activation')
ax[1][0].set_ylabel('activation')

alpha_inh = 0.5*np.sin(2*np.pi*timevec*alpha_params[0])+0.5
for i,comp_inp in enumerate(inp_combi):
    input_ = x_train[comp_inp[0]] *1.1+ x_train[comp_inp[1]]*0.9
    _,_,_,O_t = aet_dyn.euler_dyn(model,input_, dyn_params, timevec, alpha_params,DEVICE,inp_on=_io)
    
    ax[0][i].plot(timevec,O_t[:,1:].cpu().detach().numpy().T)
    
    if alpha_params[1]:
        ax[0][i].plot(timevec,alpha_inh,'k',linewidth=0.5,linestyle='-.')
    ax[0][i].set_ylim((0,1))
    fi = [np.where(idx == comp_inp[0])[0][0], np.where(idx == comp_inp[1])[0][0]]
    ax[0][i].set_title(label[fi[0]] + ' + ' + label[fi[1]])

for i,comp_inp in enumerate(inp_combi):
    input_ = x_train[comp_inp[0]] *0.9+ x_train[comp_inp[1]]*1.1
    _,_,_,O_t = aet_dyn.euler_dyn(model,input_, dyn_params, timevec, alpha_params,DEVICE,inp_on=_io)
    
    ax[1][i].plot(timevec,O_t[:,1:].cpu().detach().numpy().T)
    if alpha_params[1]:
        ax[1][i].plot(timevec,alpha_inh,'k',linewidth=0.5,linestyle='-.')
    ax[1][i].set_ylim((0,1))
    fi = [np.where(idx == comp_inp[1])[0][0], np.where(idx == comp_inp[0])[0][0]]
    ax[1][i].set_title(label[fi[0]] + ' + ' + label[fi[1]])
    ax[1][i].set_xlabel('time (s)')

#fig.suptitle('aa: '+ str(aa), fontsize=16)

fig.legend((label))
fig.tight_layout()
fig.savefig('temp_code_learned_bias_optuna.png')
