# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:24:20 2022;
last edited: Aug 15 2022

AETZ problem - model 1:
    - manual model using pytorch automatic differentiation
    - use with shifted sigmoid (slope = 2; bias = 2.5) OR learn bias (sparsity constraints recommended)
    

@author: Katharina Duecker, PhD candidate Neuronal Oscillations group, Centre for Human Brain Health, Birmingham, UK
"""

import torch
import numpy as np
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
import os

     
# Loss function classes

# cross entropy loss      
class cross_entr(object):        
    
    @staticmethod 
    # loss function
    def loss_fun(output_hat,output_):
    
        return -torch.sum(output_*torch.log(output_hat))  
    
    # activation function output layer
    @staticmethod
    def acti_fun(z):
        return softmax(z)

# binary cross entropy loss
class bin_cross_entr(object):
    
    # loss function
    @staticmethod
    def loss_fun(output_hat,output_):
    
        # binary ce: y * ln(y_hat) + (1-y)*ln(1-y_hat)
        bce = output_*torch.log(output_hat)+(1-output_)*torch.log(1-output_hat)
        
        return -torch.sum(bce)/output_.shape[0]
    
    # activation output layer
    @staticmethod
    def acti_fun(z):
        return sigmoid(z)
    

# sparsity regularization class

# regularization with KL divergence (L1 and L2 norm not implemented yet)
class KL_reg(object):
    
    # achieve sparsity using KL divergence
    # see https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf for KL divergence
    
    # average hidden activations over all inputs to estimate "pj"
    
    @staticmethod    
    def penal_loss(_data,_dims,forw_fun_,sparse_param):
    
        # 1. Calculate p_hat
        # store hidden nodes over all examples
        _phat = torch.zeros(_dims[1],)
        # loop over all examples and calculate p_j for KL diveregence
        for _ii,dtr in enumerate(_data):
            
            # current forward sweep
            _,H,_ = forw_fun_(dtr)
            
            # add hidden activations
            _phat += H
        
        # divide by number of examples -> mean
        _phat /= _data.shape[0]
    
        # 2. calculate penalty term added to loss
         
        beta, p_sparse = sparse_param   # extract sparsity parameters

        p = torch.zeros(_phat.shape[0],)+p_sparse
         
        KL_div = p*torch.log(p/_phat) + (1-p)*torch.log((1-p)/(1-_phat)) # calculate distance between _phat and "goal" p (small number)
        
        return _phat, beta*torch.sum(KL_div)
        
    # penalty term added to bias 
    @staticmethod
    def penal_bias(_phat,sparse_param):
        
        beta, p_sparse = sparse_param
        
        p = torch.zeros(_phat.shape[0],)+p_sparse
        
        pnl = beta*(-p/_phat + ((1-p)/(1-_phat)))
        
        return pnl

# activation functions ######

def sigmoid(z,sig_params = [1,0]):      
    slope_, bias_ = sig_params
    return 1.0/(1.0+torch.exp(-slope_*(z-bias_)))        
        
def softmax(z):
    return torch.exp(z-torch.max(z))/sum(torch.exp(z-torch.max(z)))

# create A,E,T,Z stimuli
def mkstim(noise_=False):
    # create stimuli
    A = np.zeros((28,28))
    A[8:23,6:9] = 1
    A[5:8,8:19] = 1
    A[16:19,8:19] = 1
    A[8:23,19:22] = 1
    
    
    E = np.zeros((28,28))
    E[4:23,6:9] = 1
    E[4:7,8:21] = 1
    E[12:15,8:21] = 1
    E[20:23,8:21] = 1
    
    T = np.zeros((28,28))
    T[6:10,6:22] = 1
    T[8:23,12:16] = 1
    
    Z = np.zeros((28,28))
    Z[6:8, 6:22] = 1
    Z[21:23, 6:22] = 1
    
    ru = 8
    cu = np.arange(18,22)
    rl = 20
    
    for i in np.arange(13):
        Z[ru,cu] = 1
        ru +=1
        cu -=1
        rl -=1
    
    
    # 2. Place letters in larger image
    BIGA = np.zeros((4,56,56))
    BIGE = np.zeros((4,56,56))
    BIGT = np.zeros((4,56,56))
    BIGZ = np.zeros((4,56,56))
    #BIGEMPTY = np.zeros((4,56,56))
    
    s =  np.arange(0,56).reshape((2,-1))     # split in half

    q = 0                # quadrant counter
    
    # loop over height
    for h in range(int(BIGA.shape[0]/2)):
        # looper over width
        for w in range((int(BIGA.shape[0]/2))):

            BIGA[q,s[h,0]:s[h,-1]+1,s[w,0]:s[w,-1]+1] += A
            BIGE[q,s[h,0]:s[h,-1]+1,s[w,0]:s[w,-1]+1] += E
            BIGT[q,s[h,0]:s[h,-1]+1,s[w,0]:s[w,-1]+1] += T
            BIGZ[q,s[h,0]:s[h,-1]+1,s[w,0]:s[w,-1]+1] += Z
            q += 1
            
    I = torch.from_numpy(np.concatenate((BIGA,BIGE,BIGT,BIGZ)))
    
    O = torch.cat((torch.tile(torch.tensor((1,0,0,0)),(4,1)),torch.tile(torch.tensor((0,1,0,0)),(4,1)),torch.tile(torch.tensor((0,0,1,0)),(4,1)),torch.tile(torch.tensor((0,0,0,1)),(4,1))))

    

    # 3. Optional: adverserial attack
    
    if noise_:
        
        stim = torch.from_numpy(np.concatenate((BIGA,BIGE,BIGT,BIGZ)))
        label = torch.cat((torch.tile(torch.tensor((1,0,0,0)),(4,1)),torch.tile(torch.tensor((0,1,0,0)),(4,1)),torch.tile(torch.tensor((0,0,1,0)),(4,1)),torch.tile(torch.tensor((0,0,0,1)),(4,1))))
        
        num_it = 10
        
        I = stim
        O = label
        
        for i in range(num_it):
            
            # values should be between 0 and 1
            I_noise = torch.abs_(stim - torch.normal(0.4,0.1,stim.shape)*0.1)
            
            I = torch.cat((I,I_noise),dim=0)
            
            O = torch.cat((O,label),dim=0)

    return I, O


# NN class
class netw(object):
    
    # initialize network
    def __init__(self,data,dims,train_param,sig_param,sparse_param,reg_method=KL_reg,lossfun=bin_cross_entr):
        
        # network size & dimensions
        self.n_lay = len(dims)
        self.dims = dims
        
        # training data
        self.data = data
        
        
        # hidden layer activation + parameters
        self.actifun = sigmoid
        self.sig_param = sig_param
        
        # loss function
        self.costfun = lossfun.loss_fun
        # output function for current loss function
        self.outfun = lossfun.acti_fun
        
        # training parameters
        self.train_param = train_param
        
        # regularization
        self.reg_method = reg_method                # method
        self.sparse_param = sparse_param            # parameters
        
        # initialize weights & biases
        _m, _sigm = 0, 0.01 

        sz_qu = dims[0][0]
        
        # convolutional weight matrix
        w_conv = torch.normal(_m,_sigm,(dims[1],sz_qu, sz_qu)).requires_grad_(True)

        # fully connected
        w_fc = torch.normal(_m,_sigm,(dims[2],dims[1])).requires_grad_(True)

        # all weights
        w_ = [w_conv, w_fc]                                          
    
        # biases
        #_m, _sigm = 1, 0.01 
        
        #b_ = [torch.normal(_m,_sigm,(y,)).requires_grad_(True) for y in dims[1:]] 
        b_ = [torch.zeros((y,)).requires_grad_(True) for y in dims[1:]] 
        
        # store parameters
        self.params = [w_,b_]
        

    # forward sweep with one convolutional layer (-> weight sharing between quadrants)
    def forw_conv(self,input_):
        
        # weights, biases
        w,b = self.params
        
        # 1. convolutional layer
        # extract to be convoluted images with stride of window size
        quads_ = torch.Tensor(view_as_windows(input_.numpy(), w[0][0].shape,step= w[0][0].shape[0]))

        # pre-activations
        Z = torch.zeros((self.dims[1],))
        # activations
        H = torch.zeros((self.dims[1],))

        for i,w_h in enumerate(w[0]):
        
            
            Z[i] = torch.sum(torch.tensordot(quads_,w_h,dims=([2,3],[0,1]))) + b[0][i]
            #Z[i] = torch.max(torch.tensordot(quads_,w_h,dims=([2,3],[0,1]))) + b[0][i]
            
            H[i] = self.actifun(Z[i],self.sig_param)    
    
            
        # 2. output layer
        O = self.outfun(torch.matmul(w[1],H) + b[-1])
        
        return Z, H, O
    
    # backpropagation: get gradients of w and b
    def back_prop(self,input_,output_):
        
        _,_,O = self.forw_conv(input_)
        
        # loss + sparsity penalty
        if self.sparse_param[0]:
            _phat,KL_sp = self.reg_method.penal_loss(self.data, self.dims, self.forw_conv, self.sparse_param)
            _loss = self.costfun(O,output_) + KL_sp
        # loss without sparsity
        else:
            _loss = self.costfun(O,output_)
            
        _loss.backward()
        
        w,b = self.params
        
        # calculate gradients
        nabla_b = [b_.grad for b_ in b]
        nabla_w = [w_.grad for w_ in w]
        
        # if sparsity, update gradients
        if self.sparse_param[0]:
            nabla_b[0] += self.reg_method.penal_bias(_phat,self.sparse_param)


        return _loss, nabla_b, nabla_w
    
    # gradient descent: update weights and biases
    def grad_desc(self,eta_,mini_b_,ub=True):
        
        w,b = self.params
        
        loss_ = 0
        for input_,output_ in zip(mini_b_[0],mini_b_[1]):
            
            # accumulate gradients for each mini batch
            l, nabla_b_mini, nabla_w_mini = self.back_prop(input_, output_)
            loss_ += l
            
        loss_ /= len(mini_b_[0])
        
       
        with torch.no_grad():
          
         # update bias (if biases are learned)
            if ub:
                
                for i,b_ in enumerate(b):
                    
                    b_ -= (eta_ /len(mini_b_[0]))*nabla_b_mini[i]
                    b_.grad.detach_()
                    b_.grad.zero_()
                        

            # update weights
            for i,w_ in enumerate(w):
                w_ -= (eta_ /len(mini_b_[0]))*nabla_w_mini[i]                

                
                w_.grad.detach_()
                w_.grad.zero_()


            return loss_
    
    # training function
    def train_nn(self,data_train,output_train,mini_size=32,update_bias=True):
        
        eta_,n_epochs = self.train_param
        
        # store loss
        loss_hist = torch.zeros((n_epochs))
        
        
        # loop over epochs
        for i in range(n_epochs):
            
            
            # shuffle data
            idx = torch.randperm(data_train.shape[0])
            
            num_batch = int(data_train.shape[0]/mini_size)
            size_batch = int(data_train.shape[0]/num_batch)

            
            # start mini-batch counter
            x = 0
            
            for n in range(num_batch):
                
                loss = 0
                mini_b = []
                mini_b = [data_train[idx[x:x+size_batch]],output_train[idx[x:x+size_batch]]]
                x += size_batch
            
                loss += self.grad_desc(eta_,mini_b, ub = update_bias)
            
                loss_hist[i] += loss

                
        return loss_hist
    
    # forward dynamics/discretization, euler integration
    def forw_dyn(self,input_,params_,t_,alpha_params):
        
        # inputs:
            # input_: input image
            # params_: hyperparameters
            # t_: time vector for integration
            # alpha_params: alpha frequency & amplitude
            
        # discretization & dynamics parameters
        tau_h,tau_R,R,T,h_start,R_start = params_
        
        # weights and biases
        _w,_b = self.params
        
        # alpha frequency & amplitude
        _af,_aa = alpha_params
        
        with torch.no_grad():
            # initialize empty matrices
            dt = np.diff(t_)[0]
            dhdt = torch.ones((_b[0].shape[0],len(t_)+1))*h_start
            dRdt = torch.ones((_b[0].shape[0],len(t_)+1))*R_start
            dOdt = torch.zeros((_b[1].shape[0],len(t_)+1))
            ddotdt = torch.zeros((_b[0].shape[0],len(t_)+1))
            
            # alpha inhibition
            alpha_inh = _aa*np.sin(2*np.pi*_af*t_)+_aa
            
            # preactivation (dot product of input and first weight matrix)
            Z,_,_ = self.forw_conv(input_)
            
            # adjust initial adaptation term (threshold)
            dRdt *= torch.max(Z)                           
            
            # scale for adaptation
            r_scale = R*torch.max(Z).detach()
            
            
            for _it,t in enumerate(t_):
                
                # pre-activation
                ddotdt[:,_it+1] = (Z + dhdt[:,_it] - dRdt[:,_it] - alpha_inh[_it])/T
                
                # dynamics hidden layer
                dhdt[:,_it+1] = dhdt[:,_it] + dt/tau_h * (-dhdt[:,_it] + self.actifun(ddotdt[:,_it+1],self.sig_param))      
                
                # adaptation term
                dRdt[:,_it+1] = dRdt[:,_it] + dt/tau_R * (-dRdt[:,_it] + r_scale*dhdt[:,_it+1])
                
                # output layer
                dOdt[:,_it+1] = self.outfun(torch.matmul(_w[-1],dhdt[:,_it+1]) + _b[-1])
                
        return dhdt, dRdt, dOdt, ddotdt
    
    