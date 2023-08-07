# %% [code]
# %% [code]
# %% [code]
# %% [code]
# -*- coding: utf-8 -*-

import torch
from torch import nn
import aet_stim
#import mnist_stim

# sigmoid activation with option to stretch and shift
def sigmoid(z,sig_param):

    _slope,_bias = sig_param

    return 1.0/(1.0+torch.exp(-_slope*(z+_bias)))

def CE_loss(output_hat,output_):

    return -torch.sum(output_*torch.log(output_hat)) 

# weight & bias initialization
def init_params(model,weight_init='normal'):
    
    for m in model.modules():
        # if module is a conv or linear layer, set weights
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if weight_init == 'uni':
                nn.init.xavier_uniform_(m.weight)#,mean=0,std=0.2)
            elif weight_init == 'normal':
                nn.init.normal_(m.weight,mean=0,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model
        
    
class net(nn.Module):
    
    def __init__(self, params, lfun):

            super(net,self).__init__()

            dims,lr,mini_sz,num_ep,reg,sig_param = params

            ## NETWORK ARCHITECTURE

            # convolutional & fully connected layer
            if sig_param[1]: # when using set bias, don't learn
                self.conv1 = nn.Conv2d(1,dims[1], dims[0], stride=dims[0],bias=False)
                self.fc1 = nn.Linear(dims[1], dims[-1],bias=False)
            else:
                self.conv1 = nn.Conv2d(1,dims[1], dims[0], stride=dims[0],bias=True)
                self.fc1 = nn.Linear(dims[1], dims[-1],bias=True)

            self.acti1 = sigmoid
            self.pool1 = torch.sum
            self.sig_param = sig_param
            self.dims = dims

            # Fully connected layer
            self.actiout = lfun[1]
            
            ## TRAINING PARAMETERS
            # loss function
            self.lossfun = lfun[0]

            # regularizer
            self.reg = reg

            # mini batch size & number of epochs
            self.mini_sz = mini_sz
            self.num_ep = num_ep
            
        
        # forward sweep 
    def forw_conv(self,data):
        
        ## convolutional layer
        
        if len(data.shape) ==3:
            data = data.reshape(1,1,56,56)
            
        y = self.conv1(data)
        
        # pool over quadrants
        Z = self.pool1(y,dim=(-2,-1)).squeeze()
        
        # activation
        H = sigmoid(Z,self.sig_param).squeeze()
               
        # fully connect layer and activation
        O = self.actiout(self.fc1(H))

        
        return Z,H,O
    
    


    # sparsity regularizer https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    def bias_regularizer(self,data):
        
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _beta, _rho = self.reg
        rho_hat = torch.zeros((self.dims[1],)).to(DEVICE)
        for inp in data:
            
            Z,H,_ = self.forw_conv(inp)
            
            # calculate hidden activations
            rho_hat += H
        
        rho_hat /= data.shape[0]
        
        # regularizer: KL divergence between hidden activations & rho (added to loss)
        regu_loss = _beta*torch.sum(_rho*torch.log(_rho/rho_hat) + (1-_rho)*torch.log((1-_rho)/(1-rho_hat)))
        
        # penalty bias gradient
        regu_bias = _beta * (-_rho/rho_hat + ((1-_rho)/(1-rho_hat)))
        
        return regu_loss, regu_bias
    
    
    def train(self,optimizer,dataset='aet',noise=False,print_loss=True):
        
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dataset == 'mnist':
            data,output = mnist_stim.make_stim()
        elif dataset == 'aet':
            data,output = aet_stim.mkstim(noise)
        else:
            data,output = dataset
            
        data = data.to(DEVICE)
        output = output.to(DEVICE)

        loss = torch.zeros((self.num_ep),).to(DEVICE)

        for e in range(self.num_ep):

            if dataset == 'mnist':
                mini_idx = mnist_stim.make_minib(data.shape[0],mini_sz=self.mini_sz)
            elif dataset == 'aet':
                mini_idx = aet_stim.make_minib(data.shape[0],mini_sz=self.mini_sz)
            else:
                mini_idx = aet_stim.make_minib(data.shape[0],mini_sz=self.mini_sz)

            for mb in range(len(mini_idx)):

                # forward
                _,_,y = self.forw_conv(data[mini_idx[mb]])

                # regularizer (if sparsity params are defined)
                if self.reg[0]:
                    _regu = self.bias_regularizer(data)
                else:
                    _regu = torch.zeros(2)

                # loss + sparsity penalty
                _loss = self.lossfun(output[mini_idx[mb]],y) + _regu[0]
                
                optimizer.zero_grad()
                # accumulate gradients for minibatch
                _loss.backward()



                # add sparsity penalty to bias
                if self.reg[0]:
                    bias = self.get_parameter('conv1.bias')
                    bias.grad += _regu[1]

                 # update after mini batch
                optimizer.step()

            loss[e] = _loss#.mean()

            if print_loss:
                print(f'epoch: {e}, loss: {loss[e]}')

        del data, output
        
        return loss

        

    