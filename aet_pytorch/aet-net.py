# %% [code]
# %% [code]
# %% [code]
# -*- coding: utf-8 -*-

import torch
from torch import nn
import aet_stim

# sigmoid activation with option to stretch and shift
def sigmoid(z,sig_param):

    _slope,_bias = sig_param

    return 1.0/(1.0+torch.exp(-_slope*(z+_bias)))

def CE_loss(output_hat,output_):

    return -torch.sum(output_*torch.log(output_hat)) 
    
class net(nn.Module):
    
    def __init__(self, params, lfun):

            super(net, self).__init__()

            dims,lr,mini_sz,num_ep,reg,sig_param = params

            ## NETWORK ARCHITECTURE

            # convolutional layer, taking one input, number of output channels is size of hidden layer
            self.conv1 = nn.Conv2d(1, dims[1], dims[0], stride=dims[0])
            self.acti1 = sigmoid
            self.pool1 = torch.sum
            self.sig_param = sig_param
            self.dims = dims

            # Fully connected layer
            self.fc1 = nn.Linear(dims[1], dims[-1])
            self.actiout = lfun[1]
            
            ## TRAINING PARAMETERS
            # loss function
            self.lossfun = lfun[0]

            # regularizer
            self.reg = reg

            # mini batch size & number of epochs
            self.mini_sz = mini_sz
            self.num_ep = num_ep
            
            # if bias is set, don't learn biases
            if sig_param[1]:
                self.conv1.bias.data.zero_()
                self.conv1.bias.requires_grad = False

            
        
        # forward sweep 
    def forw_conv(self,data):
        
        ## convolutional layer
        y = self.conv1(data)
        
        # pool over quadrants
        Z = self.pool1(y,dim=(1,2))
        
        # activation
        H = sigmoid(Z,self.sig_param)
               
        # fully connect layer and activation
        O = self.actiout(self.fc1(H))

        
        return Z,H,O
    
    


    # sparsity regularizer https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    def bias_regularizer(self,data):
        
        DEVICE = torch.cuda.current_device()
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
    
    def train(self,optimizer,noise,print_loss=True):
        
        DEVICE = torch.cuda.current_device()
        data,output = aet_stim.mkstim(noise)
        data = data.to(DEVICE)
        output = output.to(DEVICE)

        loss = torch.zeros((self.num_ep),).to(DEVICE)

        for e in range(self.num_ep):

            optimizer.zero_grad()
            x_mini,y_mini = aet_stim.make_minib(data,output,DEVICE)
            

            for mb in range(x_mini.shape[0]):

                for input_,output_ in zip(x_mini[mb],y_mini[mb]):

                    # forward
                    _,_,y = self.forw_conv(input_)

                    # regularizer (if sparsity params are defined)
                    if self.reg:
                        _regu = self.bias_regularizer(data)
                    else:
                        _regu = torch.zeros(2)
                        
                    # loss + sparsity penalty
                    _loss = self.lossfun(y,output_) + _regu[0]

                    # accumulate gradients for minibatch
                    _loss.backward()

                    loss[e] += _loss

                    # add sparsity penalty to bias
                    if self.reg:
                        bias = self.get_parameter('conv1.bias')
                        bias.grad += _regu[1]

            optimizer.step()

            if print_loss:
                print(f'epoch: {e}, cumulative loss: {loss[e]}')


        return loss
    
        

    