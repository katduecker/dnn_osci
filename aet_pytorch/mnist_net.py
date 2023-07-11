# %% [code]
import torch
from torch import nn
import numpy as np
from collections import OrderedDict

from itertools import combinations

from collections import OrderedDict


def bias_regularizer(model,data,kl_reg_,dims):

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _beta, _rho = kl_reg_

        H = model.FC2.forward(model.FC1.forward(model.CONV_QCOMP.forward(data)))

        rho_hat = torch.mean(H,dim=0)

        # regularizer: KL divergence between hidden activations & rho (added to loss)
        regu_loss = _beta*torch.sum(_rho*torch.log(_rho/rho_hat) + (1-_rho)*torch.log((1-_rho)/(1-rho_hat)))

        # penalty bias added to bias gradient
        regu_bias = _beta * (-_rho/rho_hat + ((1-_rho)/(1-rho_hat)))

        return regu_loss, regu_bias

def train_model(model,dims,data,output,num_ep,mini_sz,kl_reg_,optimizer,criterion):

    # shuffle data
    shuff_idx = np.random.permutation(data.shape[0])

    # make minibatch
    _num_minib = int(data.shape[0]/mini_sz)

    c = 0               # index counter
    mn = 0              # mini batch counter
    # make empty list
    mini_idx = [None]*_num_minib

    while c <= data.shape[0]-mini_sz:
        mini_idx[mn] = [shuff_idx[c:c+mini_sz]]
        c += mini_sz
        mn +=1

    loss_epo = torch.zeros((num_ep,))
    accu_epo = torch.zeros((num_ep,))
    for e in range(num_ep):

        for i in range(len(mini_idx)):

            # forward
            y = model(data[mini_idx[i]])

            optimizer.zero_grad()

            # sparsity regularizer?
            if kl_reg_[0]:
                regu_loss, regu_bias = bias_regularizer(model,data,kl_reg_,dims)

                optimizer.zero_grad()

                # loss + sparsity penalty
                _loss = criterion(output[mini_idx[i]],y) + regu_loss

                # accumulate gradients for minibatch
                _loss.backward()
                model.FC2.fc.bias.grad += regu_bias 
            else:
                _loss = criterion(output[mini_idx[i]],y) 
                # accumulate gradients for minibatch
                _loss.backward()


             # update after mini batch
            optimizer.step()

        loss_epo[e] = _loss
        #print(f'loss epoch {e}: {_loss}')
        # accuracy: average activation of correct node
        accu_epo[e] = torch.mean(y[output[mini_idx[i]].to(bool)])

    return loss_epo, accu_epo

def init_params(model):
    
    for m in model.modules():
        # if module is a conv or linear layer, set weights
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)#,mean=0,std=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    return model

# NN building blocks

class CNN_layer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size = 3, learn_bias=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=int(kernel_size/2),bias=learn_bias)    
        self.nonlin = nn.Sigmoid()
        self.pool = nn.MaxPool2d(3, stride=2)
        
    def forward(self,input):
        
        x = self.conv(input)
        x = self.pool(x)
        y = self.nonlin(x)
        
        return y

class CNN_q_comp(nn.Module):
    
    def __init__(self, in_channels, out_channels, quadrant_size = 28, learn_bias=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=quadrant_size,stride=quadrant_size,bias=learn_bias)    
        self.nonlin = nn.Sigmoid()
        self.pool = torch.sum
        
    def forward(self,input):
        
        x = self.conv(input)
        x = self.pool(x,dim=(-2,-1))
        y = self.nonlin(x)
        
        return y

class FC_layer(nn.Module):
    def __init__(self, in_channels, out_channels, actifun, learn_bias=True):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels,bias=learn_bias)    
        self.nonlin = actifun
        
    def forward(self,input):
        
        x = self.nonlin(self.fc(input))

        return x# AET net
    
class mnist_FC_model(nn.Module):
    
    def __init__(self,lay_size):
        
        super(mnist_FC_model,self).__init__()
        
        self.lay_size = lay_size
        
    def get_model(self,lb =True):

        model = nn.Sequential(OrderedDict([
        ('CONV_QCOMP', CNN_q_comp(1, self.lay_size[1],learn_bias=lb)),
        ('FC1', FC_layer(self.lay_size[1], self.lay_size[2],nn.Sigmoid(),learn_bias=lb)),
        ('FC2', FC_layer(self.lay_size[2], self.lay_size[3],nn.Softmax(dim=-1)))]))

        model = init_params(model)
        
        return model
    
