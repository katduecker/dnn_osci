import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import math


def make_stim(train_data=True):
    # download MNIST
    mnist = datasets.MNIST(root='./data', train=train_data, transform=torchvision.transforms.ToTensor(),target_transform=F.one_hot,download=True)

    # %% [code] {"execution":{"iopub.status.busy":"2022-09-14T15:00:28.827244Z","iopub.execute_input":"2022-09-14T15:00:28.827630Z","iopub.status.idle":"2022-09-14T15:00:29.137653Z","shell.execute_reply.started":"2022-09-14T15:00:28.827599Z","shell.execute_reply":"2022-09-14T15:00:29.136417Z"},"jupyter":{"outputs_hidden":false}}
    # one-hot encode
    y = mnist.target_transform(mnist.targets)

    # position images into quadrants
    size_quadr_img = tuple(mnist.data.shape*np.array((1,2,2)))

    sq = mnist.data.shape[1]

    # first quadrant
    x1 = torch.zeros(size_quadr_img)
    x1[:,0:sq,0:sq] += mnist.data

    # second quadrant
    x2 = torch.zeros(size_quadr_img)
    x2[:,0:sq,sq:] += mnist.data        

    # third quadrant
    x3 = torch.zeros(size_quadr_img)
    x3[:,sq:,0:sq] += mnist.data   

    # fourth quadrant
    x4 = torch.zeros(size_quadr_img)
    x4[:,sq:,sq:] += mnist.data

    # concatenate quadrants
    if train_data:
        x = torch.concat((x1,x2,x3,x4))
        x = x.reshape(-1,1,56,56)
        
        # repeat targets
        y = torch.tile(y,(4,1))

    else:
        x1 = x1.reshape(-1,1,56,56)
        x2 = x2.reshape(-1,1,56,56)
        x3 = x3.reshape(-1,1,56,56)
        x4 = x4.reshape(-1,1,56,56)
        
        x = torch.concat((x1,x2,x3,x4),dim=1)
        
        y = y.reshape(-1,1,10)
        
        y = torch.tile(y,(1,4,1))

    y = y.float()
    
    return x, y

def make_minib(data_sz,mini_sz,set_sz=60000):

    shuff_idx = torch.randperm(data_sz)
    
    shuff_idx = shuff_idx[:set_sz]

    _num_minib = int(set_sz/mini_sz)
    
    c = 0               # index counter
    mn = 0              # mini batch counter
    # make empty list
    mini_idx = [None]*_num_minib

    while c <= set_sz-mini_sz:
        mini_idx[mn] = [shuff_idx[c:c+mini_sz]]
        c += mini_sz
        mn +=1
    
    return mini_idx    
