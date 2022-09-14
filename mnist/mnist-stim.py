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

    
    return x, y

def make_minib(data,output,device,mini_sz=-1,set_sz=60000):

    shuff_idx = torch.randperm(data.shape[0])

    data = data[shuff_idx[:set_sz]]
    output = output[shuff_idx[:set_sz]]

    _num_minib = int(data.shape[0]/mini_sz)

    x_mini = torch.empty(_num_minib,mini_sz,data.shape[1],data.shape[2],data.shape[3]).to(device)
    y_mini = torch.empty(_num_minib,mini_sz,output.shape[1]).to(device)
    mc = 0   # mini batch counter
    for m in range(_num_minib):
        x_mini[m] = data[mc:mini_sz+mc]
        y_mini[m] = output[mc:mini_sz+mc]

        mc += mini_sz
    
    if mc < data.shape[0]-1:
        x_mini[m+1] = data[mc:-1]
        y_mini[m+1] = output[mc:-1]
    
    return x_mini, y_mini