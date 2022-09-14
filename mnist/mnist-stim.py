# %% [code] {"execution":{"iopub.status.busy":"2022-09-14T14:50:09.424917Z","iopub.execute_input":"2022-09-14T14:50:09.425686Z","iopub.status.idle":"2022-09-14T14:50:09.431432Z","shell.execute_reply.started":"2022-09-14T14:50:09.425645Z","shell.execute_reply":"2022-09-14T14:50:09.429969Z"},"jupyter":{"outputs_hidden":false}}
import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F

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
    x = torch.concat((x1,x2,x3,x4))

    # repeat targets
    y = torch.tile(y,(4,1))
    
    return x, y
