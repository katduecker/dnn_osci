# %% [code] {"execution":{"iopub.status.busy":"2022-09-14T14:50:09.424917Z","iopub.execute_input":"2022-09-14T14:50:09.425686Z","iopub.status.idle":"2022-09-14T14:50:09.431432Z","shell.execute_reply.started":"2022-09-14T14:50:09.425645Z","shell.execute_reply":"2022-09-14T14:50:09.429969Z"},"jupyter":{"outputs_hidden":false}}
import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F

import math


def make_stim():
    # download MNIST
    mnist_trainset = datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(),target_transform=F.one_hot,download=True)

    # %% [code] {"execution":{"iopub.status.busy":"2022-09-14T15:00:28.827244Z","iopub.execute_input":"2022-09-14T15:00:28.827630Z","iopub.status.idle":"2022-09-14T15:00:29.137653Z","shell.execute_reply.started":"2022-09-14T15:00:28.827599Z","shell.execute_reply":"2022-09-14T15:00:29.136417Z"},"jupyter":{"outputs_hidden":false}}
    # one-hot encode
    y_train = mnist_trainset.target_transform(mnist_trainset.targets)

    # position images into quadrants
    size_quadr_img = tuple(mnist_trainset.data.shape*np.array((1,2,2)))

    sq = mnist_trainset.data.shape[1]

    x_train = torch.zeros(size_quadr_img)

    sq = mnist_trainset.data.shape[1]

    # first quadrant
    x_train1 = torch.zeros(size_quadr_img)
    x_train1[:,0:sq,0:sq] += mnist_trainset.data

    # second quadrant
    x_train2 = torch.zeros(size_quadr_img)
    x_train2[:,0:sq,sq:] += mnist_trainset.data        

    # third quadrant
    x_train3 = torch.zeros(size_quadr_img)
    x_train3[:,sq:,0:sq] += mnist_trainset.data   

    # fourth quadrant
    x_train4 = torch.zeros(size_quadr_img)
    x_train4[:,sq:,sq:] += mnist_trainset.data

    # concatenate quadrants
    x_train = torch.concat((x_train1,x_train2,x_train3,x_train4))

    # repeat targets
    y_train = torch.tile(y_train,(4,1))