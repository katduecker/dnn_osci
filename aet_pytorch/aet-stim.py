# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2022-09-13T09:31:57.549869Z","iopub.execute_input":"2022-09-13T09:31:57.550927Z","iopub.status.idle":"2022-09-13T09:31:59.540438Z","shell.execute_reply.started":"2022-09-13T09:31:57.550834Z","shell.execute_reply":"2022-09-13T09:31:59.538623Z"}}
# -*- coding: utf-8 -*-

import torch
import numpy as np


# create A,E,T,Z stimuli
def mkstim(noise_=False):
    # create stimuli
    A = torch.zeros((28,28))
    A[8:23,6:9] = 1
    A[5:8,8:19] = 1
    A[16:19,8:19] = 1
    A[8:23,19:22] = 1


    E = torch.zeros((28,28))
    E[4:23,6:9] = 1
    E[4:7,8:21] = 1
    E[12:15,8:21] = 1
    E[20:23,8:21] = 1

    T = torch.zeros((28,28))
    T[6:10,6:22] = 1
    T[8:23,12:16] = 1

    Z = torch.zeros((28,28))
    Z[6:8, 6:22] = 1
    Z[21:23, 6:22] = 1

    ru = 8
    cu = torch.arange(18,22)
    rl = 20

    for i in torch.arange(13):
        Z[ru,cu] = 1
        ru +=1
        cu -=1
        rl -=1


    # 2. Place letters in larger image
    BIGA = torch.zeros((4,56,56))
    BIGE = torch.zeros((4,56,56))
    BIGT = torch.zeros((4,56,56))
    BIGZ = torch.zeros((4,56,56))

    s =  torch.arange(0,56).reshape((2,-1))     # split in half

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

    I = torch.cat((BIGA.reshape(4,1,56,56),BIGE.reshape(4,1,56,56),BIGT.reshape(4,1,56,56),BIGZ.reshape(4,1,56,56)))

    O = torch.cat((torch.tile(torch.tensor((1.,0.,0.,0.)),(4,1)),torch.tile(torch.tensor((0.,1.,0.,0.)),(4,1)),torch.tile(torch.tensor((0.,0.,1.,0.)),(4,1)),torch.tile(torch.tensor((0.,0.,0.,1.)),(4,1))))



    # add noise to images
    if noise_:

        stim = torch.from_numpy(np.concatenate((BIGA,BIGE,BIGT,BIGZ)))
        stim = stim.reshape(-1,1,56,56)
        label = torch.cat((torch.tile(torch.tensor((1,0,0,0)),(4,1)),torch.tile(torch.tensor((0,1,0,0)),(4,1)),torch.tile(torch.tensor((0,0,1,0)),(4,1)),torch.tile(torch.tensor((0,0,0,1)),(4,1))))

        num_it = 10

        I = stim
        O = label

        for i in range(num_it):

            # values should be between 0 and 1
            I_noise = torch.abs_(stim - torch.normal(0.4,0.1,stim.shape)*0.5)
            I_noise = I_noise.reshape(-1,1,56,56)
            I = torch.cat((I,I_noise),dim=0)

            O = torch.cat((O,label),dim=0)
            
            

    return I, O

# make mini batches 
def make_minib(data,output,device,mini_sz=1):

    shuff_idx = torch.randperm(data.shape[0])

    data = data[shuff_idx]
    output = output[shuff_idx]

    _num_minib = int(data.shape[0]/mini_sz)

    x_mini = torch.empty(_num_minib,mini_sz,data.shape[1],data.shape[2],data.shape[3]).to(device)
    y_mini = torch.empty(_num_minib,mini_sz,output.shape[1]).to(device)
    mc = 0   # mini batch counter
    for m in range(_num_minib):
        x_mini[m] = data[mc:mini_sz+mc]
        y_mini[m] = output[mc:mini_sz+mc]

        mc += mini_sz

    return x_mini, y_mini