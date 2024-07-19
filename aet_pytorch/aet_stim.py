# %% [code]
# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2022-09-13T09:31:57.549869Z","iopub.execute_input":"2022-09-13T09:31:57.550927Z","iopub.status.idle":"2022-09-13T09:31:59.540438Z","shell.execute_reply.started":"2022-09-13T09:31:57.550834Z","shell.execute_reply":"2022-09-13T09:31:59.538623Z"}}
# -*- coding: utf-8 -*-

import torch
import numpy as np


# create A,E,T stimuli
def mkstim(noise=0, num_it=10):
    # create stimuli
    A = torch.zeros((28, 28))
    A[8:23, 6:9] = 1
    A[5:8, 8:19] = 1
    A[16:19, 8:19] = 1
    A[8:23, 19:22] = 1

    E = torch.zeros((28, 28))
    E[4:23, 6:9] = 1
    E[4:7, 8:21] = 1
    E[12:15, 8:21] = 1
    E[20:23, 8:21] = 1

    T = torch.zeros((28, 28))
    T[5:10, 5:23] = 1
    T[7:24, 12:16] = 1
    
    # align brightness of stimuli
    A = A * (torch.sum(T) / torch.sum(A))
    E = E * (torch.sum(T) / torch.sum(E))

    #     Z = torch.zeros((28,28))
    #     Z[6:8, 6:22] = 1
    #     Z[21:23, 6:22] = 1

    #     ru = 8
    #     cu = torch.arange(18,22)
    #     rl = 20

    #     for i in torch.arange(13):
    #         Z[ru,cu] = 1
    #         ru +=1
    #         cu -=1
    #         rl -=1

    # 2. Place letters in larger image
    BIGA = torch.zeros((4, 56, 56))
    BIGE = torch.zeros((4, 56, 56))
    BIGT = torch.zeros((4, 56, 56))
    #     BIGZ = torch.zeros((4,56,56))

    s = torch.arange(0, 56).reshape((2, -1))  # split in half

    q = 0  # quadrant counter

    # loop over height
    for h in range(int(BIGA.shape[0] / 2)):
        # looper over width
        for w in range((int(BIGA.shape[0] / 2))):

            BIGA[q, s[h, 0] : s[h, -1] + 1, s[w, 0] : s[w, -1] + 1] += A
            BIGE[q, s[h, 0] : s[h, -1] + 1, s[w, 0] : s[w, -1] + 1] += E
            BIGT[q, s[h, 0] : s[h, -1] + 1, s[w, 0] : s[w, -1] + 1] += T
            #             BIGZ[q,s[h,0]:s[h,-1]+1,s[w,0]:s[w,-1]+1] += Z
            q += 1

    I = torch.cat(
        (
            BIGA.reshape(4, 1, 56, 56),
            BIGE.reshape(4, 1, 56, 56),
            BIGT.reshape(4, 1, 56, 56),
        )
    )  # ,BIGZ.reshape(4,1,56,56)))

    O = torch.cat(
        (
            torch.tile(torch.tensor((1.0, 0.0, 0.0)), (4, 1)),
            torch.tile(torch.tensor((0.0, 1.0, 0.0)), (4, 1)),
            torch.tile(torch.tensor((0.0, 0.0, 1.0)), (4, 1)),
        )
    )
    # O = torch.cat((torch.tile(torch.tensor((1.,0.,0.,0.)),(4,1)),torch.tile(torch.tensor((0.,1.,0.,0.)),(4,1)),torch.tile(torch.tensor((0.,0.,1.,0.)),(4,1)),torch.tile(torch.tensor((0.,0.,0.,1.)),(4,1))))

    # add noise to images
    if noise:

        stim = torch.from_numpy(np.concatenate((BIGA, BIGE, BIGT)))  # ,BIGZ)))
        stim = stim.reshape(-1, 1, 56, 56)
        label = torch.cat(
            (
                torch.tile(torch.tensor((1.0, 0.0, 0.0)), (4, 1)),
                torch.tile(torch.tensor((0.0, 1.0, 0.0)), (4, 1)),
                torch.tile(torch.tensor((0.0, 0.0, 1.0)), (4, 1)),
            )
        )

        # label = torch.cat((torch.tile(torch.tensor((1,0,0,0)),(4,1)),torch.tile(torch.tensor((0,1,0,0)),(4,1)),torch.tile(torch.tensor((0,0,1,0)),(4,1)),torch.tile(torch.tensor((0,0,0,1)),(4,1))))
        I = torch.abs(stim + torch.normal(0.4, 0.1, stim.shape) * noise)
        O = label

        for i in range(num_it):

            # values should be between 0 and 1
            I_noise = torch.abs(stim + torch.normal(0.4, 0.1, stim.shape) * noise)
            I_noise = I_noise.reshape(-1, 1, 56, 56)
            I = torch.cat((I, I_noise), dim=0)

            O = torch.cat((O, label), dim=0)

    for stim in I:
        stim /= torch.max(stim)

    return I, O


# make mini batches
def make_minib(data_sz, mini_sz):

    shuff_idx = torch.randperm(data_sz)

    _num_minib = int(data_sz / mini_sz)

    c = 0  # index counter
    mn = 0  # mini batch counter
    # make empty list
    mini_idx = [None] * _num_minib

    while c <= data_sz - mini_sz:
        mini_idx[mn] = [shuff_idx[c : c + mini_sz]]
        c += mini_sz
        mn += 1

    return mini_idx
