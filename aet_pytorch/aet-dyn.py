# %% [code]
import numpy as np
import torch

# forward dynamics/discretization, euler integration
def euler_dyn(model,input_,params_,t_,alpha_params,DEVICE,dyn_inp=False):

    # inputs:
        # input_: input image
        # params_: hyperparameters
        # t_: time vector for integration
        # alpha_params: alpha frequency & amplitude

    # discretization & dynamics parameters
    tau_h,tau_R,R,T,h_start,R_start = params_


    # alpha frequency & amplitude
    _af,_aa = alpha_params

    with torch.no_grad():

        # initialize empty matrices
        dt = np.diff(t_)[0]
        dhdt = (torch.ones((model.dims[1],len(t_)+1))*h_start).to(DEVICE)
        dRdt = (torch.ones((model.dims[1],len(t_)+1))*R_start).to(DEVICE)
        dOdt = (torch.zeros((model.dims[2],len(t_)+1))).to(DEVICE)
        dZdt = (torch.zeros((model.dims[1],len(t_)+1))).to(DEVICE)

        # alpha inhibition
        alpha_inh = _aa*np.sin(2*np.pi*_af*t_)+_aa

        # preactivation (dot product of input and first weight matrix)
        Z,_,_ = model.forw_conv(input_)
        # create boxcar function if the input is dynamic
        if dyn_inp:
            boxcar = np.zeros_like(t_)
            boxcar[50:] = 1

        # adjust initial adaptation term (threshold)
        dRdt *= torch.max(Z)                           

        # scale for adaptation
        r_scale = R*torch.max(Z).detach()


        for _it,t in enumerate(t_):

            # dynamic input: multiply input with boxcar
            if dyn_inp:
                Z,_,_ = model.forw_conv(input_*boxcar[_it])
            
            # pre-activation
            dZdt[:,_it+1] = (Z + dhdt[:,_it] - dRdt[:,_it] - alpha_inh[_it])/T

            # dynamics hidden layer
            dhdt[:,_it+1] = dhdt[:,_it] + dt/tau_h * (-dhdt[:,_it] + model.acti1(dZdt[:,_it+1],model.sig_param))      

            # adaptation term
            dRdt[:,_it+1] = dRdt[:,_it] + dt/tau_R * (-dRdt[:,_it] + r_scale*dhdt[:,_it+1])

            # output layer
            dOdt[:,_it+1] = model.actiout(model.fc1(dhdt[:,_it+1]))

    return dZdt, dhdt, dRdt, dOdt