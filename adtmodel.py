import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# Transfer function definition
def sigmoid(x):
    return 1/(1+np.exp(-x))
# Adaptive Threshold Model 
def adtmodel( y, t ):
    h , f = y
    T = 0.05
    I = 1.1
    A = 3
    tau_h=10  # h time scale
    tau_f=100 # threshold time scale 
    dhdt = 1/tau_h*( -h + sigmoid((I + h -A*f)/T)  )
    dfdt = 1/tau_f*( - f + h )
    dydt = [dhdt, dfdt ]
    return dydt
I=1.1
y0 = [0.0, 0.0]
t = np.linspace(0, 500, 1000)
solhu = odeint(adtmodel, y0, t)
plt.figure(figsize=(15, 5))
plt.plot(t, I*np.ones(len(t)),label='Input')
plt.plot(t, solhu[:, 0],label='Hidden Cell Activity' )
plt.plot(t, solhu[:, 1],label='Threshold Variable')
plt.legend(loc='best')
plt.xlabel('time')
plt.grid()
plt.show()
