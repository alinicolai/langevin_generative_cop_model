

import numpy as np


def compute_com_from_cop_LPFA(cop, frequency, pendulum=None, **kwargs):
    
    """ LPFA method """
        
    if pendulum is not None:
        C = (2*np.pi)**2 / pendulum
    else:
        C= [4.2,4.2]    # corresponds to pendulum coeff (2pi)^2 / 4.2 = 9.4

    freqs = np.fft.rfftfreq(n=len(cop),d=1/frequency)

    com = []
    ndim = len(cop.shape)

    if ndim == 1 :
        
        cop = cop.reshape((-1,1))

    for axis in range(cop.shape[1]):
        fourier = np.fft.rfft(cop[:,axis]) 
        
        phi_coef = 1/(1+C[axis]*(freqs**2))
        
        fourier = fourier * phi_coef

        x = np.fft.irfft(fourier)[:len(cop)]
        com.append(x) 
    
    if ndim == 1 :
        return np.array(com[0])
    else: 
        com = np.array(com).T
    
    return com

