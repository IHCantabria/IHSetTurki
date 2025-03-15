import numpy as np
from numba import jit

@jit
def turki(EF, Hb, theb, BeachL, dt, kk, alp0):
    """
    Turki et al. 2013 model
    """
    rhos = 2650
    rho = 1025
    a = 0.3
    g = 9.81
    gamb = theb * np.pi/180 - alp0 * np.pi/180
     
    hc = (2.8*Hb)**0.67

    hc[hc < 0.1] = 0.1

    KK = (0.5*kk*0.07)/((rhos-rho)*g*(1-a))*3600
    
    R0 = 0
    Fc = 4*KK*(1/BeachL)*(EF/hc)
    P = (4/BeachL)*Fc*np.cos(2*gamb)
    Q = Fc*np.sin(2*gamb)
        
    intP0 = np.cumsum(P)*dt
    intQ0 = np.cumsum(Q*np.exp(intP0))*dt

    R = np.exp(-intP0)*(R0+intQ0)
    alps = np.arctan(2*R/BeachL) + alp0 * np.pi/180
    
    alps = alps * 180/np.pi
    
    return alps, R

def turki_njit(EF, Hb, theb, BeachL, dt, kk, alp0):
    """
    Turki et al. 2013 model
    """
    rhos = 2650
    rho = 1025
    a = 0.3
    g = 9.81
    gamb = theb * np.pi/180 - alp0 * np.pi/180
     
    hc = (2.8*Hb)**0.67
    for i in range(len(hc)):
        if hc[i] < 0.1:
           hc[i] = 0.1

    KK = (0.5*kk*0.07)/((rhos-rho)*g*(1-a))*3600
    
    R0 = 0
    Fc = 4*KK*(1/BeachL)*(EF/hc)
    P = (4/BeachL)*Fc*np.cos(2*gamb)
    Q = Fc*np.sin(2*gamb)
        
    intP0 = np.cumsum(P)*dt
    intQ0 = np.cumsum(Q*np.exp(intP0))*dt

    R = np.exp(-intP0)*(R0+intQ0)
    alps = np.arctan(2*R/BeachL) + alp0 * np.pi/180
    
    alps = alps * 180/np.pi
    
    return alps, R