import numpy as np
from numba import jit

# @jit
# def turki(EF, Hb, theb, BeachL, dt, kk, alp0):
#     """
#     Turki et al. 2013 model
#     """
#     rhos = 2650
#     rho = 1025
#     a = 0.3
#     g = 9.81
#     gamb = theb * np.pi/180 - alp0 * np.pi/180
     
#     hc = (2.8*Hb)**0.67

#     hc[hc < 0.1] = 0.1

#     KK = (0.5*kk*0.07)/((rhos-rho)*g*(1-a))*3600
    
#     R0 = 0
#     Fc = 4*KK*(1/BeachL)*(EF/hc)
#     P = (4/BeachL)*Fc*np.cos(2*gamb)
#     Q = Fc*np.sin(2*gamb)
        
#     intP0 = np.cumsum(P)*dt
#     intQ0 = np.cumsum(Q*np.exp(intP0))*dt

#     R = np.exp(-intP0)*(R0+intQ0)
#     alps = np.arctan(2*R/BeachL) + alp0 * np.pi/180
    
#     alps = alps * 180/np.pi
    
#     return alps, R


import numpy as np
from numba import njit
import math

@njit(fastmath=True, cache=True)
def turki(EF, Hb, theb, BeachL, dt, kk, alp0):
    # Parâmetros constantes
    rhos = 2650.0
    rho = 1025.0
    one_minus_a = 1.0 - 0.3
    g = 9.81
    # ângulo alfa0 em radianos
    alp0_rad = alp0 * math.pi / 180.0
    # coeficiente KK
    KK = (0.5 * kk * 0.07) / ((rhos - rho) * g * one_minus_a) * 3600.0

    # Número de pontos
    n = EF.shape[0]
    # Alocações
    alps = np.empty(n, dtype=np.float64)
    R = np.empty(n, dtype=np.float64)
    # Vetores temporários
    P = np.empty(n, dtype=np.float64)
    Q = np.empty(n, dtype=np.float64)
    intP = np.empty(n, dtype=np.float64)
    intQ = np.empty(n, dtype=np.float64)

    # Cálculo inicial por índice
    for i in range(n):
        # profundidade de ruptura hc
        hc = (2.8 * Hb[i]) ** 0.67
        if hc < 0.1:
            hc = 0.1
        # ângulo beta em radianos e diferença gamb
        gamb = theb[i] * math.pi / 180.0 - alp0_rad
        # Fc
        Fc = 4.0 * KK * (1.0 / BeachL) * (EF[i] / hc)
        # P e Q
        P[i] = (4.0 / BeachL) * Fc * math.cos(2.0 * gamb)
        Q[i] = Fc * math.sin(2.0 * gamb)

    # Integração cumulativa
    # intP[0]
    intP[0] = P[0] * dt[0]
    intQ[0] = Q[0] * math.exp(intP[0]) * dt[0]
    # laço de 1 a n-1
    for i in range(1, n):
        # acumulado de P
        intP[i] = intP[i-1] + P[i] * dt[i]
        # acumulado de Q modulado por exp(intP[i])
        intQ[i] = intQ[i-1] + Q[i] * math.exp(intP[i]) * dt[i]

    # Cálculo de R e ângulo final
    for i in range(n):
        # R = exp(-intP) * intQ (R0=0)
        R[i] = math.exp(-intP[i]) * intQ[i]
        # alpha final em rad + alp0_rad
        alps[i] = math.atan(2.0 * R[i] / BeachL) + alp0_rad
        # converter para graus
        alps[i] = alps[i] * 180.0 / math.pi

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