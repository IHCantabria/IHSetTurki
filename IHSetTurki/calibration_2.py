import numpy as np
from .turki import turki
from IHSetUtils.CoastlineModel import CoastlineModel
from IHSetUtils import hunt

class cal_Turki_2(CoastlineModel):
    """
    cal_Turki_2
    
    Configuration to calibfalse,and run the Turki et al. (2013) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Turki et al. (2013)',
            mode='calibration',
            model_type='RT',
            model_key='Turki'
        )
        self.setup_forcing()

    def setup_forcing(self):
        self.switch_Yini = self.cfg['switch_Yini']
        if self.switch_Yini == 0:
            self.alp0 = self.Obs_splited[0]
        self.D50 = self.cfg['D50']
        self.BeachL = self.cfg['BeachL']
        
        self.depthb[self.hb < 0.5] = 0.91
        self.depthb_s[self.hb_s < 0.5] = 0.91
        self.hb[self.hb < 0.5] = 0.5
        self.tp[self.tp < 3] = 3
        self.L = hunt(self.tp, self.depthb)

        rhos = 2650
        rho = 1025
        g = 9.81
        ss = rhos/rho
        gama = 0.55
        Ub_cr = (0.014*self.tp*((ss-1)**2)*(g**2)*(self.D50))**(1/3)
        Hcr = (2**0.5/np.pi)*Ub_cr*self.tp*np.sinh((2*np.pi*self.depthb)/self.L)
        H_dif = self.hb-Hcr
        idx = np.where(H_dif <= 0)
        H_dif[idx] = 0.5
        self.EF = 1 / 8 * rho * g * H_dif ** 2.5 * (g / gama) ** 0.5
        self.EF_s = self.EF[self.idx_calibration]

    def init_par(self, population_size: int):
        if self.switch_Yini == 0:
            lowers = np.array([self.lb[0]])
            uppers = np.array([self.ub[0]])
        else:
            lowers = np.array([self.lb[0], np.min(self.Obs) - 360])
            uppers = np.array([self.ub[0], 1.5*np.max(self.Obs) + 360])
        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers


    def model_sim(self, par: np.ndarray) -> np.ndarray:
        kk = par[0]
        if self.switch_Yini== 0:
            alp0 = self.alp0
        else:
            alp0 = par[1]
        Ymd, _ = turki(self.EF_s,
                        self.hb_s,
                        self.dirb_s,
                        self.BeachL,
                        self.dt_s,
                        kk,
                        alp0)
        return Ymd[self.idx_obs_splited]

    def run_model(self, par: np.ndarray) -> np.ndarray:
        kk = par[0]
        if self.switch_Yini== 0:
            alp0 = self.alp0
        else:
            alp0 = par[1]
        Ymd, _ = turki(self.EF,
                        self.hb,
                        self.dirb,
                        self.BeachL,
                        self.dt,
                        kk,
                        alp0)
        return Ymd
    
    def _set_parameter_names(self):
        if self.switch_Yini == 0:
            self.par_names = [r'k_k']
        elif self.switch_Yini == 1:
            self.par_names = [r'k_k', r'alpha_0']
        