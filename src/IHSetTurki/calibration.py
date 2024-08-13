import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from IHSetTurki import turki
from IHSetCalibration import objective_functions
from IHSetUtils import BreakingPropagation, hunt

class cal_Turki(object):
    """
    cal_Turki
    
    Configuration to calibfalse,and run the Turki et al. (2013) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path, **kwargs):

        self.path = path
        
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = cfg['dt'].values
        self.switch_alpha_ini = cfg['switch_alpha_ini'].values

        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)

        self.Hs = wav['Hs'].values
        self.Tp = wav['Tp'].values
        self.Dir = wav['Dir'].values
        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)
                
        self.depth = cfg['depth'].values
        self.angleBathy = cfg['bathy_angle'].values
        self.D50 = cfg['D50'].values
        self.BeachL = cfg['BeachL'].values
        
        breakType = "spectral"
        self.Hb, self.theb, self.depthb = BreakingPropagation(self.Hs, self.Tp, self.Dir, np.repeat(self.depth, (len(self.Hs))), np.repeat(self.angleBathy, (len(self.Hs))),  breakType)
        if breakType == "mono":
            Bcoef = 0.78
        elif breakType == "spectral":
            Bcoef = 0.45
        
        self.depthb[self.Hb == 0] = 0.5 / Bcoef
        self.Hb[self.Hb == 0] = 0.5
        self.L = hunt(self.Tp, self.depthb)
        
        rhos = 2650
        rho = 1025
        g = 9.81  
        
        ss = rhos/rho
        gama = 0.55
        Ub_cr = (0.014*self.Tp*((ss-1)**2)*(g**2)*(self.D50))**(1/3)
        Hcr = (2**0.5/np.pi)*Ub_cr*self.Tp*np.sinh((2*np.pi*self.depthb)/self.L)
        H_dif = self.Hb-Hcr
        H_dif = H_dif
        ind = np.where(H_dif <= 0)
        H_dif[ind] = 0.5
        self.EF = 1 / 8 * rho * g * H_dif ** 2.5 * (g / gama) ** 0.5
        
        self.Obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))

        self.split_data()

        if self.switch_alpha_ini == 0:
            self.alp0 = self.Obs_splited[0]

        cfg.close()
        wav.close()
        ens.close()
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        if self.switch_alpha_ini == 0:
            def model_simulation(par):
                kk = par['kk']
                alp0 = par['alp0']

                Ymd, _ = turki.turki(self.EF_splited,
                                    self.Hb_splited,
                                    self.theb_splited,
                                    self.BeachL,
                                    self.dt,
                                    kk,
                                    self.alp0)
                
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('kk', 0.001, 10),
                Uniform('alp0', np.min(self.Obs), np.max(self.Obs))
            ]
            self.model_sim = model_simulation

        elif self.switch_alpha_ini == 1:
            def model_simulation(par):
                kk = par['kk']          
                alp0 = par['alp0']

                Ymd, _ = turki.turki(self.EF_splited,
                                    self.Hb_splited,
                                    self.theb_splited,
                                    self.BeachL,
                                    self.dt,
                                    kk,
                                    alp0)

                return Ymd[self.idx_obs_splited]
            self.params = [
                Uniform('kk', 0.001, 10),
                Uniform('alp0', np.min(self.Obs), np.max(self.Obs))
            ]
            self.model_sim = model_simulation

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.EF_splited = self.EF[idx]
        self.Hb_splited = self.Hb[idx]
        self.theb_splited = self.theb[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    # def split_data(self):
    #     """
    #     Split the data into calibration and validation datasets.
    #     """
    #     idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
    #     self.idx_validation = idx

    #     idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
    #     self.idx_calibration = idx
    #     self.EF_splited = self.EF[idx]
    #     self.Hb_splited = self.Hb[idx]
    #     self.theb_splited = self.theb[idx]
    #     self.time_splited = self.time[idx]

    #     idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
    #     self.Obs_splited = self.Obs[idx]
    #     self.time_obs_splited = self.time_obs[idx]

    #     mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
    #     self.idx_obs_splited = mkIdx(self.time_obs_splited)
    #     self.observations = self.Obs_splited

    #     # Validation
    #     idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))
    #     self.idx_validation_obs = idx
    #     mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
    #     self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
