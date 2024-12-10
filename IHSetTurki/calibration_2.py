import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .turki import turki
import json
from scipy.stats import circmean
from IHSetUtils import BreakingPropagation, hunt

class cal_Turki_2(object):
    """
    cal_Turki_2
    
    Configuration to calibfalse,and run the Turki et al. (2013) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['Turki'])

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.switch_Yini = cfg['switch_Yini']
        self.lb = cfg['lb']
        self.ub = cfg['ub']
        self.D50 = cfg['D50']
        self.BeachL = cfg['BeachL']
        
        self.switch_brk = cfg['switch_brk']
        if self.switch_brk == 1:
            self.bathy_angle = cfg['bathy_angle']
            self.breakType = cfg['break_type']
            self.depth = cfg['depth']

        self.calibr_cfg = fo.config_cal(cfg)

        self.hs = np.mean(data.hs.values, axis=1)
        self.tp = np.mean(data.tp.values, axis=1)
        self.dir = circmean(data.dir.values, axis=1, high=360, low=0)
        self.time = pd.to_datetime(data.time.values)
        self.Obs = data.rot.values
        self.Obs = self.Obs[~data.mask_nan_rot]
        self.time_obs = pd.to_datetime(data.time_obs.values)
        self.time_obs = self.time_obs[~data.mask_nan_rot]
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])

        data.close()
        
        if self.switch_brk == 0:
            self.depthb = self.hb / 0.55
            self.hb = self.hs
            self.dirb = self.dir
        elif self.switch_brk == 1:
            self.hb, self.dirb, self.depthb = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)
        
        self.depthb[self.hb < 0.5] = 1
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

        self.split_data()

        if self.switch_Yini == 0:
            self.alp0 = self.Obs_splited[0]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        self.dt = np.hstack((self.dt, self.dt[-1]))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))
        self.dt_splited = np.hstack((self.dt_splited, self.dt_splited[-1]))

        if self.switch_Yini == 0:
            def model_simulation(par):
                kk = par[0]

                Ymd, _ = turki(self.EF_splited,
                                self.hb_splited,
                                self.dirb_splited,
                                self.BeachL,
                                self.dt_splited,
                                kk,
                                self.alp0)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                kk = par[0]
                
                Ymd, _ = turki(self.EF,
                                self.hb,
                                self.dirb,
                                self.BeachL,
                                self.dt,
                                kk,
                                self.alp0)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([self.lb[0]])
                log_upper_bounds = np.array([self.ub[0]])
                population = np.zeros((population_size, 1))
                for i in range(1):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            def model_simulation(par):
                kk = par[0]
                alp0 = par[1]

                Ymd, _ = turki(self.EF_splited,
                                self.hb_splited,
                                self.dirb_splited,
                                self.BeachL,
                                self.dt_splited,
                                kk,
                                alp0)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                kk = par[0]
                alp0 = par[1]

                Ymd, _ = turki(self.EF,
                                self.hb,
                                self.dirb,
                                self.BeachL,
                                self.dt,
                                kk,
                                alp0)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([self.lb[0], 0.5*np.min(self.Obs)])
                log_upper_bounds = np.array([self.ub[0], 1.5*np.max(self.Obs)])
                population = np.zeros((population_size, 2))
                for i in range(2):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds

            self.init_par = init_par            

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """ 
        ii = np.where(self.time>=self.start_date)[0][0]
        self.hb = self.hb[ii:]
        self.dirb = self.dirb[ii:]
        self.EF = self.EF[ii:]
        self.time = self.time[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.hb_splited = self.hb[idx]
        self.dirb_splited = self.dirb[idx]
        self.EF_splited = self.EF[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))

        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))
        self.idx_validation_obs = idx[0]
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)