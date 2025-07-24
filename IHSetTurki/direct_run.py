# import numpy as np
# import xarray as xr
# import fast_optimization as fo
# import pandas as pd
# from .turki import turki_njit
# import json
# from scipy.stats import circmean
# from IHSetUtils import BreakingPropagation, hunt

# class Turki_run(object):
#     """
#     Turki_run
    
#     Configuration to run the Turki et al. (2013) Shoreline Rotation Model.
    
#     This class reads input datasets, performs its calibration.
#     """

#     def __init__(self, path):

#         self.path = path
#         self.name = 'Turki et al. (2013)'
#         self.mode = 'standalone'
#         self.type = 'RT'
     
#         data = xr.open_dataset(path)
        
#         cfg = json.loads(data.attrs['run_Turki'])
#         self.cfg = cfg

#         self.switch_Yini = cfg['switch_Yini']
#         self.D50 = cfg['D50']
#         self.BeachL = cfg['BeachL']
        
#         self.switch_brk = cfg['switch_brk']
#         if self.switch_brk == 1:
#             self.bathy_angle = circmean(data.phi.values, high=360, low=0)
#             self.breakType = cfg['break_type']
#             self.depth = np.mean(data.waves_depth.values)

#         self.hs = np.mean(data.hs.values, axis=1)
#         self.tp = np.mean(data.tp.values, axis=1)
#         self.dir = circmean(data.dir.values, axis=1, high=360, low=0)
#         self.time = pd.to_datetime(data.time.values)
#         self.Obs = data.rot.values
#         self.Obs = self.Obs[~data.mask_nan_rot]
#         self.time_obs = pd.to_datetime(data.time_obs.values)
#         self.time_obs = self.time_obs[~data.mask_nan_rot]
        
#         self.start_date = pd.to_datetime(cfg['start_date'])
#         self.end_date = pd.to_datetime(cfg['end_date'])

#         data.close()
        
#         if self.switch_brk == 0:
#             self.hb = self.hs
#             self.depthb = self.hb / 0.55
#             self.dirb = self.dir
#         elif self.switch_brk == 1:
#             self.hb, self.dirb, self.depthb = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)
        
#         self.depthb[self.hb < 0.5] = 1
#         self.hb[self.hb < 0.5] = 0.5
#         self.tp[self.tp < 3] = 3

#         self.L = hunt(self.tp, self.depthb)

#         rhos = 2650
#         rho = 1025
#         g = 9.81

#         ss = rhos/rho
#         gama = 0.55
#         Ub_cr = (0.014*self.tp*((ss-1)**2)*(g**2)*(self.D50))**(1/3)
#         Hcr = (2**0.5/np.pi)*Ub_cr*self.tp*np.sinh((2*np.pi*self.depthb)/self.L)
#         H_dif = self.hb-Hcr
#         idx = np.where(H_dif <= 0)
#         H_dif[idx] = 0.5
#         self.EF = 1 / 8 * rho * g * H_dif ** 2.5 * (g / gama) ** 0.5

#         self.split_data()

#         if self.switch_Yini == 1:
#             self.alp0 = self.Obs[0]

#         mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

#         self.idx_obs = mkIdx(self.time_obs)

#         # Now we calculate the dt from the time variable
#         mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
#         self.dt = mkDT(np.arange(0, len(self.time)-1))
#         self.dt = np.hstack((self.dt, self.dt[-1]))

#         if self.switch_Yini == 0:
#             def run_model(par):
#                 kk = par[0]
#                 alp0 = par[1]
                
#                 Ymd, _ = turki_njit(self.EF,
#                                 self.hb,
#                                 self.dirb,
#                                 self.BeachL,
#                                 self.dt,
#                                 kk,
#                                 alp0)
#                 return Ymd
            
#             self.run_model = run_model

#         elif self.switch_Yini == 1:
#             def run_model(par):
#                 kk = par[0]

#                 Ymd, _ = turki_njit(self.EF,
#                                 self.hb,
#                                 self.dirb,
#                                 self.BeachL,
#                                 self.dt,
#                                 kk,
#                                 self.alp0)
#                 return Ymd
            
#             self.run_model = run_model

#     def run(self, par):
#         self.full_run = self.run_model(par)
#         if self.switch_Yini == 1:
#             self.par_names = [r'k_k']
#             self.par_values = par
#         elif self.switch_Yini == 0:
#             self.par_names = [r'k_k', r'alpha_0']
#             self.par_values = par

#         # self.calculate_metrics()

#     def calculate_metrics(self):
#         self.metrics_names = fo.backtot()[0]
#         self.indexes = fo.multi_obj_indexes(self.metrics_names)
#         self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)

#     def split_data(self):
#         """
#         Split the data into calibration and validation datasets.
#         """ 
#         ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
#         self.hb = self.hb[ii]
#         self.dirb = self.dirb[ii]
#         self.EF = self.EF[ii]
#         self.time = self.time[ii]

#         ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
#         self.Obs = self.Obs[ii]
#         self.time_obs = self.time_obs[ii]


import numpy as np
from .turki import turki_njit
from IHSetUtils.CoastlineModel import CoastlineModel
from IHSetUtils import hunt

class Turki_run(CoastlineModel):
    """
    Turki_run
    
    Configuration to run the Turki et al. (2013) Shoreline Rotation Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Turki et al. (2013)',
            mode='standalone',
            model_type='RT',
            model_key='run_Turki'
        )
        self.setup_forcing()

    def setup_forcing(self):
        self.switch_Yini = self.cfg['switch_Yini']
        if self.switch_Yini == 1:
            self.alp0 = self.Obs[0]
        self.D50 = self.cfg['D50']
        self.BeachL = self.cfg['BeachL']
        
        self.depthb[self.hb < 0.5] = 0.91
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
        
    def run_model(self, par: np.ndarray) -> np.ndarray:
        kk = par[0]
        if self.switch_Yini == 1:
            alp0 = self.alp0
        else:
            alp0 = par[1]
        Ymd, _ = turki_njit(self.EF,
                            self.hb,
                            self.dirb,
                            self.BeachL,
                            self.dt,
                            kk,
                            alp0)
        return Ymd
    
    def _set_parameter_names(self):
        if self.switch_Yini == 1:
            self.par_names = [r'k_k']
        elif self.switch_Yini == 0:
            self.par_names = [r'k_k', r'alpha_0']
        