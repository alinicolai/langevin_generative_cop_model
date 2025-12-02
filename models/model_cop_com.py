


import numpy as np
from models.forces import get_forces_dict_cop, get_forces_dict_com

from models.model_tools.signals_derivation import compute_discrete_derivative
from models.model_tools.OLS_fit import fit_OLS

class ModelCoupledCoPCoM():

    def __init__(self, list_forces_cop=[], list_forces_com=[], fitted_com_variable = "com_acc", fitted_cop_variable="cop_acc"):

        self.list_forces_cop = list_forces_cop
        
        self.list_forces_com = list_forces_com

        self.fitted_cop_variable = fitted_cop_variable

        self.fitted_com_variable = fitted_com_variable

    def get_ready_data(self, cop, com, frequency, com_acc=None):

        dt = 1/frequency

        cop_spd = compute_discrete_derivative(cop, order=1, dt=dt)
        cop_acc = compute_discrete_derivative(cop, order=2, dt=dt)

        if com_acc is None:
            com_acc = compute_discrete_derivative(com, order=2, dt=dt)

        n_min = min(len(cop_acc), len(com_acc))

        cop_spd = cop_spd[:n_min]
        cop_acc = cop_acc[:n_min]
        cop = cop[:n_min]        

        com_spd = compute_discrete_derivative(com, order=1, dt=dt)
        com = com[:n_min]
        com_spd = com_spd[:n_min]
        com_acc = com_acc[:n_min] 

        return cop, com, cop_spd, cop_acc, com_spd, com_acc               
    
    def fit_pendulum(self, cop, com, frequency, com_acc=None):
        
        dt = 1/frequency

        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc)

        forces_dict = get_forces_dict_com(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc)
        results = fit_OLS(forces_dict=forces_dict, Y=com_acc, dt=dt, list_forces=["pendulum"])
        self.pendulum = np.array([results[i]["coefs"]["pendulum"] for i in range(2)])
        self.sigma_pendulum = np.array([results[i]["coefs"]["sigma"] for i in range(2)])

    def fit(self, cop, com, frequency, com_acc=None):
        
        self.fit_pendulum(cop=cop, com=com, com_acc=com_acc, frequency=frequency)

        if len(self.list_forces_com)>0:
            self.fit_com(com=com, com_acc=com_acc, cop=cop, frequency=frequency)

        if len(self.list_forces_cop)>0:
            self.fit_cop(cop=cop, com=com, com_acc=com_acc, frequency=frequency)
            

        self.fit_results = {'cop': {'ML': self.fit_cop_results[0],
                                    'AP': self.fit_cop_results[1]},
                            'com': {'ML': self.fit_com_results[0],
                                    'AP': self.fit_com_results[1]}
                            }
        

    def fit_cop_1D(self, cop, com, frequency, com_acc=None):
        
        dt = 1/frequency
        
        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc)

        Y = cop_acc

        # Load forces
        forces_dict = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, com_acc=com_acc, cop_acc=cop_acc, frequency=frequency, pendulum=self.pendulum, coef_eq=self.coef_eq, coef_spd_eq=self.coef_spd_eq)

        # Fit
        self.fit_cop_results = fit_OLS(forces_dict=forces_dict, Y=Y, dt=dt, forces_delay=self.forces_delay_cop, list_forces=self.list_forces_cop, dim=1)

        return self.fit_cop_results

    def fit_cop(self, cop, frequency, com=None, com_acc=None):

        dt = 1/frequency
        
        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc)

        Y = cop_acc

        ## Compute forces ##
        forces_dict = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, frequency=frequency)

        self.fit_cop_results = fit_OLS(forces_dict=forces_dict, Y=Y, dt=dt, list_forces=self.list_forces_cop)

        return self.fit_cop_results


    def fit_com(self, com, com_acc, cop, frequency):
        
        dt = 1/frequency

        cop, com, cop_spd, cop_acc, com_spd, com_acc = self.get_ready_data(cop, com, frequency, com_acc=com_acc)

        Y = com_acc

        ## Compute forces ##
        forces_dict = get_forces_dict_com(cop=cop, cop_spd=cop_spd, cop_acc=cop_acc, com=com, com_spd=com_spd, com_acc=com_acc)
                
        ## Fit
        self.fit_com_results = fit_OLS(forces_dict=forces_dict, Y=Y, dt=dt, list_forces=self.list_forces_com)

        return self.fit_com_results
    

    def generate(self, true_cop, true_com, frequency, length_factor=1):

        start = 10
        dt = 1./(frequency)

        self.generate_start = start

        cop = np.zeros((int(len(true_cop)*length_factor),2))
        cop_spd = np.zeros((int(len(true_cop)*length_factor),2))
        
        cop[:start] = true_cop[:start]
        cop_spd[:start] = compute_discrete_derivative(true_cop, 1, dt)[:start]
        
        com = np.zeros((int(len(true_cop)*length_factor),2))
        com[:start] = true_com[:start]
        
        com_spd = np.zeros((int(len(true_cop)*length_factor),2))
        com_spd[:start] = compute_discrete_derivative(true_com, 1, dt)[:start]

        com_acc = np.zeros((int(len(true_cop)*length_factor),2))
        com_acc[:start-1] = compute_discrete_derivative(true_com, 2, dt)[:start-1]
        
        cop_acc = np.zeros((int(len(true_cop)*length_factor),2))
        cop_acc[:start-1] = compute_discrete_derivative(true_com, 2, dt)[:start-1]      
        
        self.generative_results = {
                            "forces_output_cop":{},
                            "forces_output_com":{}
                            }

        for t in range(start,len(cop)+1): 

            ##################
            ## Generate COP ##
            ##################

            forces_dict_cop = get_forces_dict_cop(cop=cop, cop_spd=cop_spd, com=com, com_spd=com_spd, frequency=frequency)

            estimated_forces_cop = np.array([0,0])

            for f in self.list_forces_cop:
   
                coefs_force = np.array([self.fit_cop_results[axis]["coefs"][f] for axis in range(2)])

                predicted_force = forces_dict_cop[f][t-1] * coefs_force
                    
                estimated_forces_cop = estimated_forces_cop + predicted_force
                
                self.generative_results["forces_output_cop"][f] = predicted_force

            # sigma * dBt/dt ~ N(0,(sigma²/dt))
            noise = np.random.randn(2) * np.array([self.fit_cop_results[axis]["coefs"]["sigma"] for axis in range(2)]) * (1/np.sqrt(dt))
        
            self.generative_results["forces_output_cop"]["perturbation"] = noise

            estimated_forces_cop = estimated_forces_cop + noise

            cop_acc[t-1] = estimated_forces_cop
            dspd_cop = estimated_forces_cop * dt
            dpos_cop = cop_spd[t-1]*dt    
            
            if t<len(cop):
                cop_spd[t] = cop_spd[t-1] + dspd_cop         

                cop[t]=cop[t-1]+dpos_cop

            ##################
            ## Generate COM ##
            ##################
        
            estimated_forces_com = np.array([0,0])

            for f in self.list_forces_com:

                forces_dict_com = get_forces_dict_com(cop=cop, cop_spd=cop_spd, cop_acc=cop_acc, com=com, com_spd=com_spd, com_acc=com_acc)
                
                coefs_force = np.array([self.fit_com_results[axis]["coefs"][f] for axis in range(2)])

                predicted_force = forces_dict_com[f][t-1] * coefs_force
     
                estimated_forces_com = estimated_forces_com + predicted_force
 
                self.generative_results["forces_output_com"][f] = predicted_force
    
            # sigma * dBt/dt ~ N(0,(sigma²/dt))
            noise = np.random.randn(2) * np.array([self.fit_com_results[axis]["coefs"]["sigma"] for axis in range(2)]) * (1/np.sqrt(dt))
            self.generative_results["forces_output_com"]["perturbation"] = noise

            estimated_forces_com = estimated_forces_com + noise
     
            com_acc[t-1] = estimated_forces_com
            dspd_com = (estimated_forces_com)*dt

            dpos_com = com_spd[t-1]*dt 
            
            if t<len(cop):
                com_spd[t] = com_spd[t-1] + dspd_com 
                
                com[t]=com[t-1]+dpos_com

        return cop, com

