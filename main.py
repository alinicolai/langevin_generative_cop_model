
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 1 22:19:42 2025

@author: Alice Nicolaï
"""

import numpy as np
import pandas
import os

from resampling.swarii import SWARII
from models.com_approximation import compute_com_from_cop_LPFA
from models.fit import fit_langevin_model
from models.plot import plot_langevin_model

datapath = 'data'
data_filepath = os.path.join('data', 'example_cop_recording', 'example_cop_table.csv')

# Load example data file
data_table = pandas.read_csv(data_filepath) 
time = data_table.iloc[:,0].values.reshape(-1,1)
cop_ml = data_table.iloc[:,1].values.reshape(-1,1)
cop_ap = data_table.iloc[:,2].values.reshape(-1,1)
cop = np.concatenate([cop_ml, cop_ap], axis=1)

# Resample CoP at 20 Hz and filter noise using the Swarii algorithm
frequency = 20
time_cop = np.concatenate([time, cop], axis=1)
preprocessed_cop = SWARII.resample(data=time_cop, 
                                   desired_frequency=frequency)

# Center the signal
preprocessed_cop = preprocessed_cop - np.mean(preprocessed_cop)

# Compute estimated CoP using the LPFA method  
estimated_LPFA_com = compute_com_from_cop_LPFA(preprocessed_cop, frequency=frequency)

# Remove first and last points to avoid edge effects
estimated_LPFA_com = estimated_LPFA_com[frequency:-frequency]
preprocessed_cop = preprocessed_cop[frequency:-frequency]

model_name = 'total_recall'

cop_params_table, generated_cop, generated_com = fit_langevin_model(model_name=model_name, 
                                                                     cop=preprocessed_cop, 
                                                                     frequency=frequency,
                                                                     estimated_LPFA_com=estimated_LPFA_com)

cop_params_table.to_csv(model_name+'_fitted_parameters_table.csv', 
                        index=False)

plot_langevin_model(true_cop=preprocessed_cop, 
                    generated_cop=generated_cop, 
                    frequency=frequency, 
                    savepath='',
                    model_name=model_name,
                    estimated_LPFA_com=estimated_LPFA_com, 
                    generated_com=generated_com)