#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 1 22:37:14 2025

@author: Alice Nicolaï
"""

import pandas

from models.model_cop_com import ModelCoupledCoPCoM
from models.model_cop import ModelCoP

def fit_langevin_model(model_name, cop, frequency, estimated_LPFA_com=None):
    
    # model name must be either 'global_recall' or 'total_recall'

    if model_name=="total_recall":

        list_forces_cop = ['global_recall', 'damping', 'local_position_push', 'local_velocity_push']
        list_forces_com = ["pendulum"]   
        model_instance = ModelCoupledCoPCoM(list_forces_cop=list_forces_cop,
                                            list_forces_com=list_forces_com)         
            
        model_instance.fit(cop=cop, com=estimated_LPFA_com, frequency=frequency)
        generated_cop, generated_com = model_instance.generate(true_cop=cop, 
                                                               true_com=estimated_LPFA_com, 
                                                               frequency=frequency)
    
    elif model_name=="global_recall":
        
        list_forces_cop = ["global_recall", "damping"]
        model_instance = ModelCoP(list_forces_cop = list_forces_cop)
                        
        model_instance.fit(cop=cop, frequency=frequency)
        generated_cop = model_instance.generate(true_cop=cop, frequency=frequency)
                
        generated_com = None
        
    fitted_ML_cop_parameters = model_instance.fit_cop_results[0]["coefs"]
    fitted_AP_cop_parameters = model_instance.fit_cop_results[1]["coefs"]
    
    rows = []
    
    for name, value in fitted_ML_cop_parameters.items():
        rows.append([f"{name}, ML", float(value)])
    
    for name, value in fitted_AP_cop_parameters.items():
        rows.append([f"{name}, AP", float(value)])
    
    fitted_parameters_table = pandas.DataFrame(rows, 
                                               columns=["Parameter name", "Value"])
    

    return fitted_parameters_table, generated_cop, generated_com
        
