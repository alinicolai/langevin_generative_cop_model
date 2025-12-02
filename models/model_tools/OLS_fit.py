

import numpy as np

from statsmodels.regression.linear_model import OLS




def fit_OLS(forces_dict, Y, dt, list_forces, forces_delay={}, dim=2):

    fit_results = [] 

    if len(forces_delay)==0:
        max_delay = 0
    else:
        max_delay = max(forces_delay.values())
        

    if dim==2:
        for axis in range(2):
    
            forces_axis = np.array([forces_dict[f][max_delay-forces_delay[f]:-forces_delay[f],axis] if f in forces_delay and forces_delay[f]>0 else forces_dict[f][max_delay:,axis] for f in list_forces]).T
    
            OLS_results =  OLS_1D(list_forces=list_forces, forces=forces_axis, Y=Y[max_delay:,axis], dt=dt)
    
            fit_results.append(OLS_results)
            
    elif dim==1:
        
        forces = np.array([forces_dict[f][max_delay-forces_delay[f]:-forces_delay[f]] if f in forces_delay and forces_delay[f]>0 else forces_dict[f][max_delay:] for f in list_forces]).T

        fit_results =  OLS_1D(list_forces=list_forces, forces=forces, Y=Y[max_delay:], dt=dt)


    return fit_results



def OLS_1D(list_forces, forces, Y, dt, forces_to_ignore=[]):

    coefs = {}
    confidence_interval = {}
    
    Y = Y.reshape(-1,1)
    solution = np.linalg.pinv(forces) @ Y
    estimated_Y = forces @ solution
    error = np.var(Y - estimated_Y)

    noise_param = np.sqrt(error*dt)
    
    scr = np.sum((Y-estimated_Y)**2)
    sct = np.sum((Y-np.mean(Y))**2)
    R2 = 1-(scr/sct) 

    rmse = np.sqrt( np.mean((Y-estimated_Y)**2) )

    n = len(Y)
    k = forces.shape[1]
    adj_r2 = 1 - (1-R2) * ((n-1)/(n-k-1))

    for i, f in enumerate(list_forces):
        coefs[f]=float(solution[i])
        
    coefs["sigma"] = noise_param

    ols = OLS(endog=Y, exog=forces)
    results = ols.fit()

    fit_infos = {
            
                "coefs":coefs,
                "fitted":Y[:,0],
                "prediction":estimated_Y,
                "confidence_intervals":confidence_interval,
                "r2":R2,
                "adj_r2":adj_r2,
                "rmse":rmse,
                "residuals":Y-estimated_Y,
                "forces":forces,
                "forces_dict":{list_forces[i]:forces[:,i] for i in range(len(list_forces))},
                "AIC":results.aic,
                "loglikelihood":results.llf,
                "BIC":results.bic,
            
                }

    return fit_infos

