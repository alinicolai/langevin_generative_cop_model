
# import numpy as np
# from scipy.signal import convolve2d



def get_forces_dict_cop(cop, cop_spd, frequency, com=None, com_spd=None):
    
    forces_dict = {"damping" : -cop_spd, 
                  "global_recall" : -(cop),
                  }
    
    if com is not None:
        forces_dict.update({
                            "local_position_push": (com),
                            "local_velocity_push": com_spd,     
                            "local_recall": (com-cop),
                            })      
    
    return forces_dict


def get_forces_dict_com(cop, com, cop_spd, cop_acc, com_spd, com_acc):
    

    forces_dict = {
                   "pendulum": com-cop,

                  }
    
    return forces_dict
 
    