#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 23:54:17 2025

@author: alice
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
font = {'family' : 'serif'} 
matplotlib.rc('font', **font)

def plot_langevin_model(true_cop, generated_cop, frequency, savepath, model_name,
                        estimated_LPFA_com=None, generated_com=None):

    fig, ax = plt.subplots(2, 2, figsize=(13,5))  # plus de sharey=True

    time = np.arange(len(true_cop)) / frequency
    generated_time = np.arange(len(generated_cop)) / frequency

    for i, axis in enumerate(["ML", "AP"]):

        ax[0,i].plot(time, true_cop[:,i], color="darkblue", label='Recorded CoP')
        if estimated_LPFA_com is not None and generated_com is not None:
            ax[0,i].plot(time, estimated_LPFA_com[:,i], color="orangered", label='Estimated LPFA CoM')

        ax[1,i].plot(generated_time, generated_cop[:,i], color='darkblue', label='Simulated CoP')
        if generated_com is not None:
            ax[1,i].plot(generated_time, generated_com[:,i], color='orangered', label='Simulated CoM')

        top_ylim = ax[0,i].get_ylim()
        bottom_ylim = ax[1,i].get_ylim()
        global_ylim = [min(top_ylim[0], bottom_ylim[0]), max(top_ylim[1], bottom_ylim[1])]
        # add margin
        margin = 0.15 * (global_ylim[1]-global_ylim[0])
        global_ylim = [global_ylim[0]-margin, global_ylim[1]+margin]

        ax[0,i].set_ylim(global_ylim)
        ax[1,i].set_ylim(global_ylim)

        for axj in [ax[0,i], ax[1,i]]:
            axj.plot([time[0], time[-1]], [0,0], linestyle="--", color="grey")
            axj.plot([time[3], time[3]], global_ylim, linestyle="--", color="grey")

        ax[0,i].set_title(f"Preprocessed CoP trajectory ({axis} axis)", fontsize=10)
        ax[0,i].set_xlabel("Time (s)", fontsize=10)
        ax[0,i].set_ylabel("Position (cm)", fontsize=10)

        ax[1,i].set_title(f"Model simulated CoP trajectory ({axis} axis)", fontsize=10)
        ax[1,i].set_xlabel("Time (s)", fontsize=10)
        ax[1,i].set_ylabel("Position (cm)", fontsize=10)

        lines, labels = ax[0,i].get_legend_handles_labels()
        if labels:
            ax[0,i].legend(fontsize=8, loc='upper right')
        lines, labels = ax[1,i].get_legend_handles_labels()
        if labels:
            ax[1,i].legend(fontsize=8, loc='upper right')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.savefig(os.path.join(savepath, model_name+'_model_plot.pdf'))
    plt.close(fig)


    
