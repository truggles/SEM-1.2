#!/usr/bin/env python3

import numpy as np
import os
from glob import glob
import pandas as pd
import matplotlib
from datetime import datetime



def plot_peak_demand_system(out_file_name, save_dir, save_name):

    # Open out file as df
    df = pd.read_csv(out_file_name)
    print(df.head())

    # Find the peak hour
    peak_idxs = df[ df['demand (kW)'] == np.max(df['demand (kW)'])].index
    assert(len(peak_idxs) == 1), f"\n\nThere are multiple instances of peak demand value, {peak_idxs}\n\n"
    peak_idx = peak_idxs[0]
    print(peak_idx)
    print(peak_idx, df.iloc[peak_idx-3:peak_idx+3])

    return

    plt.close()
    matplotlib.rcParams["figure.figsize"] = (6.4, 4.8)
    fig, ax = plt.subplots()
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])
    plt.title(kwargs['title'])

    ax.fill_between(kwargs['x_vals'], 0., kwargs['nuclear'], color='red', label=f'nuclear {kwargs["legend_app"]}')
    if 'renewables' in kwargs.keys():
        ax.fill_between(kwargs['x_vals'], kwargs['nuclear'], kwargs['nuclear']+kwargs['renewables'], color='green', label=f'renewables {kwargs["legend_app"]}')
    else:
        ax.fill_between(kwargs['x_vals'], kwargs['nuclear'], kwargs['nuclear']+kwargs['wind'], color='blue', label=f'wind {kwargs["legend_app"]}')
        ax.fill_between(kwargs['x_vals'], kwargs['nuclear']+kwargs['wind'], kwargs['nuclear']+kwargs['wind']+kwargs['solar'], color='yellow', label=f'solar {kwargs["legend_app"]}')


    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig.savefig(f'{save_dir}/{save_name}.png')



if '__main__' in __name__:

    out_file_name = 'Output_Data/fuel_test_20200210_v1_Case1_Nuclear/'
    out_file_name += 'fuel_test_20200210_v1_Case1_Nuclear_Run_034_fuelD0.02074kWh_solarX-1_windX-1_nukeX1_battX-1_electoX1_elecEffX1.csv'
    save_dir = 'out_plots'
    save_name = 'test1'
    plot_peak_demand_system(out_file_name, save_dir, save_name)

