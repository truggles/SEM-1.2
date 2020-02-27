#!/usr/bin/env python3

import numpy as np
import os
from glob import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime



def plot_peak_demand_system(out_file_name, save_dir, save_name):

    # Open out file as df
    df = pd.read_csv(out_file_name)

    # Find the peak hour
    peak_idxs = df[ df['demand (kW)'] == np.max(df['demand (kW)'])].index
    assert(len(peak_idxs) == 1), f"\n\nThere are multiple instances of peak demand value, {peak_idxs}\n\n"
    peak_idx = peak_idxs[0]
    max_demand = np.max(df['demand (kW)'])

    plt.close()
    matplotlib.rcParams["figure.figsize"] = (6, 5)
    fig, ax = plt.subplots()
    ax.set_xlabel('Hours')
    ax.set_ylabel('Power (kW)')
    lo = peak_idx - 24*4
    hi = peak_idx + 24*3
    dfs = df.iloc[lo:hi]
    xs = dfs['time (hr)']
    cap_nuke = np.max(df['dispatch nuclear (kW)'])

    ax.fill_between(xs, 0., dfs['dispatch nuclear (kW)'], color='red', alpha=0.2, label='Nuclear Dispatch')
    ax.fill_between(xs, dfs['dispatch nuclear (kW)'] - dfs['dispatch to fuel h2 storage (kW)'], dfs['dispatch nuclear (kW)'], facecolor='none', edgecolor='blue', hatch='xxx', label='Electrolyzer Consumption')

    ax.plot(xs, dfs['demand (kW)'], 'k-', linewidth=2, label='Demand')
    ax.plot(xs, np.ones(len(xs))*cap_nuke, 'r-', linewidth=1, label='Capacity Nuclear')

    ax.set_ylim(0, max_demand*1.5)
    ax.set_xlim(lo, hi)
    plt.tight_layout()
    plt.legend(ncol=2)
    fig.savefig(f'{save_dir}/{save_name}.pdf')



if '__main__' in __name__:

    out_file_name = 'Output_Data/fuel_test_20200210_v1_Case1_Nuclear/'
    out_file_name += 'fuel_test_20200210_v1_Case1_Nuclear_Run_034_fuelD0.02074kWh_solarX-1_windX-1_nukeX1_battX-1_electoX1_elecEffX1.csv'
    save_dir = 'out_plots'
    save_name = 'test1'
    plot_peak_demand_system(out_file_name, save_dir, save_name)

