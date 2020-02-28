#!/usr/bin/env python3

import numpy as np
import os
from glob import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


def get_fuel_demands(start, end, steps):
    fuel_demands = [0., start,]
    while True:
        if fuel_demands[-1] > end:
            break
        fuel_demands.append(round(fuel_demands[-1]*steps,5))
    return fuel_demands



def plot_peak_demand_system(out_file_name, techs, save_dir, save_name, ldc=False):

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
    if ldc:
        dfs = dfs.sort_values('demand (kW)', ascending=False)
        ax.set_xlabel('Fraction of time')
        xs = np.linspace(0, 1, len(dfs.index))
    else:
        xs = dfs['time (hr)']
    cap_nuke = np.max(df['dispatch nuclear (kW)'])

    bottom = np.zeros(len(xs))
    if 'solar' in techs:
        ax.fill_between(xs, bottom, dfs['dispatch solar (kW)'], color='yellow', alpha=0.2, label='Power from solar')
        bottom += dfs['dispatch solar (kW)'].values
    if 'wind' in techs:
        ax.fill_between(xs, bottom, dfs['dispatch wind (kW)'], color='blue', alpha=0.2, label='Power from wind')
        bottom += dfs['dispatch wind (kW)'].values
    if 'nuclear' in techs:
        ax.fill_between(xs, bottom, dfs['dispatch nuclear (kW)'], color='tan', alpha=0.5, label='Power from nuclear')
        bottom += dfs['dispatch nuclear (kW)'].values
    if 'storage' in techs:
        ax.fill_between(xs, bottom, dfs['dispatch from storage (kW)'], color='magenta', alpha=0.2, label='Power from storage')
        bottom += dfs['dispatch from storage (kW)'].values

    bottom2 = np.zeros(len(xs))
    ax.fill_between(xs, 0., dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)'], facecolor='none', edgecolor='black', hatch='/////', label='Power to demand')
    bottom2 += dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)']
    ax.fill_between(xs, bottom2, bottom2 + dfs['dispatch to fuel h2 storage (kW)'], facecolor='none', edgecolor='purple', hatch='xxxxx', label='Power to electrolyzer')
    bottom2 += dfs['dispatch to fuel h2 storage (kW)']
    if 'storage' in techs:
        ax.fill_between(xs, bottom2, bottom2 + dfs['dispatch to storage (kW)'], facecolor='none', edgecolor='magenta', hatch='xxx', label='Power to storage')
        bottom2 += dfs['dispatch to storage (kW)']
    ax.fill_between(xs, dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)'], dfs['demand (kW)'], color='red', alpha=0.8, label='Unmet demand')
    #ax.fill_between(xs, bottom2, bottom2 + dfs['dispatch unmet demand (kW)'], facecolor='none', edgecolor='red', hatch='|||||', label='Unmet demand')


    ax.plot(xs, dfs['demand (kW)'], 'k-', linewidth=2, label='Demand')

    ax.plot(xs, np.ones(len(xs))*cap_nuke, 'r-', linewidth=1, label='Capacity Nuclear')

    ax.set_ylim(0, 2.25)#cap_nuke*1.3)
    if ldc:
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(lo, hi-1)
    plt.tight_layout()
    plt.legend(ncol=2)
    fig.savefig(f'{save_dir}/{save_name}.pdf')



if '__main__' in __name__:

    save_dir = 'out_plots'
    date = '20200210_v1'
    base = 'Output_Data/'

    cases = {
            "Case1_Nuclear" : ['nuclear',], 
            #"Case0_NuclearFlatDemand", 
            #"Case2_NuclearStorage", 
            #"Case3_WindStorage", 
            #"Case4_SolarStorage", 
            #"Case5_WindSolarStorage", 
            #"Case6_NuclearWindSolarStorage",
    }

    possible_dem_vals = get_fuel_demands(0.01, 10, 1.2) # start, end, steps

    tgt_fuel_dems = [
            '0.012',
            '0.02489',
            '0.05161',
            '0.07432',
            '0.10702',
            '0.2219',
            #'0.31954',
            #'1.14497',
            #'10.20862',
    ]

    for case, techs in cases.items():

        print(f"Plotting for {case} with techs {techs}")
        for idx, dem in enumerate(reversed(possible_dem_vals)):
            if str(dem) not in tgt_fuel_dems:
                continue
            out_file_name = f'{base}fuel_test_{date}_{case}/'
            out_file_name += f'fuel_test_{date}_{case}_Run_{idx:03d}_fuelD{dem}kWh_solarX-1_windX-1_nukeX1_battX-1_electoX1_elecEffX1.csv'
            save_name = f"{case}_fuelD{dem}kWh"
            plot_peak_demand_system(out_file_name, techs, save_dir, save_name)
            print("Now LDC")
            plot_peak_demand_system(out_file_name, techs, save_dir, 'ldc_'+save_name, True)



























