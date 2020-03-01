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


def plot_peak_demand_grid(out_file_dir, out_file_name, tgt_fuel_dems, case, techs, save_dir, set_max=-1, ldc=False):

    plt.close()
    matplotlib.rcParams["figure.figsize"] = (7, 9)
    matplotlib.rcParams["font.size"] = 9
    matplotlib.rcParams['lines.linewidth'] = 0.5
    matplotlib.rcParams['hatch.linewidth'] = 0.5
    axs = []
    for j, dem in enumerate(reversed(tgt_fuel_dems)):
        i = len(tgt_fuel_dems) - j + 1
        this_file = out_file_dir + out_file_name.replace('fuelDXXX', f'fuelD{dem}')
        if j == 0:
            axs.append( plt.subplot(7, 2, 2 * i - 1) )
            axs[-1].set_xlabel('Hours')
        else:
            axs.append( plt.subplot(7, 2, 2 * i - 1, sharex=axs[0]) )
            plt.setp(axs[-1].get_xticklabels(), visible=False)
        axs[-1].set_ylabel('Power (kW)')

        plot_peak_demand_system(axs[-1], this_file, info[0], save_dir, info[1])
        
        if j == 0:
            axs.append( plt.subplot(7, 2, 2 * i, sharey=axs[-1]) )
            plt.setp(axs[-1].get_yticklabels(), visible=False)
            axs[-1].set_xlabel('Fraction of time')
        else:
            axs.append( plt.subplot(7, 2, 2 * i, sharey=axs[-1], sharex=axs[1]) )
            plt.setp(axs[-1].get_yticklabels(), visible=False)
            plt.setp(axs[-1].get_xticklabels(), visible=False)
        plot_peak_demand_system(axs[-1], this_file, info[0], save_dir, info[1], True)

    plt.tight_layout()
    horiz = -1 if case != 'Case6_NuclearWindSolarStorage' else -1.1
    plt.legend(ncol=3, loc='upper left', bbox_to_anchor=(horiz, 2))
    fig = plt.gcf()
    fig.savefig(f"{save_dir}/{case}.png")
    fig.savefig(f"{save_dir}/pdf/{case}.pdf")


def plot_peak_demand_system(ax, out_file_name, techs, save_dir, set_max=-1, ldc=False):

    # Open out file as df
    full_file_name = glob(out_file_name)
    assert(len(full_file_name) == 1)
    df = pd.read_csv(full_file_name[0])

    # Find the peak hour
    peak_idxs = df[ df['demand (kW)'] == np.max(df['demand (kW)'])].index
    assert(len(peak_idxs) == 1), f"\n\nThere are multiple instances of peak demand value, {peak_idxs}\n\n"
    peak_idx = peak_idxs[0]
    max_demand = np.max(df['demand (kW)'])

    lo = peak_idx - 24*4
    hi = peak_idx + 24*3
    dfs = df.iloc[lo:hi]
    if ldc:
        dfs = dfs.sort_values('demand (kW)', ascending=False)
        #ax.set_xlabel('Fraction of time')
        xs = np.linspace(0, 1, len(dfs.index))
    else:
        xs = dfs['time (hr)']
    cap_nuke = np.max(df['dispatch nuclear (kW)'])

    fblw = 0.25
    bottom = np.zeros(len(xs))
    if 'solar' in techs:
        ax.fill_between(xs, bottom, bottom + dfs['dispatch solar (kW)'], color='yellow', alpha=0.4, label='Power from solar', lw=fblw)
        bottom += dfs['dispatch solar (kW)'].values
    if 'wind' in techs:
        ax.fill_between(xs, bottom, bottom + dfs['dispatch wind (kW)'], color='blue', alpha=0.2, label='Power from wind', lw=fblw)
        bottom += dfs['dispatch wind (kW)'].values
    if 'nuclear' in techs:
        ax.fill_between(xs, bottom, bottom + dfs['dispatch nuclear (kW)'], color='tan', alpha=0.5, label='Power from nuclear', lw=fblw)
        bottom += dfs['dispatch nuclear (kW)'].values
    if 'storage' in techs:
        ax.fill_between(xs, bottom, bottom + dfs['dispatch from storage (kW)'], color='magenta', alpha=0.2, label='Power from storage', lw=fblw)
        bottom += dfs['dispatch from storage (kW)'].values

    bottom2 = np.zeros(len(xs))
    ax.fill_between(xs, 0., dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)'], facecolor='none', edgecolor='black', hatch='/////', label='Power to demand', lw=fblw)
    bottom2 += dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)']
    ax.fill_between(xs, bottom2, bottom2 + dfs['dispatch to fuel h2 storage (kW)'], facecolor='none', edgecolor='green', hatch='xxxxx', label='Power to electrolyzer', lw=fblw)
    bottom2 += dfs['dispatch to fuel h2 storage (kW)']
    if 'storage' in techs:
        ax.fill_between(xs, bottom2, bottom2 + dfs['dispatch to storage (kW)'], facecolor='none', edgecolor='magenta', hatch='xxxxx', label='Power to storage', lw=fblw)
        bottom2 += dfs['dispatch to storage (kW)']
    ax.fill_between(xs, dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)'], dfs['demand (kW)'], color='red', alpha=0.8, label='Unmet demand', lw=fblw)
    #ax.fill_between(xs, bottom2, bottom2 + dfs['dispatch unmet demand (kW)'], facecolor='none', edgecolor='red', hatch='|||||', label='Unmet demand')


    ax.plot(xs, dfs['demand (kW)'], 'k-', linewidth=fblw, label='Demand')

    bottom3 = np.zeros(len(xs))
    lab = 'Gen.'
    if 'solar' in techs:
        lab += ' Solar'
        ax.plot(xs, bottom3 + dfs['dispatch solar (kW)'] + dfs['cutailment solar (kW)'], 'y-', linewidth=fblw, label=lab)
        bottom3 += dfs['dispatch solar (kW)'] + dfs['cutailment solar (kW)']
        lab = lab.replace('Solar', 'Sol.')
    if 'wind' in techs:
        if 'solar' in techs:
            lab += ' + Wind'
        else:
            lab += ' Wind'
        ax.plot(xs, bottom3 + dfs['dispatch wind (kW)'] + dfs['cutailment wind (kW)'], 'b-', linewidth=fblw, label=lab)
        bottom3 += dfs['dispatch wind (kW)'] + dfs['cutailment wind (kW)']
    if 'nuclear' in techs:
        if 'solar' in techs or 'wind' in techs:
            lab += ' + Nuclear Cap.'
        else:
            lab = 'Capacity Nuclear'
        ax.plot(xs, bottom3 + np.ones(len(xs))*cap_nuke, 'r-', linewidth=fblw, label=lab)

    if set_max == -1:
        set_max = ax.get_ylim()[1]
    ax.set_ylim(0, set_max)
    if ldc:
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(lo, hi-1)




if '__main__' in __name__:

    save_dir = 'out_plots'
    date = '20200228_v1'
    base = 'Output_Data/'
    cases = {
            #"Case0_NuclearFlatDemand" : [['nuclear',], -1],
            "Case1_Nuclear" : [['nuclear',], 2],
            "Case2_NuclearStorage" : [['nuclear','storage'], 2],
            "Case3_WindStorage" : [['wind', 'storage'], 6],
            "Case4_SolarStorage" : [['solar', 'storage'], 5],
            "Case5_WindSolarStorage" : [['wind', 'solar', 'storage'], 3.5],
            "Case6_NuclearWindSolarStorage" : [['nuclear', 'wind', 'solar', 'storage'], 3],
    }

    #possible_dem_vals = get_fuel_demands(0.01, 10, 1.2) # start, end, steps

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

    for case, info in cases.items():

        print(f"Plotting for {case} with techs {info[0]} and max = {info[1]}")
        #for idx, dem in enumerate(reversed(possible_dem_vals)):
        #    if str(dem) not in tgt_fuel_dems:
        #        continue
        out_file_dir = f'{base}fuel_test_{date}_{case}/'
        out_file_name = f'fuel_test_{date}_{case}_Run_*_fuelDXXXkWh_*.csv'
        plot_peak_demand_grid(out_file_dir, out_file_name, tgt_fuel_dems, case, info[0], save_dir, info[1], True)




