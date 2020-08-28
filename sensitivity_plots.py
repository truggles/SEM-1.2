#!/usr/bin/env python3

import numpy as np
import csv
import subprocess
import os
from glob import glob
from shutil import copy2, move
from collections import OrderedDict
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import copy
from helpers import get_fuel_demands, get_fuel_fractions
from end_use_fractions import add_detailed_results
from analytic_fuels import kWh_to_GGE, kWh_LHV_per_kg_H2
matplotlib.rcParams.update({'font.size': 12.5})
matplotlib.rcParams.update({'lines.linewidth': 2})


def marker_list():
    return ['o', 'v', '^', 's', 'P', '*', 'H', 'X', 'd', '<', '>']

# Based on common color blindness
# https://www.nature.com/articles/nmeth.1618/figures/2
# Skip black and reserve it for other specific lines
def color_list():
    l = [
            np.array([230, 159, 0]), # orange
            np.array([86, 180, 233]), # Sky blue
            np.array([0, 158, 115]), # Bluish green
            np.array([240, 228, 66]), # Yellow
            np.array([0, 114, 178]), # Blue
            np.array([213, 94, 0]), # Vermillion
            np.array([204, 121, 167]), # Reddish purple
    ]
    return [i/255. for i in l]


# Poorly written, the args are all required and are below.
#save_name, save_dir
def costs_plot_alt(tgt_shift, var='fuel demand (kWh)', **kwargs):
    cases = {
            0 : 'Case7_NatGasCCS',
            1 : 'Case9_NatGasCCSWindSolarStorage',
            2 : 'Case5_WindSolarStorage',
            }

    dfs = kwargs['dfs']

    colors = color_list()
    plt.close()
    #if 'Case7' in kwargs['save_name']:
    #    y_max = 0.08
    #if 'Case5' in kwargs['save_name']:
    #    y_max = 0.12
    #if 'Case9' in kwargs['save_name']:
    #    y_max = 0.08
    y_max = 0.12
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(11,6), sharey=True)

    

    line_types = ['solid', (0, (1, 1)), (0, (2, 1)), (0, (3, 3)), (0, (5, 5)),
            (0, (3, 1, 1, 1)),
            (0, (3, 1, 1, 1, 1, 1)),
            ]

    for i in range(3):
        axs[i].set_xlim(0.0, 1.0)
        axs[i].yaxis.set_ticks_position('both')
        cnt = 0
        for shift, df in dfs[cases[i]].items():
            if shift != 'nominal' and tgt_shift not in shift:
                continue
            # Electricity cost
            axs[i].plot(df[var], df['mean price ($/kWh)'], linestyle=line_types[cnt], label=f'electric load: {shift}', color=colors[0])
            cnt += 1

        cnt = 0
        for shift, df in dfs[cases[i]].items():
            if not (shift == 'nominal' or tgt_shift in shift):
                continue
            tot_eff_fuel_process = EFFICIENCY_FUEL_ELECTROLYZER * EFFICIENCY_FUEL_CHEM_CONVERSION
            # Add the cost of electric power to the fuel load
            axs[i].plot(df[var], df['fuel_load_cost'], linestyle=line_types[cnt], label=f'flexible load: {shift}', color=colors[1])
            cnt += 1

        cnt = 0
        for shift, df in dfs[cases[i]].items():
            if not (shift == 'nominal' or tgt_shift in shift):
                continue
            avg_elec_cost = df['mean price ($/kWh)'] * (1. - df[var]) + df['fuel_load_cost'] * df[var]
            axs[i].plot(df[var], avg_elec_cost, linestyle=line_types[cnt], label=f'mean cost: {shift}', color='black')
            cnt += 1


    axs[0].set_ylim(0, y_max)
    axs[0].set_ylabel(r'cost (\$/kWh$_{e}$)')
    axs[1].set_xlabel(kwargs['x_label'])

    horiz = 1.07
    vert = 1
    horiz = 0.4
    vert = 1.3
    axs[1].legend(ncol=3, loc='upper center', framealpha = 1.0, bbox_to_anchor=(horiz, vert))
    #plt.tight_layout()
    plt.subplots_adjust(top=.8, left=.1, bottom=.13, right=.95)


    fig.savefig(f'{kwargs["save_dir"]}{kwargs["save_name"]}.png')
    #fig.savefig(f'{kwargs["save_dir"]}{kwargs["save_name"]}.pdf')
    print(f'{kwargs["save_dir"]}{kwargs["save_name"]}.png')



def prep_csv(case, f_name_base):
    fixed = 'natgas_ccs'
    if case == 'Case5_WindSolarStorage':
        fixed = 'nuclear'
    df = pd.read_csv(f'results_sens/{f_name_base}.csv', index_col=False)
    df = df.sort_values('fuel demand (kWh)', axis=0)
    df = df.reset_index()
    df['fuel load / available power'] = df['dispatch to fuel h2 storage (kW)'] / (
            df['dispatch wind (kW)'] + df['curtailment wind (kW)'] + 
            df['dispatch solar (kW)'] + df['curtailment solar (kW)'] + 
            df[f'dispatch {fixed} (kW)'] + df[f'curtailment {fixed} (kW)']
            )
    df['fuel load / total load'] = df['dispatch to fuel h2 storage (kW)'] / (
            df['dispatch to fuel h2 storage (kW)'] + 1. # electric power demand = 1 
            )

    # Stacked components
    f_elec = df['fixed cost fuel electrolyzer ($/kW/h)'] * df['capacity fuel electrolyzer (kW)'] / df['fuel demand (kWh)']
    f_chem = df['fixed cost fuel chem plant ($/kW/h)'] * df['capacity fuel chem plant (kW)'] / df['fuel demand (kWh)']
    f_store = df['fixed cost fuel h2 storage ($/kWh/h)'] * df['capacity fuel h2 storage (kWh)'] / df['fuel demand (kWh)']
    v_chem = df['var cost fuel chem plant ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_CONVERSION / df['fuel demand (kWh)']
    v_co2 = df['var cost fuel co2 ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_CONVERSION / df['fuel demand (kWh)']
    f_tot = f_elec+f_chem+f_store+v_chem+v_co2
    tot_eff_fuel_process = EFFICIENCY_FUEL_ELECTROLYZER * EFFICIENCY_FUEL_CHEM_CONVERSION
    df['fuel_load_cost'] = (df['fuel price ($/kWh)'] - f_tot) * tot_eff_fuel_process

    for i in range(len(df.index)):
        if df.loc[i, 'fuel demand (kWh)'] == 0.0 or df.loc[i, 'mean demand (kW)'] == 0.0:
            print(f"Dropping idx {i}: fuel {df.loc[i, 'fuel demand (kWh)']} elec {df.loc[i, 'mean demand (kW)']}")
            df = df.drop([i,])
    df = df.reset_index()

    df.to_csv(f'results_sens/{f_name_base}_tmp.csv', index=False)



# HERE


# Efficiencies so I don't have to pull them from the cfgs for the moment, FIXME
EFFICIENCY_FUEL_ELECTROLYZER=0.607 # Updated 4 March 2020 based on new values
EFFICIENCY_FUEL_CHEM_CONVERSION=0.682

date = '20200826'
f_map = {
        'Case7_NatGasCCS' : {
            'version' : 'v2',
            'shifts' : {
                'nominal' : '',
                'electrolyzer 75%' : 'EL0.75',
                'electrolyzer 50%' : 'EL0.5',
                'natGas+CCS 75%' : 'NG0.75',
                'natGas+CCS 50%' : 'NG0.5',
            },
        },
        'Case5_WindSolarStorage' : {
            'version' : 'v3',
            'shifts' : {
                'nominal' : '',
                'electrolyzer 75%' : 'EL0.75',
                'electrolyzer 50%' : 'EL0.5',
                'wind 75%' : 'WIND0.75',
                'wind 50%' : 'WIND0.5',
                'solar 75%' : 'SOL0.75',
                'solar 50%' : 'SOL0.5',
            },
        },
        'Case9_NatGasCCSWindSolarStorage' : {
            'version' : 'v4',
            'shifts' : {
                'nominal' : '',
                'electrolyzer 75%' : 'EL0.75',
                'electrolyzer 50%' : 'EL0.5',
                'wind 75%' : 'WIND0.75',
                'wind 50%' : 'WIND0.5',
                'solar 75%' : 'SOL0.75',
                'solar 50%' : 'SOL0.5',
                'natGas+CCS 75%' : 'NG0.75',
                'natGas+CCS 50%' : 'NG0.5',
            },
        },
}
prep_CSV = True
prep_CSV = False

plot = True
#plot = False

if prep_CSV:
    print("\nPreparing the csvs")
    for case, info in f_map.items():
        print(f"\n{case}")
        for shift, tag in info['shifts'].items():
            print(f"Shift: {shift}, tag {tag}")
            f_name_base = f"Results_fuel_test_{date}_{info['version']}{tag}_{case}_1_1"
            prep_csv(case, f_name_base)

    
if plot:

    print("\nPreparing the csvs")
    m = {
            'x_label' : 'flexible load (kW) / total load (kW)',
            'x_lim' : [0., 1.],
            'x_type' : 'linear',
    }

    var = 'fuel load / total load'

    df_map = OrderedDict()
    for case, info in f_map.items():
        print(f"\n{case}")
        df_map[case] = OrderedDict()
        for shift, tag in info['shifts'].items():
            print(f"Shift: {shift}, tag {tag}")
            f_name_base = f"Results_fuel_test_{date}_{info['version']}{tag}_{case}_1_1"
            df = pd.read_csv(f'results_sens/{f_name_base}_tmp.csv', index_col=False)
            df_map[case][shift] = df



    save_dir = f'./plots_{date}_sensitivity/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


    kwargs = {}
    kwargs['dfs'] = df_map
    kwargs['save_dir'] = save_dir
    kwargs['stacked_min'] = m['x_lim'][0]
    kwargs['stacked_max'] = m['x_lim'][1]
    kwargs['x_vals'] = df[var]
    kwargs['x_label'] = m['x_label']
    kwargs['x_type'] = m['x_type']
    kwargs['x_var'] = var


    tot_load = df['dispatch to fuel h2 storage (kW)'] + 1. # electric power demand = 1 


    ### Fuel cost compare scatter and use to fill electricity costs in stacked
    #kwargs['save_name'] = 'stackedCostPlot' + m['app']
    #costs_plot(var, **kwargs)
    #kwargs['ALT'] = True
    #costs_plot(var, **kwargs)
    #del kwargs['ALT']

    shifts = ['electrolyzer', 'wind', 'solar', 'natGas+CCS']
    for shift in shifts:
        kwargs['save_name'] = f'costPlot_{shift}'
        costs_plot_alt(shift, var, **kwargs)




