#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

save_type = 'png'
save_type = 'pdf'
adj = 1.

us_mean_dem = 450 # GW
us_mean_dem = 1 # kW

# Create the 1) initial optimization df and 2) the results df
def split_df(df):

    master, results = [], []
    for idx in df.index:
        name = df.loc[idx, 'case name']
        info = name.split('_')
        if info[-1] == info[-2].replace('lead',''):
            #print("master", info)
            master.append(idx)
        else:
            #print("results", info)
            results.append(idx)

    df_master = df.loc[master]
    df_results = df.loc[results]
    assert(len(df_master.index) + len(df_results.index) == len(df.index))
    return df_master, df_results


def plot_caps(df, unmet, app):

    plt.close()
    fig, ax = plt.subplots(figsize=(4.5*adj,4*adj))

    for i, idx in enumerate(df.index):
        info = np.array([df.loc[idx, 'capacity wind (kW)'],
                df.loc[idx, 'capacity solar (kW)'],
                df.loc[idx, 'capacity nuclear (kW)'],
                df.loc[idx, 'capacity storage (kW)']])
        ax.plot(range(4), info*us_mean_dem, markersize=10, marker='o', linestyle='dashed', label=f'Year {i+1}')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylabel('Nameplate Capacity (kW or kWh)')
    plt.xticks(range(4), ['Wind (kW)', 'Solar (kW)', 'Nuclear (kW)', 'Storage (kWh)'])
    plt.legend(loc='lower right')
    if app == 'ERCOT':
        ax.set_ylim(0, 3)
        plt.legend(loc='upper left', ncol=3)
    plt.tight_layout()
    plt.savefig('plots/single_cap_'+str(unmet).replace('.','p')+'_'+app+'.'+save_type)


def plot_unmet(df, unmet, app):

    plt.close()
    fig, ax = plt.subplots(figsize=(4.5*adj,4*adj))

    years = OrderedDict()
    max_v = 0.
    all_v = []
    all_v2 = []
    for idx in df.index:
        info = df.loc[idx, 'case name'].split('_')
        if df.loc[idx, 'dispatch unmet demand (kW)'] > max_v:
            max_v = df.loc[idx, 'dispatch unmet demand (kW)']
        if info[-2] in years.keys(): # This checks the leading (master) year
            years[info[-2]].append(df.loc[idx, 'dispatch unmet demand (kW)'])
        else:
            years[info[-2]] = [df.loc[idx, 'dispatch unmet demand (kW)'],]
        all_v.append((unmet - df.loc[idx, 'dispatch unmet demand (kW)'])/unmet)
        all_v2.append(df.loc[idx, 'dispatch unmet demand (kW)'])

    print(f'Coefficient of Variation Unmet Dem: {np.std(all_v2)/np.mean(all_v2)}')

    i = 0
    for k, v in years.items():
        #ax.scatter([i for _ in range(len(v))], v, marker='o', label=f'Year {i+1}')
        ax.plot([i for _ in range(len(v))], v, 'o', markersize=10, label=f'Year {i+1}', alpha=0.5)
        i += 1


    ax.plot(np.linspace(-1, len(years), 10), np.ones(10)*unmet, 'k--', label='Target Unmet\nDemand')

    #ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylim(0, max_v*1.5)
    ax.set_xlim(-0.25, 0.25+len(years)-1)
    ax.set_ylabel('Normalized Unmet Demand')
    plt.xticks(range(len(v)+1), [f'Year {i+1}\nConfig.' for i in range(len(v)+1)])
    plt.legend(loc='upper right')
    if app == 'ERCOT':
        plt.xticks(range(len(v)+1), [f'Yr {i+1} Cfg.' for i in range(len(v)+1)], rotation=90)
        ax.set_ylim(0, 0.01)
        plt.legend(loc='upper left', ncol=3)
    plt.tight_layout()
    plt.savefig('plots/single_unmet_'+str(unmet).replace('.','p')+'_'+app+'.'+save_type)

def print_variance(df):
        info = ['capacity wind (kW)', 'capacity solar (kW)',
                'capacity nuclear (kW)', 'capacity storage (kW)']

        for col in df.columns:
            if col not in info:
                continue
            print(col, round(np.std(df[col]),4), round(np.std(df[col])/np.mean(df[col]),4), 
                    round((max(df[col]) - min(df[col]))/np.mean(df[col]),4), 
                    round(np.mean(df[col]),4),
                    round(np.mean(df[col])*us_mean_dem,4))
    

file_map = { # Results file : associated unmet demand
    '../results/Results_reliability_20191201_v1_wind.csv' : [0.001, 'CONUS'],
    '../results/Results_reliability_20191219_v1TX0.999_wind.csv' : [0.001, 'ERCOT'],
    #'Results_reliability_20191201_v2_wind.csv' : 0.0003,
}

for f, info in file_map.items():
    print(info, f)
    df = pd.read_csv(f, index_col=False)
    master, results = split_df(df)
    
    plot_caps(master, info[0], info[1])
    plot_unmet(results, info[0], info[1])
    print_variance(master)



