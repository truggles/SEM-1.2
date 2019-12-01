#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


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


def plot_caps(df):

    plt.close()
    fig, ax = plt.subplots()

    for i, idx in enumerate(df.index):
        info = [df.loc[idx, 'capacity wind (kW)'],
                df.loc[idx, 'capacity solar (kW)'],
                df.loc[idx, 'capacity nuclear (kW)'],
                df.loc[idx, 'capacity storage (kW)']]
        ax.plot(range(4), info, markersize=10, marker='o', linestyle='dashed', label=f'Year {i+1}')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylabel('Capacity (kW or kWh)')
    plt.xticks(range(4), ['Wind (kW)', 'Solar (kW)', 'Nuclear (kW)', 'Storage (kWh)'])
    plt.legend(loc='lower right')
    plt.savefig('plots/single_cap.png')


def plot_unmet(df, unmet_target):

    plt.close()
    fig, ax = plt.subplots()

    years = OrderedDict()
    for idx in df.index:
        info = df.loc[idx, 'case name'].split('_')
        if info[-2] in years.keys(): # This checks the leading (master) year
            years[info[-2]].append(df.loc[idx, 'dispatch unmet demand (kW)'])
        else:
            years[info[-2]] = [df.loc[idx, 'dispatch unmet demand (kW)'],]

    i = 0
    for k, v in years.items():
        #ax.scatter([i for _ in range(len(v))], v, marker='o', label=f'Year {i+1}')
        ax.plot([i for _ in range(len(v))], v, 'o', markersize=10, label=f'Year {i+1}')
        i += 1


    ax.plot(np.linspace(-1, len(years), 100), np.ones(100)*unmet_target, 'k-', label='Target Unmet\nDemand')

    #ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylim(0, unmet_target*4)
    ax.set_xlim(-0.25, 0.25+len(years)-1)
    ax.set_ylabel('Unmet Demand (kW)')
    plt.xticks(range(4), [f'Year {i+1}\nOptimization' for i in range(4)])
    plt.legend(loc='upper right')
    plt.savefig('plots/single_unmet.png')

df = pd.read_csv('Results_reliability_20191201_v1_wind.csv', index_col=False)
master, results = split_df(df)

plot_caps(master)
plot_unmet(results, 0.001)



