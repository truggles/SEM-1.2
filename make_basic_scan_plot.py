#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from collections import OrderedDict


def return_file_info_map(region):
    assert(region in ['CONUS', 'TEXAS'])

    info_map = { # region : # f_path, header rows
        'CONUS': {
            'demand': ['Input_Data/Lei_Solar_Wind/US_demand_unnormalized.csv', 8, 'demand (MW)', 'year'],
            'wind': ['Input_Data/Lei_Solar_Wind/US_capacity_wind_25pctTop_unnormalized.csv', 5, 'wind 25% top', 'Year'], 
            'solar': ['Input_Data/Lei_Solar_Wind/US_capacity_solar_25pctTop_unnormalized.csv', 5, 'solar 25% top', 'Year'],
        },
        'TEXAS': { 
            'demand': ['Input_Data/TEXAS/TX_demand_unnormalized.csv', 6, 'demand (MW)', 'year'],
            'wind': ['Input_Data/TEXAS/windCF_lei_TI_nonormalized.csv', 5, 'solar capacity', 'year'],
            'solar': ['Input_Data/TEXAS/solarCF_lei_TI_nonormalized.csv', 5, 'solar capacity', 'year'],
        }
    }
    return info_map[region]


# Normalize demand
def get_dem_wind_solar(im):

    demand = pd.read_csv(im['demand'][0], header=im['demand'][1])
    demand[im['demand'][2]] = demand[im['demand'][2]]/np.mean(demand[im['demand'][2]])
    wind = pd.read_csv(im['wind'][0], header=im['wind'][1])
    solar = pd.read_csv(im['solar'][0], header=im['solar'][1])

    return demand, wind, solar


def get_renewable_fraction(year, wind_factor, solar_factor, im, demand, wind, solar, zero_negative=True):
    
    d_profile = demand.loc[ demand[im['demand'][3]] == year, im['demand'][2] ].values
    w_profile = wind.loc[ wind[im['wind'][3]] == year, im['wind'][2] ].values
    s_profile = solar.loc[ solar[im['solar'][3]] == year, im['solar'][2] ].values

    final_profile = d_profile - wind_factor * w_profile - solar_factor * s_profile
    if zero_negative:
        final_profile = np.where(final_profile >= 0, final_profile, 0.)

    return np.mean(d_profile) - np.mean(final_profile)




def plot_matrix(matrix, solar_values, wind_values, save_name):


    fig, ax = plt.subplots()#figsize=(4.5, 4))
    #im = ax.imshow(matrix,interpolation='none',origin='lower')
    im = ax.imshow(matrix,interpolation='spline16',origin='lower')

    # Contours
    levels = [.25, .5, .75, 1.]
    cs = ax.contour(matrix, levels, colors='k', origin='lower')
    # inline labels
    ax.clabel(cs, inline=1, fontsize=10)

    wind_labs, solar_labs = [], []
    for v in wind_values:
        if int(v)==v:
            wind_labs.append(v)
        else:
            wind_labs.append('')
    for v in solar_values:
        if int(v)==v:
            solar_labs.append(v)
        else:
            solar_labs.append('')
    plt.xticks(range(len(wind_values)), wind_labs, rotation=90)
    plt.yticks(range(len(solar_values)), solar_labs)
    plt.xlabel("Wind Cap./Mean Demand")
    plt.ylabel("Solar Cap./Mean Demand")
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(f"Renewable Energy Fraction (%)")
    plt.title(f"")
    plt.tight_layout()


    plt.savefig(f"plots_new/{save_name}.png")
    plt.clf()



def get_annual_df(year, df, tgt, im):

    print(len(df))
    df2 = df[ df[ im[tgt][3]] == year]
    print(len(df2))
    return df2


# demand_threshold is in percent
def return_ordered_df(demand, wind, solar, im, demand_threshold):

    #rank_mthd='ordinal'
    rank_mthd='min'
    to_map = OrderedDict()
    to_map['demand'] = demand[im['demand'][2]].values
    to_map['demand_rank'] = rankdata(demand[im['demand'][2]].values, method=rank_mthd)
    to_map['demand_pct_of_max'] = demand[im['demand'][2]].values/np.max(demand[im['demand'][2]])
    to_map['wind'] = wind[im['wind'][2]].values
    to_map['wind_rank'] = rankdata(wind[im['wind'][2]].values, method=rank_mthd)
    to_map['wind_pct_of_max'] = wind[im['wind'][2]].values/np.max(wind[im['wind'][2]])
    to_map['solar'] = solar[im['solar'][2]].values
    to_map['solar_rank'] = rankdata(solar[im['solar'][2]].values, method=rank_mthd)
    to_map['solar_pct_of_max'] = solar[im['solar'][2]].values/np.max(solar[im['solar'][2]])

    df = pd.DataFrame(to_map)
    df = df.sort_values(by='demand_rank', ascending=0)

    df = df.drop(df[df['demand_rank'] < (len(df.index) * (100. - demand_threshold)/100.)].index, axis=0)
    return df


def make_ordering_plot(dfs, save_name, wind_factor=1., solar_factor=1.):

    plt.close()
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), 'k-')
    for year, df in dfs.items():
        print(year)
        ax.plot(df['demand']-df['wind']*wind_factor, df['solar']*solar_factor, '.', alpha=0.2, label=year)
    ax.set_xlabel('demand - wind gen')
    ax.set_ylabel('solar gen')
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig(f"plots_new/{save_name}_dem_min_wind_vs_solar.png")
    plt.clf()

    plt.close()
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), 'k-')
    for year, df in dfs.items():
        print(year)
        ax.plot(df['demand']-df['solar']*solar_factor, df['wind']*wind_factor, '.', alpha=0.2, label=year)
    ax.set_xlabel('demand - solar gen')
    ax.set_ylabel('wind gen')
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig(f"plots_new/{save_name}_dem_min_solar_vs_wind.png")
    plt.clf()


    plt.close()
    fig, ax = plt.subplots()
    for year, df in dfs.items():
        print(year)
        ax.plot(df['demand'], df['solar'], '.', alpha=0.2, label=year)
    ax.set_xlabel('demand')
    ax.set_ylabel('solar CF')
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig(f"plots_new/{save_name}_solar.png")
    plt.clf()

    plt.close()
    fig, ax = plt.subplots()
    for year, df in dfs.items():
        print(year)
        ax.plot(df['demand'], df['wind'],  '.', alpha=0.2, label=year)
    ax.set_xlabel('demand')
    ax.set_ylabel('wind CF')
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig(f"plots_new/{save_name}_wind.png")
    plt.clf()


    plt.close()
    fig, ax = plt.subplots()
    for year, df in dfs.items():
        print(year)
        ax.plot(df['demand_rank'], df['solar_rank'], '.', alpha=0.2, label=year)
    ax.set_xlabel('demand rank')
    ax.set_ylabel('solar rank')
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig(f"plots_new/{save_name}_solar_rank.png")
    plt.clf()

    plt.close()
    fig, ax = plt.subplots()
    for year, df in dfs.items():
        print(year)
        ax.plot(df['demand_rank'], df['wind_rank'],  '.', alpha=0.2, label=year)
    ax.set_xlabel('demand rank')
    ax.set_ylabel('wind rank')
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig(f"plots_new/{save_name}_wind_rank.png")
    plt.clf()



region = 'CONUS'
im = return_file_info_map(region)
demand, wind, solar = get_dem_wind_solar(im)




test_ordering = True
#test_ordering = False
if test_ordering:
    dfs = OrderedDict()
    years = [2016,2017,2018]
    for year in years:
        d_yr = get_annual_df(year, demand, 'demand', im)
        w_yr = get_annual_df(year, wind, 'wind', im)
        s_yr = get_annual_df(year, solar, 'solar', im)
        dfs[year] = return_ordered_df(d_yr, w_yr, s_yr, im, 100)

    make_ordering_plot(dfs, f'ordering_{region}')
        



make_scan = True
make_scan = False
if make_scan:

    year = 2018
    mean = np.mean(demand.loc[ demand[im['demand'][3]] == year, im['demand'][2] ])
    print(f"Mean normalized demand for year {year} = {mean}")
    
    to_explore = [0, 4, 0.25]
    r_vals = np.arange(to_explore[0], to_explore[1], to_explore[2])
    tgt_len = len(r_vals)
    
    matrix = []
    
    zero_negative=True
    for i, solar_factor in enumerate(r_vals):
        matrix.append([])
        for wind_factor in r_vals:
            val = get_renewable_fraction(year, wind_factor, solar_factor, im, demand, wind, solar, zero_negative)
            #print(f" - Wind:Solar:RenewableFraction {wind_factor}:\t{solar_factor}:\t{round(val,3)}")
            matrix[i].append(val)
    
    ary = np.array(matrix)
    plot_matrix(matrix, r_vals, r_vals, 'test')




