#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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




region = 'CONUS'
im = return_file_info_map(region)
demand, wind, solar = get_dem_wind_solar(im)
print(demand.head())


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




