#!/usr/bin/env python3

import os
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import analytic_fuels as af



def scan_electricity_and_electrolyzer_costs(system, electricity_info, electrolyzer_info, base):

    print(f"Initial electrolyzer cost: {system['FIXED_COST_ELECTROLYZER']['value']}")

    info1 = electricity_info
    info2 = electrolyzer_info
    z = np.zeros((info1[-1], info2[-1]))
    for i, electricity in enumerate(np.linspace(info1[0], info1[1], info1[2])):
        for j, electrolyzer in enumerate(np.linspace(info2[0], info2[1], info2[2])):
            system['FIXED_COST_ELECTROLYZER']['value'] = electrolyzer
            cost = af.get_fuel_system_costs(syst, electricity)
            z[i][j] = cost

    plot_2D(z, electrolyzer_info, electricity_info, 'Electrolyzer CapEx ($/kW/h)', 'Electricity Costs ($/kWh)', 
            'Fuel Cost ($/kWh)', 'scan_electricity_and_electrolyzer_costs_kWh', base)
    plot_2D(z*33.4, electrolyzer_info, electricity_info, 'Electrolyzer CapEx ($/kW/h)', 'Electricity Costs ($/kWh)', 
            'Fuel Cost ($/GGE)', 'scan_electricity_and_electrolyzer_costs_GGE', base)



def plot_2D(z, x_axis_info, y_axis_info, x_label, y_label, z_label, save_name, base):

    info1 = y_axis_info
    info2 = x_axis_info
    x_ticks_loc = []
    x_ticks_val = []
    y_ticks_loc = []
    y_ticks_val = []
    for i, y_val in enumerate(np.linspace(info1[0], info1[1], info1[2])):
        if round(y_val,2) == y_val:
            y_ticks_loc.append(i)
            y_ticks_val.append(str(round(y_val,2)))
    for j, x_val in enumerate(np.linspace(info2[0], info2[1], info2[2])):
        if round(x_val,2) == x_val:
            x_ticks_loc.append(j)
            x_ticks_val.append(str(round(x_val,2)))

    fig, ax = plt.subplots()
    im = ax.imshow(z, origin='lower', interpolation='spline16', vmin=0)

    n_levels = 5
    cs = ax.contour(z, n_levels, colors='w')
    ax.clabel(cs, inline=1, fontsize=12, fmt='%1.1f')

    plt.xticks(x_ticks_loc, x_ticks_val)
    plt.yticks(y_ticks_loc, y_ticks_val)
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(z_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #plt.tight_layout()
    plt.savefig(base+save_name+'.png')
    plt.clf()





date = 20200204
base = f'plots_analytic_fuels_{date}/'
if not os.path.exists(base):
    os.makedirs(base)

syst = af.return_fuel_system()

cost = af.get_fuel_system_costs(syst, 0.06)


electricity_info = [0.0, 0.2, 51]
electrolyzer_info = [0, 0.05, 51]
scan_electricity_and_electrolyzer_costs(syst,
        electricity_info, electrolyzer_info, base)


