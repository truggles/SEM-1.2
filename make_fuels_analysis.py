#!/usr/bin/env python3

import os
from collections import OrderedDict
import copy
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import analytic_fuels as af


kWh_to_GGE = 33.4


# Plot stacked bars representing total fuel costs for multiple scenarios
def stacked_plots(systems, system_labels, electricity_costs, save_name, base):

    order = [
            'fixed cost chem plant',
            'var cost chem plant',
            'fixed cost electrolyzer',
            'var cost electricity',
            'var cost CO2',
    ]
    items = OrderedDict()
    for o in order:
        items[o] = []
    for system, electricity_cost in zip(systems, electricity_costs):
        for item in items.keys():
            if item == 'fixed cost electrolyzer':
                items[item].append(system['FIXED_COST_ELECTROLYZER']['value'] / system['EFFICIENCY_CHEM_PLANT']['value'])
            if item == 'fixed cost chem plant':
                items[item].append(system['FIXED_COST_CHEM_PLANT']['value'])
            if item == 'var cost chem plant':
                items[item].append(system['VAR_COST_CHEM_PLANT']['value'])
            if item == 'var cost CO2':
                items[item].append(system['VAR_COST_CO2']['value'])
            if item == 'var cost electricity':
                items[item].append(electricity_cost / (system['EFFICIENCY_ELECTROLYZER']['value'] * system['EFFICIENCY_CHEM_PLANT']['value']))

    for units, SF in {'kWh': 1, 'GGE': kWh_to_GGE}.items():
        plt.close()
        fig, ax = plt.subplots()

        width = 0.35 # Matplotlib example width
        btm = np.zeros(len(systems))
        for item, vals in items.items():
            vect = np.array(vals)
            plt.bar(np.arange(len(vect)), vect*SF, width, bottom=btm, label=item)
            btm += vect*SF
        plt.xticks(np.arange(len(system_labels)), system_labels)
        ax.set_ylabel(f'fuel cost ($/{units})')
        ax.set_ylim(0, max(btm)*1.5)
        plt.legend()
        plt.savefig(f'{base}_{save_name}_{units}.png')
        plt.clf()




def scan_electricity_and_electrolyzer_costs(system, electricity_info, electrolyzer_info, base):


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
    plot_2D(z*kWh_to_GGE, electrolyzer_info, electricity_info, 'Electrolyzer CapEx ($/kW/h)', 'Electricity Costs ($/kWh)', 
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
print("Default System:")
for k1, v1 in syst.items():
    print(f" - {k1}")
    for k2, v2 in v1.items():
        if k2 == 'ref':
            continue
        print(f" --- {k2} = {v2}")

cost = af.get_fuel_system_costs(syst, 0.06)
print(f"Default electrolyzer cost: {round(syst['FIXED_COST_ELECTROLYZER']['value'],4)} $/h/kW")
print(f"Default fuel cost: {round(cost,4)} $/kWh")


electricity_info = [0.0, 0.2, 51]
electrolyzer_info = [0, 0.05, 51]
scan_electricity_and_electrolyzer_costs(syst,
        electricity_info, electrolyzer_info, base)

systems = []
systems.append( copy.deepcopy(syst) )
systems.append( copy.deepcopy(syst) )
systems[-1]['FIXED_COST_ELECTROLYZER']['value'] = 0.01
system_labels = ['Default', 'Expensive Electrolyzer']
electricity_costs = [0.05, 0.1]
stacked_plots(systems, system_labels, electricity_costs, 'bar_chart_test', base)





