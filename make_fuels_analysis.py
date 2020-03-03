#!/usr/bin/env python3

import os
from collections import OrderedDict
import copy
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

import analytic_fuels as af


kWh_to_GGE = 33.4


# Plot stacked bars representing total fuel costs for multiple scenarios
def stacked_plots(systems, system_labels, electricity_costs, save_name, base):
    assert(len(systems) == len(system_labels))
    assert(len(system_labels) == len(electricity_costs))

    order = [
            'fixed: electrolyzer + compressor',
            'fixed: chem plant',
            'variable: chem plant',
            'variable: CO2',
            'variable: electricity',
    ]
    items = OrderedDict()
    for o in order:
        items[o] = []
    default_total = 0.
    for system, electricity_cost, syst_name in zip(systems, electricity_costs, system_labels):
        for item in items.keys():
            if item == 'fixed: electrolyzer + compressor':
                items[item].append((system['FIXED_COST_ELECTROLYZER']['value'] + system['FIXED_COST_COMPRESSOR']['value']) / (system['EFFICIENCY_CHEM_CONVERSION']['value'] * system['FIXED_COST_ELECTROLYZER']['capacity factor']))
            if item == 'fixed: chem plant':
                items[item].append(system['FIXED_COST_CHEM_PLANT']['value'])
            if item == 'variable: chem plant':
                items[item].append(system['VAR_COST_CHEM_PLANT']['value']) # Values from Konig already incorporate chem plant eff.
            if item == 'variable: CO2':
                items[item].append(system['VAR_COST_CO2']['value']) # Does not depend on chem plant eff.
            if item == 'variable: electricity':
                items[item].append(electricity_cost / (system['EFFICIENCY_ELECTROLYZER_COMP']['value'] * system['EFFICIENCY_CHEM_CONVERSION']['value']))
            if syst_name == 'baseline':
                default_total += items[item][-1]
                print(syst_name, item, round(items[item][-1],4))
    if 'baseline' in system_labels:
        print(f"baseline system total: {round(default_total,4)} $/kWh")

    for units, SF in {'kWh': 1, 'GGE': kWh_to_GGE}.items():
        plt.close()
        adj = 1.0 if len(systems) < 6 else 1.4
        fig, ax = plt.subplots(figsize=(6.4*adj, 4.8))

        width = 0.35 # Matplotlib example width
        btm = np.zeros(len(systems))
        for item, vals in items.items():
            vect = np.array(vals)
            plt.bar(np.arange(len(vect)), vect*SF, width, bottom=btm, label=item.replace('CO2',r'CO$_{2}$'))
            btm += vect*SF
        plt.xticks(np.arange(len(system_labels)), system_labels)
        ax.set_ylabel(f'electrofuel cost ($/{units})')
        if 'kWh' in units:
            ax.set_ylabel(r'electrofuel cost (\$/kWh$_{LHV}$)')
        ax.set_ylim(0, max(btm)*1.5)
        plt.legend()
        plt.grid(axis='y')
        #plt.tight_layout()
        plt.subplots_adjust(left=.1, right=.95, top=.95, bottom=0.2)
        plt.savefig(f'{base}{save_name}_{units}.png')
        plt.savefig(f'{base}/pdf/{save_name}_{units}.pdf')
        plt.clf()




def scan_electricity_costs_and_electrolyzer_CFs(system, electricity_info, electrolyzer_info, base):


    info1 = electricity_info
    info2 = electrolyzer_info
    z = np.zeros((info1[-1], info2[-1]))
    for i, electricity in enumerate(np.linspace(info1[0], info1[1], info1[2])):
        for j, electrolyzer in enumerate(np.linspace(info2[0], info2[1], info2[2])):
            system['FIXED_COST_ELECTROLYZER']['capacity factor'] = electrolyzer
            cost = af.get_fuel_system_costs(syst, electricity)
            z[i][j] = cost

    plot_2D(z, electrolyzer_info, electricity_info, 'Electrolyzer CF', 'Electricity Costs ($/kWh)', 
            'Fuel Cost ($/kWh)', 'scan_electricity_costs_and_electrolyzer_CFs_kWh', base)
    plot_2D(z*kWh_to_GGE, electrolyzer_info, electricity_info, 'Electrolyzer CF', 'Electricity Costs ($/kWh)', 
            'Fuel Cost ($/GGE)', 'scan_electricity_costs_and_electrolyzer_CFs_GGE', base)



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
    rounder_x = 2 if 'scan_electricity_and_electrolyzer_costs' in save_name else 1
    rounder_y = 2
    for i, y_val in enumerate(np.linspace(info1[0], info1[1], info1[2])):
        if round(y_val,rounder_y) == round(y_val,10):
            y_ticks_loc.append(i)
            y_ticks_val.append(str(round(y_val,rounder_y)))
    for j, x_val in enumerate(np.linspace(info2[0], info2[1], info2[2])):
        if round(x_val,rounder_x) == round(x_val,10):
            x_ticks_loc.append(j)
            x_ticks_val.append(str(round(x_val,rounder_x)))

    fig, ax = plt.subplots()
    im = ax.imshow(z, origin='lower', interpolation='spline16', vmin=0, aspect='auto')

    n_levels = 5
    cs = ax.contour(z, n_levels, colors='w')
    ax.clabel(cs, inline=1, fontsize=12, fmt='%1.2f')

    plt.xticks(x_ticks_loc, x_ticks_val)
    plt.yticks(y_ticks_loc, y_ticks_val)
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(z_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #plt.tight_layout()

    # Add baseline reference
    # Hardcoded FIXME
    x, y = -1, -1
    if 'scan_electricity_and_electrolyzer_costs' in save_name:
        x = 0.014309606239901077
        y = us_avg
    if 'scan_electricity_costs_and_electrolyzer_CFs' in save_name:
        x = 1.0
        y = us_avg
    x_spacing = (info2[1] - info2[0])/info2[2]
    y_spacing = (info1[1] - info1[0])/info1[2]
    x_loc = (x-info2[0])/x_spacing
    y_loc = (y-info1[0])/y_spacing
    if x_loc >= info2[2]:
        x_loc = info2[2]-1.5
    if y_loc >= info1[2]:
        y_loc = info1[2]-1.5
    #print(save_name, x_loc, y_loc, info2[2], info1[2])

    if x != -1 and y != -1:
        ax.scatter( x_loc, y_loc, s=320, marker='*', color='gold')

    plt.savefig(base+save_name+'.png')
    plt.savefig(base+'pdf/'+save_name+'.pdf')
    plt.clf()




# Nov 2019 industrial cost of electricity 0.0673 $/kWh US Avg https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
# https://www.eia.gov/electricity/annual/html/epa_01_01.html
# Table 1.1. Total Electric Power Industry Summary Statistics, 2018 and 2017
# Average Price of Electricity to Ultimate Customers (cents/kWh)
# https://www.eia.gov/electricity/annual/pdf/epa.pdf
us_avg = 0.0692

date = 20200303
base = f'plots_analytic_fuels_{date}/'
if not os.path.exists(base):
    os.makedirs(base)
    os.makedirs(base+'pdf/')

syst = af.return_fuel_system()
print("baseline System:")
for k1, v1 in syst.items():
    print(f" - {k1}")
    for k2, v2 in v1.items():
        if k2 == 'ref':
            continue
        print(f" --- {k2} = {v2}")

verbose = True
cost = af.get_fuel_system_costs(syst, us_avg, verbose)
print(f"baseline electrolyzer cost: {round(syst['FIXED_COST_ELECTROLYZER']['value'],4)} $/h/kW")
print(f"baseline electrofuel cost: {round(cost,4)} $/kWh")
print(f"               or: {round(cost*kWh_to_GGE,4)} $/GGE")
cost = af.get_fuel_system_costs(syst, 0.0, verbose)
print(f"free electricity electrofuel cost: {round(cost,4)} $/kWh")
print(f"                        or: {round(cost*kWh_to_GGE,4)} $/GGE")


electricity_info = [0.0, 0.2, 51]
electrolyzer_info = [0, 0.05, 51]
syst = af.return_fuel_system() # Get fresh system
scan_electricity_and_electrolyzer_costs(syst,
        electricity_info, electrolyzer_info, base)

electrolyzer_CF_info = [0.2, 1.00, 41]
syst = af.return_fuel_system() # Get fresh system
scan_electricity_costs_and_electrolyzer_CFs(syst,
        electricity_info, electrolyzer_CF_info, base)


systems, system_labels, electricity_costs = [], [], []

syst = af.return_fuel_system() # Get fresh system
system_labels.append('baseline')
electricity_costs.append(us_avg)
systems.append( copy.deepcopy(syst) )

system_labels.append('free\nelectricity')
electricity_costs.append(0.0)
systems.append( copy.deepcopy(syst) )

system_labels.append('1/2 cost\nelectrolyzer')
electricity_costs.append(us_avg)
systems.append( copy.deepcopy(syst) )
systems[-1]['FIXED_COST_ELECTROLYZER']['value'] = systems[-1]['FIXED_COST_ELECTROLYZER']['value'] * 0.5

system_labels.append('DAC CO$_{2}$')
electricity_costs.append(us_avg)
systems.append( copy.deepcopy(syst) )
systems[-1]['VAR_COST_CO2']['co2 cost'] = 600
systems[-1]['VAR_COST_CO2'] = af.var_cost_of_CO2(**systems[-1]['VAR_COST_CO2'])

system_labels.append('50% CF\nelectrolyzer\n+ 1/2 cost\nelectricity')
electricity_costs.append(us_avg*0.5)
systems.append( copy.deepcopy(syst) )
systems[-1]['FIXED_COST_ELECTROLYZER']['capacity factor'] = 0.5

system_labels.append('20% CF\nelectrolyzer\n+ free\nelectricity')
electricity_costs.append(0.0)
systems.append( copy.deepcopy(syst) )
systems[-1]['FIXED_COST_ELECTROLYZER']['capacity factor'] = 0.2

stacked_plots(systems, system_labels, electricity_costs, 'bar_chart', base)





