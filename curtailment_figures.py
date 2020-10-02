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
from datetime import datetime, timedelta
import copy
from helpers import get_fuel_demands, get_fuel_fractions
from end_use_fractions import add_detailed_results
from analytic_fuels import kWh_to_GGE, kWh_LHV_per_kg_H2
matplotlib.rcParams.update({'font.size': 12.5})
matplotlib.rcParams.update({'lines.linewidth': 3})


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



def simple_plot(x, ys, labels, save, logY=False, ylims=[-1,-1], **kwargs):

    print("Plot {} using x,y = {},{}".format(save, kwargs['x_label'],kwargs['y_label']))

    plt.close()
    matplotlib.rcParams["figure.figsize"] = (6.4, 4.8)
    fig, ax = plt.subplots()
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])

    markers = marker_list()
    for y, label, mark in zip(ys, labels, markers):
        #print(label)
        #for i, j in zip(x, y):
        #    print(i, j)

        # Skip adding empty and null arrays to legend
        ary = np.array(y)
        if np.isnan( ary ).sum() == len(y) or \
                (ary == 0.0).sum() == len(y):
            #print(f"Skipping: {ary}")
            #ax.scatter([], [])
            ax.plot([], [])
            continue
        if 'h2_only' in kwargs.keys() and (label == 'chemical plant' or 'storage' in label):
            ax.plot([], [])
            continue


        if label == 'h2 storage':
            #ax.scatter(x, y, label=r'H$_{2}$ storage', marker=markers[-1], color='C7')
            ax.plot(x, y, label=r'H$_{2}$ storage', color='C7')
        else:
            #ax.scatter(x, y, label=label, marker=mark)
            ax.plot(x, y, label=label)

    if logY:
        plt.yscale('log', nonposy='clip')

    if ylims != [-1,-1]:
        ax.set_ylim(ylims[0], ylims[1])
    
    plt.xscale(kwargs['x_type'], nonposx='clip')
    ax.set_xlim(kwargs['stacked_min'], kwargs['stacked_max'])

    #plt.tight_layout()
    if not 'systemCFsEF' in save:
        plt.grid()

    if 'systemCFsEF' in save:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel(ax.get_ylabel(), labelpad=-2)
        plt.legend(ncol=2, loc='lower left', framealpha = 1.0)
    elif 'systemCFs' in save:
        plt.legend(ncol=3, loc='upper left', framealpha = 1.0)
    elif 'systemCosts' in save or 'powerSystemCosts' in save:
        if 'CostsExp' in save:
            plt.legend(loc='upper left', framealpha = 1.0)
        else:
            plt.legend(loc='upper right', framealpha = 1.0)
    else:
        plt.legend(loc='upper left', framealpha = 1.0)
    fig.savefig('{}/{}.png'.format(kwargs['save_dir'], save))
    fig.savefig('{}/{}.pdf'.format(kwargs['save_dir'], save))







# Poorly written, the args are all required and are below.
#x_vals, nuclear, wind, solar, x_label, y_label, 
#legend_app, stacked_min, stacked_max, save_name, save_dir):
def stacked_plot(**kwargs):

    plt.close()
    matplotlib.rcParams["figure.figsize"] = (6.4, 4.8)
    fig, ax = plt.subplots()
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])

    colors = color_list()
    tot = np.zeros(len(kwargs['x_vals']))
    if 'stackedEndUseFraction' in kwargs["save_name"]:
        if 'ALT' in kwargs and kwargs['ALT'] == True:
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['elec_load']*kwargs['dem_renew_frac'], color=colors[0], linewidth=0, label=f'power to electric load - renew')
            tot += kwargs['elec_load']*kwargs['dem_renew_frac']
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['elec_load']*(1. - kwargs['dem_renew_frac']), color=colors[0], linewidth=0, alpha=0.5, label=f'power to electric load - dispatch')
            tot += kwargs['elec_load']*(1. - kwargs['dem_renew_frac'])
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['fuel_load']*kwargs['electro_renew_frac'], color=colors[1], linewidth=0, label=f'power to flexible load - renew')
            tot += kwargs['fuel_load']*kwargs['electro_renew_frac']
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['fuel_load']*(1. - kwargs['electro_renew_frac']), color=colors[1], linewidth=0, alpha=0.5, label=f'power to flexible load - dispatch')
            tot += kwargs['fuel_load']*(1. - kwargs['electro_renew_frac'])
            #ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['battery_losses'], color='blue', linewidth=0, label=f'battery losses')
            #tot += kwargs['battery_losses']
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['renewable_curt'], color=colors[5], linewidth=0, label=f'curtailed - renew')
            tot += kwargs['renewable_curt']
            #for ff, y in zip(kwargs['x_vals'], tot):
            #    print(ff, y)
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['nuclear_curt'], color=colors[5], linewidth=0, alpha=0.5, label=f'unused - dispatch')
            tot += kwargs['nuclear_curt']
            #for ff, y in zip(kwargs['x_vals'], tot):
            #    print(ff, y)

            if 'ylim' in kwargs.keys():
                y_max = kwargs['ylim'][1]
            else:
                y_max = 9999
            #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, kwargs['x_var']], 0, y_max, color=colors[-1], alpha=0.2)
            #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, kwargs['x_var']], 0, y_max, facecolor='none', edgecolor='gray', hatch='....', linewidth=0, alpha=0.5)
            #ax.fill_between(xs, 0., dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)'], facecolor='none', edgecolor='black', hatch='/////', label='power to electric load', lw=fblw)
    

        else:
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['elec_load'], color=colors[0], linewidth=0, label=f'power to electric load')
            tot += kwargs['elec_load']
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['fuel_load'], color=colors[1], linewidth=0, label=f'power to flexible load')
            tot += kwargs['fuel_load']
            #ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['battery_losses'], color='blue', linewidth=0, label=f'battery losses')
            #tot += kwargs['battery_losses']
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['renewable_curt'], color=colors[5], linewidth=0, label=f'curtailed - renew')
            tot += kwargs['renewable_curt']
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['nuclear_curt'], color=colors[5], linewidth=0, alpha=0.5, label=f'unused - dispatch')
            tot += kwargs['nuclear_curt']

    else:
        #if not (np.isnan( kwargs['nuclear'] ).sum() == len(kwargs['nuclear']) or (kwargs['nuclear'] == 0.0).sum() == len(kwargs['nuclear'])):
        ax.fill_between(kwargs['x_vals'], 0., kwargs['nuclear'], color=colors[5], linewidth=0, label=f'dispatchable {kwargs["legend_app"]}')
        tot += kwargs['nuclear']
        if 'renewables' in kwargs.keys():
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['renewables'], color=colors[1], linewidth=0, label=f'renewables {kwargs["legend_app"]}')
            tot += kwargs['renewables']
        else:
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['wind'], color=colors[1], linewidth=0, label=f'wind {kwargs["legend_app"]}')
            tot += kwargs['wind']
            ax.fill_between(kwargs['x_vals'], tot, tot+kwargs['solar'], color=colors[0], linewidth=0, label=f'solar {kwargs["legend_app"]}')
            tot += kwargs['solar']
        if 'stackedGenerationElecNorm' in kwargs["save_name"]:
            ax.plot(kwargs['x_vals'], tot, 'k-', label='total available gen.')
            ax.plot(kwargs['x_vals'], np.ones(len(kwargs['x_vals'])), 'k--', label='firm electric load')
            ax.plot(kwargs['x_vals'], np.ones(len(kwargs['x_vals'])) / (1. - kwargs['x_vals']), 'k:', label='firm electric +\nflexible load')
            #for x, y in zip(kwargs['x_vals'], tot):
            #    print(x, y)

            if 'ylim' in kwargs.keys():
                y_max = kwargs['ylim'][1]
            else:
                y_max = 9999
            #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, kwargs['x_var']], 0, y_max, color=colors[-1], alpha=0.2)
            #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, kwargs['x_var']], 0, y_max, facecolor='none', edgecolor='gray', hatch='....', linewidth=0, alpha=0.5)
            #ax.fill_between(xs, 0., dfs['demand (kW)'] - dfs['dispatch unmet demand (kW)'], facecolor='none', edgecolor='black', hatch='/////', label='power to electric load', lw=fblw)




    plt.xscale(kwargs['x_type'], nonposx='clip')
    ax.set_xlim(kwargs['stacked_min'], kwargs['stacked_max'])

    if 'logy' in kwargs.keys():
        if kwargs['logy'] == True:
            plt.yscale('log', nonposy='clip')
            ax.set_ylim(min(0.1, ax.get_ylim()[0]), max(100, ax.get_ylim()[1]))

    #plt.tight_layout()

    if 'ylim' in kwargs.keys():
        ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
        if 'stackedEndUseFraction' in kwargs["save_name"]:
            plt.legend(loc='upper right', ncol=1, framealpha = 1.0)
        elif 'stackedGenerationElecNorm' in kwargs["save_name"]:
            plt.legend(loc='upper left', ncol=1, framealpha = 1.0)
        else:
            plt.legend(ncol=3, framealpha = 1.0)
    else:
        plt.legend(framealpha = 1.0)
    ax.yaxis.set_ticks_position('both')
    #plt.grid()
    fig.savefig(f'{kwargs["save_dir"]}/{kwargs["save_name"]}.png')
    fig.savefig(f'{kwargs["save_dir"]}/{kwargs["save_name"]}.pdf')







# Poorly written, the args are all required and are below.
#save_name, save_dir
def costs_plot(var='fuel demand (kWh)', **kwargs):

    dfs = kwargs['dfs']

    colors = color_list()
    plt.close()
    y_max = 30
    if 'h2_only' in kwargs.keys():
        if 'ALT' in kwargs.keys():
            y_max = 12
        else:
            y_max = 8
    matplotlib.rcParams["figure.figsize"] = (7.5, 4)
    if not 'h2_only' in kwargs.keys():
        matplotlib.rcParams["figure.figsize"] = (7.5, 5)
    fig, axs = plt.subplots(ncols=3, sharedy=True, dpi=400)

    # The left y-axis will use $/GGE for electrofuel or $/kg for H2
    if 'h2_only' in kwargs.keys():
        conversion = kWh_LHV_per_kg_H2 # Convert for main y-axis
        conversion *= EFFICIENCY_FUEL_CHEM_CONVERSION # The cost is set based on liquid hydrocarbon
                                                      # output, so must be scaled for H2 only
        ax.set_ylabel(r'cost (\$/kg$_{H2}$)')
    else:
        conversion = kWh_to_GGE
        ax.set_ylabel(r'cost (\$/GGE)')

    ax.set_xlabel(kwargs['x_label'])
    
    appA = ''
    appB = ''
    ep = 'electric power'
    if 'ALT' in kwargs.keys():
        appA = ' (marginal cost)'
        appB = appA #'\n(marginal cost)'
        ep = 'power'


    # Electricity cost
    #ax.scatter(df[var], df['mean price ($/kWh)'], color='black', label=r'electricity cost (\$/kWh$_{e}$)', marker='v')
    #ax.plot(df[var], df['mean price ($/kWh)'], 'k--', label='power to electric\n'+r'load (\$/kWh$_{e}$)')

    # $/GGE fuel line use Dual Value
    #ax.scatter(df[var], df['fuel price ($/kWh)'], color='black', label='total electrofuel\n'+r'cost (\$/kWh$_{LHV}$)')
    lab = r'H$_{2}$ production total' if 'h2_only' in kwargs.keys() else 'electrofuel production total'
    ax.plot(df[var], df['fuel price ($/kWh)'] * conversion, 'k-', label=lab+appB)
    for ff, cost in zip(df[var], df['fuel price ($/kWh)'] * conversion):
        if round(ff,2) == 0.50:
            print("tot", ff, cost)
    #    break

    # Stacked components
    f_elec = df['fixed cost fuel electrolyzer ($/kW/h)'] * df['capacity fuel electrolyzer (kW)'] / df['fuel demand (kWh)']
    f_chem = df['fixed cost fuel chem plant ($/kW/h)'] * df['capacity fuel chem plant (kW)'] / df['fuel demand (kWh)']
    f_store = df['fixed cost fuel h2 storage ($/kWh/h)'] * df['capacity fuel h2 storage (kWh)'] / df['fuel demand (kWh)']
    f_tot = (f_elec+f_chem+f_store)
    v_chem = df['var cost fuel chem plant ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_CONVERSION / df['fuel demand (kWh)']
    v_co2 = df['var cost fuel co2 ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_CONVERSION / df['fuel demand (kWh)']
    #for ff, cost, c2 in zip(df[var], f_elec * conversion, f_elec * EFFICIENCY_FUEL_CHEM_CONVERSION):
    #    print(ff, cost, c2)

    f_elec *= conversion
    f_chem *= conversion
    f_store *= conversion
    f_tot *= conversion
    v_chem *= conversion
    v_co2 *= conversion


    # Build stack
    lab = 'fixed cost: electrolysis plant' # if 'h2_only' in kwargs.keys() else 'fixed: electrolysis\nplant'
    ax.fill_between(df[var], 0, f_elec, label=lab, color=colors[0])
    for ff, cost in zip(df[var], f_elec):
        if round(ff,2) == 0.50:
            print("electrolysis", ff, cost)
    #    break

    if 'h2_only' not in kwargs.keys():
        ax.fill_between(df[var], f_elec, f_elec+f_chem, label='fixed cost: chemical plant', color=colors[1])
        ax.fill_between(df[var], f_elec+f_chem, f_elec+f_chem+f_store, label='fixed cost: storage', color=colors[2]) # fixed cost storage set at 2.72E-7
        ax.fill_between(df[var], f_tot, f_tot+v_chem, label='variable cost: chemical plant', color=colors[3])
        ax.fill_between(df[var], f_tot+v_chem, f_tot+v_chem+v_co2, label='variable cost: CO$_{2}$ feedstock', color=colors[4])


    ax.fill_between(df[var], f_tot+v_chem+v_co2, df['fuel price ($/kWh)'] * conversion, label=f'{ep}'+appA, color=colors[5], alpha=.8, hatch='\\\\')

    if 'ALT' in kwargs.keys():
        tot_eff_fuel_process = EFFICIENCY_FUEL_ELECTROLYZER * EFFICIENCY_FUEL_CHEM_CONVERSION
        avg_elec_cost = (df['mean price ($/kWh)'] * (1. - df[var]) + df['fuel_load_cost'] * df[var]) / tot_eff_fuel_process
        #for ff, cost, c2, c3 in zip(df[var], avg_elec_cost * conversion, avg_elec_cost * EFFICIENCY_FUEL_CHEM_CONVERSION, avg_elec_cost * tot_eff_fuel_process):
        #    print(ff, cost, c2, c3)
        ax.fill_between(df[var], f_tot+v_chem+v_co2, f_tot+v_chem+v_co2 + avg_elec_cost * conversion, label='power (system-wide cost)', hatch='////', alpha=0.3, color=colors[4])
        lab = r'H$_{2}$ production total' if 'h2_only' in kwargs.keys() else 'electrofuel production total'
        ax.plot(df[var], f_tot+v_chem+v_co2 + avg_elec_cost * conversion, color='black', linestyle='--', label=lab+' (system-wide cost)')
        for ff, cost in zip(df[var], f_tot+v_chem+v_co2 + avg_elec_cost * conversion):
            if round(ff,2) == 0.50:
                print("tot mean", ff, cost)
        #    break
        

    n = len(df.index)-1
    print(f" --- Stacked cost plot for fractional fuel demand = {round(df.loc[1, 'fuel demand (kWh)'],4)}           {round(df.loc[n, 'fuel demand (kWh)'],4)}:")
    print(f" ----- fixed cost electrolyzer          {round(f_elec[1],6)}         {round(f_elec[n],6)}")
    print(f" ----- fixed cost chem                  {round(f_chem[1],6)}         {round(f_chem[n],6)}")
    print(f" ----- fixed cost storage               {round(f_store[1],6)}         {round(f_store[n],6)}")
    print(f" ----- variable chem                    {round(v_chem[1],6)}         {round(v_chem[n],6)}")
    print(f" ----- variable CO2                     {round(v_co2[1],6)}          {round(v_co2[n],6)}")
    print(f" ----- variable electrolyzer            {round(df.loc[1, 'fuel price ($/kWh)']-(f_tot[1]+v_chem[1]+v_co2[1]),6)}             {round(df.loc[n, 'fuel price ($/kWh)']-(f_tot[n]+v_chem[n]+v_co2[n]),6)}")
    print(f" ----- TOTAL:                           {round(df.loc[1, 'fuel price ($/kWh)'],6)}           {round(df.loc[n, 'fuel price ($/kWh)'],6)}")
    print(f" ----- mean cost electricity            {round(df.loc[1, 'mean price ($/kWh)'],6)}           {round(df.loc[n, 'mean price ($/kWh)'],6)}")

    # Highlight transition region
    #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, var], 0, y_max, color=colors[-1], alpha=0.15)
    #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, var], 0, y_max, facecolor='none', edgecolor='gray', hatch='....', linewidth=0, alpha=0.5)


    # To print the estimated MEM electricity cost per point
    #ax.scatter(df[var], df['fuel price ($/kWh)'], color='black', label='_nolegend_')
    #ax.scatter(df[var], df['mean price ($/kWh)'], color='black', label='_nolegend_', marker='v')
    #ax.plot(df[var], df['mean price ($/kWh)'], 'k--', label='_nolegend_')
    ax.plot(df[var], df['fuel price ($/kWh)'] * conversion, 'k-', label='_nolegend_')
    if 'ALT' in kwargs.keys():
        ax.plot(df[var], f_tot+v_chem+v_co2 + avg_elec_cost * conversion, color='black', linestyle='--', label='_nolegend_')



    plt.xscale(kwargs['x_type'], nonposx='clip')
    if kwargs['x_type'] == 'log':
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(myLogFormat))

    if var == 'fuel demand (kWh)' or var == 'dispatch to fuel h2 storage (kW)':
        ax.set_xlim(df.loc[1, var], df.loc[len(df.index)-1, var])
    else:
        plt.xscale('linear')
        ax.set_xlim(0.0, 1.0)

    ax.set_ylim(0, y_max)
    #plt.grid()
    if 'h2_only' not in kwargs.keys():
        plt.legend(loc='upper right', ncol=1, framealpha = 1.0)
    else:
        plt.legend(loc='upper right', ncol=1, framealpha = 1.0)
        plt.subplots_adjust(left=0.12, right=.85)

    # 2nd y-axis for $/kWh_e
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(r'cost (\$/kWh$_{LHV}$)')
    ax2.set_ylim(0, ax.get_ylim()[1] / kWh_to_GGE)

    #plt.tight_layout()
    app2 = '_ALT' if 'ALT' in kwargs.keys() else ''
    ax.set_rasterized(True)
    ax2.set_rasterized(True)
    fig.savefig(f'{kwargs["save_dir"]}{kwargs["save_name"]}{app2}.png')
    fig.savefig(f'{kwargs["save_dir"]}{kwargs["save_name"]}{app2}.pdf')
    print(f'{kwargs["save_dir"]}{kwargs["save_name"]}.png')


# Poorly written, the args are all required and are below.
#save_name, save_dir
def costs_plot_alt(var='fuel demand (kWh)', **kwargs):

    df = kwargs['df']

    colors = color_list()
    plt.close()
    #y_max = 0.7
    y_max = 0.15
    matplotlib.rcParams["figure.figsize"] = (6.4, 4.8)
    fig, ax = plt.subplots()

    ax.set_xlabel(kwargs['x_label'])
    
    #ax.set_ylabel(r'cost (\$/kWh$_{LHV}$ or \$/kWh$_{e}$)')
    ax.set_ylabel(r'cost (\$/kWh$_{e}$)')


    # Electricity cost
    #ax.scatter(df[var], df['mean price ($/kWh)'], label=r'electricity cost (\$/kWh$_{e}$)', marker='v')
    ax.plot(df[var], df['mean price ($/kWh)'], label=r'marginal cost: firm electric load', color=colors[0])

    # Highlight transition region
    #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, var], 0, y_max, color=colors[-1], alpha=0.15)
    #ax.fill_between(df.loc[ff_1_idx:ff_2_idx, var], 0, y_max, facecolor='none', edgecolor='gray', hatch='....', linewidth=0, alpha=0.5)
    ax.plot(df[var], df['mean price ($/kWh)'], label='_nolegend_', color=colors[0])
    #for ff, cost in zip(df[var], df['mean price ($/kWh)']):
    #    if round(ff, 2) == 0 or round(ff, 2) == 1:
    #        print('firm', ff, cost)

    ## $/GGE fuel line use Dual Value
    #ax.scatter(df[var], df['fuel price ($/kWh)'], label='total electrofuel\n'+r'cost (\$/kWh$_{LHV}$)')
    #ax.plot(df[var], df['fuel price ($/kWh)'], label='total electrofuel\n'+r'cost (\$/kWh$_{LHV}$)')

    tot_eff_fuel_process = EFFICIENCY_FUEL_ELECTROLYZER * EFFICIENCY_FUEL_CHEM_CONVERSION
    # Add the cost of electric power to the fuel load
    #ax.scatter(df[var], (df['fuel price ($/kWh)'] - f_tot) * tot_eff_fuel_process, label=r'fuel load electricity cost (\$/kWh$_{e}$)', marker='o')
    ax.plot(df[var], df['fuel_load_cost'], label=r'marginal cost: flexible load', color=colors[1])
    #for ff, cost in zip(df[var], df['fuel_load_cost']):
    #    if round(ff, 2) == 0 or round(ff, 2) == 1:
    #        print('flex', ff, cost)

    avg_elec_cost = df['mean price ($/kWh)'] * (1. - df[var]) + df['fuel_load_cost'] * df[var]
    ax.plot(df[var], avg_elec_cost, 'k--', label=r'system-wide cost', linewidth=2)
    #for ff, cost in zip(df[var], avg_elec_cost):
    #    if round(ff, 2) == 0 or round(ff, 2) == 1:
    #        print('mean', ff, cost)



    plt.xscale(kwargs['x_type'], nonposx='clip')

    if var == 'fuel demand (kWh)' or var == 'dispatch to fuel h2 storage (kW)':
        ax.set_xlim(df.loc[1, var], df.loc[len(df.index)-1, var])
    else:
        plt.xscale('linear')
        ax.set_xlim(0.0, 1.0)

    ax.set_ylim(0, y_max)
    ax.yaxis.set_ticks_position('both')
    #plt.grid()
    plt.legend(loc='upper left', framealpha = 1.0)


    #plt.tight_layout()
    fig.savefig(f'{kwargs["save_dir"]}{kwargs["save_name"]}.png')
    fig.savefig(f'{kwargs["save_dir"]}{kwargs["save_name"]}.pdf')
    print(f'{kwargs["save_dir"]}{kwargs["save_name"]}.png')













if '__main__' in __name__:

    import sys
    print(f"\nRunning {sys.argv[0]}")
    print(f"Input arg list {sys.argv}")

    
    run_sem = False
    make_results_file = False
    make_plots = False
    if 'run_sem' in sys.argv:
        run_sem = True
    if 'make_results_file' in sys.argv:
        make_results_file = True
    if 'make_plots' in sys.argv:
        make_plots = True

    date = '20200209' # default
    case = 'Case6_NuclearWindSolarStorage' # default
    version1 = 'v3'
    multiplication_factor = 0.01 # default for step_size in new get_fuel_fractions method
    n_jobs = 1
    job_num = 1
    full_year = False # default to run over April only
    h2_only = False # if h2_only is True, costs for CO2, H2 storage, and chem plant are set to 1e-9
    fixed_solar = 1
    fixed_wind = 1
    fixed_electrolyzer = 1
    fixed_natGasCCS = 1
    for arg in sys.argv:
        if 'date' in arg:
            date = arg.split('_')[1]
        if 'Case' in arg:
            case = arg
        if 'version' in arg:
            version1 = arg.split('_')[1]
        if 'factor' in arg:
            multiplication_factor = float(arg.split('_')[1])
        if 'nJobs' in arg:
            n_jobs = int(arg.split('_')[1])
        if 'jobNum' in arg:
            job_num = int(arg.split('_')[1])
        if 'FULL_YEAR' in arg:
            full_year = True
        if 'H2_ONLY' in arg:
            h2_only = True
        if 'FIXED_SOLAR' in arg:
            fixed_solar = float(arg.split('_')[-1])
        if 'FIXED_WIND' in arg:
            fixed_wind = float(arg.split('_')[-1])
        if 'FIXED_ELECTROLYZER' in arg:
            fixed_electrolyzer = float(arg.split('_')[-1])
        if 'FIXED_NATGASCCS' in arg:
            fixed_natGasCCS = float(arg.split('_')[-1])



    # Efficiencies so I don't have to pull them from the cfgs for the moment, FIXME
    EFFICIENCY_FUEL_ELECTROLYZER=0.607 # Updated 4 March 2020 based on new values
    EFFICIENCY_FUEL_CHEM_CONVERSION=0.682

    save_dir = f'./plots_{date}_{version1}_NEW/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


    cases = [
        "Case7_NatGasCCS",
        "Case9_NatGasCCSWindSolarStorage",
        "Case5_WindSolarStorage",
    ]

    axs = []

    dfs = {}

    for case in cases:

        input_file = 'fuel_test_20200802_AllCases_EIAPrices.csv'
        #input_file = 'fuel_test_20200802_AllCases_EIAPrices_100PctReli.csv'
        if 'Case0' in case:
            input_file = 'fuel_test_20200302_Case0_NuclearFlatDemand.csv'
        version = f'{version1}_{case}'
        global_name = 'fuel_test_{}_{}_{}_{}'.format(date, version, n_jobs, job_num)
        path = 'Output_Data/{}/'.format(global_name)
        results = path+'results/'
        results_search = 'Output_Data/fuel_test_{}_{}*/results/'.format(date, version)

        # Print settings:
        print(f'\nGlobal name                    {global_name}')
        print(f'Output path                    {path}')
        print(f'Results path                   {results}')
        print(f'Demand multiplication factor:  {round(multiplication_factor,3)}')
        print(f'H2_ONLY:                       {h2_only}')
        print(f'Number of jobs:                {n_jobs}')
        print(f'Job number:                    {job_num}')
        print(f'\n - RUN_SEM =          {run_sem}')
        print(f' - MAKE_RESULTS_FILE ={make_results_file}')
        print(f' - MAKE_PLOTS =       {make_plots}\n')


        fixed = 'natgas_ccs'
        if case == 'Case5_WindSolarStorage':
            fixed = 'nuclear'
        print(f"\nPlotting using {fixed} as the dispatchable tech\n")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        df = pd.read_csv('results/Results_{}_app.csv'.format(global_name), index_col=False)
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
        df.to_csv('results/Results_{}_tmp.csv'.format(global_name))
        for i in range(len(df.index)):
            if df.loc[i, 'fuel demand (kWh)'] == 0.0 or df.loc[i, 'mean demand (kW)'] == 0.0:
                print(f"Dropping idx {i}: fuel {df.loc[i, 'fuel demand (kWh)']} elec {df.loc[i, 'mean demand (kW)']}")
                df = df.drop([i,])

        # Stacked components
        f_elec = df['fixed cost fuel electrolyzer ($/kW/h)'] * df['capacity fuel electrolyzer (kW)'] / df['fuel demand (kWh)']
        f_chem = df['fixed cost fuel chem plant ($/kW/h)'] * df['capacity fuel chem plant (kW)'] / df['fuel demand (kWh)']
        f_store = df['fixed cost fuel h2 storage ($/kWh/h)'] * df['capacity fuel h2 storage (kWh)'] / df['fuel demand (kWh)']
        v_chem = df['var cost fuel chem plant ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_CONVERSION / df['fuel demand (kWh)']
        v_co2 = df['var cost fuel co2 ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_CONVERSION / df['fuel demand (kWh)']
        f_tot = f_elec+f_chem+f_store+v_chem+v_co2
        tot_eff_fuel_process = EFFICIENCY_FUEL_ELECTROLYZER * EFFICIENCY_FUEL_CHEM_CONVERSION
        df['fuel_load_cost'] = (df['fuel price ($/kWh)'] - f_tot) * tot_eff_fuel_process

        dfs[case] = df





    ###########################################################################
    ###             PLOTTING                                                ###
    ###########################################################################




    m = {
            'x_label' : 'flexible load (kW) / total load (kW)',
            'app' : '',
            'x_lim' : [0., 1.],
            'x_type' : 'linear',
    }
    k = 'fuel load / total load'

    kwargs = {}
    kwargs['dfs'] = dfs
    kwargs['save_dir'] = save_dir
    kwargs['stacked_min'] = m['x_lim'][0]
    kwargs['stacked_max'] = m['x_lim'][1]
    kwargs['x_vals'] = df[k]
    kwargs['x_label'] = m['x_label']
    kwargs['x_type'] = m['x_type']
    kwargs['x_var'] = k
    if h2_only:
        kwargs['h2_only'] = True


    tot_load = dfs[cases[0]]['dispatch to fuel h2 storage (kW)'] + 1. # electric power demand = 1 


    ### Fuel cost compare scatter and use to fill electricity costs in stacked
    kwargs['save_name'] = 'stackedCostPlot' + m['app']
    #costs_plot(k, **kwargs)
    kwargs['ALT'] = True
    costs_plot(k, **kwargs)
    exit()
    del kwargs['ALT']
    kwargs['save_name'] = 'costPlot' + m['app']
    costs_plot_alt(k, **kwargs)
    


    kwargs['save_name'] = 'stackedGenerationElecNorm' + m['app']
    kwargs['nuclear'] = df[f'dispatch {fixed} (kW)'] + df[f'curtailment {fixed} (kW)']
    kwargs['wind'] = df['dispatch wind (kW)'] + df['curtailment wind (kW)']
    kwargs['solar'] = df['dispatch solar (kW)'] + df['curtailment solar (kW)']
    kwargs['y_label'] = 'total available generation (kW) /\nfirm electric load (kW)'
    kwargs['legend_app'] = ''
    kwargs['ylim'] = [0, 5]
    stacked_plot(**kwargs)

    
    ### Stacked curtailment fraction plot
    del kwargs['nuclear']
    del kwargs['wind']
    del kwargs['solar']

    
    ### Stacked curtailment fraction plot - new y-axis, Total Generation
    kwargs['save_name'] = 'stackedEndUseFractionTotGenAlt' + m['app']
    kwargs['nuclear_curt'] = df[f'curtailment {fixed} (kW)'] / tot_load
    kwargs['renewable_curt'] = (df['curtailment wind (kW)'] + df['curtailment solar (kW)']) / tot_load
    kwargs['battery_losses'] = (df['dispatch to storage (kW)'] - df['dispatch from storage (kW)']) / tot_load
    kwargs['fuel_load'] = df['dispatch to fuel h2 storage (kW)'] / tot_load
    kwargs['elec_load'] = 1. / tot_load
    kwargs['y_label'] = 'total available generation (kW) / total load (kW)'
    kwargs['legend_app'] = ''
    kwargs['ylim'] = [0, 3]
    kwargs['ALT'] = True
    kwargs['dem_renew_frac'] = df['dem_renew_frac']
    kwargs['electro_renew_frac'] = df['electro_renew_frac']
    if 'Case1' in kwargs['save_dir']:
        kwargs['ylim'] = [0, 2]
    stacked_plot(**kwargs)
    del kwargs['nuclear_curt']
    del kwargs['renewable_curt']
    del kwargs['battery_losses']
    del kwargs['fuel_load']
    del kwargs['elec_load']
    del kwargs['dem_renew_frac']
    del kwargs['electro_renew_frac']
    del kwargs['ALT']

    


    ## Fuel system capacity factor ratios
    ylims = [0.0, 1.]

    kwargs['df'] = df


    # EF system capacity factor ratios
    kwargs['y_label'] = 'capacity factor'
    simple_plot(df[k].values,
            [df['dispatch to fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_ELECTROLYZER/df['capacity fuel electrolyzer (kW)'].values, 
                df['dispatch from fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_CHEM_CONVERSION/df['capacity fuel chem plant (kW)'].values,
                df['fuel h2 storage (kWh)'].values/df['capacity fuel h2 storage (kWh)'].values,
                ], # y values 
            ['electrolysis facility', 'chemical plant', r'H$_{2}$ storage',], # labels
            'systemCFsEF' + m['app'], False, ylims, **kwargs)
        

