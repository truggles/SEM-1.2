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



# Use Pandas to retrieve the output values b/c it handles
# fully populated tables well
def get_cap_and_costs_fuels(path, file_name):
    return get_all_cap_and_costs_fuels(path+'/'+file_name)


def get_all_cap_and_costs_fuels(file_name):
    dta = pd.read_csv(file_name, index_col=0, header=None,
                   dtype={
                        'case name':np.str,
                        'problem status':np.str,
                        #'fuel cost ($/GGE)':np.float64,
                        #'fuel demand (kWh)':np.float64,
                        #'system cost ($/kW/h)':np.float64,
                        #'capacity nuclear (kW)':np.float64,
                        #'capacity solar (kW)':np.float64,
                        #'capacity wind (kW)':np.float64,
                        #'capacity fuel electrolyzer (kW)':np.float64,
                        #'capacity fuel chem plant (kW)':np.float64,
                        #'capacity fuel h2 storage (kWh)':np.float64,
                        #'dispatch unmet demand (kW)':np.float64,
                       }).T

    return dta


# Use normal python csv functions b/c this is a sparsely populated
# csv file
def get_SEM_csv_file(file_name):

    with open(file_name, 'r') as f:
        info = list(csv.reader(f, delimiter=","))
    
    return info



# Multiplier is either applied to fuel cost or fuel demand
# based on 'do_demand_constraint'
def set_fuel_info(cfg, global_name, fuel_str, multiplier, do_demand_constraint, run_one_year):

    new_cfg = []

    case_data_line = -999 # Starts really negative so the 2nd 'if' is never triggered until ready
    fuel_value_position = -999
    fuel_demand_position = -999
    end_month_position = -999
    for i, line in enumerate(cfg):

        if line[0] == 'GLOBAL_NAME':
            line[1] = global_name

        if line[0] == 'CASE_NAME':
            case_data_line = i
            fuel_value_position = line.index('FUEL_VALUE')
            fuel_demand_position = line.index('FUEL_DEMAND')
            end_month_position = line.index('END_MONTH')
            print(" --- demand pos: {}, value pos {}, multiplier {}x, do_demand_constraint {}, run_one_year {}".format(
                    fuel_demand_position, fuel_value_position, multiplier, do_demand_constraint, run_one_year))
        
        if i == case_data_line+2:
            # Set case name
            line[0] = fuel_str
            if do_demand_constraint:
                line[fuel_value_position] = 0
                line[fuel_demand_position] = multiplier
            else:
                line[fuel_value_position] = multiplier
                line[fuel_demand_position] = 0
            if run_one_year:
                line[end_month_position] = 12 # End on December
        new_cfg.append(line)

    return new_cfg


def write_file(file_name, cfg):

    with open(file_name, 'w') as f:
        for line in cfg:
            to_write = ''
            for val in line:
                to_write += str(val)+','
            f.write(to_write+'\n')
        f.close()


def get_output_file_names(path):

    #print("Looking here for csv files: {}*.csv".format(path))
    files = glob(path+'*.csv')
    files.sort()
    if len(files) > 1:
        print("This many files were found matching {}*.csv: {}".format(path, len(files)))
    return files

def get_results(files, global_name):

    results = {}

    keys = []
    for f in files:
        info = get_all_cap_and_costs_fuels(f)
        if not hasattr(info, 'capacity nuclear (kW)'):
            info['capacity nuclear (kW)'] = 0.
            info['dispatch nuclear (kW)'] = 0.
            info['curtailment nuclear (kW)'] = 0.
        if not hasattr(info, 'capacity wind (kW)'):
            info['capacity wind (kW)'] = 0.
            info['dispatch wind (kW)'] = 0.
            info['curtailment wind (kW)'] = 0.
        if not hasattr(info, 'capacity solar (kW)'):
            info['capacity solar (kW)'] = 0.
            info['dispatch solar (kW)'] = 0.
            info['curtailment solar (kW)'] = 0.
        if not hasattr(info, 'capacity storage (kWh)'):
            info['capacity storage (kWh)'] = 0.
            info['dispatch to storage (kW)'] = 0.
            info['dispatch from storage (kW)'] = 0.
            info['energy storage (kWh)'] = 0.
        #print(info)
        keys.append(info['case name'].values[0])
        results[info['case name'].values[0]] = [
                       info['problem status'].values[0],
                       info['fuel cost ($/GGE)'].values[0],
                       info['fuel demand (kWh)'].values[0],
                       info['system cost ($ or $/kWh)'].values[0],
                       info['capacity nuclear (kW)'].values[0],
                       info['capacity solar (kW)'].values[0],
                       info['capacity wind (kW)'].values[0],
                       info['capacity storage (kWh)'].values[0],
                       info['capacity fuel electrolyzer (kW)'].values[0],
                       info['capacity fuel chem plant (kW)'].values[0],
                       info['capacity fuel h2 storage (kWh)'].values[0],
                       info['dispatch to fuel h2 storage (kW)'].values[0],
                       info['dispatch from fuel h2 storage (kW)'].values[0],
                       info['dispatch unmet demand (kW)'].values[0],
                       info['dispatch nuclear (kW)'].values[0],
                       info['dispatch wind (kW)'].values[0],
                       info['dispatch solar (kW)'].values[0],
                       info['dispatch to storage (kW)'].values[0],
                       info['dispatch from storage (kW)'].values[0],
                       info['energy storage (kWh)'].values[0],
                       info['curtailment nuclear (kW)'].values[0],
                       info['curtailment wind (kW)'].values[0],
                       info['curtailment solar (kW)'].values[0],
                       info['fixed cost fuel electrolyzer ($/kW/h)'].values[0],
                       info['fixed cost fuel chem plant ($/kW/h)'].values[0],
                       info['fixed cost fuel h2 storage ($/kWh/h)'].values[0],
                       info['var cost fuel electrolyzer ($/kW/h)'].values[0],
                       info['var cost fuel chem plant ($/kW/h)'].values[0],
                       info['var cost fuel co2 ($/kW/h)'].values[0],
                       info['fuel h2 storage (kWh)'].values[0],
                       info['fuel price ($/kWh)'].values[0],
                       info['mean price ($/kWh)'].values[0],
                       info['max price ($/kWh)'].values[0]
        ]

    print('Writing results to "Results_{}.csv"'.format(global_name))
    ofile = open('Results_{}.csv'.format(global_name), 'w')
    keys = sorted(keys)
    ofile.write('case name,problem status,fuel cost ($/GGE),fuel demand (kWh),system cost ($/kW/h),capacity nuclear (kW),capacity solar (kW),capacity wind (kW),capacity storage (kWh),capacity fuel electrolyzer (kW),capacity fuel chem plant (kW),capacity fuel h2 storage (kWh),dispatch to fuel h2 storage (kW),dispatch from fuel h2 storage (kW),dispatch unmet demand (kW),dispatch nuclear (kW),dispatch wind (kW),dispatch solar (kW),dispatch to storage (kW),dispatch from storage (kW),energy storage (kWh),curtailment nuclear (kW),curtailment wind (kW),curtailment solar (kW),fixed cost fuel electrolyzer ($/kW/h),fixed cost fuel chem plant ($/kW/h),fixed cost fuel h2 storage ($/kWh/h),var cost fuel electrolyzer ($/kW/h),var cost fuel chem plant ($/kW/h),var cost fuel co2 ($/kW/h),fuel h2 storage (kWh),fuel price ($/kWh),mean price ($/kWh),max price ($/kWh)\n')
    for key in keys:
        to_print = ''
        for info in results[key]:
            to_print += str(info)+','
        ofile.write("{},{}\n".format(key, to_print))
    ofile.close()
    return results

# Get info from file, so we don't have to repeat get_results
# many many times
def simplify_results(results_file, reliability_values, wind_values, solar_values):
    ifile = open(results_file, 'r')

    simp = {}
    for reliability in reliability_values:
        simp[reliability] = {}
        for solar in solar_values:
            simp[reliability][solar] = {}
            for wind in wind_values:
                simp[reliability][solar][wind] = [0.0, 0]

    for line in ifile:
        if 'case name' in line: continue # skip hearder line
        info = line.split(',')
        reli = float(info[2])
        solar = float(info[5])
        wind = float(info[6])
        unmet = float(info[7])

        # Remove entries which were from Step 1 which calculated
        # capacities with a target reliability
        if round(reli, 10) == round(unmet, 10): continue

        if reli == 0.0: continue # TMP FIXME
        to_add = abs(unmet/reli - 1.)
        simp[reli][solar][wind][0] += to_add
        simp[reli][solar][wind][1] += 1

    for reli in reliability_values:
        for solar in solar_values:
            for wind in wind_values:
                if simp[reli][solar][wind][1] == 0: continue
                simp[reli][solar][wind][0] = simp[reli][solar][wind][0]/simp[reli][solar][wind][1] # np.sqrt(simp[reli][solar][wind])

    return simp


def simple_plot(save_dir, x, ys, labels, x_label, y_label, title, save, logY=False, ylims=[-1,-1]):

    print("Plotting x,y = {},{}".format(x_label,y_label))

    fuel_x = 'fuel demand (kWh)'.replace('kWh','kWh/h')
    if x_label == 'fuel demand (kWh)':
        x_label = fuel_x

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)

    for y, label in zip(ys, labels):
        #print(label)
        #print(y)
        ax.scatter(x, y, label=label)

    if logY:
        plt.yscale('log', nonposy='clip')
        y_min = 999
        y_max = -999
        for y in ys:
            y_tmp = y[np.nonzero(y)]
            y_tmp = y_tmp[np.isfinite(y_tmp)]
            if min(y_tmp) < y_min:
                y_min = min(y_tmp)
            if max(y_tmp) > y_max:
                y_max = max(y_tmp)

        if not (y_min == y_max):
            #if 'dual' in title:
            #    ax.set_ylim(y_min*.5, 1.0)
            #    plt.tick_params(axis='y', which='minor')
            #    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
            #else:
            print(y_min, y_max)
            ax.set_ylim(y_min*.5, y_max*2)

            #y_tmp = y[np.nonzero(y)]
            #y_tmp = y_tmp[np.isfinite(y_tmp)]
            #ax.set_ylim(min(y_tmp)*.5, max(y_tmp)*2)

    if ylims != [-1,-1]:
        ax.set_ylim(ylims[0], ylims[1])

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(x[np.nonzero(x)])*.5, max(x)*2)

    plt.tight_layout()
    plt.grid()
    plt.legend()
    fig.savefig('{}/{}.png'.format(save_dir, save))



# FIXME - just taking the last set of y values for the 2nd axis.  This could be much better...
def simple_plot_with_2nd_yaxis(save_dir, x, ys, labels, x_label, y_label_1, y_label_2, title, save):

    print("Plotting x,y1 = {},{} and x,y2 = {},{}".format(x_label,y_label_1,x_label,y_label_2))

    fuel_x = 'fuel demand (kWh)'.replace('kWh','kWh/h')
    if x_label == 'fuel demand (kWh)':
        x_label = fuel_x

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label_1)
    plt.title(title)

    # First y-axis
    for i in range(len(ys)-1): # skip last set of y values
        ax.scatter(x, ys[i], label=labels[i])
    plt.legend(loc='upper left')
    ax.set_ylim(ax.get_ylim()[0]/1.5, ax.get_ylim()[1]*1.3)
    plt.grid()
    
    # Second y-axis
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(y_label_2, color='C3')  # we already handled the x-label with ax1
    ax2.scatter(x, ys[-1], color='C3', marker='o', label=labels[-1])
    ax2.tick_params(axis='y', labelcolor='C3')
    plt.legend(loc='upper right')
    ax2.set_ylim(ax2.get_ylim()[0]/1.5, ax2.get_ylim()[1]*1.3)

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(x[np.nonzero(x)])*.5, max(x)*2)

    plt.tight_layout()
    plt.legend()
    fig.savefig('{}/{}.png'.format(save_dir, save))


# The addition here compared to the simple_plot is coloring/sizing the dots
def biv_curtailment_cost_plot(tot_curtailment, fuel_costs, fuel_demand, x_label, save_dir, save_name):

    print("Plotting biv_curtailment_cost_plot")

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel('Fuel Cost ($/kWh)')
    plt.title('Total Curtailment vs. Fuel Costs')

    sc = ax.scatter(tot_curtailment, fuel_costs, c=fuel_demand, norm=matplotlib.colors.LogNorm(), s=250)
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel("Fuel Demand (kWh/h)")

    plt.grid()
    
    #ax.set_xlim(0., 1.0)
    #ax.set_ylim(.17, .26)

    plt.tight_layout()

    fig.savefig('{}/{}.png'.format(save_dir, save_name))



# Poorly written, the args are all required and are below.
#x_vals, nuclear, wind, solar, x_label, y_label, 
#title, legend_app, stacked_min, stacked_max, save_name, save_dir):
def stacked_plot(**kwargs):

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])
    plt.title(kwargs['title'])

    ax.fill_between(kwargs['x_vals'], 0., kwargs['nuclear'], color='red', label=f'nuclear {kwargs["legend_app"]}')
    if 'renewables' in kwargs.keys():
        ax.fill_between(kwargs['x_vals'], kwargs['nuclear'], kwargs['nuclear']+kwargs['renewables'], color='green', label=f'renewables {kwargs["legend_app"]}')
    else:
        ax.fill_between(kwargs['x_vals'], kwargs['nuclear'], kwargs['nuclear']+kwargs['wind'], color='blue', label=f'wind {kwargs["legend_app"]}')
        ax.fill_between(kwargs['x_vals'], kwargs['nuclear']+kwargs['wind'], kwargs['nuclear']+kwargs['wind']+kwargs['solar'], color='yellow', label=f'solar {kwargs["legend_app"]}')

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(kwargs['stacked_min'], kwargs['stacked_max'])

    if 'logy' in kwargs.keys():
        if kwargs['logy'] == True:
            plt.yscale('log', nonposy='clip')
            ax.set_ylim(min(0.1, ax.get_ylim()[0]), max(100, ax.get_ylim()[1]))

    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig.savefig(f'{kwargs["save_dir"]}/{kwargs["save_name"]}.png')


# Poorly written, the args are all required and are below.
#save_name, save_dir
def costs_plot(df, **kwargs):

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('fuel demand (kWh/h)')
    ax.set_ylabel('fuel cost ($/kWh)')
    plt.title('Cost Breakdown For Fuels')

    # $/GGE fuel line above
    # subtract off the base-case (least fuel demand, should be zero in the future FIXME) system cost
    dollars_per_fuel = df['system cost ($/kW/h)'] - df.loc[df['fuel demand (kWh)'] == 0.0, 'system cost ($/kW/h)'].values
    # divide by fuel produced in that scenario
    dollars_per_fuel = dollars_per_fuel / df['fuel demand (kWh)']
    ax.scatter(df['fuel demand (kWh)'], dollars_per_fuel, color='black', label='Fuel Cost ($/kWh)')


    # Stacked components
    f_elec = df['fixed cost fuel electrolyzer ($/kW/h)'] * df['capacity fuel electrolyzer (kW)'] / df['fuel demand (kWh)']
    f_chem = df['fixed cost fuel chem plant ($/kW/h)'] * df['capacity fuel chem plant (kW)'] / df['fuel demand (kWh)']
    #f_store = df['fixed cost fuel h2 storage ($/kWh/h)'] * df['capacity fuel h2 storage (kWh)'] / df['fuel demand (kWh)']
    f_tot = f_elec+f_chem#+f_store
    v_elec = df['var cost fuel electrolyzer ($/kW/h)'] * df['dispatch to fuel h2 storage (kW)'] * EFFICIENCY_FUEL_ELECTROLYZER / df['fuel demand (kWh)']
    v_chem = df['var cost fuel chem plant ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_PLANT / df['fuel demand (kWh)']
    v_co2 = df['var cost fuel co2 ($/kW/h)'] * df['dispatch from fuel h2 storage (kW)'] * EFFICIENCY_FUEL_CHEM_PLANT / df['fuel demand (kWh)']


    # Build stack
    ax.fill_between(df['fuel demand (kWh)'], 0, f_elec, label='fixed cost electrolyzer')
    ax.fill_between(df['fuel demand (kWh)'], f_elec, f_elec+f_chem, label='fixed cost chem plant')
    #ax.fill_between(df['fuel demand (kWh)'], f_elec+f_chem, f_elec+f_chem+f_store, label='fixed cost storage') # fixed cost storage set at 2.72E-7
    ax.fill_between(df['fuel demand (kWh)'], f_tot, f_tot+v_chem, label='variable cost chem plant')
    ax.fill_between(df['fuel demand (kWh)'], f_tot+v_chem, f_tot+v_chem+v_co2, label='variable cost CO$_{2}$')
    #ax.fill_between(df['fuel demand (kWh)'], f_tot+v_chem+v_co2, f_tot+v_chem+v_co2+v_elec, label='variable cost electrolyzer') # variable cost electrolyzer is set at 1.0E-6 b/c MEM handles electric costs already


    # Fill remainder with assumed electricity cost from MEM
    ax.fill_between(df['fuel demand (kWh)'], f_tot+v_chem+v_co2, dollars_per_fuel, label='variable cost electrolyzer (MEM electricity)')
    # To print the estimated MEM electricity cost per point
    #print(dollars_per_fuel)
    #print(dollars_per_fuel - (f_tot+v_chem+v_co2))
    ax.scatter(df['fuel demand (kWh)'], dollars_per_fuel, color='black', label='_nolegend_')


    plt.xscale('log', nonposx='clip')
    ax.set_xlim(df.loc[1, 'fuel demand (kWh)'], df.loc[len(df.index)-1, 'fuel demand (kWh)'])
    #ax.set_ylim(0, max(dollars_per_fuel.loc[1:len(dollars_per_fuel)-1])*1.7)
    ax.set_ylim(0, 0.55)

    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.grid()
    fig.savefig(f'{kwargs["save_dir"]}{kwargs["save_name"]}.png')
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

    date = '20191022' # default
    case = 'Case6_NuclearWindSolarStorage' # default
    version = 'v3'
    for arg in sys.argv:
        if 'date' in arg:
            date = arg.split('_')[1]
        if 'Case' in arg:
            case = arg
        if 'version' in arg:
            version = arg.split('_')[1]

    input_file = f'fuel_test_191017_{case}.csv'
    version = f'{version}_{case}'
    global_name = 'fuel_test_{}_{}'.format(date, version)
    path = 'Output_Data/{}/'.format(global_name)
    results = path+'results/'

    # Print settings:
    print(f'\nGlobal name {global_name}')
    print(f'Output path {path}')
    print(f'Results path {results}')
    print(f'Input File: {input_file}')
    print(f'\n - RUN_SEM={run_sem}')
    print(f' - MAKE_RESULTS_FILE={make_results_file}')
    print(f' - MAKE_PLOTS={make_plots}\n')

    # Efficiencies so I don't have to pull them from the cfgs for the moment, FIXME
    EFFICIENCY_FUEL_ELECTROLYZER=0.676783005
    EFFICIENCY_FUEL_CHEM_PLANT=0.659
    EFFICIENCY_COMBINED = 0.446 # 0.676783005 * 0.659
    MEAN_JAN_2016_WIND_CF = 0.429287634 # From 1st month of 2016 wind, all Jan

    do_demand_constraint = True # All true for now
    run_one_year = True # If true, run all cases for 1 full year instead of just Jan 2016



    if run_sem:
        multipliers = []
        multipliers = [0., 0.01,]
        #multipliers = [0., 1,]
        while True:
            if multipliers[-1] > 10:
                break
            #multipliers.append(round(multipliers[-1]*1.1,5))
            multipliers.append(round(multipliers[-1]*1.2,5))
            #multipliers.append(round(multipliers[-1]*1.5,5))
        print("Length of multipliers {}".format(len(multipliers)))
        print(multipliers)

        for i, multiplier in enumerate(multipliers):

            if not run_sem:
                break

            if do_demand_constraint:
                fuel_str = f'Run_{i:03}_fuel_demand_'+str(multiplier)+'kWh'
            else:
                fuel_str = f'Run_{i:03}_fuel_cost_'+str(round(fuel_multiplier,6)).replace('.','p')+'USD'

            # 1st Step
            cfg = get_SEM_csv_file(input_file)
            case_name = fuel_str
            case_file = case_name+'.csv'
            cfg = set_fuel_info(cfg, global_name, fuel_str, multiplier, do_demand_constraint, run_one_year)
            write_file(case_file, cfg)
            subprocess.call(["python", "Simple_Energy_Model.py", case_file])

            files = get_output_file_names(path+'{}_2019'.format(global_name))
            print(files)

            # Copy output file
            if not os.path.exists(results):
                os.makedirs(results)
            move(files[-1], results)
            os.remove(case_file)


    if make_results_file:
        base = os.getcwd()
        results = base+'/'+results
        files = get_output_file_names(results+'{}_2019'.format(global_name))
        results = get_results(files, global_name)

    if not make_plots:
        assert(False), "Kill before plotting"

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    df = pd.read_csv('Results_{}.csv'.format(global_name), index_col=False)

    save_dir = './plots_{}/'.format(version)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    plot_map = { # title / save : x, y, x_title, y_title
        'fuel demand vs. hourly dispatch' : ['fuel demand (kWh)', 'dispatch from fuel h2 storage (kW)'],
        'fuel demand vs. system cost' : ['fuel demand (kWh)', 'system cost ($/kW/h)'],
        'fuel demand vs. capacity nuclear' : ['fuel demand (kWh)', 'capacity nuclear (kW)'],
        'fuel demand vs. capacity solar' : ['fuel demand (kWh)', 'capacity solar (kW)'],
        'fuel demand vs. capacity wind' : ['fuel demand (kWh)', 'capacity wind (kW)'],
        'fuel demand vs. capacity electrolyzer' : ['fuel demand (kWh)', 'capacity fuel electrolyzer (kW)'],
        'fuel demand vs. capacity chem plant' : ['fuel demand (kWh)', 'capacity fuel chem plant (kW)'],
        'fuel demand vs. capacity h2 storage' : ['fuel demand (kWh)', 'capacity fuel h2 storage (kWh)'],
        'fuel demand vs. dispatch unmet demand' : ['fuel demand (kWh)', 'dispatch unmet demand (kW)'],
        #'fuel cost vs. hourly dispatch' : ['fuel cost ($/GGE)', 'dispatch from fuel h2 storage (kW)'],

    }


    #for k, v in plot_map.items():
    #    logY = True
    #    simple_plot(save_dir, df[v[0]].values, [df[v[1]].values,], [v[1],], v[0], v[1], k, k.replace('.','').replace(' ','_'), logY)
    #    simple_plot(save_dir, df[v[0]].values, [df[v[1]].values/df[v[0]].values,], [v[1]+'/'+v[0],], v[0], v[1]+'/'+v[0], 
    #            k+'/fuel demand (kWh)', k.replace('.','').replace(' ','_')+'_div_fuel_dem', logY)


    # $/GGE fuel
    # subtract off the base-case (least fuel demand, should be zero in the future FIXME) system cost
    dollars_per_fuel = df['system cost ($/kW/h)'] - df.loc[df['fuel demand (kWh)'] == 0.0, 'system cost ($/kW/h)'].values
    # divide by fuel produced in that scenario
    dollars_per_fuel = dollars_per_fuel / df['fuel demand (kWh)']
    simple_plot(save_dir, df['fuel demand (kWh)'].values, [dollars_per_fuel.values,], ['Cost of Fuel ($/kWh)',], 'fuel demand (kWh)', 'Cost of Fuel ($/kWh)', 
            'fuel demand vs. fuel cost', 'fuelDemandVsFuelCost')

    # Fuel price dual_value
    ylims = [0.05, 1000]
    simple_plot(save_dir, df['fuel demand (kWh)'].values, [df['mean price ($/kWh)'].values, df['max price ($/kWh)'].values, df['fuel price ($/kWh)'].values,], ['Mean Demand Dual ($/kWh)', 'Max Demand Dual ($/kWh)', 'Fuel Price Dual ($/kWh)',], 'fuel demand (kWh)', 'Dual Values ($/kWh)', 
            'fuel demand vs. dual values', 'fuelDemandVsDualValues', True, ylims)


    # Curtailment vs. fuel cost with marker color as fuel fraction
    tot_curtailment = (df['curtailment nuclear (kW)'] + df['curtailment wind (kW)'] + \
            df['curtailment solar (kW)'])
            #/ (df['capacity nuclear (kW)'] + df['capacity wind (kW)'] + \
            #df['capacity solar (kW)'])
    biv_curtailment_cost_plot(tot_curtailment.loc[1:len(df.index)-1], \
            df['fuel price ($/kWh)'].loc[1:len(df.index)-1], df['fuel demand (kWh)'].loc[1:len(df.index)-1], \
            'Total Curtailment', save_dir, 'bivariate_curtailment_vs_cost_with_demand_markers')


    # Curtailment vs. fuel cost with marker color as fuel fraction
    tot_curtailment = (df['curtailment nuclear (kW)'].fillna(0) + df['curtailment wind (kW)'].fillna(0) + \
            df['curtailment solar (kW)'].fillna(0)) / \
            (df['capacity nuclear (kW)'].fillna(0) + df['capacity wind (kW)'].fillna(0) + \
            df['capacity solar (kW)'].fillna(0))
    biv_curtailment_cost_plot(tot_curtailment.loc[1:len(df.index)-1], \
            df['fuel price ($/kWh)'].loc[1:len(df.index)-1], df['fuel demand (kWh)'].loc[1:len(df.index)-1], \
            'Total Curtailment/Total Capacity', save_dir, 'bivariate_curtailment_div_cap_vs_cost_with_demand_markers')



    ### These should be defaults for all of them
    ### Reset them if needed
    idx_max = 125
    kwargs = {}
    kwargs['save_dir'] = save_dir
    kwargs['stacked_min'] = min(df.loc[0:idx_max, 'fuel demand (kWh)'].values[np.nonzero(df.loc[0:idx_max, 'fuel demand (kWh)'].values)])
    kwargs['stacked_max'] = max(df.loc[0:idx_max, 'fuel demand (kWh)'].values)
    kwargs['x_vals'] = df.loc[0:idx_max, 'fuel demand (kWh)']
    kwargs['x_label'] = 'fuel demand (kWh/h)'


    # Fuel cost compare scatter and use to fill electricity costs in stacked
    kwargs['save_name'] = 'stackedCostPlot'
    costs_plot(df, **kwargs)


    ### Stacked dispatch plot
    kwargs['nuclear'] = df.loc[0:idx_max, 'dispatch nuclear (kW)']
    kwargs['wind'] = df.loc[0:idx_max, 'dispatch wind (kW)']
    kwargs['solar'] = df.loc[0:idx_max, 'dispatch solar (kW)']
    kwargs['y_label'] = 'normalized dispatch (kW)'
    kwargs['title'] = 'Normalized Dispatch'
    kwargs['legend_app'] = 'annual dispatch'
    kwargs['save_name'] = 'stackedDispatchNormalized'
    kwargs['logy'] = True
    stacked_plot(**kwargs)
    del kwargs['logy']
    

    ### Stacked generation capacity plot
    kwargs['nuclear'] = df.loc[0:idx_max, 'capacity nuclear (kW)']
    kwargs['wind'] = df.loc[0:idx_max, 'capacity wind (kW)']
    kwargs['solar'] = df.loc[0:idx_max, 'capacity solar (kW)']
    kwargs['y_label'] = 'normalized capacity (kW)'
    kwargs['title'] = 'Normalized Capacity'
    kwargs['legend_app'] = 'capacity'
    kwargs['save_name'] = 'stackedCapacityNormalized'
    kwargs['logy'] = True
    stacked_plot(**kwargs)
    del kwargs['logy']



    ### Stacked curtailment plot
    kwargs['nuclear'] = df.loc[0:idx_max, 'curtailment nuclear (kW)']
    kwargs['wind'] = df.loc[0:idx_max, 'curtailment wind (kW)']
    kwargs['solar'] = df.loc[0:idx_max, 'curtailment solar (kW)']
    kwargs['y_label'] = 'curtailment of dispatch (kW)'
    kwargs['title'] = 'Curtailment of Dispatchable Power'
    kwargs['legend_app'] = 'curtailment'
    kwargs['save_name'] = 'stackedCurtailment'
    stacked_plot(**kwargs)

    


    ### Stacked curtailment / capacity plot
    kwargs['nuclear'] = df['curtailment nuclear (kW)']/df['capacity nuclear (kW)']
    kwargs['wind'] = df['curtailment wind (kW)']/df['capacity wind (kW)']
    kwargs['solar'] = df['curtailment solar (kW)']/df['capacity solar (kW)']
    kwargs['nuclear'].fillna(value=0, inplace=True)
    kwargs['wind'].fillna(value=0, inplace=True)
    kwargs['solar'].fillna(value=0, inplace=True)
    #kwargs['renewables'] = kwargs['wind'] + kwargs['solar']
    kwargs['y_label'] = 'curtailment of dispatch / capacity'
    kwargs['title'] = 'Curtailment of Dispatchable Power / Capacities'
    kwargs['legend_app'] = 'curtailment/capacity'
    kwargs['save_name'] = 'stackedCurtailmentDivCapacity'
    stacked_plot(**kwargs)


    ### Stacked curtailment / dispatch plot
    kwargs['nuclear'] = df['curtailment nuclear (kW)']/df['dispatch nuclear (kW)']
    kwargs['wind'] = df['curtailment wind (kW)']/df['dispatch wind (kW)']
    kwargs['solar'] = df['curtailment solar (kW)']/df['dispatch solar (kW)']
    kwargs['nuclear'].fillna(value=0, inplace=True)
    kwargs['wind'].fillna(value=0, inplace=True)
    kwargs['solar'].fillna(value=0, inplace=True)
    #kwargs['renewables'] = kwargs['wind'] + kwargs['solar']
    kwargs['y_label'] = 'curtailment of dispatch / dispatch'
    kwargs['title'] = 'Curtailment of Dispatchable Power / Dispatch'
    kwargs['legend_app'] = 'curtailment/dispatch'
    kwargs['save_name'] = 'stackedCurtailmentDivDispatch'
    kwargs['stacked_max'] = max(df['fuel demand (kWh)'].values)
    stacked_plot(**kwargs)

    
    



    
    # Fuel system capacities ratios
    simple_plot_with_2nd_yaxis(save_dir, df['fuel demand (kWh)'].values,
            [df['capacity fuel electrolyzer (kW)'].values/df['fuel demand (kWh)'].values, 
                df['capacity fuel chem plant (kW)'].values/df['fuel demand (kWh)'].values, 
                df['capacity fuel h2 storage (kWh)'].values/df['fuel demand (kWh)'].values], # y values
            ['cap electrolyzer/fuel dem.', 'cap chem plant/fuel dem.', 'cap H2 storage/fuel dem.'], # labels
            'fuel demand (kWh/h)', 'Fuel System Capacities in (kW) / Fuel Demand (kWh/h)', 'Storage Capacities in (kWh) / Fuel Demand (kWh/h)',
            'Ratios of Fuel System Capacities / Fuel Demand', 'ratiosFuelSystemVsFuelCost')


    # Fuel system capacity factor ratios
    # The way the Core_Model.py is set up currently, efficiencies only need to be applied for the chem plant - 6 Sept 2019
    ylims = [0.0, 1.1]
    simple_plot(save_dir, df['fuel demand (kWh)'].values,
            [df['dispatch to fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_ELECTROLYZER/df['capacity fuel electrolyzer (kW)'].values, 
                df['dispatch from fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_CHEM_PLANT/df['capacity fuel chem plant (kW)'].values,], # y values 
            ['electrolyzer capacity factor', 'chem plant capacity factor'], # labels
            'fuel demand (kWh)', 'Fuel System Capacity Factors', 
            'Fuel System Capacity Factors', 'ratiosFuelSystemCFsVsFuelCost', False, ylims)


    # All system capacity factor ratios
    # The way the Core_Model.py is set up currently, efficiencies only need to be applied for the chem plant - 6 Sept 2019
    simple_plot(save_dir, df['fuel demand (kWh)'].values,
            [df['dispatch to fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_ELECTROLYZER/df['capacity fuel electrolyzer (kW)'].values, 
                df['dispatch from fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_CHEM_PLANT/df['capacity fuel chem plant (kW)'].values,
                df['fuel h2 storage (kWh)'].values/df['capacity fuel h2 storage (kWh)'].values,
                df['dispatch nuclear (kW)'].values/df['capacity nuclear (kW)'].values,
                df['dispatch wind (kW)'].values/df['capacity wind (kW)'].values,
                df['dispatch solar (kW)'].values/df['capacity solar (kW)'].values,
                df['energy storage (kWh)'].values/df['capacity storage (kWh)'].values,
                ], # y values 
            ['electrolyzer', 'chem plant', 'h2 storage',
                'nuclear', 'wind', 'solar', 'battery storage'], # labels
            'fuel demand (kWh)', 'System Capacity Factors', 
            'System Capacity Factors', 'ratiosSystemCFsVsFuelCost', False, ylims)



    # Relative curtailment based on available power
    # This version factors out the wind CF of 0.43
    nuclear = df['curtailment nuclear (kW)']/df['capacity nuclear (kW)']
    wind = df['curtailment wind (kW)']/(df['curtailment wind (kW)']+df['dispatch wind (kW)'])
    solar = df['curtailment solar (kW)']/(df['curtailment solar (kW)']+df['dispatch solar (kW)'])
    #for idx, val in nuclear.items():
    #    if np.isnan(nuclear.at[idx]):
    #        nuclear.at[idx] = 0
    simple_plot(save_dir, df['fuel demand (kWh)'].values,
            [nuclear.values, wind.values, solar.values], # y values 
            ['nuclear curtailment / capacity', 'wind curtailment / available power', 'solar curtailment / available power'], # labels
            'fuel demand (kWh)', 'curtailment of dispatch (kW) / available power (kW)', 
            'Relative Curtailment of Generation Capacities', 'ratiosCurtailmentDivAvailablePower')

    



