import numpy as np
import csv
import subprocess
import os
from glob import glob
from shutil import copy2, move
from collections import OrderedDict
import pandas as pd



# Use Pandas to retrieve the output values b/c it handles
# fully populated tables well
def get_cap_and_costs_fuels(path, file_name):
    return get_all_cap_and_costs_fuels(path+'/'+file_name)


def get_all_cap_and_costs_fuels(file_name):
    dta = pd.read_csv(file_name, index_col=0, header=None,
                   dtype={
                        'case name':np.str,
                        'problem status':np.str,
                        'fuel cost ($/GGE)':np.float64,
                        'fuel demand (kWh)':np.float64,
                        'system cost ($/kW/h)':np.float64,
                        'capacity nuclear (kW)':np.float64,
                        'capacity solar (kW)':np.float64,
                        'capacity wind (kW)':np.float64,
                        'capacity fuel electrolyzer (kW)':np.float64,
                        'capacity fuel chem plant (kW)':np.float64,
                        'capacity fuel h2 storage (kW)':np.float64,
                        'dispatch unmet demand (kW)':np.float64,
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
def set_fuel_info(cfg, global_name, fuel_str, multiplier, do_demand_constraint):

    new_cfg = []

    cnt = 1
    case_data_line = -999 # Starts really negative so the 2nd 'if' is never triggered until ready
    fuel_value_position = -999
    fuel_demand_position = -999
    for line in cfg:

        if line[0] == 'GLOBAL_NAME':
            line[1] = global_name

        if line[0] == 'CASE_NAME':
            case_data_line = cnt
            fuel_value_position = line.index('FUEL_VALUE')
            fuel_demand_position = line.index('FUEL_DEMAND')
            print("fuel info --- demand at position {}, value at position {}, multiplier {}x, do_demand_constraint {}".format(
                    fuel_demand_position, fuel_value_position, multiplier, do_demand_constraint))
        
        if cnt == case_data_line+2:
            # Set case name
            line[0] = fuel_str
            if do_demand_constraint:
                line[fuel_value_position] = 0
                line[fuel_demand_position] = multiplier
            else:
                line[fuel_value_position] = multiplier
                line[fuel_demand_position] = 0
        cnt += 1
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

    print("Looking here for csv files: {}".format(path))
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
        print(info)
        keys.append(info['case name'].values[0])
        results[info['case name'].values[0]] = [
                       info['problem status'].values[0],
                       info['fuel cost ($/GGE)'].values[0],
                       info['fuel demand (kWh)'].values[0],
                       info['system cost ($ or $/kWh)'].values[0],
                       info['capacity nuclear (kW)'].values[0],
                       info['capacity solar (kW)'].values[0],
                       info['capacity wind (kW)'].values[0],
                       info['capacity fuel electrolyzer (kW)'].values[0],
                       info['capacity fuel chem plant (kW)'].values[0],
                       info['capacity fuel h2 storage (kW)'].values[0],
                       info['dispatch to fuel h2 storage (kW)'].values[0],
                       info['dispatch from fuel h2 storage (kW)'].values[0],
                       info['dispatch unmet demand (kW)'].values[0]
        ]

    print('Writing results to "Results_{}.csv"'.format(global_name))
    ofile = open('Results_{}.csv'.format(global_name), 'w')
    keys = sorted(keys)
    ofile.write('case name,problem status,fuel cost ($/GGE),fuel demand (kWh),system cost ($/kW/h),capacity natgas (kW),capacity solar (kW),capacity wind (kW),capacity fuel electrolyzer (kW),capacity fuel chem plant (kW),capacity fuel h2 storage (kW),dispatch to fuel h2 storage (kW),dispatch from fuel h2 storage (kW),dispatch unmet demand (kW)\n')
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


def simple_plot(x, y, x_label, y_label, title, save, add_one=False):

    print("Plotting x,y = {},{}".format(x_label,y_label))

    if add_one:
        y = y + 1.

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)

    ax.scatter(x, y)

    plt.yscale('log', nonposy='clip')
    plt.xscale('log', nonposx='clip')
    
    ax.set_xlim(min(x)*.5, max(x)*2)
    if not (min(y) == max(y)):
        ax.set_ylim(min(y[np.nonzero(y)])*.5, max(y)*2)

    plt.tight_layout()
    plt.grid()
    fig.savefig('plots/{}.png'.format(save))


if '__main__' in __name__:

    do_demand_constraint = True

    input_file = 'zFuels_case_input_test_190827.csv'
    version = 'v9'
    global_name = 'fuel_test_20190905_{}'.format(version)
    path = 'Output_Data/{}/'.format(global_name)
    results = path+'results/'

    run_sem = True
    multipliers = []
    multipliers = [0.001,]
    while True:
        if multipliers[-1] > 10:
            break
        multipliers.append(multipliers[-1]*1.1)
    if run_sem:
        print("Length of multipliers {}".format(len(multipliers)))
        print(multipliers)

    for multiplier in multipliers:

        if not run_sem:
            break

        if do_demand_constraint:
            fuel_str = 'fuel_demand_'+str(multiplier)+'kWh'
        else:
            fuel_str = 'fuel_cost_'+str(round(fuel_multiplier,6)).replace('.','p')+'USD'

        # 1st Step
        cfg = get_SEM_csv_file(input_file)
        case_name = fuel_str
        case_file = case_name+'.csv'
        cfg = set_fuel_info(cfg, global_name, fuel_str, multiplier, do_demand_constraint)
        write_file(case_file, cfg)
        subprocess.call(["python", "Simple_Energy_Model.py", case_file])

        files = get_output_file_names(path+'{}_2019'.format(global_name))

        # Copy output file
        if not os.path.exists(results):
            os.makedirs(results)
        move(files[-1], results)


    base = '/Users/truggles/IDrive-Sync/Carnegie/SEM-1.2_CIW/'
    results = base+results
    files = get_output_file_names(results+'{}_2019'.format(global_name))
    results = get_results(files, global_name)

    import matplotlib.pyplot as plt
    df = pd.read_csv('Results_{}.csv'.format(global_name), index_col=False)

    plot_map = { # title / save : x, y, x_title, y_title
        'fuel demand vs. hourly dispatch' : ['fuel demand (kWh)', 'dispatch from fuel h2 storage (kW)'],
        'fuel demand vs. system cost' : ['fuel demand (kWh)', 'system cost ($/kW/h)'],
        'fuel demand vs. capacity natgas' : ['fuel demand (kWh)', 'capacity natgas (kW)'],
        #'fuel demand vs. capacity solar' : ['fuel demand (kWh)', 'capacity solar (kW)'],
        'fuel demand vs. capacity wind' : ['fuel demand (kWh)', 'capacity wind (kW)'],
        'fuel demand vs. capacity electrolyzer' : ['fuel demand (kWh)', 'capacity fuel electrolyzer (kW)'],
        'fuel demand vs. capacity chem plant' : ['fuel demand (kWh)', 'capacity fuel chem plant (kW)'],
        'fuel demand vs. capacity h2 storage' : ['fuel demand (kWh)', 'capacity fuel h2 storage (kW)'],
        'fuel demand vs. dispatch unmet demand' : ['fuel demand (kWh)', 'dispatch unmet demand (kW)'],
        #'fuel cost vs. hourly dispatch' : ['fuel cost ($/GGE)', 'dispatch from fuel h2 storage (kW)'],

    }


    for k, v in plot_map.items():
        add_one = True
        simple_plot(df[v[0]].values, df[v[1]].values, v[0], v[1], k, k.replace('.','').replace(' ','_'), add_one)
        simple_plot(df[v[0]].values, df[v[1]].values/df[v[0]].values, v[0], v[1]+'/'+v[0], 
                k+'/fuel demand (kWh)', k.replace('.','').replace(' ','_')+'_div_fuel_dem')






