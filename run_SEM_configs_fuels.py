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




def set_fuel_cost(cfg, fuel_cost_multiplier):

    new_cfg = []

    cnt = 1
    fuel_value_position = 0
    for line in cfg:
        if cnt == 166:
            fuel_value_position = line.index('FUEL_VALUE')
            print("fuel_value_position {} will be set to {}".format(fuel_value_position, fuel_cost_multiplier))
        if cnt == 168:
            line[0] = line[0]+'_'+str(round(fuel_cost_multiplier,4)).replace('.','p')+'X'
            line[fuel_value_position] = fuel_cost_multiplier
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
    if len(files) > 1:
        print("This many files were found matching {}*.csv: {}".format(path, len(files)))
    return files

def get_results(files):

    results = {}

    keys = []
    for f in files:
        info = get_all_cap_and_costs_fuels(f)
        print(info)
        keys.append(info['case name'].values[0])
        results[info['case name'].values[0]] = [
                       info['problem status'].values[0],
                       float(info['case name'].values[0].split('_')[1].replace('X','').replace('p','.')), # reliability value
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

    print('Writing results to "Results.csv"')
    ofile = open('Results.csv', 'w')
    keys = sorted(keys)
    ofile.write('case name,problem status,fuel cost multiplier,system cost ($/kW/h),capacity natgas (kW),capacity solar (kW),capacity wind (kW),capacity fuel electrolyzer (kW),capacity fuel chem plant (kW),capacity fuel h2 storage (kW),dispatch to fuel h2 storage (kW),dispatch from fuel h2 storage (kW),dispatch unmet demand (kW)\n')
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


if '__main__' in __name__:

    fuel_multipliers = [0.0001, 0.001, 0.01, 0.1, 1.0]

    input_file = 'yFuels_case_input_test_190827.csv'
    path = 'Output_Data/test_190829_v1/'
    results = path+'results/'

    #for fuel_multiplier in fuel_multipliers:

    #    fuel_str = 'fuel_'+str(round(fuel_multiplier,4)).replace('.','p')+'X'

    #    # 1st Step
    #    cfg = get_SEM_csv_file(input_file)
    #    case_name = fuel_str
    #    case_file = case_name+'.csv'
    #    cfg = set_fuel_cost(cfg, fuel_multiplier)
    #    write_file(case_file, cfg)
    #    subprocess.call(["python", "Simple_Energy_Model.py", case_file])

    #    files = get_output_file_names(path+'test_190829_v1_2019')

    #    # Copy output file
    #    if not os.path.exists(results):
    #        os.makedirs(results)
    #    move(files[-1], results)


    base = '/Users/truggles/IDrive-Sync/Carnegie/SEM-1.2_CIW/'
    results = base+results
    #files = get_output_file_names(results+'test_190829_v1_2019')
    #results = get_results(files)

    import matplotlib.pyplot as plt
    df = pd.read_csv('Results.csv', index_col=False)
    fig, ax = plt.subplots()
    ax.plot(df['fuel cost multiplier'], df['dispatch from fuel h2 storage (kW)'])
    plt.xlabel('fuel cost multiplier')
    plt.ylabel('hourly dispatch from fuel h2 storage (kW)')
    plt.xscale('log', nonposx='clip')
    fig.savefig('plot.png')
