#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import subprocess
import os
from glob import glob
from shutil import copy2
from collections import OrderedDict
import re



# Use Pandas to retrieve the output values b/c it handles
# fully populated tables well
def get_cap_and_costs(path, file_name):
    return get_all_cap_and_costs(path+'/'+file_name)

def get_all_cap_and_costs(file_name):
    dta = pd.read_csv(file_name, index_col=0, header=None,
                   dtype={
                        'case name':np.str,
                        'problem status':np.str,
                        'system cost ($/kW/h)':np.float64,
                        'capacity natgas (kW)':np.float64,
                        'capacity nuclear (kW)':np.float64,
                        'capacity storage (kWh)':np.float64,
                        'capacity solar (kW)':np.float64,
                        'capacity wind (kW)':np.float64,
                        'dispatch unmet demand (kW)':np.float64,
                       }).T

    return dta


# Use normal python csv functions b/c this is a sparsely populated
# csv file
def get_SEM_csv_file(file_name):

    with open(file_name, 'r') as f:
        info = list(csv.reader(f, delimiter=","))
    
    return info


# These can be set for each and every run
def set_all_values(cfg, global_name, case_name, start_year, end_year, reliability,
        cap_solar, cap_wind, cap_NG, cap_nuclear, cap_storage, var_cost_unmet_demand):

    new_cfg = []

    case_data_line = -999 # Starts really negative so the 2nd 'if' is never triggered until ready
    case_name_position = -999
    reliability_position = -999
    start_year_position = -999
    end_year_position = -999
    cap_solar_position = -999
    cap_wind_position = -999     
    cap_NG_position = -999
    cap_nuclear_position = -999
    cap_storage_position = -999
    var_cost_unmet_demand_position = -999

    for i, line in enumerate(cfg):

        if line[0] == 'GLOBAL_NAME':
            line[1] = global_name

        if line[0] == 'CASE_NAME':
            case_data_line = i
            case_name_position = line.index('CASE_NAME')
            reliability_position = line.index('SYSTEM_RELIABILITY')
            start_year_position = line.index('START_YEAR')
            end_year_position = line.index('END_YEAR')
            cap_solar_position = line.index('CAPACITY_SOLAR')
            cap_wind_position = line.index('CAPACITY_WIND')
            cap_NG_position = line.index('CAPACITY_NATGAS')
            cap_nuclear_position = line.index('CAPACITY_NUCLEAR')
            cap_storage_position = line.index('CAPACITY_STORAGE')
            var_cost_unmet_demand_position = line.index('VAR_COST_UNMET_DEMAND')

        if i == case_data_line+2:
            line[case_name_position] = case_name
            line[reliability_position] = reliability
            line[start_year_position] = start_year
            line[end_year_position] = end_year
            line[cap_solar_position] = cap_solar
            line[cap_wind_position] = cap_wind
            line[cap_NG_position] = cap_NG
            line[cap_nuclear_position] = cap_nuclear
            line[cap_storage_position] = cap_storage
            line[var_cost_unmet_demand_position] = var_cost_unmet_demand

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

    files = glob(path+'*.csv')
    if len(files) > 1:
        print("This many files were found matching {}*.csv: {}".format(path, len(files)))
    return files

def del_pickle_files(path):

    files = glob(path+'*.pickle')
    for f in files:
        print(f"deleting pickle file: {f}")
        os.remove(f)

def del_csv_files(path):

    files = glob(path+'*.csv')
    for f in files:
        print(f"deleting LONG csv file: {f}")
        os.remove(f)


def get_results(files, global_name):

    results = {}

    keys = []
    for f in files:
        info = get_all_cap_and_costs(f)
        keys.append(info['case name'].values[0])
        if not hasattr(info, 'capacity storage (kW)'):
            info['capacity storage (kW)'] = 0.
        results[info['case name'].values[0]] = [
                       info['problem status'].values[0],
                       float(info['case name'].values[0].split('_')[1].replace('p','.')), # reliability value
                       info['system cost ($ or $/kWh)'].values[0],
                       #info['capacity natgas (kW)'].values[0],
                       info['capacity nuclear (kW)'].values[0],
                       info['capacity storage (kW)'].values[0],
                       info['capacity solar (kW)'].values[0],
                       info['capacity wind (kW)'].values[0],
                       info['dispatch unmet demand (kW)'].values[0]
        ]


    # Remove * in name that was introduced for searching regex
    save_name = re.sub("\*","",global_name)
    print(f'Writing results to "results/Results_{save_name}.csv"')
    ofile = open(f'results/Results_{save_name}.csv', 'w')
    
    keys = sorted(keys)
    ofile.write('case name,problem status,target reliability,system cost ($/kW/h),capacity nuclear (kW),capacity storage (kW),capacity solar (kW),capacity wind (kW),dispatch unmet demand (kW)\n')
    for key in keys:
        to_print = ''
        for info in results[key]:
            to_print += str(info)+','
        ofile.write("{},{}\n".format(key, to_print))
    ofile.close()
    return results

# Get info from file, so we don't have to repeat get_results
# many many times
def simplify_results(results_file):
    df = pd.read_csv(results_file, index_col=False)
    #print(df.head())

    reliability_values, wind_values, solar_values = [], [], []
    for idx in df.index:
        if df.loc[idx, 'problem status'] != 'optimal':
            continue
        if round(df.loc[idx, 'target reliability'],4) not in reliability_values:
            reliability_values.append(round(df.loc[idx, 'target reliability'],4))
        if round(df.loc[idx, 'capacity wind (kW)'],2) not in wind_values:
            wind_values.append(round(df.loc[idx, 'capacity wind (kW)'],2))
        if round(df.loc[idx, 'capacity solar (kW)'],2) not in solar_values:
            solar_values.append(round(df.loc[idx, 'capacity solar (kW)'],2))

    simp = {}
    for reliability in reliability_values:
        simp[reliability] = {}
        for solar in solar_values:
            simp[reliability][solar] = {}
            for wind in wind_values:            # rel vals, unmet, cap storage, cap nuclear, std dev, abs rel diff, rel diff, unmet, storage, nuclear
                simp[reliability][solar][wind] = [[], [], [], [], [], [], 0., 0., 0., 0., 0., 0., 0., np.nan, np.nan]

    for idx in df.index:
        
        # Skip non-optimal solutions:
        if df.loc[idx, 'problem status'] != 'optimal':
            print(f" ... skipping {df.loc[idx, 'case name']} at idx: {idx} soln: {df.loc[idx, 'problem status']}")
            continue

        reli = round(df.loc[idx, 'target reliability'],4)
        solar = round(df.loc[idx, 'capacity solar (kW)'],2)
        wind = round(df.loc[idx, 'capacity wind (kW)'],2)
        unmet = df.loc[idx, 'dispatch unmet demand (kW)']
        cap_storage = df.loc[idx, 'capacity storage (kW)']
        cap_nuclear = df.loc[idx, 'capacity nuclear (kW)']

        # Remove entries which were from Step 1 with fixed
        # target reliability
        year_info = df.loc[idx, 'case name'].split('_')
        for val in year_info:
            if 'lead' in val:
                lead_year = val.replace('lead','')
        if year_info[-1] == lead_year:
            simp[reli][solar][wind][4].append(cap_storage)
            simp[reli][solar][wind][5].append(cap_nuclear)
            continue

        if reli == 0.0:
            to_add = unmet
        else:
            to_add = (unmet - (1. - reli))/(1. - reli)
        simp[reli][solar][wind][0].append(to_add)
        simp[reli][solar][wind][1].append(unmet)
        simp[reli][solar][wind][2].append(cap_storage)
        simp[reli][solar][wind][3].append(cap_nuclear)

    adj = 6
    for reli in reliability_values:
        for solar in solar_values:
            for wind in wind_values:
                if len(simp[reli][solar][wind][0]) == 0: continue
                tot_abs, tot = 0., 0.
                for val in simp[reli][solar][wind][0]:
                    tot_abs += abs(val)
                    tot += val
                tot_abs /= len(simp[reli][solar][wind][0])
                tot /= len(simp[reli][solar][wind][0])
                simp[reli][solar][wind][adj+0] = np.std(simp[reli][solar][wind][0])
                simp[reli][solar][wind][adj+1] = tot_abs
                simp[reli][solar][wind][adj+2] = tot
                y = np.array(simp[reli][solar][wind][0])
                simp[reli][solar][wind][adj+3] = np.sqrt(np.mean(y**2)) # RMS Error
                simp[reli][solar][wind][adj+4] = np.mean(simp[reli][solar][wind][1])
                simp[reli][solar][wind][adj+5] = np.mean(simp[reli][solar][wind][2])
                simp[reli][solar][wind][adj+6] = np.mean(simp[reli][solar][wind][3])
                if np.mean(simp[reli][solar][wind][4]) > 0:
                    simp[reli][solar][wind][adj+7] = np.std(simp[reli][solar][wind][4])/np.mean(simp[reli][solar][wind][4])
                if np.mean(simp[reli][solar][wind][5]) > 0:
                    simp[reli][solar][wind][adj+8] = np.std(simp[reli][solar][wind][5])/np.mean(simp[reli][solar][wind][5])

    return simp



# Get info from file, so we don't have to repeat get_results
# many many times
def simplify_qmu_results(results_file):
    df = pd.read_csv(results_file, index_col=False)
    #print(df.head())


    # Get the default values from initial optimization
    # Default values from initial optimization
    def_nuclear, def_storage = 0., 0.
    for idx in df.index:
        if df.loc[idx, 'problem status'] != 'optimal':
            continue
        if 'storageSF_Def' in df.loc[idx, 'case name']:
            def_nuclear = df.loc[idx, 'capacity nuclear (kW)']
            def_storage = df.loc[idx, 'capacity storage (kW)']


    # Get the scale factor values based on capacities w.r.t. initial values
    nuclearSF_values, storageSF_values = [], []
    for idx in df.index:
        if df.loc[idx, 'problem status'] != 'optimal':
            continue
        if 'storageSF_Def' in df.loc[idx, 'case name']:
            continue

        nuclearSF = round(float(df.loc[idx, 'case name'].split('nukeSF')[-1].strip('_').split('_')[0].replace('p','.')),2)
        if nuclearSF not in nuclearSF_values:
            nuclearSF_values.append(nuclearSF)
        storageSF = round(float(df.loc[idx, 'case name'].split('storageSF')[-1].strip('_').split('_')[0].replace('p','.')),2)
        if storageSF not in storageSF_values:
            storageSF_values.append(storageSF)

    nuclearSF_values.sort()
    storageSF_values.sort()


    # Create mapping of results to the associated nuclear and storage SF parameter values
    simp = {}
    for nuke in nuclearSF_values:
        simp[nuke] = {}
        for storage in storageSF_values: # unmet, cost
            simp[nuke][storage] = [[], []]

    for idx in df.index:
        # Skip non-optimal solutions:
        if df.loc[idx, 'problem status'] != 'optimal':
            print(f" ... skipping {df.loc[idx, 'case name']} at idx: {idx} soln: {df.loc[idx, 'problem status']}")
            continue
        if 'storageSF_Def' in df.loc[idx, 'case name']:
            continue

        nuclearSF = round(float(df.loc[idx, 'case name'].split('nukeSF')[-1].strip('_').split('_')[0].replace('p','.')),2)
        storageSF = round(float(df.loc[idx, 'case name'].split('storageSF')[-1].strip('_').split('_')[0].replace('p','.')),2)
        unmet = df.loc[idx, 'dispatch unmet demand (kW)']
        cost = df.loc[idx, 'system cost ($/kW/h)']

        simp[nuclearSF][storageSF][0].append(unmet)
        simp[nuclearSF][storageSF][1].append(cost)

    return simp


def reconfigure_and_run(path, results, case_name_base, input_file, global_name, 
        lead_year_code, year_code, reliability, solar, wind, cap_NG, 
        cap_nuclear, cap_storage, var_cost_unmet_demand):
    # Get new copy of SEM cfg
    case_name = case_name_base+'_lead'+lead_year_code+'_'+year_code
    case_file = case_name+'.csv'
    cfg = get_SEM_csv_file(input_file)
    cfg = set_all_values(cfg, global_name, case_name, years[year_code][0], years[year_code][1], 
            reliability, solar, wind, cap_NG, cap_nuclear, cap_storage, var_cost_unmet_demand)
    write_file(case_file, cfg)
    subprocess.call(["python", "Simple_Energy_Model.py", case_file])

    # Read results
    files = get_output_file_names(path+'/'+global_name+'_2019')
    # Try to read results, if Gurobi failed ungracefully, try running again
    # If it fails a second time, give up. (don't want to get stuck in some while loop
    # waiting for Gurobi to suceed on an impossible model)
    if len(files) == 0:
        print(f"ERROR: XXX Initial solve failed, trying again for {global_name} {case_file}")
        cnt = 0
        while cnt < 10:
            print(f"\n --- Entering retry loop: {cnt}\n")
            subprocess.call(["python", "Simple_Energy_Model.py", case_file])
            files = get_output_file_names(path+'/'+global_name+'_2019')
            if len(files) > 0:
                break
            # Else retry up to 10 times
            cnt += 1

    f_name = files[-1].split('/')[-1]
    dta = get_cap_and_costs(path, f_name)
    print("Results file:", f_name, dta['case name'])

    # Copy output file, Delete results files
    if not os.path.exists(results):
        os.makedirs(results)
    copy2(files[-1], results)
    os.remove(files[-1])
    os.remove(case_file)
    del_pickle_files(path)
    del_csv_files(path)

    return dta


# 2D matrix showing reliability over different wind and solar builds
# mthd is 1, 2, 3, 4 and refers to the results array
# 1 = std dev
# 2 = abs rel diff
# 3 = rel diff
# 4 = RMS Error
# 5 = Unmet
# 6 = Cap Storage
# 7 = Cap Nuclear
def reliability_matrix(mthd, results, reliability, solar_values, wind_values, save_name):

    assert(mthd in range(1,10))
    names = {
            1 : 'Std Dev',
            2 : 'Abs Rel Diff',
            3 : 'Rel Diff',
            4 : 'RMS Error',
            5 : 'Mean Unmet Demand (kWh)',
            6 : 'Mean Cap Storage (kWh)',
            7 : 'Mean Cap Nuclear (kW)',
            8 : 'Std Dev div Mean Cap Storage (kWh)',
            9 : 'Std Dev div Mean Cap Nuclear (kW)',
    }
    
    print(f"Reliability {reliability} using method {mthd}, {names[mthd]}")
    reli_matrix = np.zeros((len(solar_values),len(wind_values)))
    for solar in solar_values:
        for wind in wind_values:
            reli_matrix[solar_values.index(solar)][wind_values.index(wind)] = results[reliability][solar][wind][mthd+5] # This was shifted by adding more lists to front of main list

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ### FIXME - crazy soln is making one cell stick out over all the others
    if reliability == 0.9997:
        false_z_max = reli_matrix.flatten()
        false_z_max.sort()
        im = ax.imshow(reli_matrix,interpolation='none',origin='lower',vmax=false_z_max[-2])
    else:
        im = ax.imshow(reli_matrix,interpolation='none',origin='lower')

    plt.xticks(range(len(wind_values)), wind_values, rotation=90)
    plt.yticks(range(len(solar_values)), solar_values)
    plt.xlabel("Wind Capacity w.r.t Dem. Mean")
    plt.ylabel("Solar Capacity w.r.t Dem. Mean")
    cbar = ax.figure.colorbar(im)
    app = ' of (unmet - target)/target' if mthd <= 4 else ''
    cbar.ax.set_ylabel(f"{names[mthd]}{app}")
    plt.title(f"{names[mthd]}{app}")
    plt.tight_layout()
    # Modify save_name to make more LaTeX-able
    if 'ZS' in save_name:
        save_name = 'ZeroStorage'
    else:
        save_name = 'Normal'
    plt.savefig("plots_reli/reliability_uncert_{}_for_target_{}_{}.png".format(save_name, str(reliability).replace('.','p'), names[mthd].replace(' ','_').replace('(','').replace(')','')))
    plt.clf()


# QMU matrix
def plot_qmu_matrix(results, reliability, save_name, mthd):

    #assert(mthd in range(3))
    names = {
            0 : 'Mean Unmet Demand (kWh)',
            1 : 'Mean System Cost (cents/kWh)',
            2 : 'Fraction with Unmet Demand > Target',
            3 : 'Fraction with Unmet Demand > 0',
    }
    
    # Get nuclear SFs and storage SFs from results map
    nuclearSFs = list(results.keys())
    storageSFs = list(results[nuclearSFs[0]].keys())
    nuclearSFs.sort()
    storageSFs.sort()
    
    print(f"QMU plotting for reliability {reliability} for type {mthd} = {names[mthd]}")
    qmu_matrix = np.zeros((len(storageSFs),len(nuclearSFs)))
    for storage in storageSFs:
        for nuclear in nuclearSFs:
            if mthd == 2:
                ary = np.array(results[nuclear][storage][0])
                val = len(np.where(ary > 1. - reliability)[0])
                qmu_matrix[storageSFs.index(storage)][nuclearSFs.index(nuclear)] = 1.-val/len(ary)
            elif mthd == 3:
                ary = np.array(results[nuclear][storage][0])
                val = len(np.where(ary > 0)[0])
                qmu_matrix[storageSFs.index(storage)][nuclearSFs.index(nuclear)] = 1.-val/len(ary)
            elif mthd == 1:
                qmu_matrix[storageSFs.index(storage)][nuclearSFs.index(nuclear)] = np.mean(results[nuclear][storage][mthd])*100
            else:
                qmu_matrix[storageSFs.index(storage)][nuclearSFs.index(nuclear)] = np.mean(results[nuclear][storage][mthd])

    #fig, ax = plt.subplots(figsize=(4.5, 4))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(qmu_matrix,interpolation='none',origin='lower')

    # Annotate if this is cost (mthd 1)
    # Loop over data dimensions and create text annotations.
    if mthd == 1:
        # This is the 100% meet 100% criteria matrix for coloring the text RED
        qmu_matrix2 = np.zeros((len(storageSFs),len(nuclearSFs)))
        for storage in storageSFs:
            for nuclear in nuclearSFs:
                ary = np.array(results[nuclear][storage][0])
                val = len(np.where(ary > 0)[0])
                qmu_matrix2[storageSFs.index(storage)][nuclearSFs.index(nuclear)] = 1.-val/len(ary)
        min_val = 999
        min_val_100_reli = 999
        for i in range(len(storageSFs)):
            for j in range(len(nuclearSFs)):
                if round(qmu_matrix[i, j],2) < min_val_100_reli and not qmu_matrix2[i, j] < 1.0:
                    min_val_100_reli = round(qmu_matrix[i, j],2)
                if round(qmu_matrix[i, j],2) < min_val:
                    min_val = round(qmu_matrix[i, j],2)
        for i in range(len(storageSFs)):
            for j in range(len(nuclearSFs)):
                txt_color = "w" if qmu_matrix2[i, j] < 1.0 else "k"
                if round(qmu_matrix[i, j],2) == min_val_100_reli and not qmu_matrix2[i, j] < 1.0:
                    txt_color = "r"
                if round(qmu_matrix[i, j],2) == min_val:
                    txt_color = "r"
                text = ax.text(j, i, round(qmu_matrix[i, j],2),
                        ha="center", va="center", color=txt_color, fontsize=6)


    plt.xticks(range(len(nuclearSFs)), nuclearSFs, rotation=90)
    plt.yticks(range(len(storageSFs)), storageSFs)
    plt.xlabel("Nuclear Scale Factor")
    plt.ylabel("Storage Scale Factor")
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(f"{names[mthd]}")
    plt.title(f"{names[mthd]}")
    plt.tight_layout()
    plt.savefig("agu_poster/plots/qmu_matrix_{}_for_target_{}_{}.png".format(save_name, str(reliability).replace('.','p'), names[mthd].split('(')[0].strip().replace(' ','_')))
    plt.clf()

if '__main__' in __name__:

    import sys
    print(f"\nRunning {sys.argv[0]}")
    print(f"Input arg list {sys.argv}")

    run_sem = False
    make_results_file = False
    plot_results = False
    zero_storage = False # Include storage in options
    post_mazama = False # Use after "run_sem" for gathering results and plotting
    qmu_scan = False # Use with fixed wind and solar values to scan nuclear and storage SFs
    if 'run_sem' in sys.argv:
        run_sem = True
    if 'make_results_file' in sys.argv:
        make_results_file = True
    if 'plot_results' in sys.argv:
        plot_results = True
    if 'zero_storage' in sys.argv:
        zero_storage = True
    if 'post_mazama' in sys.argv:
        post_mazama = True
    if 'qmu_scan' in sys.argv:
        qmu_scan = True

    # Default scans
    reliability_values = [1.0, 0.9999, 0.9997, 0.999, 0.995, 0.99]
    wind_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
    solar_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
    #wind_values = [0.0,]# 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    #solar_values = [0.0,]# 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    reliability_values = [0.9997,]
    #wind_values = [0.0, 0.25, 1.0]
    #solar_values = [0.0, 0.25, 1.0]
    nuclear_SF = 1.0
    storage_SFs = [1.0,]
    if qmu_scan:
        storage_SFs = np.arange(1.0, 5.01, 0.25)

    date = '20191119' # default
    version = 'v11'
    for arg in sys.argv:
        if 'date' in arg:
            date = arg.split('_')[1]
        if 'version' in arg:
            version = arg.split('_')[1]
        if 'wind' in arg:
            wind_values = [float(arg.split('_')[1]),]
        if 'solar' in arg:
            solar_values = [float(arg.split('_')[1]),]
        if 'reliability' in arg and not 'analysis' in arg:
            reliability_values = [float(arg.split('_')[1]),]
        if 'nuclear_SF' in arg:
            nuclear_SF = float(arg.split('_')[-1])

    input_file = 'reliability_case_191017.csv'
    if zero_storage:
        input_file = 'reliability_case_no_storage_191017.csv'
    version = f'{version}'
    global_name = 'reliability_{}_{}'.format(date, version)
    if len(wind_values) == 1: # Add wind value to global name for mazama file sorting
        global_name = 'reliability_{}_{}_wind{}'.format(date, version, str(wind_values[-1]).replace('.','p'))
    if qmu_scan:
        global_name += '_nukeSF{}'.format(str(round(nuclear_SF,2)).replace('.','p'))
    if post_mazama:
        global_name = 'reliability_{}_{}_wind*'.format(date, version, str(wind_values[-1]).replace('.','p'))
    path = 'Output_Data/{}/'.format(global_name)
    results_path = path+'results'


    # Print settings:
    print(f'\nGlobal name {global_name}')
    print(f'Output path {path}')
    print(f'Results path {results_path}')
    print(f'Input File: {input_file}')
    print(f'Reliability Values: {reliability_values}')
    print(f'Wind Values: {wind_values}')
    print(f'Solar Values: {solar_values}')
    if qmu_scan:
        print(f'Nuclear SF: {nuclear_SF}')
        print(f'Storage SFs: {storage_SFs}')
    print(f'\n - RUN_SEM={run_sem}')
    print(f' - MAKE_RESULTS_FILE={make_results_file}')
    print(f' - PLOT_RESULTS={plot_results}')
    print(f' - ZERO_STORAGE={zero_storage}')
    print(f' - QMU_SCAN={qmu_scan}')
    print(f' - POST_MAZAMA={post_mazama}\n')





    years = {
            '15-16' : [2015, 2016],
            '16-17' : [2016, 2017],
            '17-18' : [2017, 2018],
            '18-19' : [2018, 2019],
    }



    if run_sem:
        for reliability in reliability_values:
            for solar in solar_values:
                for wind in wind_values:
                    for lead_year_code in years.keys():

                        solar_str = 'solar_'+str(round(solar,2)).replace('.','p')
                        wind_str = 'wind_'+str(round(wind,2)).replace('.','p')
                        reliability_str = 'rel_'+str(round(reliability,4)).replace('.','p')
                        nuclear_str = '_nukeSF_'+str(round(nuclear_SF,2)).replace('.','p') if qmu_scan else ''
                        storage_str = '_storageSF_Def' if qmu_scan else ''
                        case_name_base = reliability_str+'_'+wind_str+'_'+solar_str+nuclear_str+storage_str+'_'+version+'_'+date

                        # 1st Step
                        cap_NG, cap_nuclear, cap_storage, var_cost_unmet_demand = -1, -1, -1, 1 # scale unmet normally
                        dta = reconfigure_and_run(path, results_path, case_name_base, input_file, global_name, 
                            lead_year_code, lead_year_code, reliability, solar, wind, cap_NG, cap_nuclear, 
                            cap_storage, var_cost_unmet_demand)


                        print(f"\nStorage SFs: {storage_SFs}\n")
                        for storage_SF in storage_SFs: # Defaults to [1.0,] unless specified
                            print(f"\nStorage SF: {storage_SF}\n")
                            var_cost_unmet_demand = 1
                            if qmu_scan:
                                storage_str = '_storageSF_'+str(round(storage_SF,2)).replace('.','p')
                                case_name_base_new = case_name_base.replace('_storageSF_Def', storage_str)
                                var_cost_unmet_demand = 0.005
                                #var_cost_unmet_demand = 1
                            else:
                                case_name_base_new = case_name_base

                            cap_NG = -1
                            if 'capacity storage (kW)' in dta.columns:
                                cap_storage = float(dta['capacity storage (kW)'])*storage_SF # SF defaults to 1.0 unless specified
                            else:
                                cap_storage = -1
                            cap_nuclear = float(dta['capacity nuclear (kW)'])*nuclear_SF # SF defaults to 1.0 unless specified

                            # XXX FOR EIA BASELINE TEST
                            #cap_wind = float(dta['capacity wind (kW)'])
                            #cap_solar = float(dta['capacity solar (kW)'])
                            cap_wind = wind
                            cap_solar = solar
                            float_reli = -1

                            # 2nd Step - run over 3 years with defined capacities
                            year_codes = list(years.keys())
                            year_codes.remove(lead_year_code)
                            for year_code in year_codes:
                                reconfigure_and_run(path, results_path, case_name_base_new, input_file, global_name, 
                                    lead_year_code, year_code, float_reli, cap_solar, cap_wind, cap_NG, 
                                    cap_nuclear, cap_storage, var_cost_unmet_demand)

    if make_results_file:
        files = get_output_file_names(results_path+'/'+global_name.replace('_wind','')+'_2019')
        results = get_results(files, global_name)

    if plot_results:

        global_name = re.sub("\*","",global_name)
        if qmu_scan:
            results = simplify_qmu_results(f"results/Results_{global_name}.csv")
            assert(len(reliability_values) == 1)
            for mthd in [0, 1, 3]:
                plot_qmu_matrix(results, reliability_values[0], f'{date}_{version}', mthd)

        if not qmu_scan:
            results = simplify_results(f"results/Results_{global_name}.csv")
            #print(results)


            ## and plot results
            for reliability in reliability_values:
                #for mthd in [1, 5, 6, 7]:
                for mthd in [1, 5, 6, 7, 8, 9]:
                    reliability_matrix(mthd, results, reliability, solar_values, wind_values, f'{date}_{version}')



        ## Make some plots of a single fraction over reliability range
        #techs = OrderedDict()
        #techs[(0.0, 0.0)] = ["Wind Zero, Solar Zero", []]
        #techs[(0.5, 0.5)] = ["Wind 0.5, Solar 0.5", []]
        #techs[(1.0, 0.0)] = ["Wind 1.0, Solar 0.0", []]
        #techs[(0.0, 1.0)] = ["Wind 0.0, Solar 1.0", []]
        #techs[(1.0, 1.0)] = ["Wind 1.0, Solar 1.0", []]

        #inverted = sorted(reliability_values, reverse=True)
        #inverted.remove(0.0)
        #for reli in inverted:
        #    for solar in solar_values:
        #        for wind in wind_values:
        #            for name, vals in techs.items():
        #                if name[0] == wind and name[1] == solar:
        #                    vals[1].append(results[reli][wind][solar][0] * 100)



        #fig, ax = plt.subplots()
        #for name, vals in techs.items():
        #    print(name, vals)
        #    ax.plot(inverted, vals[1], 'o-', label=vals[0])

        #plt.xlabel("Target Unmet Demand: 1 - (annual delivered/annual demand)")
        #plt.ylabel("abs[(unmet dem. - target unmet dem.)/target unmet dem.]")
        #plt.title("Uncertainty in Achieving Annual Reliability Targets")
        #plt.xscale('log', nonposx='clip')
        #ax.legend()
        #plt.savefig("reliability_uncert_comparison.png")
    

