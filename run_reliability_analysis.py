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
        cap_solar, cap_wind, cap_NG, cap_nuclear, cap_storage):

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
    print(f'Writing results to "Results_{save_name}.csv"')
    ofile = open(f'Results_{save_name}.csv', 'w')
    
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
        if df.loc[idx, 'target reliability'] not in reliability_values:
            reliability_values.append(round(df.loc[idx, 'target reliability'],4))
        if df.loc[idx, 'capacity wind (kW)'] not in wind_values:
            wind_values.append(round(df.loc[idx, 'capacity wind (kW)'],2))
        if df.loc[idx, 'capacity solar (kW)'] not in solar_values:
            solar_values.append(round(df.loc[idx, 'capacity solar (kW)'],2))

    simp = {}
    for reliability in reliability_values:
        simp[reliability] = {}
        for solar in solar_values:
            simp[reliability][solar] = {}
            for wind in wind_values:            # rel vals, unmet, cap storage, cap nuclear, std dev, abs rel diff, rel diff, unmet, storage, nuclear
                simp[reliability][solar][wind] = [[], [], [], [], 0., 0., 0., 0., 0., 0., 0.]

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
            continue

        if reli == 0.0:
            to_add = unmet
        else:
            to_add = (unmet - (1. - reli))/(1. - reli)
        simp[reli][solar][wind][0].append(to_add)
        simp[reli][solar][wind][1].append(unmet)
        simp[reli][solar][wind][2].append(cap_storage)
        simp[reli][solar][wind][3].append(cap_nuclear)

    for reli in reliability_values:
        for solar in solar_values:
            for wind in wind_values:
                if len(simp[reli][solar][wind][0]) == 0: continue
                simp[reli][solar][wind][4] = np.std(simp[reli][solar][wind][0])
                tot_abs, tot = 0., 0.
                for val in simp[reli][solar][wind][0]:
                    tot_abs += abs(val)
                    tot += val
                tot_abs /= len(simp[reli][solar][wind][0])
                tot /= len(simp[reli][solar][wind][0])
                simp[reli][solar][wind][5] = tot_abs
                simp[reli][solar][wind][6] = tot
                y = np.array(simp[reli][solar][wind][0])
                simp[reli][solar][wind][7] = np.sqrt(np.mean(y**2)) # RMS Error
                simp[reli][solar][wind][8] = np.mean(simp[reli][solar][wind][1])
                simp[reli][solar][wind][9] = np.mean(simp[reli][solar][wind][2])
                simp[reli][solar][wind][10] = np.mean(simp[reli][solar][wind][3])

    return simp


def reconfigure_and_run(path, results, case_name_base, input_file, global_name, 
        lead_year_code, year_code, reliability, solar, wind, cap_NG, cap_nuclear, cap_storage):
    # Get new copy of SEM cfg
    case_name = case_name_base+'_lead'+lead_year_code+'_'+year_code
    case_file = case_name+'.csv'
    cfg = get_SEM_csv_file(input_file)
    cfg = set_all_values(cfg, global_name, case_name, years[year_code][0], years[year_code][1], reliability, solar, wind, cap_NG, cap_nuclear, cap_storage)
    write_file(case_file, cfg)
    subprocess.call(["python", "Simple_Energy_Model.py", case_file])

    # Read results
    files = get_output_file_names(path+'/'+global_name+'_2019')
    f_name = files[-1].split('/')[-1]
    print(f_name)
    dta = get_cap_and_costs(path, f_name)

    # Copy output file, Delete results files
    if not os.path.exists(results):
        os.makedirs(results)
    copy2(files[-1], results)
    os.remove(files[-1])
    os.remove(case_file)
    del_pickle_files(path)

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

    assert(mthd in range(1,8))
    names = {
            1 : 'Std Dev',
            2 : 'Abs Rel Diff',
            3 : 'Rel Diff',
            4 : 'RMS Error',
            5 : 'Mean Unmet (kWh)',
            6 : 'Mean Cap Storage (kWh)',
            7 : 'Mean Cap Nuclear (kW)',
    }
    
    print(f"Reliability {reliability} using method {mthd}, {names[mthd]}")
    reli_matrix = np.zeros((len(solar_values),len(wind_values)))
    for solar in solar_values:
        for wind in wind_values:
            reli_matrix[solar_values.index(solar)][wind_values.index(wind)] = results[reliability][solar][wind][mthd+3] # This was shifted by adding more lists to front of main list

    fig, ax = plt.subplots(figsize=(9, 8))
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
    plt.title("Target Unmet Demand: {:.2f}%".format(reliability*100))
    cbar = ax.figure.colorbar(im)
    app = ' of (unmet - tgt. unmet)/tgt. unmet' if mthd <= 4 else ''
    cbar.ax.set_ylabel(f"{names[mthd]}{app}")
    plt.tight_layout()
    # Modify save_name to make more LaTeX-able
    if 'ZS' in save_name:
        save_name = 'ZeroStorage'
    else:
        save_name = 'Normal'
    plt.savefig("plots_reli/reliability_uncert_{}_for_target_{}_{}.png".format(save_name, str(reliability).replace('.','p'), names[mthd].replace(' ','_').replace('(','').replace(')','')))
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

    # Default scans
    reliability_values = [1.0, 0.9999, 0.9997, 0.999, 0.995, 0.99]
    wind_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]
    solar_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]
    #wind_values = [0.0,]# 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    #solar_values = [0.0,]# 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    reliability_values = [0.9997,]
    #wind_values = [0.0, 0.25, 1.0]
    #solar_values = [0.0, 0.25, 1.0]

    date = '20191119' # default
    version = 'v11'
    for arg in sys.argv:
        if 'date' in arg:
            date = arg.split('_')[1]
        if 'version' in arg:
            version = arg.split('_')[1]
        if 'wind' in arg:
            wind_values = [float(arg.split('_')[1]),]
        if 'reliability' in arg and not 'analysis' in arg:
            reliability_values = [float(arg.split('_')[1]),]

    input_file = 'reliability_case_191017.csv'
    if zero_storage:
        input_file = 'reliability_case_no_storage_191017.csv'
    version = f'{version}'
    global_name = 'reliability_{}_{}'.format(date, version)
    if len(wind_values) == 1: # Add wind value to global name for mazama file sorting
        global_name = 'reliability_{}_{}_wind{}'.format(date, version, str(wind_values[-1]).replace('.','p'))
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
    print(f'\n - RUN_SEM={run_sem}')
    print(f' - MAKE_RESULTS_FILE={make_results_file}')
    print(f' - PLOT_RESULTS={plot_results}')
    print(f' - ZERO_STORAGE={zero_storage}')
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
                        case_name_base = reliability_str+'_'+wind_str+'_'+solar_str+'_'+version+'_'+date

                        # 1st Step
                        cap_NG, cap_nuclear, cap_storage = -1, -1, -1
                        dta = reconfigure_and_run(path, results_path, case_name_base, input_file, global_name, 
                            lead_year_code, lead_year_code, reliability, solar, wind, cap_NG, cap_nuclear, cap_storage)


                        cap_NG = -1
                        if 'capacity storage (kW)' in dta.columns:
                            cap_storage = float(dta['capacity storage (kW)'])
                        else:
                            cap_storage = -1
                        cap_nuclear = float(dta['capacity nuclear (kW)'])
                        float_reli = -1

                        # 2nd Step - run over 3 years with defined capacities
                        year_codes = list(years.keys())
                        year_codes.remove(lead_year_code)
                        for year_code in year_codes:
                            reconfigure_and_run(path, results_path, case_name_base, input_file, global_name, 
                                lead_year_code, year_code, float_reli, solar, wind, cap_NG, cap_nuclear, cap_storage)

    if make_results_file:
        files = get_output_file_names(results_path+'/'+global_name.replace('_wind','')+'_2019')
        results = get_results(files, global_name)

    if plot_results:

        global_name = re.sub("\*","",global_name)
        results = simplify_results(f"Results_{global_name}.csv")
        #print(results)

        ## and plot results
        for reliability in reliability_values:
            for mthd in [1, 5, 6, 7]:
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
    

