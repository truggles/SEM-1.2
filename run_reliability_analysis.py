import numpy as np
import pandas as pd
import csv
import subprocess
import os
from glob import glob
from shutil import copy2
from collections import OrderedDict



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
        cap_solar, cap_wind, cap_NG, cap_storage):

    new_cfg = []

    case_data_line = -999 # Starts really negative so the 2nd 'if' is never triggered until ready
    case_name_position = -999
    reliability_position = -999
    start_year_position = -999
    end_year_position = -999
    cap_solar_position = -999
    cap_wind_position = -999     
    cap_NG_position = -999
    cap_storage_position = -999

    for i, line in enumerate(cfg):

        if line[0] == 'GLOBAL_NAME':
            line[1] = global_name

        if line[0] == 'CASE_NAME':
            case_data_line = i
            case_name_position = line.index('CASE_NAME')
            reliability_position = line.index('RELIABILITY')
            start_year_position = line.index('START_YEAR')
            end_year_position = line.index('END_YEAR')
            cap_solar_position = line.index('CAPACITY_SOLAR')
            cap_wind_position = line.index('CAPACITY_WIND')
            cap_NG_position = line.index('CAPACITY_NATGAS')
            cap_storage_position = line.index('CAPACITY_STORAGE')

        if i == case_data_line+2:
            line[case_name_position] = case_name
            line[reliability_position] = reliability
            line[start_year_position] = start_year
            line[end_year_position] = end_year
            line[cap_solar_position] = cap_solar
            line[cap_wind_position] = cap_wind
            line[cap_NG_position] = cap_NG
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

def get_results(files):

    results = {}

    keys = []
    for f in files:
        info = get_all_cap_and_costs(f)
        keys.append(info['case name'].values[0])
        results[info['case name'].values[0]] = [
                       info['problem status'].values[0],
                       float(info['case name'].values[0].split('_')[1].replace('p','.')), # reliability value
                       info['system cost ($/kW/h)'].values[0],
                       info['capacity natgas (kW)'].values[0],
                       info['capacity solar (kW)'].values[0],
                       info['capacity wind (kW)'].values[0],
                       info['dispatch unmet demand (kW)'].values[0]
        ]

    print('Writing results to "Results.txt"')
    ofile = open('Results.txt', 'w')
    keys = sorted(keys)
    ofile.write('case name,problem status,target reliability,system cost ($/kW/h),capacity natgas (kW),capacity solar (kW),capacity wind (kW),dispatch unmet demand (kW)\n')
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


def reconfigure_and_run(path, results, case_name_base, input_file, global_name, 
        start_year, end_year, reli_float, solar, wind, cap_NG, cap_storage):
    # Get new copy of SEM cfg
    case_name = case_name_base+'_17-18'
    case_file = case_name+'.csv'
    cfg = get_SEM_csv_file(input_file)
    cfg = set_all_values(cfg, global_name, case_name, 2017, 2018, reli_float, solar, wind, cap_NG, cap_storage)
    write_file(case_file, cfg)
    subprocess.call(["python", "Simple_Energy_Model.py", case_file])

    # 3rd set results, Copy output file, Delete results files
    files = get_output_file_names(path+'/'+global_name+'_2019')
    copy2(files[-1], results)
    os.remove(files[-1])

if '__main__' in __name__:

    #reliability_values = [0.0000, 0.0001, 0.0003, 0.001, 0.01, 0.1]
    reliability_values = [0.0001,]
    wind_values = [0.0,]# 0.25, 0.5, 0.75, 1.0]
    solar_values = [0.0,]# 0.25, 0.5, 0.75, 1.0]
    years = {
            '15-16' : [2015, 2016],
            '16-17' : [2016, 2017],
            '17-18' : [2017, 2018],
    }

    input_file = 'reliability_case_191017.csv'
    global_name = 'reliability_20191118_v1'
    path = 'Output_Data/'+global_name
    results = path+'/results'

    for reliability in reliability_values:
        for solar in solar_values:
            for wind in wind_values:

                solar_str = 'solar_'+str(round(solar,2)).replace('.','p')
                wind_str = 'wind_'+str(round(wind,2)).replace('.','p')
                reliability_str = 'rel_'+str(round(reliability,4)).replace('.','p')
                case_name_base = reliability_str+'_'+wind_str+'_'+solar_str

                # 1st Step
                case_name = case_name_base+'_15-16'
                case_file = case_name+'.csv'
                cfg = get_SEM_csv_file(input_file)
                cap_NG, cap_storage = -1, -1
                cfg = set_all_values(cfg, global_name, case_name, 2015, 2016, reliability, solar, wind, cap_NG, cap_storage)
                write_file(case_file, cfg)
                subprocess.call(["python", "Simple_Energy_Model.py", case_file])

                # Copy output file
                files = get_output_file_names(path+'/'+global_name+'_2019')
                if not os.path.exists(results):
                    os.makedirs(results)
                copy2(files[-1], results)
                os.remove(files[-1])


                # Read results
                files = get_output_file_names(path+'/'+global_name+'_2019')
                f_name = files[-1].split('/')[-1]
                print(f_name)
                dta = get_cap_and_costs(path, f_name)
                print(dta.head())
                cap_NG, cap_storage = float(dta['capacity natgas (kW)']), float(dta['capacity storage (kW)'])

                # Get new copy of SEM cfg
                case_name = reliability_str+'_'+wind_str+'_'+solar_str+'_16-17'
                case_file = case_name+'.csv'
                cfg = get_SEM_csv_file(input_file)
                reli_float = -1
                cfg = set_all_values(cfg, global_name, case_name, 2016, 2017, reli_float, solar, wind, cap_NG, cap_storage)
                write_file(case_file, cfg)
                subprocess.call(["python", "Simple_Energy_Model.py", case_file])


                # 2nd set results, Copy output file, Delete results files
                files = get_output_file_names(path+'/'+global_name+'_2019')
                copy2(files[-1], results)
                os.remove(files[-1])


                # Get new copy of SEM cfg
                case_name = reliability_str+'_'+wind_str+'_'+solar_str+'_17-18'
                case_file = case_name+'.csv'
                cfg = get_SEM_csv_file(input_file)
                cfg = set_all_values(cfg, global_name, case_name, 2017, 2018, reli_float, solar, wind, cap_NG, cap_storage)
                write_file(case_file, cfg)
                subprocess.call(["python", "Simple_Energy_Model.py", case_file])

                # 3rd set results, Copy output file, Delete results files
                files = get_output_file_names(path+'/'+global_name+'_2019')
                copy2(files[-1], results)
                os.remove(files[-1])


                # Get new copy of SEM cfg
                case_name = reliability_str+'_'+wind_str+'_'+solar_str+'_18-19'
                case_file = case_name+'.csv'
                cfg = get_SEM_csv_file(input_file)
                cfg = set_all_values(cfg, global_name, case_name, 2018, 2019, reli_float, solar, wind, cap_NG, cap_storage)
                write_file(case_file, cfg)
                subprocess.call(["python", "Simple_Energy_Model.py", case_file])

                # 3rd set results, Copy output file, Delete results files
                files = get_output_file_names(path+'/'+global_name+'_2019')
                copy2(files[-1], results)
                os.remove(files[-1])

    assert(False)
    #results = '/Users/truggles/IDrive-Sync/Carnegie/SEM-1.2/Output_Data/tests_Jul25_v1/results'
    #files = get_output_file_names(results+'/tests_Jul25_v1_2019')
    #results = get_results(files)
    results = simplify_results("Results.txt", reliability_values, wind_values, solar_values)

    ## Take 2D container from get_hourly_info_per_week()
    ## and plot results
    #def plot_daily_over_weeks_surface(hourly_info, save, angle_z=30, angle_plane=50):

    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Make data.
    X = wind_values
    Y = solar_values
    X, Y = np.meshgrid(X, Y)


    for reliability in reliability_values:
        if reliability == 0.0: continue
        Z = np.zeros((len(wind_values),len(solar_values)))
        for solar in solar_values:
            for wind in wind_values:
                Z[solar_values.index(solar)][wind_values.index(wind)] = results[reliability][solar][wind][0] * 100.

        print(reliability)
        print(Z)

        fig, ax = plt.subplots()
        im = ax.imshow(Z,interpolation='none',extent=[-0.125,1.125,-0.125,1.125],origin='lower', vmin=0.)

        plt.xticks(wind_values, wind_values)
        plt.yticks(wind_values, wind_values)
        plt.xlabel("Wind Capacity w.r.t Dem. Mean")
        plt.ylabel("Solar Capacity w.r.t Dem. Mean")
        plt.title("Reliability Uncert. for Target Unmet Demand: {:.2f}%".format(reliability*100))
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel("Relative reliability uncert. (%)")
        plt.savefig("reliability_uncert_for_target_{}.png".format(str(reliability).replace('.','p')))
        plt.clf()


    # Make some plots of a single fraction over reliability range
    techs = OrderedDict()
    techs[(0.0, 0.0)] = ["Wind Zero, Solar Zero", []]
    techs[(0.5, 0.5)] = ["Wind 0.5, Solar 0.5", []]
    techs[(1.0, 0.0)] = ["Wind 1.0, Solar 0.0", []]
    techs[(0.0, 1.0)] = ["Wind 0.0, Solar 1.0", []]
    techs[(1.0, 1.0)] = ["Wind 1.0, Solar 1.0", []]

    inverted = sorted(reliability_values, reverse=True)
    inverted.remove(0.0)
    for reli in inverted:
        for solar in solar_values:
            for wind in wind_values:
                for name, vals in techs.items():
                    if name[0] == wind and name[1] == solar:
                        vals[1].append(results[reli][wind][solar][0] * 100)



    fig, ax = plt.subplots()
    for name, vals in techs.items():
        print(name, vals)
        ax.plot(inverted, vals[1], 'o-', label=vals[0])

    plt.xlabel("Target Unmet Demand: 1 - (annual delivered/annual demand)")
    plt.ylabel("abs[(unmet dem. - target unmet dem.)/target unmet dem.] (%)")
    plt.title("Uncertainty in Achieving Annual Reliability Targets")
    plt.xscale('log', nonposx='clip')
    ax.legend()
    plt.savefig("reliability_uncert_comparison.png")
    

