import numpy as np
import csv
import subprocess
import os
from glob import glob
from shutil import copy2, move
from collections import OrderedDict
import pandas as pd
import os



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
                        #'capacity fuel h2 storage (kW)':np.float64,
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
                       info['dispatch unmet demand (kW)'].values[0],
                       info['dispatch nuclear (kW)'].values[0],
                       info['dispatch wind (kW)'].values[0],
                       info['dispatch solar (kW)'].values[0],
                       info['curtailment nuclear (kW)'].values[0],
                       info['curtailment wind (kW)'].values[0],
                       info['curtailment solar (kW)'].values[0]
        ]

    print('Writing results to "Results_{}.csv"'.format(global_name))
    ofile = open('Results_{}.csv'.format(global_name), 'w')
    keys = sorted(keys)
    ofile.write('case name,problem status,fuel cost ($/GGE),fuel demand (kWh),system cost ($/kW/h),capacity nuclear (kW),capacity solar (kW),capacity wind (kW),capacity fuel electrolyzer (kW),capacity fuel chem plant (kW),capacity fuel h2 storage (kW),dispatch to fuel h2 storage (kW),dispatch from fuel h2 storage (kW),dispatch unmet demand (kW),dispatch nuclear (kW),dispatch wind (kW),dispatch solar (kW),curtailment nuclear (kW),curtailment wind (kW),curtailment solar (kW)\n')
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


def simple_plot(x, ys, labels, x_label, y_label, title, save, logY=False):

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
            if max(y_tmp) < y_max:
                y_max = max(y_tmp)

        if not (y_min == y_max):
            ax.set_ylim(y_min*.5, y_max*2)

            #y_tmp = y[np.nonzero(y)]
            #y_tmp = y_tmp[np.isfinite(y_tmp)]
            #ax.set_ylim(min(y_tmp)*.5, max(y_tmp)*2)

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(x[np.nonzero(x)])*.5, max(x)*2)

    plt.tight_layout()
    plt.grid()
    plt.legend()
    fig.savefig('plots/{}.png'.format(save))


if '__main__' in __name__:

    # Efficiencies so I don't have to pull them from the cfgs for the moment, FIXME
    EFFICIENCY_FUEL_ELECTROLYZER=0.676783005
    EFFICIENCY_FUEL_CHEM_PLANT=0.659
    MEAN_JAN_2016_WIND_CF = 0.429287634 # From 1st month of 2016 wind, all Jan

    do_demand_constraint = True

    input_file = 'zOnlyNukeFuels_case_input_test_190827.csv'
    input_file = 'zFuels_case_input_test_190827.csv'
    version = 'v12'
    global_name = 'fuel_test_20190905_{}'.format(version)
    path = 'Output_Data/{}/'.format(global_name)
    results = path+'results/'

    run_sem = False
    multipliers = []
    multipliers = [0., 0.0001,]
    while True:
        if multipliers[-1] > 1000:
            break
        multipliers.append(round(multipliers[-1]*1.1,5))
    if run_sem:
        print("Length of multipliers {}".format(len(multipliers)))
        print(multipliers)

    for i, multiplier in enumerate(multipliers):

        if not run_sem:
            break

        if do_demand_constraint:
            fuel_str = f'run_{i:03}_fuel_demand_'+str(multiplier)+'kWh'
        else:
            fuel_str = f'run_{i:03}_fuel_cost_'+str(round(fuel_multiplier,6)).replace('.','p')+'USD'

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
        os.remove(case_file)


    #base = '/Users/truggles/IDrive-Sync/Carnegie/SEM-1.2_CIW/'
    base = '/Users/truggles/IDrive-Sync/Carnegie/SEM-1.2_HOME/'
    results = base+results
    #files = get_output_file_names(results+'{}_2019'.format(global_name))
    #results = get_results(files, global_name)

    import matplotlib.pyplot as plt
    df = pd.read_csv('Results_{}.csv'.format(global_name), index_col=False)

    plot_map = { # title / save : x, y, x_title, y_title
        'fuel demand vs. hourly dispatch' : ['fuel demand (kWh)', 'dispatch from fuel h2 storage (kW)'],
        'fuel demand vs. system cost' : ['fuel demand (kWh)', 'system cost ($/kW/h)'],
        'fuel demand vs. capacity nuclear' : ['fuel demand (kWh)', 'capacity nuclear (kW)'],
        #'fuel demand vs. capacity solar' : ['fuel demand (kWh)', 'capacity solar (kW)'],
#        'fuel demand vs. capacity wind' : ['fuel demand (kWh)', 'capacity wind (kW)'],
        'fuel demand vs. capacity electrolyzer' : ['fuel demand (kWh)', 'capacity fuel electrolyzer (kW)'],
        'fuel demand vs. capacity chem plant' : ['fuel demand (kWh)', 'capacity fuel chem plant (kW)'],
        'fuel demand vs. capacity h2 storage' : ['fuel demand (kWh)', 'capacity fuel h2 storage (kW)'],
        'fuel demand vs. dispatch unmet demand' : ['fuel demand (kWh)', 'dispatch unmet demand (kW)'],
        #'fuel cost vs. hourly dispatch' : ['fuel cost ($/GGE)', 'dispatch from fuel h2 storage (kW)'],

    }


    #for k, v in plot_map.items():
    #    logY = True
    #    simple_plot(df[v[0]].values, [df[v[1]].values,], [v[1],], v[0], v[1], k, k.replace('.','').replace(' ','_'), logY)
    #    simple_plot(df[v[0]].values, [df[v[1]].values/df[v[0]].values,], [v[1]+'/'+v[0],], v[0], v[1]+'/'+v[0], 
    #            k+'/fuel demand (kWh)', k.replace('.','').replace(' ','_')+'_div_fuel_dem', logY)


    # $/GGE fuel
    # subtract off the base-case (least fuel demand, should be zero in the future FIXME) system cost
    dollars_per_fuel = df['system cost ($/kW/h)'] - df.loc[df['fuel demand (kWh)'] == 0.0, 'system cost ($/kW/h)'].values
    # divide by fuel produced in that scenario
    dollars_per_fuel = dollars_per_fuel / df['fuel demand (kWh)']
    simple_plot(df['fuel demand (kWh)'].values, [dollars_per_fuel.values,], ['Cost of Fuel ($/kWh)',], 'fuel demand (kWh)', 'Cost of Fuel ($/kWh)', 
            'fuel demand vs. fuel cost', 'fuel_demand_vs_fuel_cost')



    # Stacked generation dispatch plot
    # In this base-case scenario, there is zero solar, so ignore it for now in plotting FIXME
    norm_nuclear = df['dispatch nuclear (kW)']
    norm_wind = df['dispatch wind (kW)']
    norm_solar = df['dispatch solar (kW)']

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('fuel demand (kWh/h)')
    ax.set_ylabel('normalized dispatch (kW)')
    plt.title('Normalized Dispatch')

    ax.fill_between(df['fuel demand (kWh)'], 0., norm_nuclear, color='red', label='annual nuclear dispatch')
    ax.fill_between(df['fuel demand (kWh)'], norm_nuclear, norm_nuclear+norm_wind, color='blue', label='annual wind dispatch')
    #ax.fill_between(df['fuel demand (kWh)'], norm_nuclear+norm_wind, norm_nuclear+norm_wind+norm_solar, color='yellow')

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(df['fuel demand (kWh)'].values[np.nonzero(df['fuel demand (kWh)'].values)]), max(df['fuel demand (kWh)'].values))

    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig.savefig('plots/stacked_dispatch_normalized.png')

    


    # Stacked generation capacity plot
    # In this base-case scenario, there is zero solar, so ignore it for now in plotting FIXME
    norm_nuclear = df['capacity nuclear (kW)']
    norm_wind = df['capacity wind (kW)']
    norm_solar = df['capacity solar (kW)']

    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('fuel demand (kWh/h)')
    ax.set_ylabel('normalized capacity (kW)')
    plt.title('Normalized Dispatch')

    ax.fill_between(df['fuel demand (kWh)'], 0., norm_nuclear, color='red', label='nuclear capacity')
    ax.fill_between(df['fuel demand (kWh)'], norm_nuclear, norm_nuclear+norm_wind, color='blue', label='wind capacity')
    #ax.fill_between(df['fuel demand (kWh)'], norm_nuclear+norm_wind, norm_nuclear+norm_wind+norm_solar, color='yellow')

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(df['fuel demand (kWh)'].values[np.nonzero(df['fuel demand (kWh)'].values)]), max(df['fuel demand (kWh)'].values))

    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig.savefig('plots/stacked_capacity_normalized.png')

    


    # Stacked curtailment plot
    # In this base-case scenario, there is zero solar, so ignore it for now in plotting FIXME
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('fuel demand (kWh/h)')
    ax.set_ylabel('curtailment of dispatch (kWh)')
    plt.title('Curtailment of Generation Capacities')

    ax.fill_between(df['fuel demand (kWh)'], 0., df['curtailment nuclear (kW)'], color='red', label='curtailment nuclear')
    ax.fill_between(df['fuel demand (kWh)'], df['curtailment nuclear (kW)'], df['curtailment nuclear (kW)']+df['curtailment wind (kW)'], color='blue', label='curtailment wind')

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(df['fuel demand (kWh)'].values[np.nonzero(df['fuel demand (kWh)'].values)]), max(df['fuel demand (kWh)'].values))

    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig.savefig('plots/stacked_curtailment.png')
    


    # Stacked curtailment / capacity plot
    # In this base-case scenario, there is zero solar, so ignore it for now in plotting FIXME
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('fuel demand (kWh/h)')
    ax.set_ylabel('curtailment of dispatch / capacity')
    plt.title('Curtailment of Generation Capacities')

    curt_div_dis_nuclear = df['curtailment nuclear (kW)']/df['capacity nuclear (kW)']
    curt_div_dis_wind = df['curtailment wind (kW)']/df['capacity wind (kW)']
    for idx, val in curt_div_dis_nuclear.items():
        if np.isnan(curt_div_dis_nuclear.at[idx]):
            curt_div_dis_nuclear.at[idx] = 0
    ax.fill_between(df['fuel demand (kWh)'], 0., curt_div_dis_nuclear, color='red', label='curtailment/capacity nuclear')
    ax.fill_between(df['fuel demand (kWh)'], curt_div_dis_nuclear, curt_div_dis_nuclear+curt_div_dis_wind, color='blue', label='curtailment/capacity wind')

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(df['fuel demand (kWh)'].values[np.nonzero(df['fuel demand (kWh)'].values)]), max(df['fuel demand (kWh)'].values))

    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig.savefig('plots/stacked_curtailment_div_capacity.png')
    


    # Stacked curtailment / dispatch plot
    # In this base-case scenario, there is zero solar, so ignore it for now in plotting FIXME
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('fuel demand (kWh/h)')
    ax.set_ylabel('curtailment of dispatch / dispatch')
    plt.title('Curtailment of Generation Capacities')

    curt_div_dis_nuclear = df['curtailment nuclear (kW)']/df['dispatch nuclear (kW)']
    curt_div_dis_wind = df['curtailment wind (kW)']/df['dispatch wind (kW)']
    for idx, val in curt_div_dis_nuclear.items():
        if np.isnan(curt_div_dis_nuclear.at[idx]):
            curt_div_dis_nuclear.at[idx] = 0
    ax.fill_between(df['fuel demand (kWh)'], 0., curt_div_dis_nuclear, color='red', label='curtailment/dispatch nuclear')
    ax.fill_between(df['fuel demand (kWh)'], curt_div_dis_nuclear, curt_div_dis_nuclear+curt_div_dis_wind, color='blue', label='curtailment/dispatch wind')

    plt.xscale('log', nonposx='clip')
    ax.set_xlim(min(df['fuel demand (kWh)'].values[np.nonzero(df['fuel demand (kWh)'].values)]), max(df['fuel demand (kWh)'].values))

    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig.savefig('plots/stacked_curtailment_div_dispatch.png')
    



    
    # Fuel system capacities ratios
    simple_plot(df['fuel demand (kWh)'].values,
            [df['capacity fuel electrolyzer (kW)'].values/df['fuel demand (kWh)'].values, 
                df['capacity fuel chem plant (kW)'].values/df['fuel demand (kWh)'].values, 
                df['capacity fuel h2 storage (kW)'].values/df['fuel demand (kWh)'].values], # y values
            ['cap electrolyzer (kW/h) / fuel demand (kWh/h)', 'cap chem plant (kW/h) / fuel demand (kWh/h)', 'cap H2 storage (kWh) / fuel demand (kWh/h)'], # labels
            'fuel demand (kWh)', 'Fuel System Capacities (kW/h or kWh) / Fuel Demand (kWh/h)', 
            'Ratios of Fuel System Capacities / Fuel Demand', 'ratios_fuel_system_vs_fuel_cost', True) # logY=True


    # Fuel system capacity factor ratios
    # The way the Core_Model.py is set up currently, efficiencies only need to be applied for the chem plant - 6 Sept 2019
    simple_plot(df['fuel demand (kWh)'].values,
            [df['dispatch to fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_ELECTROLYZER/df['capacity fuel electrolyzer (kW)'].values, 
                df['dispatch from fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_CHEM_PLANT/df['capacity fuel chem plant (kW)'].values,], # y values 
            ['electrolyzer capacity factor', 'chem plant capacity factor'], # labels
            'fuel demand (kWh)', 'Fuel System Capacity Factors', 
            'Fuel System Capacity Factors', 'ratios_fuel_system_CFs_vs_fuel_cost')


    # All system capacity factor ratios
    # The way the Core_Model.py is set up currently, efficiencies only need to be applied for the chem plant - 6 Sept 2019
    simple_plot(df['fuel demand (kWh)'].values,
            [df['dispatch to fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_ELECTROLYZER/df['capacity fuel electrolyzer (kW)'].values, 
                df['dispatch from fuel h2 storage (kW)'].values*EFFICIENCY_FUEL_CHEM_PLANT/df['capacity fuel chem plant (kW)'].values,
                df['dispatch nuclear (kW)'].values/df['capacity nuclear (kW)'].values,
                df['dispatch wind (kW)'].values/df['capacity wind (kW)'].values], # y values 
            ['electrolyzer capacity factor', 'chem plant capacity factor',
                'nuclear capacity factor', 'wind capacity factor'], # labels
            'fuel demand (kWh)', 'System Capacity Factors', 
            'System Capacity Factors', 'ratios_system_CFs_vs_fuel_cost')



    # Relative curtailment based on available power
    # This version factors out the wind CF of 0.43
    curt_div_dis_nuclear = df['curtailment nuclear (kW)']/df['capacity nuclear (kW)']
    curt_div_dis_wind = df['curtailment wind (kW)']/(df['curtailment wind (kW)']+df['dispatch wind (kW)'])
    #for idx, val in curt_div_dis_nuclear.items():
    #    if np.isnan(curt_div_dis_nuclear.at[idx]):
    #        curt_div_dis_nuclear.at[idx] = 0
    simple_plot(df['fuel demand (kWh)'].values,
            [curt_div_dis_nuclear.values, curt_div_dis_wind.values], # y values 
            ['nuclear curtailment / capacity', 'wind curtailment / available power'], # labels
            'fuel demand (kWh)', 'curtailment of dispatch (kW) / available power (kW)', 
            'Relative Curtailment of Generation Capacities', 'ratios_curtailment_div_available_power')

    



