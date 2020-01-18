#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from collections import OrderedDict
import pickle
from glob import glob
import os
from shutil import copy2


def return_file_info_map(region):
    assert(region in ['CONUS', 'TEXAS'])

    info_map = { # region : # f_path, header rows
        'CONUS': {
            'demand': ['Input_Data/Lei_Solar_Wind/US_demand_unnormalized.csv', 8, 'demand (MW)', 'year'],
            'wind': ['Input_Data/Lei_Solar_Wind/US_capacity_wind_25pctTop_unnormalized.csv', 5, 'wind 25% top', 'Year'], 
            'solar': ['Input_Data/Lei_Solar_Wind/US_capacity_solar_25pctTop_unnormalized.csv', 5, 'solar 25% top', 'Year'],
        },
        'TEXAS': { 
            'demand': ['Input_Data/TEXAS/TX_demand_unnormalized.csv', 6, 'demand (MW)', 'year'],
            'wind': ['Input_Data/TEXAS/windCF_lei_TI_nonormalized.csv', 5, 'solar capacity', 'year'],
            'solar': ['Input_Data/TEXAS/solarCF_lei_TI_nonormalized.csv', 5, 'solar capacity', 'year'],
        }
    }
    return info_map[region]


### FIXME
assert(False)
# Take all yearly inputs and split into two sets:
# 1 - all hours with demand < threshold
# 2 - all hours with demand > threshold
def run_correlations(collection, threshold, TEXAS):

    # Arrays to hold >= and < threshold, defined on each year of inputs
    solar_gtr, wind_gtr, dem_gtr = [], [], []

    # Loop over all years and hours and split
    # The split is defined w.r.t. demand percentile
    for i, year_info in collection.items():
        
        thr = np.percentile(year_info['dem']['Demand'].values, threshold)

        for i, dem in enumerate(year_info['dem']['Demand'].values):
            if dem >= thr:
                solar_gtr.append(year_info['solar']['Solar'].values[i])
                wind_gtr.append(year_info['wind']['Wind'].values[i])
                dem_gtr.append(year_info['dem']['Demand'].values[i])


    # Make DataFrams for easy use of df.corr()
    df_gtr = pd.DataFrame({'Demand': dem_gtr, 'Solar': solar_gtr, 'Wind': wind_gtr})
    plot_corr(df_gtr, f'greater{threshold}_{app}')

def plot_corr(df, save_name):
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    #corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5pt'})
    f = plt.figure()
    plt.imshow(corr, origin='lower', vmax=1, vmin=-1)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    ax = plt.gca()
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    cb.ax.set_ylim(-1, 1)
    #plt.title('Correlation Matrix', fontsize=16);

    print(corr)

    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.index):
            #print(i, j, col1, col2, round(corr.loc[col1, col2],2))
            text = ax.text(j, i, round(corr.loc[col1, col2],2),
                    ha="center", va="center", color='k', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'correlation_matrix_{save_name}.png')






# Sort output pngs into folders for easy scrolling
def make_dirs(base, tgt):
    files = glob(tgt)
    to_make = []
    for f in files:
        print(f)
        f1 = f.strip('.png')
        info = f1.split('_')
        for piece in info:
            if 'solarSF' in piece or 'windSF' in piece:
                if not os.path.exists(base+'/'+piece):
                    os.makedirs(base+'/'+piece)
                copy2(f, base+'/'+piece)

def get_dem_wind_solar(im):

    demand = pd.read_csv(im['demand'][0], header=im['demand'][1])
    wind = pd.read_csv(im['wind'][0], header=im['wind'][1])
    solar = pd.read_csv(im['solar'][0], header=im['solar'][1])

    return demand, wind, solar


def get_renewable_fraction(year, wind_factor, solar_factor, im, demand, wind, solar, zero_negative=True):
    
    d_profile = demand.loc[ demand[im['demand'][3]] == year, im['demand'][2] ].values
    w_profile = wind.loc[ wind[im['wind'][3]] == year, im['wind'][2] ].values
    s_profile = solar.loc[ solar[im['solar'][3]] == year, im['solar'][2] ].values

    final_profile = d_profile - wind_factor * w_profile - solar_factor * s_profile
    if zero_negative:
        final_profile = np.where(final_profile >= 0, final_profile, 0.)

    return np.mean(d_profile) - np.mean(final_profile)



# Ideas for 2nd x and y axis showing generation potential (CF x cap)
# Spines or 2nd axes - https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
# Parasite axes - https://matplotlib.org/3.1.0/gallery/axisartist/demo_parasite_axes2.html
# For now, try second row of text
def plot_matrix(plot_base, matrix, solar_values, wind_values, cf_wind, cf_solar, save_name):


    fig, ax = plt.subplots()#figsize=(4.5, 4))
    im = ax.imshow(matrix,interpolation='none',origin='lower')
    #im = ax.imshow(matrix,interpolation='spline16',origin='lower')

    # Contours
    levels = [25, 50, 75, 100]
    cs = ax.contour(matrix, levels, colors='k', origin='lower')
    # inline labels
    ax.clabel(cs, inline=1, fontsize=10)

    wind_labs, solar_labs = [], []
    for v in wind_values:
        if int(v)==v:
            wind_labs.append("%.1f\n(%.2f)" % (v, v * cf_wind))
        else:
            wind_labs.append('')
    for v in solar_values:
        if int(v)==v:
            solar_labs.append("%.1f\n(%.2f)" % (v, v * cf_solar))
        else:
            solar_labs.append('')
    plt.xticks(range(len(wind_values)), wind_labs, rotation=90)
    plt.yticks(range(len(solar_values)), solar_labs)
    plt.xlabel("Normalized Wind Capacity\n(Total Generation)")
    plt.ylabel("Normalized Solar Capacity\n(Total Generation)")
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(f"Renewable Energy Fraction (%)")
    plt.title(f"")
    plt.tight_layout()


    plt.savefig(f"{plot_base}/{save_name}.png")
    plt.clf()


def plot_matrix_thresholds(plot_base, matrix, solar_values, wind_values, save_name):


    plt.close()
    fig, ax = plt.subplots()#figsize=(4.5, 4))
    #im = ax.imshow(matrix,interpolation='none',origin='lower')
    im = ax.imshow(matrix,interpolation='none',origin='lower')

    wind_labs, solar_labs = [], []
    for v in wind_values:
        if int(v)==v:
            wind_labs.append("%.1f" % (v))
        else:
            wind_labs.append('')
    for v in solar_values:
        if int(v)==v:
            solar_labs.append("%.1f" % (v))
        else:
            solar_labs.append('')
    plt.xticks(range(len(wind_values)), wind_labs, rotation=90)
    plt.yticks(range(len(solar_values)), solar_labs)
    plt.xlabel("Normalized Wind Capacity")
    plt.ylabel("Normalized Solar Capacity")
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(f"Spread in Dem - Wind - Solar")
    plt.title(f"")
    plt.tight_layout()


    plt.savefig(f"{plot_base}/{save_name}.png")
    plt.clf()


def get_CF(df, name, im):

    return np.mean(df.loc[ df[im[name][3]] == year, im[name][2] ].values)

# only for demand
def print_vals(df, year, im):

    print(f"Peak demand events for year {year}")
    print("year:month:day:hour:demand")
    dfX = df.sort_values(by=['demand (MW)'], ascending=False)
    for i, idx in enumerate(dfX.index):
        print(f"{dfX.loc[idx, 'year']}:{dfX.loc[idx, 'month']}:{dfX.loc[idx, 'day']}:{dfX.loc[idx, 'hour']}:{dfX.loc[idx, 'demand (MW)']}")
        if i > 9: break

def get_annual_df(year, df, tgt, im):

    df2 = df[ df[ im[tgt][3]] == year]

    # Normalize
    if tgt == 'demand':
        df2[im[tgt][2]] = df2[im[tgt][2]]/np.mean(df2[im[tgt][2]])
        print_vals(df2, year, im)
    return df2


# demand_threshold is in percent
def return_ordered_df(demand, wind, solar, im, demand_threshold):

    #rank_mthd='ordinal'
    rank_mthd='min'
    to_map = OrderedDict()
    to_map['demand'] = demand[im['demand'][2]].values
    to_map['demand_rank'] = rankdata(demand[im['demand'][2]].values, method=rank_mthd)
    to_map['demand_pct_of_max'] = demand[im['demand'][2]].values/np.max(demand[im['demand'][2]])
    to_map['wind'] = wind[im['wind'][2]].values
    to_map['wind_rank'] = rankdata(wind[im['wind'][2]].values, method=rank_mthd)
    to_map['wind_pct_of_max'] = wind[im['wind'][2]].values/np.max(wind[im['wind'][2]])
    to_map['solar'] = solar[im['solar'][2]].values
    to_map['solar_rank'] = rankdata(solar[im['solar'][2]].values, method=rank_mthd)
    to_map['solar_pct_of_max'] = solar[im['solar'][2]].values/np.max(solar[im['solar'][2]])

    df = pd.DataFrame(to_map)
    df = df.sort_values(by='demand_rank', ascending=0)

    df = df.drop(df[df['demand_rank'] < (len(df.index) * (100. - demand_threshold)/100.)].index, axis=0)
    return df


def get_range(vect):
    return np.max(vect) - np.min(vect)


# Return the position of the integrated threshold based on total demand.
# Total demand is normalized and is 8760 or 8784 based on leap years.
# Integrate down from the max values.
def get_integrated_threshold(vals, threshold_pct):

    int_threshold = len(vals) * (1. - threshold_pct)
    int_tot = 0.
    hours = 0
    prev_val = vals[-1] # to initialize
    for i in range(len(vals)):
        current = hours * (prev_val - vals[-(i+1)])
        #print(f"{i} --- int_tot {int_tot}   hours {hours}   prev_val {prev_val}   current val {vals[-(i+1)]}   current {current}")
        if current + int_tot < int_threshold:
            hours += 1
            prev_val = vals[-(i+1)]
            int_tot += current
            continue

        # Else, we overshoot the target
        # Find location between values which would meet target
        tot_needed = int_threshold - int_tot
        # Find fraction of 'current' needed
        frac = tot_needed / current
        # Return that frac as a distance between val i and val i-1 (going bkwards)
        dist = prev_val - vals[-(i+1)]
        to_return = prev_val + dist * frac
        #print(f"  === tot_needed {tot_needed}   frac {frac}   dist {dist}   to_return {to_return}")
        return to_return




def make_ordering_plotsX(dfs, save_name, wind_factor, solar_factor, thresholds, threshold_pct, cnt, base):
    plt.close()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
    vects = []
    for t in thresholds:
        vects.append([])
    vects.append([]) # One extra for threshold_pct
    for year, df in dfs.items():
        vals = df['demand'].values - df['solar'].values * solar_factor - df['wind'].values * wind_factor
        axs[0].plot(vals, df['solar'], '.', alpha=0.2, label=year)
        vals.sort()
        pct = get_integrated_threshold(vals, threshold_pct)
        axs[0].plot([vals[-1]  for _ in range(10)], np.arange(0,1,.1), color=axs[0].lines[-1].get_color(), linestyle='-') 
        axs[0].plot([pct for _ in range(10)], np.arange(0,1,.1), color=axs[0].lines[-1].get_color(), linestyle='--') 
        for i, t in enumerate(thresholds):
            vects[i].append(vals[-1 * t])
        vects[-1].append(pct)
    axs[0].set_xlabel(f'demand - (wind x {round(wind_factor,3)}) - (solar x {round(solar_factor,3)})')
    axs[0].set_ylabel('solar CF')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, 2)
    for year, df in dfs.items():
        vals = df['demand'].values - df['solar'].values * solar_factor - df['wind'].values * wind_factor
        axs[1].plot(vals, df['wind'], '.', alpha=0.2, label=year)
        vals.sort()
        pct = get_integrated_threshold(vals, threshold_pct)
        axs[1].plot([vals[-1]  for _ in range(10)], np.arange(0,1,.1), color=axs[1].lines[-1].get_color(), linestyle='-') 
        axs[1].plot([pct for _ in range(10)], np.arange(0,1,.1), color=axs[1].lines[-1].get_color(), linestyle='--') 
    axs[1].set_xlabel(f'demand - (wind x {round(wind_factor,3)}) - (solar x {round(solar_factor,3)})')
    axs[1].set_ylabel('wind CF')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base}/{save_name}_dem_min_solar_vs_solarCF_cnt{cnt:03}_solarSF{str(round(solar_factor,3)).replace('.','p')}_windSF{str(round(wind_factor,3)).replace('.','p')}.png")
    plt.clf()

    out = []
    out2 = []
    out3 = []
    for vect in vects:
        out.append(get_range(vect))
        out2.append(np.std(vect))
        distances = []
        for i in range(len(vect)):
            for j in range(len(vect)):
                if i == j:
                    continue
                distances.append(vect[i] - vect[j])
        out3.append(np.std(distances))


    return out, out2, out3



def make_ordering_plot(dfs, save_name, wind_factor=1., solar_factor=1., cnt=0):

    #plt.close()
    #fig, ax = plt.subplots()
    #ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), 'k-')
    #for year, df in dfs.items():
    #    print(year)
    #    ax.plot(df['demand']-df['wind']*wind_factor, df['solar']*solar_factor, '.', alpha=0.2, label=year)
    #ax.set_xlabel('demand - wind gen')
    #ax.set_ylabel('solar gen')
    #ax.set_ylim(0, ax.get_ylim()[1])
    #plt.legend()
    #plt.savefig(f"plots_new/{save_name}_dem_min_wind_vs_solar.png")
    #plt.clf()

    plt.close()
    fig, ax = plt.subplots()
    #ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), 'k-')
    max_vals = []
    hundredth_vals = []
    for year, df in dfs.items():
        ax.plot(df['demand']-df['solar']*solar_factor, df['solar'], '.', alpha=0.2, label=year)
        vals = df['demand'].values - df['solar'].values * solar_factor
        vals.sort()
        ax.plot([vals[-1]  for _ in range(10)], np.arange(0,1,.1), color=plt.gca().lines[-1].get_color(), linestyle='-') 
        ax.plot([vals[-100] for _ in range(10)], np.arange(0,1,.1), color=plt.gca().lines[-1].get_color(), linestyle='--') 
        #ax.plot([vals[-50] for _ in range(10)], np.arange(0,1,.1), color=plt.gca().lines[-1].get_color(), linestyle='-.') 
        max_vals.append(vals[-1])
        hundredth_vals.append(vals[-100])
    ax.set_xlabel('demand - solar gen')
    ax.set_ylabel('wind gen')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(1.3, 2)
    plt.legend()
    plt.savefig(f"plots_new/{save_name}_dem_min_solar_vs_solarCF_cnt{cnt:03}_solarSF{str(round(solar_factor,3)).replace('.','p')}_windSF{str(round(wind_factor,3)).replace('.','p')}.png")
    plt.clf()

    return max_vals, hundredth_vals

    #plt.close()
    #fig, ax = plt.subplots()
    #for year, df in dfs.items():
    #    ax.plot(df['demand'], df['solar'], '.', alpha=0.2, label=year)
    #    vals = df['demand'].values
    #    vals.sort()
    #    ax.plot([vals[-1]  for _ in range(10)], np.arange(0,1,.1), color=plt.gca().lines[-1].get_color(), linestyle='-') 
    #    ax.plot([vals[-100] for _ in range(10)], np.arange(0,1,.1), color=plt.gca().lines[-1].get_color(), linestyle='--') 
    #    #ax.plot([vals[-50] for _ in range(10)], np.arange(0,1,.1), color=plt.gca().lines[-1].get_color(), linestyle='-.') 
    #ax.set_xlabel('demand')
    #ax.set_ylabel('solar CF')
    #ax.set_ylim(0, ax.get_ylim()[1])
    #ax.set_xlim(1.5, ax.get_xlim()[1])
    #plt.legend()
    #plt.savefig(f"plots_new/{save_name}_solar.png")
    #plt.clf()

    #plt.close()
    #fig, ax = plt.subplots()
    #for year, df in dfs.items():
    #    print(year)
    #    ax.plot(df['demand'], df['wind'],  '.', alpha=0.2, label=year)
    #ax.set_xlabel('demand')
    #ax.set_ylabel('wind CF')
    #ax.set_ylim(0, ax.get_ylim()[1])
    #plt.legend()
    #plt.savefig(f"plots_new/{save_name}_wind.png")
    #plt.clf()


    #plt.close()
    #fig, ax = plt.subplots()
    #for year, df in dfs.items():
    #    print(year)
    #    ax.plot(df['demand_rank'], df['solar_rank'], '.', alpha=0.2, label=year)
    #ax.set_xlabel('demand rank')
    #ax.set_ylabel('solar rank')
    #ax.set_ylim(0, ax.get_ylim()[1])
    #plt.legend()
    #plt.savefig(f"plots_new/{save_name}_solar_rank.png")
    #plt.clf()

    #plt.close()
    #fig, ax = plt.subplots()
    #for year, df in dfs.items():
    #    print(year)
    #    ax.plot(df['demand_rank'], df['wind_rank'],  '.', alpha=0.2, label=year)
    #ax.set_xlabel('demand rank')
    #ax.set_ylabel('wind rank')
    #ax.set_ylim(0, ax.get_ylim()[1])
    #plt.legend()
    #plt.savefig(f"plots_new/{save_name}_wind_rank.png")
    #plt.clf()



region = 'CONUS'
region = 'TEXAS'
im = return_file_info_map(region)
demand, wind, solar = get_dem_wind_solar(im)



to_explore = [0, 3.001, 0.1]
r_vals = np.arange(to_explore[0], to_explore[1], to_explore[2])
tgt_len = len(r_vals)
thresholds = [1, 3, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
int_threshold = 0.999

plot_base = 'plots_new_Jan17'
if not os.path.exists(plot_base):
    os.makedirs(plot_base)

test_ordering = True
#test_ordering = False
make_plots = True
#make_plots = False
make_scan = True
make_scan = False
run_correlations = True
#run_correlations = False



if test_ordering:
    dfs = OrderedDict()
    years = [2016,2017,2018]
    years = [y for y in range(2005, 2019)]
    #years = [y for y in range(2005, 2009)]
    for year in years:
        d_yr = get_annual_df(year, demand, 'demand', im)
        w_yr = get_annual_df(year, wind, 'wind', im)
        s_yr = get_annual_df(year, solar, 'solar', im)
        dfs[year] = return_ordered_df(d_yr, w_yr, s_yr, im, 100)


    wind_factor=0.
    steps = r_vals
    mapper = OrderedDict()
    cnt = 0
    for solar_factor in steps:
        mapper[str(round(solar_factor,2))] = OrderedDict()
        print(f"Solar factor {solar_factor}")
        for wind_factor in steps:
            vects1, vects2, vects3 = make_ordering_plotsX(dfs, f'ordering_{region}', wind_factor, solar_factor, thresholds, int_threshold, cnt, plot_base)
            mapper[str(round(solar_factor,2))][str(round(wind_factor,2))] = [vects1, vects2, vects3]
            cnt += 1

    #print("Solar Wind max_range 100th_range")
    #for solar, info in mapper.items():
    #    for wind, vals in info.items():
    #        print(solar, wind, vals)
        
    pickle_file = open('tmp5.pkl', 'wb')
    pickle.dump(mapper, pickle_file)
    pickle_file.close()

    # Sort plots
    tgt = plot_base+'/ordering*'
    make_dirs(plot_base, tgt)

if make_plots:
    pickle_in = open('tmp5.pkl','rb')
    study_regions = pickle.load(pickle_in)
    steps = r_vals


    thresholds.append(int_threshold)
    for t, threshold in enumerate(thresholds):
        print(threshold)
        # Range of dem - wind - solar
        matrix = []
        for i, solar_factor in enumerate(steps):
            matrix.append([])
            for wind_factor in steps:
                val = study_regions[str(round(solar_factor,2))][str(round(wind_factor,2))][0][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(plot_base, matrix, r_vals, r_vals, f'threshold_range_{threshold:03}')

        # Std dev of dem - wind - solar
        matrix = []
        for i, solar_factor in enumerate(steps):
            matrix.append([])
            for wind_factor in steps:
                val = study_regions[str(round(solar_factor,2))][str(round(wind_factor,2))][1][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(plot_base, matrix, r_vals, r_vals, f'threshold_std_{threshold:03}')

        # Std dev of all distances between dem - wind - solar (closest to MEM reliability)
        # results I have shown
        matrix = []
        for i, solar_factor in enumerate(steps):
            matrix.append([])
            for wind_factor in steps:
                val = study_regions[str(round(solar_factor,2))][str(round(wind_factor,2))][2][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(plot_base, matrix, r_vals, r_vals, f'threshold_std_of_distances_{threshold:03}')


if make_scan:

    year = 2018
    # Normalize demand for this year
    mean = np.mean(demand.loc[ demand[im['demand'][3]] == year, im['demand'][2] ])
    demand[ im['demand'][2] ] = demand[ im['demand'][2] ] / mean
    print(f"Mean normalized demand for year {year} = {mean}")
    
    to_explore = [0, 3.001, 0.05]
    r_vals = np.arange(to_explore[0], to_explore[1], to_explore[2])
    tgt_len = len(r_vals)
    
    matrix = []
    
    zero_negative=True
    #zero_negative=False
    for i, solar_factor in enumerate(r_vals):

        matrix.append([])
        for wind_factor in r_vals:
            print(solar_factor, wind_factor)
            val = get_renewable_fraction(year, wind_factor, solar_factor, im, demand, wind, solar, zero_negative)
            #print(f" - Wind:Solar:RenewableFraction {wind_factor}:\t{solar_factor}:\t{round(val,3)}")
            matrix[i].append(val*100)
    
    ary = np.array(matrix)

    cf_wind = get_CF(wind, 'wind', im)
    cf_solar = get_CF(solar, 'solar', im)
    plot_matrix(plot_base, matrix, r_vals, r_vals, cf_wind, cf_solar, 'testX')



if run_correlations:



