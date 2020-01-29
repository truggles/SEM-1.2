#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata, normaltest
from collections import OrderedDict
import pickle
from glob import glob
import os
from shutil import copy2


def return_file_info_map(region):
    #assert(region in ['CONUS', 'ERCOT', 'NYISO', 'TEXAS'])

    info_map = { # region : # f_path, header rows
        'CONUS': {
            'demand': ['Input_Data/ReliabilityPaper/CONUS_demand_unnormalized.csv', 8, 'demand (MW)', 'year'],
            'wind': ['Input_Data/ReliabilityPaper/CONUS_wind_top25pct_unnormalized.csv', 6, 'wind capacity', 'year'], 
            'solar': ['Input_Data/ReliabilityPaper/CONUS_solar_top25pct_unnormalized.csv', 6, 'solar capacity', 'year'],
            'years' : [2015, 2016,2017,2018],
        },
        'ERCOT': { 
            'demand': ['Input_Data/ReliabilityPaper/ERCOT_demand_unnormalized.csv', 6, 'demand (MW)', 'year'],
            'wind': ['Input_Data/ReliabilityPaper/ERCOT_wind_top25pct_unnormalized.csv', 6, 'wind capacity', 'year'],
            'solar': ['Input_Data/ReliabilityPaper/ERCOT_solar_top25pct_unnormalized.csv', 6, 'solar capacity', 'year'],
            'years' : [y for y in range(2003, 2019)],
        },
        'TEXAS': {
            'demand': ['Input_Data/ReliabilityPaper/ERCOT_demand_unnormalized.csv', 6, 'demand (MW)', 'year'],
            'wind': ['Input_Data/TEXAS/TI_wind_thresh.csv', 5, 'wind capacity', 'year'],
            'solar': ['Input_Data/TEXAS/TI_solar_thresh.csv', 5, 'solar capacity', 'year'],
            'years' : [y for y in range(2003, 2019)],
        },
        'TXv1': {
            'demand': ['Input_Data/ReliabilityPaper/ERCOT_demand_unnormalized.csv', 6, 'demand (MW)', 'year'],
            'wind': ['Input_Data/ReliabilityPaper/ERCOT_wind_top25pct_unnormalized.csv', 6, 'wind capacity', 'year'],
            'solar': ['Input_Data/TEXAS/TI_solar_thresh.csv', 5, 'solar capacity', 'year'],
            'years' : [y for y in range(2003, 2019)],
        },
        'TXv2': {
            'demand': ['Input_Data/ReliabilityPaper/ERCOT_demand_unnormalized.csv', 6, 'demand (MW)', 'year'],
            'wind': ['Input_Data/TEXAS/TI_wind_thresh.csv', 5, 'wind capacity', 'year'],
            'solar': ['Input_Data/ReliabilityPaper/ERCOT_solar_top25pct_unnormalized.csv', 6, 'solar capacity', 'year'],
            'years' : [y for y in range(2003, 2019)],
        },
        'NYISO': { 
            'demand': ['Input_Data/ReliabilityPaper/NYISO_demand_unnormalized.csv', 5, 'demand (MW)', 'year'],
            'wind': ['Input_Data/ReliabilityPaper/NYISO_wind_top25pct_unnormalized.csv', 6, 'wind capacity', 'year'],
            'solar': ['Input_Data/ReliabilityPaper/NYISO_solar_top25pct_unnormalized.csv', 6, 'solar capacity', 'year'],
            'years' : [y for y in range(2004, 2019)],
        }
    }
    return info_map[region]



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


def get_renewable_fraction(year, wind_install_cap, solar_install_cap, im, demand, wind, solar, zero_negative=True):
    
    d_profile = demand.loc[ demand[im['demand'][3]] == year, im['demand'][2] ].values
    w_profile = wind.loc[ wind[im['wind'][3]] == year, im['wind'][2] ].values
    s_profile = solar.loc[ solar[im['solar'][3]] == year, im['solar'][2] ].values

    final_profile = d_profile - wind_install_cap * w_profile - solar_install_cap * s_profile
    if zero_negative:
        final_profile = np.where(final_profile >= 0, final_profile, 0.)

    return np.mean(d_profile) - np.mean(final_profile)



# Ideas for 2nd x and y axis showing generation potential (CF x cap)
# Spines or 2nd axes - https://stackoverflow.com/questions/31803817/how-to-add-second-x-axis-at-the-bottom-of-the-first-one-in-matplotlib
# Parasite axes - https://matplotlib.org/3.1.0/gallery/axisartist/demo_parasite_axes2.html
# For now, try second row of text
#def plot_matrix(plot_base, matrix, solar_values, wind_values, cf_wind, cf_solar, save_name):
def plot_matrix(plot_base, matrix, solar_values, wind_values, save_name):


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
            wind_labs.append("%.2f" % v)
        else:
            wind_labs.append('')
    for v in solar_values:
        if int(v)==v:
            solar_labs.append("%.2f" % v)
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


def plot_matrix_thresholds(region, plot_base, matrix, solar_values, wind_values, save_name):


    plt.close()
    fig, ax = plt.subplots()#figsize=(4.5, 4))
    im = ax.imshow(matrix,interpolation='none',origin='lower')

    # Contours
    if not 'pval' in save_name:
        n_levels = np.arange(0,.1,.01)
        n_levels = np.append(n_levels, [0.025,])
        n_levels.sort()
        #cs = ax.contour(matrix, n_levels, colors='w', origin='lower')
        cs = ax.contour(matrix, n_levels, colors='w')
        # inline labels
        ax.clabel(cs, inline=1, fontsize=10)


    wind_labs, solar_labs = [], []
    for v in wind_values:
        if int(v*2)==v*2:
            wind_labs.append(f"{int(v*100)}%")
        else:
            wind_labs.append('')
    for v in solar_values:
        if int(v*4)==v*4:
            solar_labs.append(f"{int(v*100)}%")
        else:
            solar_labs.append('')
    plt.xticks(range(len(wind_values)), wind_labs, rotation=90)
    plt.yticks(range(len(solar_values)), solar_labs)
    plt.xlabel("Wind Generation as % of Total Demand")
    plt.ylabel("Solar Generation as % of Total Demand")
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(f"Spread of Thershold Locations ($\sigma$)")
    plt.title(f"")
    plt.tight_layout()


    plt.savefig(f"{plot_base}/{region}_{save_name}.png")
    plt.clf()


def get_avg_CF(dfs, name, im):
    to_avg = []
    for year, df in dfs.items():
        avg = np.mean(df[name])
        #print(year, len(df.index), name, avg)
        to_avg.append(avg)
    return np.mean(to_avg)


def get_annual_CF(df, name, im, year):

    return np.mean(df.loc[ df[im[name][3]] == year, im[name][2] ].values)

# only for demand
def print_vals(df, year, im):

    print(f"Peak demand events for year {year}")
    print("year:month:day:hour:demand")
    dfX = df.sort_values(by=['demand (MW)'], ascending=False)
    for i, idx in enumerate(dfX.index):
        print(f"{dfX.loc[idx, 'year']}:{dfX.loc[idx, 'month']}:{dfX.loc[idx, 'day']}:{dfX.loc[idx, 'hour']}:{dfX.loc[idx, 'demand (MW)']}")
        if i > 9: break



def plot_top_X_hours(dfs, top_X, save_name, wind_install_cap, solar_install_cap, cnt, base, gens=[0, 0]):

    hours = []
    months = []

    # hours to local time:
    if 'ERCOT' in base:
        adj = -6
        nm = 'CST'
    if 'NYISO' in base:
        adj = -5
        nm = 'EST'
    if 'CONUS' in base:
        adj = -6
        nm = 'CST'

    m = [[0 for _ in range(24)] for _ in range(12)]
    for year, df in dfs.items():
        mod_df = df
        mod_df['mod_dem'] = df['demand'] - df['solar'] * solar_install_cap - df['wind'] * wind_install_cap
        mod_df = mod_df.sort_values(by=['mod_dem'], ascending=False)

        for i, idx in enumerate(mod_df.index):
            if i == top_X:
                break
            hour = (mod_df.loc[idx, 'hour'] + adj)%24
            month = mod_df.loc[idx, 'month']
            hours.append(hour)
            months.append(month)
            m[month-1][hour-1] += 1

    hours_ary = np.array(hours)
    months_ary = np.array(months)
    plt.close()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    n1, bins1, patches1 = axs[0].hist(hours_ary, np.arange(0.5,24.51,1),  density=True, histtype=u'step', linewidth=4)
    n2, bins2, patches2 = axs[1].hist(months_ary, np.arange(0.5,12.51,1), density=True,  histtype=u'step', linewidth=4)

    axs[0].set_xlim(0.5,24.5)
    axs[1].set_xlim(0.5,12.5)
    axs[0].set_xlabel(f'Hours ({nm})')
    axs[1].set_xlabel('Months')
    axs[0].set_ylabel(f'Top {top_X} Hours - Normalized')
    axs[1].set_ylabel(f'Top {top_X} Hours - Normalized')

    
    plt.savefig(f"{base}/{save_name}_top{top_X:03}_cnt{cnt:03}_solarSF{str(round(gens[1],3)).replace('.','p')}_windSF{str(round(gens[0],3)).replace('.','p')}.png")

    plt.close()
    matrix = np.array(m)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix/np.sum(matrix.flatten())*100.,interpolation='none',origin='lower',aspect='auto')
    plt.xticks(range(0, 24), range(0, 24))
    plt.yticks([i for i in range(0, 12)], ('Jan','Feb','Mar','Apr','May','Jun',
        'Jul','Aug','Sep','Oct','Nov','Dec'))
    plt.xlabel(f'Hours ({nm})')
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel(f'Top {top_X} Hours (%)')
    plt.savefig(f"{base}/{save_name}_top{top_X:03}_2D_cnt{cnt:03}_solarSF{str(round(gens[1],3)).replace('.','p')}_windSF{str(round(gens[0],3)).replace('.','p')}.png")



# CONUS is short enough we want to squeeze 4 years out.
# The others are simple
def get_annual_df(region, year, df, tgt, im):

    df2 = df[ df[ im[tgt][3]] == year]
    # FIXME - add this section once we have 2019 wind / solar data for CONUS
    #if region == 'CONUS':
    #    df3 = df[ df[ im[tgt][3]] == year+1]
    #    df2a = df2[ df2['month'] >= 8]
    #    df3a = df3[ df3['month'] < 8]
    #    print(f"len df2a {len(df2a.index)}")
    #    print(f"len df3a {len(df3a.index)}")
    #    df2a = df2a.append(df3a, ignore_index=True)
    #    print(f"len df2a {len(df2a.index)}")
    #    df2 = df2a
    #    print(f"len df2 {len(df2.index)}")
    #    print(df2.head())
    #    print(df2.tail())

    # Normalize
    if tgt == 'demand':
        df2[im[tgt][2]] = df2[im[tgt][2]]/np.mean(df2[im[tgt][2]])
        #print_vals(df2, year, im)
    return df2


# demand_threshold is in percent
def return_ordered_df(demand, wind, solar, im, demand_threshold):

    #rank_mthd='ordinal'
    rank_mthd='min'
    to_map = OrderedDict()
    to_map['month'] = demand['month'].values
    to_map['day'] = demand['day'].values
    to_map['hour'] = demand['hour'].values
    to_map['demand'] = demand[im['demand'][2]].values
    #to_map['demand_rank'] = rankdata(demand[im['demand'][2]].values, method=rank_mthd)
    #to_map['demand_pct_of_max'] = demand[im['demand'][2]].values/np.max(demand[im['demand'][2]])
    to_map['wind'] = wind[im['wind'][2]].values
    #to_map['wind_rank'] = rankdata(wind[im['wind'][2]].values, method=rank_mthd)
    #to_map['wind_pct_of_max'] = wind[im['wind'][2]].values/np.max(wind[im['wind'][2]])
    to_map['solar'] = solar[im['solar'][2]].values
    #to_map['solar_rank'] = rankdata(solar[im['solar'][2]].values, method=rank_mthd)
    #to_map['solar_pct_of_max'] = solar[im['solar'][2]].values/np.max(solar[im['solar'][2]])

    df = pd.DataFrame(to_map)
    #df = df.sort_values(by='demand_rank', ascending=0)

    #df = df.drop(df[df['demand_rank'] < (len(df.index) * (100. - demand_threshold)/100.)].index, axis=0)
    return df


def get_range(vect):
    return np.max(vect) - np.min(vect)
        
# for 94% confidence in achieving X reliability goal (94% = 15years/16year based on ERCOT)
def get_2nd_highest(vect):
    vect.sort()
    return vect[-2] - np.mean(vect) # 2nd from end.  Sort defaults to ascending order.



# Return the position of the integrated threshold based on total demand.
# Total demand is normalized and is 8760 or 8784 based on leap years.
# Integrate down from the max values.
def get_integrated_threshold(vals, threshold_pct):

    int_threshold = len(vals) * (1. - threshold_pct)
    int_tot = 0.
    hours = 0
    prev_val = vals[-1] # to initialize
    for i, val in enumerate(reversed(vals)):
        current = hours * (prev_val - val)
        #print(f"{i} --- Running total: {round(int_tot,5)}   Hours: {hours}   Current val {round(val,5)}   To add? {round(current,5)}")
        if current + int_tot < int_threshold:
            hours += 1
            prev_val = val
            int_tot += current
            continue

        # Else, we overshoot the target
        # Find location between values which would meet target
        tot_needed = int_threshold - int_tot
        # Find fraction of 'current' needed
        frac = tot_needed / current
        # Return that frac as a distance between val i and val i-1 (going bkwards)
        dist = prev_val - val
        to_return = prev_val - dist * frac
        #print(f"  === tot_needed {tot_needed}   frac {frac}   dist {dist}   to_return {to_return}")
        return to_return

# Check method
def check_pct(vals, threshold_pct, threshold):

    tot = 0
    for v in reversed(vals):
        if v < threshold:
            break
        tot += v - threshold

    int_threshold = len(vals) * (1. - threshold_pct)
    print(f"Tgt: {int_threshold} == {tot}? {round(int_threshold,4) == round(tot,4)}")


def load_duration_curve_and_PDF_plots(dfs, save_name, wind_install_cap, solar_install_cap, cnt, base, gens=[0, 0], threshold_pcts=[]):

    plt.close()
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,5))
    fig.suptitle(f'Dem - wind CF x {round(wind_install_cap,2)} - solar CF x {round(solar_install_cap,2)}')
    axs[0].set_title(f'Load Duration Curve')
    axs[1].set_title(f'PDF')

    good_max = 0
    threshold_vals = []
    for year, df in dfs.items():
        mod_df = df
        mod_df['mod_dem'] = df['demand'] - df['solar'] * solar_install_cap - df['wind'] * wind_install_cap
        mod_df = mod_df.sort_values(by=['mod_dem'], ascending=False)
        axs[0].plot(np.linspace(0,1,len(mod_df.index)), mod_df['mod_dem'], linestyle='-', linewidth=0.5)
        to_bins = np.linspace(-10,2,601)
        n, bins, patches = axs[1].hist(mod_df['mod_dem'], to_bins, orientation='horizontal', histtype=u'step', color=axs[0].lines[-1].get_color(), linewidth=0.5)
        if np.max(n) > good_max:
            good_max = np.max(n)

        # If threshold_pcts has unique values
        for i, t in enumerate(threshold_pcts):
            vals = df['demand'].values - df['solar'].values * solar_install_cap - df['wind'].values * wind_install_cap
            vals.sort()
            pct = get_integrated_threshold(vals, t)
            #check_pct(vals, t, pct)
            if i == 0:
                #print(f"Adding threshold lines for {t}")
                threshold_vals.append(pct)
                axs[0].plot(np.linspace(-0.1,1,10), [pct for _ in range(10)], color=axs[0].lines[-1].get_color(), linestyle='-', linewidth=0.5) 
                axs[1].plot(np.linspace(0,1000,10), [pct for _ in range(10)], color=axs[0].lines[-1].get_color(), linestyle='-', linewidth=0.5) 
    if len(threshold_vals) > 0:
        plt.text(0.70, 0.85, f'range: {round(np.max(threshold_vals) - np.min(threshold_vals),3)}\n$\sigma$: {round(np.std(threshold_vals),3)}',
                horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes, fontsize=14)


    #axs[0].yaxis.grid(True)
    axs[0].set_xlim(-0.005, 1)
    #axs[0].set_ylim(0, axs[0].get_ylim()[1])
    axs[0].set_ylim(0, 2)
    axs[0].set_ylabel('Demand - Wind - Solar (kWh)')
    axs[0].set_xlabel('Operating duration (% of year)')
    #axs[1].yaxis.grid(True)
    axs[1].set_ylabel('Demand - Wind - Solar (kWh)')
    axs[1].set_xlabel('Hours / Bin')
    axs[1].set_xlim(0, good_max * 1.2)
    #axs[1].set_ylim(0, axs[1].get_ylim()[1])
    axs[1].set_ylim(0, 2)
    axs[1].yaxis.set_tick_params(labelleft=True)
    #axs[1].set_xlabel('Demand - Wind - Solar (kWh)')
    #plt.tight_layout()
    plt.savefig(f"{base}/{save_name}_LDC_and_PDF_cnt{cnt:03}_solarSF{str(round(gens[1],3)).replace('.','p')}_windSF{str(round(gens[0],3)).replace('.','p')}.png")


# Make box plots showing the wind and solar CFs for the top X thresholds
def make_box_plots(dfs, save_name, wind_install_cap, solar_install_cap, box_thresholds, cnt, base, gens=[0, 0]):

    to_plot = [[] for _ in range(len(box_thresholds)*2)]
    for i, threshold in enumerate(box_thresholds):
        for year, df in dfs.items():
            mod_df = df
            mod_df['mod_dem'] = df['demand'] - df['solar'] * solar_install_cap - df['wind'] * wind_install_cap
            mod_df = mod_df.sort_values(by=['mod_dem'], ascending=False)
            for j, idx in enumerate(mod_df.index):
                to_plot[i].append(mod_df.loc[idx, 'wind'])
                to_plot[len(box_thresholds)+i].append(mod_df.loc[idx, 'solar'])
                if j == threshold: break

    plt.close()
    fig, ax = plt.subplots(figsize=(7,5))
    ax.yaxis.grid(True)
    ax.set_title(f'Dem - wind CF x {round(wind_install_cap,2)} - solar CF x {round(solar_install_cap,2)}: whiskers at 5%/95%')
    medianprops = dict(linestyle='-', linewidth=2.5)
    bplot = ax.boxplot(to_plot, whis=[5, 95], showfliers=True, patch_artist=True, medianprops=medianprops)
    x_labels = []
    for val in box_thresholds:
        x_labels.append(f'Wind CFs\nTop {val} Hours')
    for val in box_thresholds:
        x_labels.append(f'Solar CFs\nTop {val} Hours')
    plt.xticks([i for i in range(1, len(box_thresholds)*2+1)], x_labels, rotation=30)
    ax.set_ylabel('Wind or Solar CF')
    plt.tight_layout()
    for patch in bplot['boxes']:
        patch.set_facecolor('lightblue')
    plt.savefig(f"{base}/{save_name}_CFs_box_plot_cnt{cnt:03}_solarSF{str(round(gens[1],3)).replace('.','p')}_windSF{str(round(gens[0],3)).replace('.','p')}.png")



def make_ordering_plotsX(dfs, save_name, wind_install_cap, solar_install_cap, thresholds, threshold_pcts, cnt, base, gens=[0, 0]):
    plt.close()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
    vects = []
    for t in thresholds:
        vects.append([])
    for t in threshold_pcts:
        vects.append([]) # One extra for threshold_pct
    for year, df in dfs.items():
        vals = df['demand'].values - df['solar'].values * solar_install_cap - df['wind'].values * wind_install_cap
        axs[0].plot(vals, df['solar'], '.', alpha=0.2, label=year)
        vals.sort()
        axs[0].plot([vals[-1]  for _ in range(10)], np.arange(0,1,.1), color=axs[0].lines[-1].get_color(), linestyle='-') 
        for i, t in enumerate(thresholds):
            vects[i].append(vals[-1 * t])

        for i, t in enumerate(threshold_pcts):
            pct = get_integrated_threshold(vals, t)
            if i == 0:
                axs[0].plot([pct for _ in range(10)], np.arange(0,1,.1), color=axs[0].lines[-1].get_color(), linestyle='--') 
            vects[len(thresholds)+i].append(pct)
    axs[0].set_xlabel(f'demand - (wind x {round(wind_install_cap,3)}) - (solar x {round(solar_install_cap,3)})')
    axs[0].set_ylabel('solar CF')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, 2)
    for year, df in dfs.items():
        vals = df['demand'].values - df['solar'].values * solar_install_cap - df['wind'].values * wind_install_cap
        axs[1].plot(vals, df['wind'], '.', alpha=0.2, label=year)
        vals.sort()
        pct = get_integrated_threshold(vals, threshold_pcts[0]) # Just plot the first one
        axs[1].plot([vals[-1]  for _ in range(10)], np.arange(0,1,.1), color=axs[1].lines[-1].get_color(), linestyle='-') 
        axs[1].plot([pct for _ in range(10)], np.arange(0,1,.1), color=axs[1].lines[-1].get_color(), linestyle='--') 
    axs[1].set_xlabel(f'demand - (wind x {round(wind_install_cap,3)}) - (solar x {round(solar_install_cap,3)})')
    axs[1].set_ylabel('wind CF')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, 2)
    plt.legend()
    plt.tight_layout()
    #if wind_install_cap == 0:
    #    plt.savefig(f"{base}/{save_name}_dem_min_solar_vs_solarCF_cnt{cnt:03}_solarSF{str(round(gens[1],3)).replace('.','p')}_windSF{str(round(gens[0],3)).replace('.','p')}.png")
    plt.clf()

    out_range = []
    out_std = []
    out_mean = []
    out_mean_to_2nd_highest = []
    out_norm_test_pval = []
    for vect in vects:
        out_range.append(get_range(vect))
        out_std.append(np.std(vect))
        out_mean.append(np.mean(vect))
        out_mean_to_2nd_highest.append(get_2nd_highest(vect))
        out_norm_test_pval.append(normaltest(vect)[1]) # save p values only


    return out_range, out_std, out_mean, out_mean_to_2nd_highest, out_norm_test_pval






#region = 'TEXAS' # Don't use
#region = 'TXv1' # Don't use
#region = 'TXv2' # Don't use
region = 'CONUS'
#region = 'NYISO'
region = 'ERCOT'
im = return_file_info_map(region)
demand, wind, solar = get_dem_wind_solar(im)


### HERE
test_ordering = True
#test_ordering = False
make_plots = True
#make_plots = False
make_scan = True
make_scan = False

thresholds = [1,]
int_thresholds = [0.9997, 0.999, 0.9999]

# Define scan space by "Total X Generation Potential" instead of installed Cap
solar_max = 1.
wind_max = 2.
steps = 41
#steps = 5
solar_gen_steps = np.linspace(0, solar_max, steps)
wind_gen_steps = np.linspace(0, wind_max, steps)
print("Wind gen increments:", wind_gen_steps)
print("Solar gen increments:", solar_gen_steps)

plot_base = f'plots_new_Jan28x_{region}'
if not os.path.exists(plot_base):
    os.makedirs(plot_base)

pkl_file = 'tmp6' # At home 41x41
pkl_file = 'Jan28_41x41'
pkl_file += f'_{region}x'

if test_ordering:
    dfs = OrderedDict()
    years = im['years']
    print(len(years))
    #years = [y for y in range(2005, 2009)]
    for year in years:
        d_yr = get_annual_df(region, year, demand, 'demand', im)
        w_yr = get_annual_df(region, year, wind, 'wind', im)
        s_yr = get_annual_df(region, year, solar, 'solar', im)
        dfs[year] = return_ordered_df(d_yr, w_yr, s_yr, im, 100)

    avg_wind_CF = get_avg_CF(dfs, 'wind', im)
    avg_solar_CF = get_avg_CF(dfs, 'solar', im)
    print(f"Avg wind CF: {avg_wind_CF}")
    print(f"Avg solar CF: {avg_solar_CF}")
    wind_cap_steps = np.linspace(0, wind_max/avg_wind_CF, steps)
    solar_cap_steps = np.linspace(0, solar_max/avg_solar_CF, steps)
    print("Wind cap increments:", wind_cap_steps)
    print("Solar cap increments:", solar_cap_steps)

    mapper = OrderedDict()
    cnt = 0
    for i, solar_install_cap in enumerate(solar_cap_steps):
        solar_gen = solar_gen_steps[i]
        mapper[str(round(solar_gen,2))] = OrderedDict()
        print(f"Solar cap {solar_install_cap}, solar gen {solar_gen}")
        for j, wind_install_cap in enumerate(wind_cap_steps):
            #if j > 0:
            #    cnt += 1
            #    continue
            wind_gen = wind_gen_steps[j]
            vect_range, vect_std, vect_mean, vect_2nd_from_top, p_val = make_ordering_plotsX(dfs, f'ordering_{region}', wind_install_cap, solar_install_cap, thresholds, int_thresholds, cnt, plot_base, [wind_gen, solar_gen])
            mapper[str(round(solar_gen,2))][str(round(wind_gen,2))] = [vect_range, vect_std, vect_mean, vect_2nd_from_top, p_val]

            plot_top_X_hours(dfs, 20, f'ordering_{region}', wind_install_cap, solar_install_cap, cnt, plot_base, [wind_gen, solar_gen])
            if wind_gen == 0:
                box_thresholds = [20, 500]
                make_box_plots(dfs, f'ordering_{region}', wind_install_cap, solar_install_cap, box_thresholds, cnt, plot_base, [wind_gen, solar_gen])

                load_duration_curve_and_PDF_plots(dfs, f'ordering_{region}', wind_install_cap, solar_install_cap, cnt, plot_base, [wind_gen, solar_gen], int_thresholds)

            cnt += 1

    #print("Solar Wind max_range 100th_range")
    #for solar, info in mapper.items():
    #    for wind, vals in info.items():
    #        print(solar, wind, vals)
        
    pickle_file = open(f'{pkl_file}.pkl', 'wb')
    pickle.dump(mapper, pickle_file)
    pickle_file.close()

    # Sort plots
    tgt = plot_base+'/ordering*'
    make_dirs(plot_base, tgt)

if make_plots:
    pickle_in = open(f'{pkl_file}.pkl','rb')
    study_regions = pickle.load(pickle_in)


    for int_threshold in int_thresholds:
        thresholds.append(int_threshold)
    for t, threshold in enumerate(thresholds):
        print(threshold)

        # Std dev of dem - wind - solar
        matrix = []
        for i, solar_install_cap in enumerate(solar_gen_steps):
            matrix.append([])
            for j, wind_install_cap in enumerate(wind_gen_steps):
                val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][1][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'threshold_std_{threshold:03}')

        # Needed dispatchable+storage
        matrix = []
        for i, solar_install_cap in enumerate(solar_gen_steps):
            matrix.append([])
            for j, wind_install_cap in enumerate(wind_gen_steps):
                val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'needed_dispatchablePlusStorage_{threshold:03}')

        # Needed overbuild in dispatchable+storage
        matrix = []
        for i, solar_install_cap in enumerate(solar_gen_steps):
            matrix.append([])
            for j, wind_install_cap in enumerate(wind_gen_steps):
                val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][1][t]
                val /= study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'overbuild_dispatchablePlusStorage_{threshold:03}')

        # Needed overbuild in dispatchable+storage 94% conf (based on ERCOT)
        matrix = []
        for i, solar_install_cap in enumerate(solar_gen_steps):
            matrix.append([])
            for j, wind_install_cap in enumerate(wind_gen_steps):
                val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][3][t]
                val /= study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][2][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'overbuild_95pct_dispatchablePlusStorage_{threshold:03}')

        # Norm test p_val (null = distribution is based on normal)
        matrix = []
        for i, solar_install_cap in enumerate(solar_gen_steps):
            matrix.append([])
            for j, wind_install_cap in enumerate(wind_gen_steps):
                val = study_regions[str(round(solar_install_cap,2))][str(round(wind_install_cap,2))][4][t]
                matrix[i].append(val)
        ary = np.array(matrix)
        plot_matrix_thresholds(region, plot_base, matrix, solar_gen_steps, wind_gen_steps, f'norm_pval_{threshold:03}')


if make_scan:

    year = 2018
    # Normalize demand for this year
    mean = np.mean(demand.loc[ demand[im['demand'][3]] == year, im['demand'][2] ])
    demand[ im['demand'][2] ] = demand[ im['demand'][2] ] / mean
    print(f"Mean normalized demand for year {year} = {mean}")
    
    matrix = []
    
    zero_negative=True
    #zero_negative=False
    for i, solar_install_cap in enumerate(solar_gen_steps):

        matrix.append([])
        for j, wind_install_cap in enumerate(wind_gen_steps):
            print(solar_install_cap, wind_install_cap)
            val = get_renewable_fraction(year, wind_install_cap, solar_install_cap, im, demand, wind, solar, zero_negative)
            #print(f" - Wind:Solar:RenewableFraction {wind_install_cap}:\t{solar_install_cap}:\t{round(val,3)}")
            matrix[i].append(val*100)
    
    ary = np.array(matrix)

    cf_wind = get_annual_CF(wind, 'wind', im, year)
    cf_solar = get_annual_CF(solar, 'solar', im, year)
    #plot_matrix(plot_base, matrix, ,solar_gen_steps, wind_gen_steps cf_wind, cf_solar, 'testX')
    plot_matrix(plot_base, matrix, solar_gen_steps, wind_gen_steps, 'testX')





