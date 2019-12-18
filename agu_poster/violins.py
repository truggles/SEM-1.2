#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.lines as mlines
import pickle



def load_inputs(n_years, TEXAS=False):
    series_map = {
            'dem' : ['Demand', 'black'],
            'wind' : ['Wind', 'blue'],
            'solar' : ['Solar', 'orange']
    }
    TX_map = { # column of interest, file path, rows to skip at start
            'Demand' : ['demand (MW)', '../Input_Data/TEXAS/TX_demand_unnormalized.csv', 6],
            'Wind' : ['solar capacity', '../Input_Data/TEXAS/solarCF_lei_TI_nonormalized.csv', 5],
            'Solar' : ['solar capacity', '../Input_Data/TEXAS/windCF_lei_TI_nonormalized.csv', 5],
            }
    
    collection = {}
    
    for i in range(1, n_years+1):
    
        collection[i] = {}
    
        for series, info in series_map.items():

            if not TEXAS:
                df = pd.read_csv(f'{series}_{i}.csv', index_col=False)
                df = df.drop(['Year', 'Month', 'Day', 'Hour'], axis=1)
            if TEXAS:
                df = pd.read_csv(TX_map[info[0]][1], index_col=False, skiprows=TX_map[info[0]][2])
                df[info[0]] = df[TX_map[info[0]][0]]
                df = df.loc[df['year'] == 2002 + i]
                print(i, series, len(df.index))
                #print(df.head())
                df = df.drop(['year', 'month', 'day', 'hour', TX_map[info[0]][0]], axis=1)

            
            ## Sanity checks
            #start = f"{df.loc[0,'Year']}"
            #start += f"-{df.loc[0,'Month']}"
            #start += f"-{df.loc[0,'Day']}"
            #start += f" {df.loc[0,'Hour']}"
            #end = f"{df.loc[len(df.index)-1,'Year']}"
            #end += f"-{df.loc[len(df.index)-1,'Month']}"
            #end += f"-{df.loc[len(df.index)-1,'Day']}"
            #end += f" {df.loc[len(df.index)-1,'Hour']}"
            #print(f'{series}_{i}.csv', len(df.index), start, end)
    
            # If demand, then normalize it, other are normalized
            if series == 'dem':
                name = info[0]
                df[name] = df[name]/np.mean(df[name])
    
            collection[i][series] = df
    
        #plt.close()
        #fig, ax = plt.subplots()
        #for series, vals in collection[i].items():
        #    ax.plot(vals[series_map[series][0]].loc[0:100], label=series, 
        #            color=series_map[series][1])
        #plt.legend()
        #plt.savefig(f'plots/collection_{i}.png')

    return collection

def violin_plot(data_to_plot, xticks, save_name):
    plt.close()
    fig, ax = plt.subplots()
    for i, data in enumerate(data_to_plot):
        parts = ax.violinplot(data, showmeans=True)

        for pc in parts['bodies']:
            #pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.2)

    plt.xticks(range(1, len(xticks)+1), xticks)
    plt.grid(axis='y')
    plt.savefig(f'plots/{save_name}')

    # If not cut at zero, make alternative plot with zero min
    # This preserves the PDf normalization for the whole distribution
    ax = plt.gca()
    if ax.get_ylim()[0] < 0:
        ax.set_ylim(0, ax.get_ylim()[1])
        fig = plt.gcf()
        plt.savefig(f'plots/alt_{save_name}')


def array_ppf_idx(ary, pp, spacing, nhours):

    tot_full = 1. * nhours # n Hours b/c of leap years
    tot_gtr_zero = np.sum(ary) * spacing
    run = tot_full - tot_gtr_zero   # Start accounting at zero and
                                    # some portion, sometimes is already zeroed
    #print(f'tot all {round(tot_full)}, tot_gtr_zero {round(tot_gtr_zero)}, frac > zero {round(tot_gtr_zero/tot_full,2)}')
    for i, v in enumerate(ary):
        run += v * spacing # To preserve integral
        if run / tot_full >= pp:
            #print(i, v, run, tot_full, tot_gtr_zero, run/tot_full)
            return i
    return -1

def array_ppfs(ary, spacing, nhours):

    tot_full = 1. * nhours # n Hours b/c of leap years
    tot_gtr_zero = np.sum(ary) * spacing
    run = tot_full - tot_gtr_zero   # Start accounting at zero and
                                    # some portion, sometimes is already zeroed
    pps = []
    for v in ary:
        run += v * spacing # To preserve integral
        pps.append(run / tot_full)
    return pps

def integrated_threshold(data, data_long, cnt, name, pp):

    plt.close()
    top = 1.7
    bottom = 0.0
    intervals = 2000
    spacing = (top - bottom)/intervals
    x_norm = np.linspace(bottom, top, intervals)

    # For normalizations
    hours = []
    for d in data_long: # Get original length of each year
        hours.append(len(d[cnt]))

    rev_cdfs = []
    pps = []
    for j, d in enumerate(data):
        test = d[cnt]

        test.sort()
        # Reverse the sorted array
        test2 = test[::-1]

        tracker = []
        prev = 999
        for val in np.linspace(top, bottom, intervals):
            to_add = 0
            for d in test2:
                if d >= val and d < prev:
                    to_add += 1
                if d < val:
                    break
            prev = val
            if len(tracker) == 0:
                prior = 0
            else:
                prior = tracker[-1]
            tracker.append(to_add + prior)
            #print(val, to_add, tracker[-1])
        #print(tracker)
        tracker.sort(reverse = True)
        rev_cdfs.append(tracker)
        pps.append(array_ppfs(tracker, spacing, hours[j]))


    ppf_idxs = []
    jjj = 0
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    for r_cdf, nhours in zip(rev_cdfs, hours):

        ax1.plot(x_norm, r_cdf, color=f'C{jjj}')

        # Get bin for percent point
        idx = array_ppf_idx(r_cdf, pp, spacing, nhours)
        ppf_idxs.append(idx)
        l = mlines.Line2D([bottom+idx*spacing,bottom+idx*spacing], [0,np.max(r_cdf)], color=f'C{jjj}')
        ax1.add_line(l)
        jjj += 1
    ax1.set_xlabel('dem - wind - solar (kW)')
    ax1.set_ylabel('# Hours >= X val')

    jjj = 0
    ax2 = fig.add_subplot(122)
    for r_cdf, pp in zip(rev_cdfs, pps):
        ax2.plot(pp, x_norm, color=f'C{jjj}')
        jjj += 1
    ax2.set_xlabel('$pp$')
    ax2.set_ylabel('dem - wind - solar (kW)')

    plt.savefig(f'plots/hist_{name.replace(" ","_")}.png')
    #plt.gcf()
    #ax.set_yscale('log')
    #ax.set_ylim(0.5, ax.get_ylim()[1])
    #plt.savefig(f'plots/hist_{name.replace(" ","_")}_log.png')

    # Check the threshold in Year A on Year B ...
    results = []
    for i, idx in enumerate(ppf_idxs):
        for j, pp in enumerate(pps):
            if i == j:
                continue
            rslt = apply_threshold_to_other(bottom, spacing, idx, pp)
            results.append(rslt)

    print(f"Results for {name}: len: {len(results)}, coeff of var: {round(np.std(results)/np.mean(results),4)}")
    return np.std(results)/np.mean(results)

# Apply a threshod from a given index to another
# year and return the unmet demand
def apply_threshold_to_other(bottom, spacing, idx, pp):

    return 1. - pp[idx]





#=================================================================================

n_years = 16
TEXAS = True
collection = load_inputs(n_years, TEXAS)

# wind, solar, idx
study_regions = OrderedDict()
#study_regions['Optimized'] = [1.0, 0.75, 0]
#study_regions['Min Std'] = [1.0, 0.5, 0]
#study_regions['Zero Renewables'] = [0., 0., 1]
#study_regions['All Solar'] = [0., 3.0, 2]
#study_regions['All Wind'] = [2.0, 0., 3]
#study_regions['Crazy'] = [4.5, 1.5, 4]

grid = [0, 5.1, 0.25]
#grid = [0, 2.1, 0.25]
initial_processing = True
#initial_processing = False

if initial_processing:
    for i in np.arange(grid[0], grid[1], grid[2]):
        for j in np.arange(grid[0], grid[1], grid[2]):
            study_regions[f'wind{i}_solar{j}'] = [i, j, 0]
    
    data_to_plot = []
    data_to_plot2 = []
    for yr in range(n_years):
        data_to_plot.append([])
        data_to_plot2.append([])
    
    for name, region in study_regions.items():
        print(f'{name}')
    
        for i, year_info in collection.items():
    
    
            ary = np.array(year_info['dem']['Demand'].values - year_info['wind']['Wind'].values*region[0] - year_info['solar']['Solar'].values*region[1])
            std = np.std(ary)
            qX = np.percentile(ary, 99.9)
    
            #print(f'  {i} --- std dev {std} qX {qX}')
            ary2 = []
            for val in ary:
                if val > 0.:
                    ary2.append(val)
            ary2 = np.array(ary2)
            data_to_plot[i-1].append(ary)
            data_to_plot2[i-1].append(ary2)
    
        #val = (max(qXs) - min(qXs))/np.mean(qXs)
        #print(val)
    
    #xticks = study_regions.keys()
    #violin_plot(data_to_plot, xticks, 'all_years.png')
    #violin_plot(data_to_plot2, xticks, 'all_years_zero_min.png')
    
    pp = 0.999
    cnt = 0
    for name in study_regions.keys():
        study_regions[name].append(integrated_threshold(
                data_to_plot2, data_to_plot, cnt, name, pp))
        cnt += 1
    
    pickle_file = open('tmp.pkl', 'wb')
    pickle.dump(study_regions, pickle_file)
    pickle_file.close()


pickle_in = open('tmp.pkl','rb')
study_regions = pickle.load(pickle_in)

for name, results in study_regions.items():
    print(name, results[-1])


matrix = []
for i, vi in enumerate(np.arange(grid[0], grid[1], grid[2])):
    matrix.append([])
    for j, vj in enumerate(np.arange(grid[0], grid[1], grid[2])):
        matrix[i].append(study_regions[f'wind{vj}_solar{vi}'][-1])

print(matrix)


fig, ax = plt.subplots()
im = ax.imshow(matrix,interpolation='none',origin='lower')
cbar = ax.figure.colorbar(im)
cbar.ax.set_ylabel(f"Unmet Demand ($sigma/$mu)")

axis_vals = np.arange(grid[0], grid[1], grid[2])
x_labs, y_labs = [], []
for v in axis_vals:
    if round(v,1)==v:
        x_labs.append(v)
    else:
        x_labs.append('')
for v in axis_vals:
    if int(v)==v:
        y_labs.append(v)
    else:
        y_labs.append('')
plt.xticks(range(len(axis_vals)), x_labs, rotation=90)
plt.yticks(range(len(axis_vals)), y_labs)
plt.xlabel("Wind Cap/Mean Demand")
plt.ylabel("Solar Cap/Mean Demand")

#plt.tight_layout()
plt.savefig('matrix.png')






