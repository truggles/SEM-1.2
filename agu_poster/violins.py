#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.lines as mlines


series_map = {
        'dem' : ['Demand', 'black'],
        'wind' : ['Wind', 'blue'],
        'solar' : ['Solar', 'orange']
}


collection = {}

for i in range(1, 5):

    collection[i] = {}

    for series, info in series_map.items():
        df = pd.read_csv(f'{series}_{i}.csv', index_col=False)
        
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

    plt.close()
    fig, ax = plt.subplots()
    for series, vals in collection[i].items():
        ax.plot(vals[series_map[series][0]].loc[0:100], label=series, 
                color=series_map[series][1])
    plt.legend()
    plt.savefig(f'plots/collection_{i}.png')


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


def array_ppf_idx(ary, pp):

    tot = np.sum(ary)
    run = 0.
    for i, v in enumerate(ary):
        run += v
        if run / tot >= pp:
            print(i, v, run, tot, run/tot)
            return i
    return -1

def integrated_threshold(data, pp):

    plt.close()
    top = 1.7
    bottom = 0.0
    intervals = 2000
    spacing = (top - bottom)/intervals

    test = data[0][0]

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
    fig, ax = plt.subplots()
    ax.plot(np.linspace(bottom, top, intervals), tracker)

    # Get bin for percent point
    idx = array_ppf_idx(tracker, pp)
    l = mlines.Line2D([bottom+idx*spacing,bottom+idx*spacing], [0,np.max(tracker)], color='red')
    ax.add_line(l)

    plt.savefig('plots/hist.png')
    plt.gcf()
    ax.set_yscale('log')
    ax.set_ylim(0.5, ax.get_ylim()[1])
    plt.savefig('plots/hist_log.png')



# wind, solar, idx
study_regions = OrderedDict()
study_regions['Optimized'] = [1.0, 0.75, 0]
study_regions['Min Std'] = [1.0, 0.5, 0]
study_regions['Zero Renewables'] = [0., 0., 1]
#study_regions['All Solar'] = [0., 3.0, 2]
#study_regions['All Wind'] = [2.0, 0., 3]
#study_regions['Crazy'] = [4.5, 1.5, 4]

data_to_plot = [[], [], [], []]
data_to_plot2 = [[], [], [], []]
for name, region in study_regions.items():
    print(f'{name}')

    stds = []
    qXs = []

    for i, year_info in collection.items():


        ary = np.array(year_info['dem']['Demand'] - year_info['wind']['Wind']*region[0] - year_info['solar']['Solar']*region[1])
        std = np.std(ary)
        qX = np.percentile(ary, 99.9)
        stds.append(std)
        qXs.append(qX)

        print(f'  {i} --- std dev {std} qX {qX}')
        ary2 = []
        for val in ary:
            if val > 0.:
                ary2.append(val)
        ary2 = np.array(ary2)
        data_to_plot[i-1].append(ary)
        data_to_plot2[i-1].append(ary2)

    val = (max(qXs) - min(qXs))/np.mean(qXs)
    print(val)

xticks = study_regions.keys()
violin_plot(data_to_plot, xticks, 'all_years.png')
violin_plot(data_to_plot2, xticks, 'all_years_zero_min.png')

pp = 0.999
integrated_threshold(data_to_plot2, pp)
