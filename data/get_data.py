#!/usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

kWh_to_GGE = 33.4

def marker_list():
    return ['o', 'v', '^', 's', 'P', '*', 'H', 'X', 'd', '<', '>']

def load_df(f_name):

    df = pd.read_excel(f_name, sheet_name='Data 1', header=2)
    return df


states = {
    'U.S.' : 'PET_PRI_GND_DCUS_NUS_W.xls',
    'California' :  'PET_PRI_GND_DCUS_SCA_W.xls',
    'Colorado' : 'PET_PRI_GND_DCUS_SCO_W.xls',
    'Florida' : 'PET_PRI_GND_DCUS_SFL_W.xls',
    'Massachusetts' : 'PET_PRI_GND_DCUS_SMA_W.xls',
    'Minnesota' : 'PET_PRI_GND_DCUS_SMN_W.xls',
    'New York' : 'PET_PRI_GND_DCUS_SNY_W.xls',
    'Ohio' : 'PET_PRI_GND_DCUS_SOH_W.xls',
    'Texas' : 'PET_PRI_GND_DCUS_STX_W.xls',
    'Washington' : 'PET_PRI_GND_DCUS_SWA_W.xls',

}


def get_gas_vals():
    gas_str = 'Weekly XXX All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)'
    out = {}
    
    for state, f_name in states.items():
    
        df = load_df(f_name)
    
        vals = []
        for idx in df.index:
            if df.loc[idx, 'Date'].year == 2018: # Industrial elec data is for 2018
                vals.append(df.loc[idx, gas_str.replace('XXX', state)])
        out[state] = [np.min(vals), np.max(vals), np.mean(vals)]

    return out


def get_elec_vals(gas_vals):

    df = pd.read_excel('eia_elec_industry_by_state_table5_c.xlsx', sheet_name='Table 5C', header=2)
    print(df.head())

    for idx in df.index:
        if df.loc[idx, 'State'] in gas_vals.keys():
            gas_vals[df.loc[idx, 'State']].append( df.loc[idx, 'Average Price (cents/kWh)']/100. ) # To USD/kWh
        elif df.loc[idx, 'State'] == 'U.S. Total':
            gas_vals['U.S.'].append( df.loc[idx, 'Average Price (cents/kWh)']/100. ) # To USD/kWh

    return gas_vals


def plot_prices(df_prices):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal())
    
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='110m',
                                         category='cultural', name=shapename)
    
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    
    # https://matplotlib.org/3.1.0/tutorials/colors/colorbar_only.html
    cmap = mpl.colors.ListedColormap(['royalblue', 'cyan',
                                      'yellow', 'orange'])
    cmap.set_over('red')
    cmap.set_under('blue')
    
    bounds = [0.5, 0.75, 1.0, 1.25, 1.5]
    c_bounds = ['blue', 'royalblue', 'cyan', 'yellow', 'orange', 'red']
    
    for astate in shpreader.Reader(states_shp).records():

        edgecolor = 'black'
        geo_state = astate.attributes['name']
        facecolor = "white"
        
        # Find matching entry in df_prices
        for idx in df_prices.index:

            prices_state = df_prices.loc[idx, 'State']
        
            if geo_state != prices_state:
                continue
        
            frac = df_prices.loc[idx, 'gas mean (USD/gallon)'] / kWh_to_GGE
            frac /= df_prices.loc[idx, 'elec mean (USD/kWh)']
            print(astate.attributes['name'], frac)
        
            # simple scheme to assign color to each state
            while True:
                for i in range(len(bounds)):
                    if frac < bounds[i]:
                        facecolor = c_bounds[i]
                        break
                if frac >= bounds[-1]:
                    facecolor = c_bounds[-1]
                break

        
        # `astate.geometry` is the polygon to plot
        ax.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor)

    cax = fig.add_axes([.2, .12, .6, 0.04]) # Start X, start Y, X width, Y width
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    boundaries=[1e-8] + bounds + [100],
                                    extend='both',
                                    extendfrac='auto',
                                    ticks=bounds,
                                    spacing='uniform',
                                    #spacing='proportional',
                                    orientation='horizontal')
    cb.set_label(r'Gas Price (USD/kWh$_{LHV}$) / Electric Price (USD/kWh)')
    fig.subplots_adjust(top=1, left=.05, right=.95)
    
    plt.savefig('geo_map_states_gas_over_elec.png')


def simple_plot(prices):
    plt.close()
    fig, ax = plt.subplots()
    markers = marker_list()
    i = 0
    for state, vals in prices.items():
        ax.scatter(vals[3], vals[2], marker=markers[i], label=state)
        i += 1
    ax.set_xlabel('Elec Price (USD/kWh)')
    ax.set_ylabel(r'Fuel Price (USD/gallon)')
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig('test_fuels_states.png')

def to_df(prices):
    df = pd.DataFrame({'State': [], 'gas min (USD/gallon)': [], 'gas max (USD/gallon)': [], 'gas mean (USD/gallon)': [], 'elec mean (USD/kWh)': []})
    for state, vals in prices.items():
        df = df.append( pd.DataFrame({'State': [state,], 'gas min (USD/gallon)': [vals[0],], 'gas max (USD/gallon)': [vals[1],], 'gas mean (USD/gallon)': [vals[2], ], 'elec mean (USD/kWh)': [vals[3], ]}), ignore_index=True )

    df.to_csv('us_gas_and_elec.csv')

    return df



gas_vals = get_gas_vals()
prices = get_elec_vals(gas_vals)

df_prices = to_df(prices)

#simple_plot(prices)

plot_prices(df_prices)


