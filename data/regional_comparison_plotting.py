#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

import analytic_fuels as af

kWh_to_GGE = 33.4
kWh_LHV_per_kg_H2 = 33.33
liters_to_gallons = 3.78541



def marker_list():
    return ['o', 'v', '^', 's', 'P', '*', 'H', 'X', 'd', '<', '>']



def plot_prices():


    df_prices = pd.read_csv('us_gas_and_elec.csv')

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
    
    plt.savefig('geo_map_states_gas_over_elec.pdf')



df = pd.read_csv('Global_elec_and_gas_prices.csv', header=3)
df = df.sort_values('Elec Price (USD/kWh)')


h2 = []
gas = []

# For each country, calculat their H2 and gasoline price
for idx in df.index:

    syst = af.return_fuel_system()
    h2.append( af.get_h2_system_costs(syst, df.loc[idx, 'Elec Price (USD/kWh)']) )
    gas.append( af.get_fuel_system_costs(syst, df.loc[idx, 'Elec Price (USD/kWh)']) )

df['h2 synth (USD/kg)'] = np.array(h2) * kWh_LHV_per_kg_H2
df['gasoline synth (USD/GGE)'] = np.array(gas) * kWh_to_GGE
df['gasoline normal (USD/gallon)'] = df['Gasoline Price (USD/l)'] * liters_to_gallons

print(df.head())

fig, ax = plt.subplots()
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)'], label='country averages')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)'], 'C1-', label='LH electrofuel')
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel('Gas Price (USD/gallon)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend()
plt.savefig('analysis_gas.pdf')

plt.close()
fig, ax = plt.subplots()
ax.plot(df['Elec Price (USD/kWh)'],  df['h2 synth (USD/kg)'], 'C2-', label=r'electrolysis to H$_{2}$')
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel('H2 Price (USD/kg)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend()
plt.savefig('analysis_h2.pdf')

plt.close()
fig, ax = plt.subplots()
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/kWh_to_GGE, label='country averages')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)']/kWh_to_GGE, 'C1-', label='LH electrofuel')
ax.plot(df['Elec Price (USD/kWh)'],  df['h2 synth (USD/kg)']/kWh_LHV_per_kg_H2, 'C2-', label=r'electrolysis to H$_{2}$')
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel(r'Fuel Price (USD/kWh$_{LHV}$)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend()
plt.savefig('analysis_fuels.pdf')

df.to_csv('analysis_fuels.csv')


# Load U.S. State's info
df2 = pd.read_csv('us_gas_and_elec.csv')
    
plt.close()
fig, ax = plt.subplots()
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)'], label='country averages')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)'], 'C1-', label='LH electrofuel')
markers = marker_list()
for i, idx in enumerate(df2.index):
    if df2.loc[idx, 'State'] == 'U.S.':
        continue
    ax.scatter(df2.loc[idx, 'elec mean (USD/kWh)'], df2.loc[idx, 'gas mean (USD/gallon)'], label=df2.loc[idx, 'State'], marker=markers[i])
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel('Gas Price (USD/gallon)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend(loc='upper left', ncol=2)
plt.savefig('analysis_gas_states.pdf')

plt.close()
fig, ax = plt.subplots()
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/kWh_to_GGE, label='country averages')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)']/kWh_to_GGE, 'C1-', label='LH electrofuel')
ax.plot(df['Elec Price (USD/kWh)'],  df['h2 synth (USD/kg)']/kWh_LHV_per_kg_H2, 'C2-', label=r'electrolysis to H$_{2}$')
for i, idx in enumerate(df2.index):
    if df2.loc[idx, 'State'] == 'U.S.':
        continue
    ax.scatter(df2.loc[idx, 'elec mean (USD/kWh)'], df2.loc[idx, 'gas mean (USD/gallon)']/kWh_to_GGE, label=df2.loc[idx, 'State'], marker=markers[i])
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel(r'Fuel Price (USD/kWh$_{LHV}$)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend(loc='upper left', ncol=2)
plt.savefig('analysis_fuels_states.pdf')

plot_prices()
