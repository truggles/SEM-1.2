#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats

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



def plot_prices(year):


    df_prices = pd.read_csv(f'us_gas_and_elec_{year}.csv')

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
            print(f"{astate.attributes['name']:>30} {round(frac,4)}")
        
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
    cb.set_label(r'gasoline price (\$/kWh$_{LHV}$) / electricity price (\$/kWh)')
    fig.subplots_adjust(top=1, left=.05, right=.95)
    
    plt.savefig(f'geo_map_states_gas_over_elec_{year}.pdf')


def check_stats(df, cor_col1, cor_col2):

    coeff = np.corrcoef(df[cor_col1], df[cor_col2])
    print(f'Pearson product-moment correlation coefficient between "{cor_col1}" and "{cor_col2}" = {coeff}\n')

    print(stats.linregress(df[cor_col1], df[cor_col2]), '\n')



def set_ax(ax, max_, y_label, x_label='electricity price ($/kWh)'):

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, max_)
    ax.set_ylim(0, ax.get_ylim()[1])
    return




def add_carbon_prices(df, carbon_price_list):

    for carbon_price in carbon_price_list:
        print(f"Adding carbon price of {carbon_price} $/ton")
        h2 = []
        synth_gas = []
        normal_gas = []
        for idx in df.index:
            syst = af.return_fuel_system()
            elec_price = df.loc[idx, 'elec mean (USD/kWh)']
            elec_adj = af.add_carbon_price(elec_price, df.loc[idx, 'CO2 Intensity (metric tons/kWh)'], carbon_price)
            h2.append( af.get_h2_system_costs(syst, elec_adj) )
            synth_gas.append( af.get_fuel_system_costs(syst, elec_adj) )
            # Wikipedia: https://en.wikipedia.org/wiki/Gasoline
            # About 19.64 pounds (8.91 kg) of carbon dioxide (CO2) are produced from burning 1 U.S. gallon (3.8 liters) of gasoline that does not contain ethanol (2.36 kg/L).
            normal_gas.append( df.loc[idx, 'gas mean (USD/gallon)'] + (8.91/1000.) * carbon_price )

        df[f'co2 {carbon_price}: gas price'] = np.array(normal_gas)
        df[f'co2 {carbon_price}: synth gas price'] = np.array(synth_gas) * kWh_to_GGE
        df[f'co2 {carbon_price}: h2 price'] = np.array(h2) * kWh_LHV_per_kg_H2

    df.to_csv('tmp.csv')
    return df






df = pd.read_csv('Global_elec_and_gas_prices.csv', header=3)
df = df.sort_values('Elec Price (USD/kWh)')

check_stats(df, 'Elec Price (USD/kWh)', 'Gasoline Price (USD/l)')
df['gasoline normal (USD/gallon)'] = df['Gasoline Price (USD/l)'] * liters_to_gallons
print(df.head())



# For each country, calculat their H2 and gasoline price
# in a new df so prices cover 0.00 $/kWh to 0.35 $/kWh
max_ = 0.35
elec = np.linspace(0.0, max_, 100)
h2 = []
gas = []
for val in elec:
    syst = af.return_fuel_system()
    h2.append( af.get_h2_system_costs(syst, val) )
    gas.append( af.get_fuel_system_costs(syst, val) )

df_synth = pd.DataFrame({
    'Elec Price (USD/kWh)': elec,
    'h2 synth (USD/kg)' : np.array(h2) * kWh_LHV_per_kg_H2,
    'gasoline synth (USD/GGE)' : np.array(gas) * kWh_to_GGE,
})



# Load U.S. State's info
for year in [2017,]:# 2018,]: # 2018 does not have carbon intensity info
    df2 = pd.read_csv(f'us_gas_and_elec_{year}.csv')

    # Add elec and gas adjustments for carbon prices
    carbon_price_list = [0, 50, 100, 200] # $/ton
    df2 = add_carbon_prices(df2, carbon_price_list)

    # Drop US avg
    if year == 2017:
        # Drop US avg
        print("Dropping US average values from plots")
        print(df2[ df2['State'] == 'U.S. Total' ])
        df2 = df2.drop( df2[ df2['State'] == 'U.S. Total'].index, axis=0 )
        
    plt.close()
    fig, ax = plt.subplots()
    ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)'], label='countries price', alpha=0.3)
    ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['gasoline synth (USD/GGE)'], 'C1-', label='LH electrofuel cost')
    markers = marker_list()
    ncol_=2
    if year == 2017:
        ax.scatter(df2['elec mean (USD/kWh)'], df2['gas mean (USD/gallon)'], label='U.S. states price', marker='D', color='C3', alpha=0.3)
    if year == 2018:
        for i, idx in enumerate(df2.index):
            if df2.loc[idx, 'State'] == 'U.S.':
                continue
            ax.scatter(df2.loc[idx, 'elec mean (USD/kWh)'], df2.loc[idx, 'gas mean (USD/gallon)'], label=df2.loc[idx, 'State']+' price', marker=markers[i])
    set_ax(ax, max_, 'gasoline price ($/gallon)')
    plt.legend(loc='upper left', ncol=ncol_)
    plt.savefig(f'analysis_gas_states_{year}.pdf')
        
    plt.close()
    fig, ax = plt.subplots()
    FCEV_mpgge = 60
    ICE_mpg = FCEV_mpgge/1.5
    dispensing_dollar_per_kg = 1.9
    ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/ICE_mpg, label='countries price', alpha=0.3)
    #ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['gasoline synth (USD/GGE)']/ICE_mpg, 'C1-', label='LH electrofuel cost')
    ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['h2 synth (USD/kg)']/kWh_LHV_per_kg_H2*kWh_to_GGE/FCEV_mpgge, 'C2--', label=r'electrolysis to H$_{2}$ cost')
    ax.plot(df_synth['Elec Price (USD/kWh)'],  (df_synth['h2 synth (USD/kg)'] + dispensing_dollar_per_kg)/kWh_LHV_per_kg_H2*kWh_to_GGE/FCEV_mpgge, 'C2-.', label='electrolysis to H'+r'$_{2}$ cost'+'\nincluding dispensing')
    markers = marker_list()
    ncol_=2
    if year == 2017:
        ax.scatter(df2['elec mean (USD/kWh)'], df2['gas mean (USD/gallon)']/ICE_mpg, label='U.S. states price', marker='D', color='C3', alpha=0.3)
    if year == 2018:
        for i, idx in enumerate(df2.index):
            if df2.loc[idx, 'State'] == 'U.S.':
                continue
            ax.scatter(df2.loc[idx, 'elec mean (USD/kWh)'], df2.loc[idx, 'gas mean (USD/gallon)']/ICE_mpg, label=df2.loc[idx, 'State']+' price', marker=markers[i])
    set_ax(ax, max_, 'fuel economy ($/mile)')
    plt.legend(loc='upper left', ncol=ncol_)
    plt.savefig(f'analysis_gas_states_fuel_econ_{year}.pdf')
    
    plt.close()
    fig, ax = plt.subplots()
    ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/kWh_to_GGE, label='countries price', alpha=0.3)
    ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['gasoline synth (USD/GGE)']/kWh_to_GGE, 'C1-', label='LH electrofuel cost')
    ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['h2 synth (USD/kg)']/kWh_LHV_per_kg_H2, 'C2-', label=r'electrolysis to H$_{2}$ cost')
    if year == 2017:
        ax.scatter(df2['elec mean (USD/kWh)'], df2['gas mean (USD/gallon)']/kWh_to_GGE, label='U.S. states price', marker='D', color='C3', alpha=0.3)
    if year == 2018:
        for i, idx in enumerate(df2.index):
            if df2.loc[idx, 'State'] == 'U.S.':
                continue
            ax.scatter(df2.loc[idx, 'elec mean (USD/kWh)'], df2.loc[idx, 'gas mean (USD/gallon)']/kWh_to_GGE, label=df2.loc[idx, 'State']+' price', marker=markers[i])
    set_ax(ax, max_, r'fuel price (\$/kWh$_{LHV}$)')
    plt.legend(loc='upper left', ncol=ncol_)
    plt.savefig(f'analysis_fuels_states_{year}.pdf')
    
    plot_prices(year)
    
    check_stats(df2, 'elec mean (USD/kWh)', 'gas mean (USD/gallon)')


