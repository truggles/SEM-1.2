#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

import analytic_fuels as af


# Parameters for fuel economy plots
eff_ratio=1.5
#eff_ratio=2
FCEV_mpge = 60
ICE_mpg = FCEV_mpge/eff_ratio
dispensing_dollar_per_kg = 1.9

# Comparison configurations
carbon_price_list = [0,200]# 50, 100, 200] # $/ton
fcev_mpges = [60, 60,] # MPGe for FVEC vs. ICE fuel economy comparison
ice_mpgs = [30, 40,] # ICE / HICE MPG
#fcev_mpges = [60,] # MPGe for FVEC vs. ICE fuel economy comparison
#ice_mpgs = [30,] # ICE / HICE MPG

def marker_list():
    return ['o', 'v', '^', 's', 'P', '*', 'H', 'X', 'd', '<', '>']



def plot_fuel_econ(df, var, axis_label, **kwargs):

    print(var, axis_label)
    print(kwargs)


    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal())
    
    lat_1 = -125
    lat_2 = -66.5
    lon_1 = 20
    lon_2 = 50
    ax.set_extent([lat_1, lat_2, lon_1, lon_2], ccrs.Geodetic())
    
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='110m',
                                         category='cultural', name=shapename)
    
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)


    # https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib
    cmap = mpl.cm.get_cmap('viridis', 10)

    if 'break-even' in kwargs and kwargs['break-even']:
        # Get limits of "good" values by looping through once
        # this is used to define the "Normalization" used for color mapping
        good_vals = []
        for astate in shpreader.Reader(states_shp).records():

            geo_state = astate.attributes['name']
            
            # Find matching entry in df
            for idx in df.index:

                prices_state = df.loc[idx, 'State']
            
                if geo_state != prices_state:
                    continue
                # Skip cases where CO2 intensity of elec is high enough that the
                # costs diverge.
                denom = df.loc[idx, var.replace('break-even', 'break-even denom')]
                if denom >= 0:
                    continue
                good_vals.append( df.loc[idx, var] )
    else:
        good_vals = df[var]


    if 'break-even' in kwargs and kwargs['break-even']:
        norm = mpl.colors.Normalize(vmin=np.min(good_vals), vmax=1000 )
    elif 'set_range' in kwargs:
        norm = mpl.colors.Normalize(vmin=kwargs['set_range'][0], vmax=kwargs['set_range'][1] )
    else:
        norm = mpl.colors.Normalize(vmin=np.min(good_vals), vmax=np.max(good_vals) )
    
    cmap.set_under('white')
    cmap.set_bad('white')
    
    for astate in shpreader.Reader(states_shp).records():

        edgecolor = 'black'
        geo_state = astate.attributes['name']
        facecolor = "white"
        
        # Find matching entry in df
        for idx in df.index:

            prices_state = df.loc[idx, 'State']
        
            if geo_state != prices_state:
                continue
        
            if 'break-even' in kwargs and kwargs['break-even']:
                # Skip cases where CO2 intensity of elec is high enough that the
                # costs diverge.
                denom = df.loc[idx, var.replace('break-even', 'break-even denom')]
                if denom >= 0:
                    continue

            val = df.loc[idx, var]
            facecolor = cmap(norm(val))
            #if prices_state == 'Washington':
            print(f"{astate.attributes['name']:>30} {round(val,4)} {var}")
        

        
        # `astate.geometry` is the polygon to plot
        ax.add_geometries([astate.geometry], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor)

    if 'text' in kwargs:
        plt.text( np.mean([lat_1, lat_2]), lon_2*1.025, kwargs['text'],
            horizontalalignment='center',
            transform=ccrs.Geodetic())

    cax = fig.add_axes([.2, .12, .6, 0.04]) # Start X, start Y, X width, Y width
    if 'break-even' in kwargs and kwargs['break-even']:
        extend = 'max'
    else:
        extend = 'neither'
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    extend=extend,
                                    orientation='horizontal')
    cb.set_label(axis_label)
    fig.subplots_adjust(top=1, left=.05, right=.95)
    
    save_str = f'geo_map_states_'
    if 'save' in kwargs:
        save_str += kwargs['save']
    plt.savefig(f'{save_str}.pdf')


def plot_prices(year):


    df_prices = pd.read_csv('../data/'+f'us_gas_and_elec_{year}.csv')

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
        
            frac = df_prices.loc[idx, 'gas mean (USD/gallon)'] / af.kWh_to_GGE
            frac /= df_prices.loc[idx, 'elec mean (USD/kWh)']
            #print(f"{astate.attributes['name']:>30} {round(frac,4)}")
        
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
    
    plt.savefig(f'geo_map_states_gas_over_elec_{year}_old.pdf')


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




def add_carbon_prices(df, carbon_price_list, ICE_mpg=40, FCEV_mpge=60):

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
            normal_gas.append( df.loc[idx, 'gas mean (USD/gallon)'] + af.co2_per_gallon_gas * carbon_price )

        df[f'co2 {carbon_price}: gas price'] = np.array(normal_gas)
        df[f'co2 {carbon_price}: synth gas price'] = np.array(synth_gas) * af.kWh_to_GGE
        df[f'co2 {carbon_price}: h2 price'] = np.array(h2) * af.kWh_LHV_per_kg_H2
        df[f'co2 {carbon_price}: gas USD/mi {ICE_mpg}'] = df[f'co2 {carbon_price}: gas price']/ICE_mpg
        df[f'co2 {carbon_price}: h2 USD/mi {FCEV_mpge}'] = (df[f'co2 {carbon_price}: h2 price'] + dispensing_dollar_per_kg)/af.kWh_LHV_per_kg_H2*af.kWh_to_GGE/FCEV_mpge

    df.to_csv('tmp.csv')
    return df


def add_break_even_prices(df, fcev_mpges, ice_mpgs):

    for fcev, ice in zip(fcev_mpges, ice_mpgs):
        print(f"Adding FEVC vs. H/ICE comparison for FCEV {fcev} MPGe vs. H/ICE {ice} MPG")
        break_even = []
        denom = []
        for idx in df.index:
            syst = af.return_fuel_system()
            info = af.calc_carbon_price_break_even(syst, dispensing_dollar_per_kg,
                df.loc[idx, 'elec mean (USD/kWh)'], df.loc[idx, 'CO2 Intensity (metric tons/kWh)'], 
                df.loc[idx, 'gas mean (USD/gallon)'], fcev, ice)
            break_even.append( info[0] )
            denom.append( info[1] )

        df[f'break-even {ice} {fcev}'] = break_even
        df[f'break-even denom {ice} {fcev}'] = denom

    df.to_csv('tmp.csv')
    return df



def carbon_price_plots(df_states, df_synth, year, carbon_prices, FCEV_mpge, ICE_mpg, dispensing_dollar_per_kg):
    assert(year == 2017), "CO2 intensity is only available for years <= 2017 for EIA data."

    plt.close()
    fig, ax = plt.subplots(figsize=(15,15))
    #ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['gasoline synth (USD/GGE)']/ICE_mpg, 'C1-', label='LH electrofuel cost')
    #ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['h2 synth (USD/kg)']/af.kWh_LHV_per_kg_H2*af.kWh_to_GGE/FCEV_mpge, 'C2--', label=r'electrolysis to H$_{2}$ cost')
    ax.plot(df_synth['Elec Price (USD/kWh)'],  (df_synth['h2 synth (USD/kg)'] + dispensing_dollar_per_kg)/af.kWh_LHV_per_kg_H2*af.kWh_to_GGE/FCEV_mpge, 'k-.', label=r'H$_{2}$ prod. + disp. cost')
    ncol_=2
    #ax.scatter(df_states['elec mean (USD/kWh)'], df_states['gas mean (USD/gallon)']/ICE_mpg, label='U.S. states price', marker='D', color='k', alpha=0.3)

    for carbon_price in carbon_prices:
        rslt = ax.scatter(df_states['elec mean (USD/kWh)'], df_states[f'co2 {carbon_price}: gas price']/ICE_mpg, label=r'CO$_{2}$ '+str(carbon_price)+': gas', marker='o', alpha=0.8)
        ax.scatter(df_states['elec mean (USD/kWh)'], (df_states[f'co2 {carbon_price}: h2 price'] + dispensing_dollar_per_kg)/af.kWh_LHV_per_kg_H2*af.kWh_to_GGE/FCEV_mpge, 
                label=r'CO$_{2}$ '+str(carbon_price)+': H$_{2}$', marker='x', alpha=0.8,
                color=rslt.get_facecolor()[0])

    ax.set_xlabel('electricity price ($/kWh)')
    ax.set_ylabel('fuel economy ($/mile)')
    ax.set_xlim(0, 0.3)
    #ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_ylim(0, 0.2)
    plt.legend(loc='upper left', ncol=ncol_)

    # Text for MPG and MPGe values
    plt.text(0.2, 0.05, f'Hybrid ICEV: {ICE_mpg} MPG\nFCEV: {FCEV_mpge} MPGe\nEff. Ratio:{round(FCEV_mpge/ICE_mpg,2)}', fontsize=20)

    plt.tight_layout()
    plt.savefig(f'analysis_gas_states_fuel_econ_{year}_co2_price_eff_ratio_{round(FCEV_mpge/ICE_mpg,2)}.pdf')



df = pd.read_csv('../data/'+'Global_elec_and_gas_prices.csv', header=3)
df = df.sort_values('Elec Price (USD/kWh)')

#check_stats(df, 'Elec Price (USD/kWh)', 'Gasoline Price (USD/l)')
df['gasoline normal (USD/gallon)'] = df['Gasoline Price (USD/l)'] * af.liters_to_gallons
#print(df.head())



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
    'h2 synth (USD/kg)' : np.array(h2) * af.kWh_LHV_per_kg_H2,
    'gasoline synth (USD/GGE)' : np.array(gas) * af.kWh_to_GGE,
})



# Load U.S. State's info
for year in [2017,]:# 2018,]: # 2018 does not have carbon intensity info
    df2 = pd.read_csv('../data/'+f'us_gas_and_elec_{year}.csv')

    # Add elec and gas adjustments for carbon prices
    df2 = add_carbon_prices(df2, carbon_price_list, ICE_mpg, FCEV_mpge)
    df2 = add_break_even_prices(df2, fcev_mpges, ice_mpgs)

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
    ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/ICE_mpg, label='countries price', alpha=0.3)
    #ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['gasoline synth (USD/GGE)']/ICE_mpg, 'C1-', label='LH electrofuel cost')
    ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['h2 synth (USD/kg)']/af.kWh_LHV_per_kg_H2*af.kWh_to_GGE/FCEV_mpge, 'C2--', label=r'electrolysis to H$_{2}$ cost')
    ax.plot(df_synth['Elec Price (USD/kWh)'],  (df_synth['h2 synth (USD/kg)'] + dispensing_dollar_per_kg)/af.kWh_LHV_per_kg_H2*af.kWh_to_GGE/FCEV_mpge, 'C2-.', label='electrolysis to H'+r'$_{2}$ cost'+'\nincluding dispensing')
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
    plt.text(0.05, 0.2, f'Hybrid ICEV: {ICE_mpg} MPG\nFCEV: {FCEV_mpge} MPGe\nEff. Ratio:{round(FCEV_mpge/ICE_mpg,2)}')
    plt.legend(loc='upper left', ncol=ncol_)
    plt.savefig(f'analysis_gas_states_fuel_econ_{year}.pdf')
    
    plt.close()
    fig, ax = plt.subplots()
    ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/af.kWh_to_GGE, label='countries price', alpha=0.3)
    ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['gasoline synth (USD/GGE)']/af.kWh_to_GGE, 'C1-', label='LH electrofuel cost')
    ax.plot(df_synth['Elec Price (USD/kWh)'],  df_synth['h2 synth (USD/kg)']/af.kWh_LHV_per_kg_H2, 'C2-', label=r'electrolysis to H$_{2}$ cost')
    if year == 2017:
        ax.scatter(df2['elec mean (USD/kWh)'], df2['gas mean (USD/gallon)']/af.kWh_to_GGE, label='U.S. states price', marker='D', color='C3', alpha=0.3)
    if year == 2018:
        for i, idx in enumerate(df2.index):
            if df2.loc[idx, 'State'] == 'U.S.':
                continue
            ax.scatter(df2.loc[idx, 'elec mean (USD/kWh)'], df2.loc[idx, 'gas mean (USD/gallon)']/af.kWh_to_GGE, label=df2.loc[idx, 'State']+' price', marker=markers[i])
    set_ax(ax, max_, r'fuel price (\$/kWh$_{LHV}$)')
    plt.legend(loc='upper left', ncol=ncol_)
    plt.savefig(f'analysis_fuels_states_{year}.pdf')
    
    plot_prices(year)
    
    #check_stats(df2, 'elec mean (USD/kWh)', 'gas mean (USD/gallon)')

    for ICE_mpg in ice_mpgs:
        carbon_price_plots(df2, df_synth, year, carbon_price_list, FCEV_mpge, ICE_mpg, dispensing_dollar_per_kg)

    # Break-even price=cost plots
    kwargs = {
        'break-even' : True,
    }
    for fcev_mpge, ice_mpg in zip(fcev_mpges, ice_mpgs):
        kwargs['save'] = f'break_even_ice{ice_mpg}_fcev{fcev_mpge}'
        kwargs['text'] = f'Hybrid ICEV: {ice_mpg} MPG\nFCEV: {fcev_mpge} MPGe\nEff. Ratio:{round(fcev_mpge/ice_mpg,2)}'
        var = f'break-even {ice_mpg} {fcev_mpge}'
        axis_label = r'break-even carbon price (\$/ton CO$_{2}$)'
        plot_fuel_econ(df2, var, axis_label, **kwargs) 
    
    # Gas price
    kwargs = {
        'save' : f'gas_price_{year}',
    }
    var = 'gas mean (USD/gallon)'
    axis_label = 'gasoline price ($/gallon)'
    plot_fuel_econ(df2, var, axis_label, **kwargs)

    # Elec price
    kwargs = {
        'save' : f'elec_price_{year}',
    }
    var = 'elec mean (USD/kWh)'
    axis_label = 'electricity price ($/kWh)'
    plot_fuel_econ(df2, var, axis_label, **kwargs)

    # Gas / Elec ratio
    kwargs = {
        'save' : f'gas_over_elec_{year}',
    }
    var = 'gas/elec (post kWh conv)'
    axis_label = r'gasoline price (\$/kWh$_{LHV}$) / electricity price (\$/kWh)'
    plot_fuel_econ(df2, var, axis_label, **kwargs)

    # Elec CO2 intensity
    kwargs = {
        'save' : f'elec_CO2_intensity_{year}',
    }
    var = 'CO2 Intensity (metric tons/kWh)'
    axis_label = r'electricity CO$_{2}$ intensity (tons/kWh)'
    plot_fuel_econ(df2, var, axis_label, **kwargs)







