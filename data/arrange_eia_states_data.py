#!/usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from state_abbreviations import get_state_abbrev_map


MMBtu_per_barrel_2017 = 5.053 # MMBtu/barrel in 2017 per EIA, see README
Gallons_per_barrel = 42 # U.S. gallons / barrel per EIA, see README
kWh_to_GGE = 33.4

def load_df(f_name):

    df = pd.read_excel(f_name, sheet_name='Data 1', header=2)
    return df


# See https://www.eia.gov/tools/faqs/faq.php?id=26&t=10
# for instructions on how to access the correct values
# Original values in $/MMBtu and we want $/gallon
def get_gas_vals_2017():

    df = pd.read_csv('eia_SEDS_1970-2017.csv')

    # Select cells of interest
    df = df[ df['MSN'] == 'MGACD']
    df = df[['State', '2017']]

    df['gas mean (USD/gallon)'] = df['2017'] * MMBtu_per_barrel_2017 / Gallons_per_barrel
    print(len(df.index))
    print(df.head())

    return df

def get_gas_vals_2018():

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



def get_elec_vals_2017(df_gas):

    df = pd.read_excel('eia_elec_industry_by_state_table5_c_2017.xlsx', sheet_name='Table 5C', header=2)
    print(df.head())

    # State to abbrev map
    abbrev = get_state_abbrev_map()
    print(len(abbrev.keys()))
    print(len(df_gas.index))

    full_states = []
    elec_prices = []

    for idx1 in df_gas.index:
        
        state = 'XXX'
        for k, v in abbrev.items():
            if v == df_gas.loc[idx1, 'State']:
                state = k
                continue
        for idx2 in df.index:
            elec_state = df.loc[idx2, 'State']
            if elec_state != state:
                continue
            full_states.append(state)
            elec_prices.append(df.loc[idx2, 'Average Price (cents/kWh)']/100.) # To USD/kWh

    df_gas['State'] = full_states
    df_gas['elec mean (USD/kWh)'] = elec_prices
    df_gas = df_gas[['State', 'gas mean (USD/gallon)', 'elec mean (USD/kWh)']]
    df_gas['gas/elec (post kWh conv)'] = df_gas['gas mean (USD/gallon)'] / kWh_to_GGE / df_gas['elec mean (USD/kWh)']
    
    df_gas = df_gas.reset_index(drop=True)

    return df_gas


# Use EIA data to calculate CO2 emissions/kWh for each state
# Only available through 2017
def add_CO2_intensity(df, year):

    assert(year <= 2017)
    df_co2 = pd.read_excel('eia_electricity_CO2_by_state_1980-2017.xlsx', sheet_name='Sheet1', header=2)
    df_co2['MMT CO2'] = df_co2[year]
    df_co2 = df_co2[['State', 'MMT CO2']]

    df_elec_tot = pd.read_excel(f'eia_elec_industry_by_state_table2_2017.xlsx', sheet_name='Table 2', header=2)
    df_elec_tot['Total MWh'] = df_elec_tot['Total']
    df_elec_tot = df_elec_tot[['State', 'Total MWh']]
    print(df_elec_tot.head())

    co2 = []
    tot_MWh = []

    for idx in df.index:
        state = df.loc[idx, 'State']

        found_co2 = False
        for idx1 in df_co2.index:
            if df_co2.loc[idx1, 'State'] == state or (state == 'U.S. Total' and df_co2.loc[idx1, 'State'] == 'State total (unadjusted)'):
                found_co2 = True
                co2.append(df_co2.loc[idx1, 'MMT CO2'])
                break
        assert(found_co2)

        found_tot_MWh = False
        for idx2 in df_elec_tot.index:
            if df_elec_tot.loc[idx2, 'State'] == state:
                found_tot_MWh = True
                tot_MWh.append(df_elec_tot.loc[idx2, 'Total MWh'])
                print(state, tot_MWh[-1])
                break
        assert(found_tot_MWh)

    df['CO2 Intensity (metric tons/kWh)'] = np.array( co2 ) * 1.0e6 / (np.array( tot_MWh ) * 1.0e3) # CO2 to tons, MWh to kWh = tons/kWh
    df['MMT CO2'] = co2
    df['Total MWh'] = tot_MWh

    df.to_csv(f'us_gas_and_elec_{year}.csv')

    return df

def get_elec_vals_2018(gas_vals):

    df = pd.read_excel('eia_elec_industry_by_state_table5_c_2018.xlsx', sheet_name='Table 5C', header=2)
    print(df.head())

    for idx in df.index:
        if df.loc[idx, 'State'] in gas_vals.keys():
            gas_vals[df.loc[idx, 'State']].append( df.loc[idx, 'Average Price (cents/kWh)']/100. ) # To USD/kWh
        elif df.loc[idx, 'State'] == 'U.S. Total':
            gas_vals['U.S.'].append( df.loc[idx, 'Average Price (cents/kWh)']/100. ) # To USD/kWh

    return gas_vals




def to_df(prices, year):
    df = pd.DataFrame({'State': [], 'gas min (USD/gallon)': [], 'gas max (USD/gallon)': [], 'gas mean (USD/gallon)': [], 'elec mean (USD/kWh)': []})
    for state, vals in prices.items():
        df = df.append( pd.DataFrame({'State': [state,], 'gas min (USD/gallon)': [vals[0],], 'gas max (USD/gallon)': [vals[1],], 'gas mean (USD/gallon)': [vals[2], ], 'elec mean (USD/kWh)': [vals[3], ]}), ignore_index=True )

    df.to_csv(f'us_gas_and_elec_{year}.csv')

    return df

df_gas = get_gas_vals_2017()
df = get_elec_vals_2017(df_gas)
df = add_CO2_intensity(df, 2017)

gas_vals = get_gas_vals_2018()
prices = get_elec_vals_2018(gas_vals)
df_prices = to_df(prices, 2018)



