#!/usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl



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




def to_df(prices):
    df = pd.DataFrame({'State': [], 'gas min (USD/gallon)': [], 'gas max (USD/gallon)': [], 'gas mean (USD/gallon)': [], 'elec mean (USD/kWh)': []})
    for state, vals in prices.items():
        df = df.append( pd.DataFrame({'State': [state,], 'gas min (USD/gallon)': [vals[0],], 'gas max (USD/gallon)': [vals[1],], 'gas mean (USD/gallon)': [vals[2], ], 'elec mean (USD/kWh)': [vals[3], ]}), ignore_index=True )

    df.to_csv('us_gas_and_elec.csv')

    return df



gas_vals = get_gas_vals()
prices = get_elec_vals(gas_vals)
df_prices = to_df(prices)



