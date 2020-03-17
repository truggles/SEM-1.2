#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import analytic_fuels as af

kWh_to_GGE = 33.4
kWh_LHV_per_kg_H2 = 33.33
liters_to_gallons = 3.78541

def marker_list():
    return ['o', 'v', '^', 's', 'P', '*', 'H', 'X', 'd', '<', '>']

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
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)'], label='elec vs. gas')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)'], 'C1-', label='elec vs. gas synth')
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel('Gas Price (USD/gallon)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend()
plt.savefig('test_gas.png')

plt.close()
fig, ax = plt.subplots()
ax.plot(df['Elec Price (USD/kWh)'],  df['h2 synth (USD/kg)'], 'C2-', label=r'elec vs. H$_{2}$')
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel('H2 Price (USD/kg)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend()
plt.savefig('test_h2.png')

plt.close()
fig, ax = plt.subplots()
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/kWh_to_GGE, label='elec vs. gas')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)']/kWh_to_GGE, 'C1-', label='elec vs. gas synth')
ax.plot(df['Elec Price (USD/kWh)'],  df['h2 synth (USD/kg)']/kWh_LHV_per_kg_H2, 'C2-', label=r'elec vs. H$_{2}$')
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel(r'Fuel Price (USD/kWh$_{LHV}$)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend()
plt.savefig('test_fuels.png')

df.to_csv('testing_out.csv')


# Load U.S. State's info
df2 = pd.read_csv('us_gas_and_elec.csv')
    
plt.close()
fig, ax = plt.subplots()
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)'], label='elec vs. gas')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)'], 'C1-', label='elec vs. gas synth')
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
plt.savefig('test_gas_states.png')

plt.close()
fig, ax = plt.subplots()
ax.scatter(df['Elec Price (USD/kWh)'],  df['gasoline normal (USD/gallon)']/kWh_to_GGE, label='elec vs. gas')
ax.plot(df['Elec Price (USD/kWh)'],  df['gasoline synth (USD/GGE)']/kWh_to_GGE, 'C1-', label='elec vs. gas synth')
ax.plot(df['Elec Price (USD/kWh)'],  df['h2 synth (USD/kg)']/kWh_LHV_per_kg_H2, 'C2-', label=r'elec vs. H$_{2}$')
for i, idx in enumerate(df2.index):
    if df2.loc[idx, 'State'] == 'U.S.':
        continue
    ax.scatter(df2.loc[idx, 'elec mean (USD/kWh)'], df2.loc[idx, 'gas mean (USD/gallon)']/kWh_to_GGE, label=df2.loc[idx, 'State'], marker=markers[i])
ax.set_xlabel('Elec Price (USD/kWh)')
ax.set_ylabel(r'Fuel Price (USD/kWh$_{LHV}$)')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
plt.legend(loc='upper left', ncol=2)
plt.savefig('test_fuels_states.png')

