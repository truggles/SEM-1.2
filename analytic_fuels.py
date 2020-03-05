import numpy as np
import pandas as pd



#----------------------------------------------------------------------
# Constants & fixed discount rate
#----------------------------------------------------------------------

DISCOUNT_RATE = 0.07
HOURS_PER_YEAR = 8760 # Reference: # days per year
BtuPerkWh = 3412.14 # https://www.eia.gov/totalenergy/data/monthly/pdf/sec13_18.pdf
MWh_per_MMBtu = 0.293 # https://www.eia.gov/totalenergy/data/monthly/pdf/sec13_18.pdf
MMBtu_per_Gallon_Gasoline = 0.114 # Btu/GGE "Fuel Economy Impact Analysis of RFG". United States Environmental Protection Agency. August 14, 2007. Retrieved Aug 27, 2019.
lhvH2Conv = 33.33 # kWh LHV / kg H2
hhvH2Conv = 39.4 # kWh HHV / kg H2



def capital_recovery_factor(discount_rate, **dic):
    dic['capital recovery factor'] = discount_rate*(1.+discount_rate)**dic['assumed lifetime'] / ((1+discount_rate)**dic['assumed lifetime'] - 1)
    return dic

def fixed_cost_per_year(**dic):
    dic['fixed cost per year'] = dic['capital cost']*dic['capital recovery factor']
    if 'capital cost (kg)' in dic.keys():
        dic['fixed cost per year (kg)'] = dic['capital cost (kg)']*dic['capital recovery factor']

    # Add fixed O&M if applicable
    if 'fixed annual OandM cost' in dic.keys():
        dic['fixed cost per year (no OandM)'] = dic['fixed cost per year']
        dic['fixed cost per year (no OandM) (kg)'] = dic['fixed cost per year (kg)']
        dic['fixed cost per year'] += dic['fixed annual OandM cost']
        dic['fixed cost per year (kg)'] += dic['fixed annual OandM cost (kg)']
    return dic

# NOTE: No difference in time value within year
def fixed_cost_per_hr(**dic):
    if not hasattr(dic, 'fixed cost per year'):
        dic = fixed_cost_per_year(**dic)
    dic['fixed cost per hr'] = dic['fixed cost per year']/HOURS_PER_YEAR
    if 'capital cost (kg)' in dic.keys():
        dic['fixed cost per hr (kg)'] = dic['fixed cost per year (kg)']/HOURS_PER_YEAR
    dic['value'] = dic['fixed cost per hr']
    return dic



#----------------------------------------------------------------------
# Cost that can be varied.
# Default values are based on cited literature
#----------------------------------------------------------------------

# "Adding engineering, construction, legal expenses and contractor’s fees, the fixed capital investment (FCI)"
# is calculated as: FCI = SF * Total Purchase Cost
# scale factor, D.H. König et al. / Fuel 159 (2015) 289–297
CHEM_PLANT_CAP_SF = 4.6

# https://www.usinflationcalculator.com/ 11 Feb 2020
USD2005_to_USD2020 = 1.31
USD2008_to_USD2020 = 1.19
USD2010_to_USD2020 = 1.17
USD2015_to_USD2020 = 1.08
USD2016_to_USD2020 = 1.07

FIXED_COST_ELECTROLYZER = {
    'capital cost (kg)' : (118.0e6)*USD2010_to_USD2020/(50000/24), # 118 M$(2010) NREL H2A / 50,000 kg H2 per day; ($/kg H2 generation)
    'capital cost' : (118.0e6)*USD2010_to_USD2020/(50000/24)/lhvH2Conv, # ($/kWh LHV H2 generation)
    'assumed lifetime' : 10, # (yr)
    'capacity factor' : 1.00, # 100%
    #'value' : 1.4300E-02 # ($/h)/kW
}



FIXED_COST_COMPRESSOR = {
    'capital cost (kg)' : (2.07e6)*USD2016_to_USD2020/(58000/24), # 2.07 M$(2016) NREL H2A / 50,000 kg H2 per day; ($/kg H2 generation)
    'capital cost' : (2.07e6)*USD2016_to_USD2020/(58000/24)/lhvH2Conv, # ($/kWh LHV H2 generation)
    'assumed lifetime' : 15, # (yr)
    'capacity factor' : 1.00, # 100%
    #'value' : 1.4300E-02 # ($/h)/kW
}



FIXED_COST_H2_STORAGE = {
    'capital cost (kg)' : 7.43e6*USD2016_to_USD2020/1160000, # 7.43 M$(2016) / 1,160,000 kg usable volume H2, source NREL H2A; ($/kg H2 storage)
    'capital cost' : 7.43e6*USD2016_to_USD2020/1160000/lhvH2Conv, # ($/kWh LHV storage)
    'fixed annual OandM cost (kg)' : 582000*USD2005_to_USD2020/1160000, # 582,000 $(2005) fixed O&M for facility
    'fixed annual OandM cost' : 582000*USD2005_to_USD2020/1160000/lhvH2Conv, # 582,000 $(2005) fixed O&M for facility
    'assumed lifetime' : 30, # (yr)
    #'value' : 2.7205E-07, # $/kWh
}



FIXED_COST_CHEM_PLANT = {
    'capital cost' : ((202+32+32)*1e6*CHEM_PLANT_CAP_SF)/(690*1000)*USD2015_to_USD2020, # ($/kW generation or conversion)
    'capital cost (kg)' : ((202+32+32)*1e6*CHEM_PLANT_CAP_SF)/56300*USD2015_to_USD2020, # ($/kW generation or conversion)
            # ($202+32+32)*4.6/690MW of liquid fuel produced for FT, Hydrocracker, RWGS) = fixex costs = cap ex*multiplier, Table 3 chem plant, D.H. König et al. / Fuel 159 (2015) 289–297
    'assumed lifetime' : 30, # (yr)
            # D.H. König et al. / Fuel 159 (2015) 289–297, pg 293
    #'value' : 1.6303E-02 # ($/h)/kW
}






VAR_COST_CHEM_PLANT = {
    #'value' : 6.91E-02, # $/kWh = 18.62*(0.069+0.038+0.016+0.001)/(MMBtu_per_Gallon_Gasoline*MWh_per_MMBtu)  # Variable O&M cost ($/MWh)
                                    # order is: maintenance, taxes & incentives, utilities, clean water
    'value' : 18.62*(0.069+0.038+0.016+0.001)/(MMBtu_per_Gallon_Gasoline*MWh_per_MMBtu) * (1./1000)*USD2015_to_USD2020, # Variable O&M cost ($/kWh)
    'value( kg)' : 6.83*(0.069+0.038+0.016+0.001)*USD2015_to_USD2020, # Variable O&M cost ($/kg)
    'ref' : 'Fig 4b, cost break down of $/GGE, excluding electrolyzer and cap annual, D.H. König et al. / Fuel 159 (2015) 289–297'

}


# $/kWh liquid hydrocarbons
def var_cost_of_CO2(**dic):
    # value = CO2 cost ($/metric ton) * (236 tons hr^-1 / 690 MW of liquid hydrocarbons ) * (1 MW / 1000 kW)
    # 'table 2, CO2 tons/hr / MW liquid hydrocarbons, D.H. König et al. / Fuel 159 (2015) 289–297'
    dic['value'] = dic['co2 cost']*(236/690)*(1/1000) # $/kWh liquid hydrocarbons
    return dic


VAR_COST_CO2 = {
    'co2 cost' : 50, # $/metric ton CO2
    #'value' : 1.71E-02, # (for $50/ton) # $/kWh liquid hydrocarbons = CO2 cost ($/metric ton) * (236 tons hr^-1 / 690 MW of liquid hydrocarbons ) * (1 MW / 1000 kW)
    'ref' : 'table 2, CO2 tons/hr / MW liquid hydrocarbons, D.H. König et al. / Fuel 159 (2015) 289–297'
}


EFFICIENCY_ELECTROLYZER_COMP = {
    'value' : .607, # LHV; Calculated from NREL H2A electrolyzer full eff. and compressor; see eta_PtCompH2 in my paper
    'ref' : 'Calculated from NREL H2A electrolyzer full eff. and compressor; see eta_PtCompH2 in my paper'
}


EFFICIENCY_CHEM_CONVERSION = {
    'value' : 0.682,
    'ref' : 'table 2, eta_CCE accounts for losses when converting H2 and CO2 into liquid hydrocarbons, D.H. König et al. / Fuel 159 (2015) 289–297'
}


EFFICIENCY_CHEM_PLANT = {
    'value' : 0.659,
    'ref' : 'table 2, eta_plant = chem plant efficiency, D.H. König et al. / Fuel 159 (2015) 289–297'
}


DECAY_RATE_H2_STORAGE = {
    'value' : 1.14E-08, # fraction per hour (0.01% per year)    
    'ref' : 'Crotogino et al., 2010, p43'
}

def return_fuel_system():
    system = {
        'FIXED_COST_ELECTROLYZER' : FIXED_COST_ELECTROLYZER,
        'FIXED_COST_COMPRESSOR' : FIXED_COST_COMPRESSOR,
        'FIXED_COST_CHEM_PLANT' : FIXED_COST_CHEM_PLANT,
        'FIXED_COST_H2_STORAGE' : FIXED_COST_H2_STORAGE,
        'VAR_COST_CHEM_PLANT' : VAR_COST_CHEM_PLANT,
        'VAR_COST_CO2' : VAR_COST_CO2,
        'EFFICIENCY_ELECTROLYZER_COMP' : EFFICIENCY_ELECTROLYZER_COMP,
        'EFFICIENCY_CHEM_PLANT' : EFFICIENCY_CHEM_PLANT,
        'EFFICIENCY_CHEM_CONVERSION' : EFFICIENCY_CHEM_CONVERSION,
        'DECAY_RATE_H2_STORAGE' : DECAY_RATE_H2_STORAGE,
    }
    for vals in ['FIXED_COST_ELECTROLYZER', 'FIXED_COST_COMPRESSOR', 'FIXED_COST_CHEM_PLANT', 'FIXED_COST_H2_STORAGE']:
        system[vals] = capital_recovery_factor(DISCOUNT_RATE, **system[vals])
        system[vals] = fixed_cost_per_hr(**system[vals])
    system['VAR_COST_CO2'] = var_cost_of_CO2(**system['VAR_COST_CO2'])
    return system

def get_h2_system_costs(system, electricity_cost, verbose=False):
    if verbose:
        print(f"Electricity price: {electricity_cost}")
    tot = 0.
    tot += system['FIXED_COST_ELECTROLYZER']['value'] / system['FIXED_COST_ELECTROLYZER']['capacity factor']
    if verbose:
        print(f" FIXED_COST_ELECTROLYZER to add: {system['FIXED_COST_ELECTROLYZER']['value'] / system['FIXED_COST_ELECTROLYZER']['capacity factor']}, new total {tot}")
    tot += system['FIXED_COST_COMPRESSOR']['value'] / system['FIXED_COST_COMPRESSOR']['capacity factor']
    if verbose:
        print(f" FIXED_COST_COMPRESSOR   to add: {system['FIXED_COST_COMPRESSOR']['value'] / system['FIXED_COST_COMPRESSOR']['capacity factor']}, new total {tot}")
    tot += electricity_cost / system['EFFICIENCY_ELECTROLYZER_COMP']['value']
    if verbose:
        print(f" ELECTRICITY COSTS       to add: {electricity_cost / system['EFFICIENCY_ELECTROLYZER_COMP']['value']}, new total {tot}")
    tot += system['FIXED_COST_H2_STORAGE']['value'] * 30 * 24 # 30 days x 24 hours for 1 month of storage capacity
    if verbose:
        print(f" FIXED_COST_H2_STORAGE   to add: {system['FIXED_COST_H2_STORAGE']['value'] * 30 * 24}, new total {tot}")
    return tot

def get_fuel_system_costs(system, electricity_cost, verbose=False):
    if verbose:
        print(f"Electricity price: {electricity_cost}")
    tot = 0.
    tot += system['FIXED_COST_ELECTROLYZER']['value'] / (system['EFFICIENCY_CHEM_CONVERSION']['value'] * system['FIXED_COST_ELECTROLYZER']['capacity factor'])
    if verbose:
        print(f" FIXED_COST_ELECTROLYZER to add: {system['FIXED_COST_ELECTROLYZER']['value'] / (system['EFFICIENCY_CHEM_CONVERSION']['value'] * system['FIXED_COST_ELECTROLYZER']['capacity factor'])}, new total {tot}")
    tot += system['FIXED_COST_COMPRESSOR']['value'] / (system['EFFICIENCY_CHEM_CONVERSION']['value'] * system['FIXED_COST_COMPRESSOR']['capacity factor'])
    if verbose:
        print(f" FIXED_COST_COMPRESSOR   to add: {system['FIXED_COST_COMPRESSOR']['value'] / (system['EFFICIENCY_CHEM_CONVERSION']['value'] * system['FIXED_COST_COMPRESSOR']['capacity factor'])}, new total {tot}")
    tot += electricity_cost / (system['EFFICIENCY_ELECTROLYZER_COMP']['value'] * system['EFFICIENCY_CHEM_CONVERSION']['value'])
    if verbose:
        print(f" ELECTRICITY COSTS       to add: {electricity_cost / (system['EFFICIENCY_ELECTROLYZER_COMP']['value'] * system['EFFICIENCY_CHEM_CONVERSION']['value'])}, new total {tot}")
    tot += system['FIXED_COST_H2_STORAGE']['value'] * 30 * 24 / system['EFFICIENCY_CHEM_CONVERSION']['value'] # 30 days x 24 hours for 1 month of storage capacity
    if verbose:
        print(f" FIXED_COST_H2_STORAGE   to add: {system['FIXED_COST_H2_STORAGE']['value'] * 30 * 24 / system['EFFICIENCY_CHEM_CONVERSION']['value']}, new total {tot}")
    tot += system['FIXED_COST_CHEM_PLANT']['value']
    if verbose:
        print(f" FIXED_COST_CHEM_PLANT   to add: {system['FIXED_COST_CHEM_PLANT']['value']}, new total {tot}")
    tot += system['VAR_COST_CHEM_PLANT']['value'] # Values from Konig already incorporate chem plant eff.
    if verbose:
        print(f" VAR_COST_CHEM_PLANT     to add: {system['VAR_COST_CHEM_PLANT']['value']}, new total {tot}")
    tot += system['VAR_COST_CO2']['value'] # Does not depend on chem plant eff.
    if verbose:
        print(f" VAR_COST_CO2            to add: {system['VAR_COST_CO2']['value']}, new total {tot}")
    return tot
