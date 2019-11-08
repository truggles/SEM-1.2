import numpy as np
import pandas as pd


DISCOUNT_RATE = 0.07
HOURS_PER_YEAR = 8765.82 # Reference: # days per year
BtuPerkWh = 3412.14 # https://www.eia.gov/totalenergy/data/monthly/pdf/sec13_18.pdf
MWh_per_MMBtu = 0.293 # https://www.eia.gov/totalenergy/data/monthly/pdf/sec13_18.pdf
MMBtu_per_Gallon_Gasoline = 0.114 # Btu/GGE "Fuel Economy Impact Analysis of RFG". United States Environmental Protection Agency. August 14, 2007. Retrieved Aug 27, 2019.


def capital_recovery_factor(discount_rate, **dic):
    dic['capital recovery factor'] = discount_rate*(1.+discount_rate)**dic['assumed lifetime'] / ((1+discount_rate)**dic['assumed lifetime'] - 1)
    return dic

def fixed_cost_per_year(**dic):
    dic['fixed cost per year'] = dic['capital cost']*dic['capital recovery factor']
    return dic

# NOTE: No difference in time value within year
def fixed_cost_per_hr(**dic):
    if not hasattr(dic, 'fixed cost per year'):
        dic = fixed_cost_per_year(**dic)
    dic['fixed cost per hr'] = dic['fixed cost per year']/HOURS_PER_YEAR
    dic['value'] = dic['fixed cost per hr']
    return dic

ELECTROLYZER_CAP_SF = 1.83
CHEM_PLANT_CAP_SF = 4.6

FIXED_COST_ELECTROLYZER = {
    'capital cost' : 850*ELECTROLYZER_CAP_SF, # ($/kW generation or conversion; $/kWh storage)
            # $850/kW*1.83 from table 3, cap ex for H2 electrolyzer * scale factor, D.H. König et al. / Fuel 159 (2015) 289–297
    'assumed lifetime' : 30, # (yr)
            # D.H. König et al. / Fuel 159 (2015) 289–297, pg 293
    #'value' : 1.4300E-02 # ($/h)/kW
}



FIXED_COST_CHEM_PLANT = {
    'capital cost' : ((202+32+32)*CHEM_PLANT_CAP_SF)/690*1000, # ($/kW generation or conversion; $/kWh storage)
            # ($202+32+32)*4.6/690MW of liquid fuel produced for FT, Hydrocracker, RWGS, Cavern) = fixex costs = cap ex*multiplier, Table 3 chem plant, D.H. König et al. / Fuel 159 (2015) 289–297
    'assumed lifetime' : 30, # (yr)
            # D.H. König et al. / Fuel 159 (2015) 289–297, pg 293
        #'value' : 1.6303E-02 # ($/h)/kW
}


FIXED_COST_H2_STORAGE = {
    'capital cost' : ((33*10**6))/(973*10**9)*1000, # ($/kW generation or conversion; $/kWh storage)
            # (33)*4.6/690MW of liquid fuel produced for FT, Hydrocracker, RWGS, Cavern) = fixex costs = cap ex*multiplier, Table 3 chem plant, D.H. König et al. / Fuel 159 (2015) 289–297
    'assumed lifetime' : 80, # (yr)
            # D.H. König et al. / Fuel 159 (2015) 289–297, pg 293
    #'Capital recovery factor' : 0.0703, # (% per year)
    #'value' : 2.7205E-07, # $/kWh
}

for dic in [FIXED_COST_ELECTROLYZER, FIXED_COST_CHEM_PLANT, FIXED_COST_H2_STORAGE]:
    dic = capital_recovery_factor(DISCOUNT_RATE, **dic)
    dic = fixed_cost_per_hr(**dic)
    print(dic)




VAR_COST_ELECTROLYZER = {
    'value' : 1.00E-06, # $/kWh, Small, but non-zero value
    'ref' : 'FIXME - find source saying cost is 99% based on electricity costs'

}


VAR_COST_CHEM_PLANT = {
    #'value' : 6.91E-02, # $/kWh = 18.62*(0.069+0.038+0.016+0.001)/(MMBtu_per_Gallon_Gasoline*MWh_per_MMBtu)  # Variable O&M cost ($/MWh)
                                    # order is: maintenance, taxes & incentives, utilities, clean water
    'value' : 18.62*(0.069+0.038+0.016+0.001)/(MMBtu_per_Gallon_Gasoline*MWh_per_MMBtu) * (1./1000), # Variable O&M cost ($/kWh)
    'ref' : 'Fig 4b, cost break down of $/GGE, excluding electrolyzer and cap annual, D.H. König et al. / Fuel 159 (2015) 289–297'

}


VAR_COST_CO2 = {
    'co2 cost' : 50, # $/metric ton CO2
    'value' : 1.71E-02, # $/kWh liquid hydrocarbons = CO2 cost ($/metric ton) * (236 tons hr^-1 / 690 MW of liquid hydrocarbons ) * (1 MW / 1000 kW)
    'ref' : 'table 2, CO2 tons/hr / MW liquid hydrocarbons, D.H. König et al. / Fuel 159 (2015) 289–297'
}


EFFICIENCY_ELECTROLYZER = {
    'value' : 0.677,
    'ref' : 'table 2, eta_PtL (44.6%) / eta_plant (65.9%) = power to liquid efficiency (total eff) / chem plant efficiency, D.H. König et al. / Fuel 159 (2015) 289–297'
}


EFFICIENCY_CHEM_PLANT = {
    'value' : 0.659,
    'ref' : 'table 2, eta_plant = chem plant efficiency, D.H. König et al. / Fuel 159 (2015) 289–297'
}


DECAY_RATE_H2_STORAGE = {
    'value' : 1.14E-08, # fraction per hour (0.01% per year)    
    'ref' : 'Crotogino et al., 2010, p43'
}
