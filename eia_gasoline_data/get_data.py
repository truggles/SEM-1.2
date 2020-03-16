import pandas as pd
import numpy as np



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

gas_str = 'Weekly XXX All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)'
out = {}

for state, f_name in states.items():

    df = load_df(f_name)

    vals = []
    for idx in df.index:
        if df.loc[idx, 'Date'].year == 2019:
            vals.append(df.loc[idx, gas_str.replace('XXX', state)])
    out[state] = [np.min(vals), np.max(vals), np.mean(vals)]

for state, vals in out.items():
    print(state, vals)

