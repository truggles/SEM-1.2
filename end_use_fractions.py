import pandas as pd
import numpy as np




# Loop over long files and add details to results file
def add_detailed_results(df_name, files, fixed = 'nuclear', storage_eff = 0.9):

    df = pd.read_csv(df_name, index_col=False)
    dem_renews, dem_fixes, electro_renews, electro_fixes = [], [], [], []

    for idx in df.index:
        run = df.loc[idx, 'case name']
        run = run[:7:] # Gets "Run_001"
        for f_name in files:
            if run not in f_name:
                continue

            #print(run, end=' ')
            print(run)
            dem_renew, dem_fix, electro_renew, electro_fix = get_single_file_details(f_name, fixed, storage_eff)
            dem_renews.append(dem_renew)
            dem_fixes.append(dem_fix)
            electro_renews.append(electro_renew)
            electro_fixes.append(electro_fix)
            break

    df['dem_renew'] = dem_renews
    df['dem_fix'] = dem_fixes
    df['dem_renew_frac'] = df['dem_renew'] / (df['dem_renew'] + df['dem_fix'])
    df['electro_renew'] = electro_renews
    df['electro_fix'] = electro_fixes
    df['electro_renew_frac'] = df['electro_renew'] / (df['electro_renew'] + df['electro_fix'])

    df.to_csv(df_name.replace('.csv', '_app.csv'), index=False)








def get_single_file_details(f_name, fixed, storage_eff):

    df = pd.read_csv(f_name)
    
    stored_energy = 0.
    stored_frac_renew = 0.
    electro_renew = 0.
    electro_fix = 0.
    dem_renew = 0.
    dem_fix = 0.
    remainder = 0. # Just to track how far off totals are




    for idx in df.index:
    
    
        # Check if we ever need to worry about wrapping from year's end to know
        # content of stored energy
        if idx == 0 and df.loc[idx, 'energy storage (kWh)'] != 0:
            print(f"\n\nStart of file and there is energy in storage: {df.loc[idx, 'energy storage (kWh)']}\n\n")
    
    
        # Dispatch from X
        # From generation
        if 'dispatch wind (kW)' in df.columns and 'dispatch solar (kW)' in df.columns:
            disp_renew = df.loc[idx, 'dispatch wind (kW)'] + df.loc[idx, 'dispatch solar (kW)']
        elif 'dispatch wind (kW)' in df.columns and 'dispatch solar (kW)' not in df.columns:
            disp_renew = df.loc[idx, 'dispatch wind (kW)']
        elif 'dispatch wind (kW)' not in df.columns and 'dispatch solar (kW)' in df.columns:
            disp_renew = df.loc[idx, 'dispatch solar (kW)']
        disp_fix = df.loc[idx, f'dispatch {fixed} (kW)']
    
        # From storage content
        # Only calculate if energy coming from storage
        if df.loc[idx, 'dispatch from storage (kW)'] > 0.:
            stored_renew = stored_energy * stored_frac_renew
            stored_fix = stored_energy * (1. - stored_frac_renew)
            renew_out = df.loc[idx, 'dispatch from storage (kW)'] * stored_frac_renew
            fix_out = df.loc[idx, 'dispatch from storage (kW)'] * (1. - stored_frac_renew)
            stored_renew -= renew_out
            stored_fix -= fix_out
            disp_renew += renew_out
            disp_fix += fix_out
            stored_energy = stored_renew + stored_fix
            stored_frac_renew = stored_renew / stored_energy
    
        disp_tot = disp_renew + disp_fix
        if disp_tot == 0.:
            disp_frac_renew = 0.
        else:
            disp_frac_renew = disp_renew / disp_tot

    
        
        # Dispatch to Y
        # To storage content
        # Only calculate if energy going into storage
        if df.loc[idx, 'dispatch to storage (kW)'] > 0.:
            stored_renew = stored_energy * stored_frac_renew
            stored_fix = stored_energy * (1. - stored_frac_renew)
            stored_renew += df.loc[idx, 'dispatch to storage (kW)'] * storage_eff * disp_frac_renew
            stored_fix += df.loc[idx, 'dispatch to storage (kW)'] * storage_eff * (1. - disp_frac_renew)
            stored_energy = stored_renew + stored_fix
            stored_frac_renew = stored_renew / stored_energy
            disp_tot -= df.loc[idx, 'dispatch to storage (kW)']
    
        # To demand
        to_demand = df.loc[idx, 'demand (kW)'] - df.loc[idx, 'dispatch unmet demand (kW)']
        dem_renew += to_demand * disp_frac_renew
        dem_fix += to_demand * (1. - disp_frac_renew)
        disp_tot -= to_demand
    
        # To electrolyzer / fuels
        electro_renew += df.loc[idx, 'dispatch to fuel h2 storage (kW)'] * disp_frac_renew
        electro_fix += df.loc[idx, 'dispatch to fuel h2 storage (kW)'] * (1. - disp_frac_renew)
        disp_tot -= df.loc[idx, 'dispatch to fuel h2 storage (kW)']
    
        remainder += disp_tot
            
    
    #print(f"electro_renew {electro_renew} electro_fix {electro_fix} dem_renew {dem_renew} dem_fix {dem_fix}")
    #print(f"Electric load % renewable {round(dem_renew/(dem_renew+dem_fix),4)*100}%")
    #print(f"Mean electric load:       {round((dem_renew+dem_fix)/8760,4)}")
    #print(f"Fuel load % renewable     {round(electro_renew/(electro_renew+electro_fix),4)*100}%")
    #print(f"Mean fuel load:           {round((electro_renew+electro_fix)/8760,4)}")
    #
    #print(f"Remainder {remainder}")

    # Return normalized versions
    l = len(df.index)
    return dem_renew/l, dem_fix/l, electro_renew/l, electro_fix/l



