import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


DATE = '20201230v1'
solar_max = 1.
wind_max = 1.
steps = 101
solar_gen_steps = np.linspace(0, solar_max, steps)
wind_gen_steps = np.linspace(0, wind_max, steps)
print("Wind gen increments:", wind_gen_steps)
print("Solar gen increments:", solar_gen_steps)


regions = ['ERCOT','PJM','NYISO','FR']
solar_vals = [0., 0.25]
wind_vals = [0., 0.25, 0.5]
wind_vals = [0., 0.25]
    
make_summary = True
make_summary = False

plot = True
#plot = False

if make_summary:
    regs = []
    solar = []
    wind = []
    hours = []
    inter = []
    intra = []
    std = []
    mean = []
    
    for region in regions:
        for HOURS_PER_YEAR in [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 200]:
            pkl_file = f'pkls/pkl_{DATE}_{steps}x{steps}_{region}_hrs{HOURS_PER_YEAR}'
            pickle_in = open(f'{pkl_file}.pkl','rb')
            study_regions = pickle.load(pickle_in)
        
            for i, solar_gen in enumerate(solar_gen_steps):
                if solar_gen not in solar_vals:
                    continue
                for j, wind_gen in enumerate(wind_gen_steps):
                    if wind_gen not in wind_vals:
                        continue
                    print(region, HOURS_PER_YEAR, solar_gen, wind_gen)
                    wind_gen = wind_gen_steps[j]
                    rls = study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][0]
    
                    # Fill a new row
                    regs.append(region)
                    solar.append(solar_gen)
                    wind.append(wind_gen)
                    hours.append(HOURS_PER_YEAR)
                    inter.append(np.std(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][5])*100.)
                    intra.append(np.mean(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][4])*100)
                    std.append(np.std(rls)*100)
                    mean.append(np.mean(rls)*100)
    
    
                    #m_rl_mean[i].append(np.mean(rls)*100)
                    #m_rl_std[i].append(np.std(rls)*100)
                    ##m_rl_50pct[i].append( (np.percentile(rls, 75) - np.percentile(rls, 25))*100)
                    ##m_rl_95pct[i].append( (np.percentile(rls, 97.5) - np.percentile(rls, 2.5))*100)
                    ##m_rl_Mto97p5pct[i].append( (np.percentile(rls, 97.5) - np.mean(rls))*100)
                    #intra[i].append(np.mean(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][4])*100.)
                    #inter[i].append(np.std(study_regions[str(round(solar_gen,2))][str(round(wind_gen,2))][5])*100.)
        
    
    
    df = pd.DataFrame({
        'region' : regs,
        'solar' : solar,
        'wind' : wind,
        'hours' : hours,
        'inter' : inter,
        'intra' : intra,
        'std' : std,
        'mean' : mean,
        })
    df.to_csv('Sensitivity_summary.csv', index=False)

if not plot:
    exit()

df = pd.read_csv('Sensitivity_summary.csv')
multi = df.set_index(['region', 'solar', 'wind']).sort_index()


for region in regions:
    fig, ax = plt.subplots()
    
    for solar in solar_vals:
        for wind in wind_vals:
            ax.plot(multi.loc[(region, solar, wind)]['hours'], multi.loc[(region, solar, wind)]['inter'], label=f'{region} s{solar}:w{wind}')

    ax.set_ylim(0, ax.get_ylim()[1])
    plt.legend()
    plt.savefig(f'plotsX/{region}.png')








