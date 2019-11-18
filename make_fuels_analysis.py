from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import analytic_fuels as af



def scan_electricity_and_electrolyzer_costs(system, electricity_info, electrolyzer_info):

    info1 = electricity_info
    info2 = electrolyzer_info
    x_ticks_loc = []
    x_ticks_val = []
    y_ticks_loc = []
    y_ticks_val = []
    #y_ticks = [-0.5]

    print(f"Initial electrolyzer cost: {system['FIXED_COST_ELECTROLYZER']['value']}")

    #x = np.outer(np.linspace(info1[0], info1[1], info1[2]), np.ones(info2[2]))
    ##y = np.outer(np.ones(info2[2]), np.linspace(info2[0], info2[1], info1[2]))
    #y = np.outer(np.ones(info1[2]), np.linspace(info2[0], info2[1], info2[2]))
    z = np.zeros((info1[-1], info2[-1]))
    for i, electricity in enumerate(np.linspace(info1[0], info1[1], info1[2])):
        if round(electricity,2) == electricity:
            y_ticks_loc.append(i)
            y_ticks_val.append(str(round(electricity,2)))
        for j, electrolyzer in enumerate(np.linspace(info2[0], info2[1], info2[2])):
            if round(electrolyzer,2) == electrolyzer:
                x_ticks_loc.append(j)
                x_ticks_val.append(str(round(electrolyzer,2)))
            system['FIXED_COST_ELECTROLYZER']['value'] = electrolyzer
            cost = af.get_fuel_system_costs(syst, electricity)
            z[i][j] = cost

    fig, ax = plt.subplots()
    im = ax.imshow(z, origin='lower', interpolation='spline16')

    #plt.xticks(x_ticks_loc, x_ticks_val, rotation=90)
    plt.xticks(x_ticks_loc, x_ticks_val)
    plt.yticks(y_ticks_loc, y_ticks_val)
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel("Fuel Cost ($/kWh)")
    ax.set_xlabel('Electrolyzer CapEx ($/kW/h)')
    ax.set_ylabel('Electricity Costs ($/kWh)')
    plt.tight_layout()
    plt.savefig('tmp.png')
    plt.clf()

syst = af.return_fuel_system()

cost = af.get_fuel_system_costs(syst, 0.06)
print(cost)

electricity_info = [0.0, 0.1, 51]
electrolyzer_info = [0, 0.05, 51]
scan_electricity_and_electrolyzer_costs(syst,
        electricity_info, electrolyzer_info)


