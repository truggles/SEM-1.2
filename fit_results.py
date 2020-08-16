#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

from scipy.stats import rankdata, normaltest
from scipy.linalg import lstsq

from sklearn.metrics import r2_score

from helpers import color_list

matplotlib.rcParams.update({'font.size': 15})

def r_squared(y_true, y_pred):

    y_true1 = np.array( y_true )
    y_pred1 = np.array( y_pred )
    SS_tot = np.sum( np.power(y_true1 - np.mean(y_true1), 2) )
    SS_reg = np.sum( np.power(y_pred1 - np.mean(y_true1), 2) )
    SS_res = np.sum( np.power(y_true1 - y_pred1, 2) )
    print(f"SS_tot {SS_tot} SS_reg {SS_reg} SS_res {SS_res}")

    r2 = 1. - SS_res / SS_tot

    return r2


def MPE(y_true, y_pred):

    y_true1 = np.array( y_true )
    y_pred1 = np.array( y_pred )

    mpe = 100. * np.mean( (y_true - y_pred) / y_true )
    print(f"MPE: {mpe}")
    return mpe


def MAPE(y_true, y_pred):

    y_true1 = np.array( y_true )
    y_pred1 = np.array( y_pred )

    mape = 100. * np.mean( np.abs((y_true - y_pred) / y_true) )
    print(f"MAPE: {mape}")
    return mape





def scatter_and_fit(regions, x, y, x_label, y_label, plot_base, save_name):

    plt.close()
    fig, ax = plt.subplots()
    
    markers = ['o', 'x', 'v']
    markers = ['o', 'o', 'o']
    colors = color_list()
    for r, x, y, m, c in zip(regions, xs, ys, markers, colors):
        print(f"\ncorr coef:\n{np.corrcoef(x, y)}")
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print("slope, intercept, r2_value, p_value, std_err")
        print(f"{slope}, {intercept}, {r_value**2}, {p_value}, {std_err}")

        M = x[:, np.newaxis]**[0, 1]
        A = np.vstack([x, np.zeros(len(x))]).T
        p, res, rnk, s = lstsq(A, y)

        ax.scatter(x, y, c=[c,], marker=m, alpha=.01, s=30, label='_nolabel_')
        ax.scatter([], [], label=r, s=20, c=[c,], marker=m)
        #ax.plot(x1, p[1] + p[0]*x1, label=f'fitted line: {r}')
        r2 = r_squared(x, y)
        mpe = MPE(x, y)
        mape = MAPE(x, y)
        print(f"{r}: {r2_score(x, y)} vs {r2}")

        cnt = 0
        print("Test yi > xi")
        for xi, yi in zip(x, y):
            if yi > xi:
                print(f" -- {cnt} {xi} {yi}")
            cnt += 1


    x_perfect = np.arange(ax.get_xlim()[1]*1.1)
    ax.plot(x_perfect, x_perfect, 'k--', label=f'f(x) = x', linewidth=2)
    ax.set_xlim(0, ax.get_xlim()[1]*1.1)
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_base}/{save_name}.png")
    plt.savefig(f"{plot_base}/{save_name}.pdf")
    plt.close()


date = '20200897v1'
save_name = 'std_vs_quad'

xs = []
ys = []


regions = ["ERCOT","PJM","NYISO"]


for r in regions:
    d = f'plots_{date}_101x101_{r}'
    df = pd.read_csv(f'{d}/csv_{r}_{save_name}.csv')
    print(df.head())
    xs.append(df['total variability'])
    ys.append(df['estimated total variability'])

x_label = r"total variability ($\sigma_{tot}$)"
y_label = r"estimated total variability ($\hat{\sigma}_{tot}$)"
plot_base = f'plots_{date}_101x101'
scatter_and_fit(regions, xs, ys, x_label, y_label, plot_base, save_name)




