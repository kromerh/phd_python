import re
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
"""
1.) Load the tally outputs. This is flux per source particle flux(E). Take the center bin energy as the energy for that bin.
"""

# read tally files in the directory
filename = ['H://hkromer//10_Experiments//02_MCNP//CurrentTarget218//CurrentTarget218_normal/tallyOutputInCell63', # S255 without magnet
            'H://hkromer//10_Experiments//02_MCNP//CurrentTarget217//CurrentTarget217_normal/tallyOutputInCell64' # S255 with magnet
            ]
title = ['Flux in tally S255 without magnets',
         'Fluy in tally S255 with magnets'
]
lineDataBegin = 2  # line where the data in the file begins


df_energy = []  # energy column
energyMade = False
df = pd.DataFrame()     # contains the flux
df_err = pd.DataFrame() # contains the error in the flux (fraction)
# create dataframe structure


# print(filename, cell)
for jj in range(0,len(filename)):
    fhandle = open(filename[jj], 'r')  # 0: without magnets, 1: with magnets
    lstFlux = []
    lst_d_Flux = []
    df_col = []  # column of the dataframe
    ll = 0  # line index
    for line in fhandle:
        if ll > lineDataBegin - 1:
            line = line.rstrip().split()
            flux = line[1]
            d_flux = line[2]
            lstFlux.append(float(flux))
            lst_d_Flux.append(float(d_flux))
        if ll > lineDataBegin - 1 and energyMade == False:
            energy = line[0]
            df_energy.append(float(energy))
        ll = ll + 1
    df[jj] = lstFlux
    df_err[jj] = lst_d_Flux
    energyMade = True
    fhandle.close()


    # print(df_cols)
    df_energy = np.array(df_energy)
    df_energy_orig = np.copy(df_energy)
    e_diff = np.diff(df_energy)
    for ii in range(0, len(df_energy_orig)):
        if ii == 0:  # first element
            df_energy[ii] = -1
        else:
            this = df_energy_orig[ii]
            prev = df_energy_orig[ii - 1]
            res = prev + abs(this - prev)/2
            df_energy[ii] = res


df['energy'] = df_energy
df_err['energy'] = df_energy

df['diff'] = abs(df[0] - df[1])
df['relDiff'] = abs(df[0] - df[1])/df[0]
#
#
# plot flux for one position
x = df['energy'].values
y = df['relDiff'].values*100
y_err = (df_err[0].values) * y

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1, 1, 1)
# x
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
# y
# minor_locator = AutoMinorLocator(2)
# ax1.yaxis.set_minor_locator(minor_locator)
ax1.bar(x, y, width=0.03, color='blue', edgecolor='black')
fig.subplots_adjust(left=0.08, bottom=0.2)
plt.xlabel(r'Tally bin center energy [MeV]', fontsize=14)
plt.ylabel(r'Relative difference [\%]', fontsize=14)
plt.title(r'Influence of the NdFeB-Magnets tally 255 cm south', fontsize=16)
ax1.text(0.1, 4, 'Relative difference in flux with \n and without magnets', fontsize=12, color='white',
        bbox=dict(facecolor='grey', edgecolor='grey', boxstyle='round,pad=0.4'))
# pyplot.xscale('log')
ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')
# ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e7))
# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

plt.xlim(0, 2.6)
plt.ylim(0, 6)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
# ticks
# y_ticks = np.arange(1.0e6, 10e6 + 1.0e6, 1.0e6)
# y = np.arange(1, 10 + 1, 1)
# plt.yticks(y_ticks, y)
# plt.legend(loc='best', title=r"\textbf{Legend}")
plt.savefig( filename[0] + '.png', dpi=600)
# plt.show()
plt.close()