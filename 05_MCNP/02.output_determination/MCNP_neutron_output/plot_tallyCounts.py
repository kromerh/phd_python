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
folder = '//fs03//LTH_Neutimag/hkromer//10_Experiments//02_MCNP//'
tallies = ['tallyOutPutInCell45','tallyOutPutInCell55','tallyOutPutInCell35', 'tallyOutPutInCell25'] # N S W E 
files = ['/CurrentTarget234/CurrentTarget234_normal/', '/CurrentTarget241/CurrentTarget241_normal/']
# title = 'Zerodensity 160 degC 2.17 MeV'
lineDataBegin = 1  # line where the data in the file begins

lst_direction = ['north', 'south', 'west', 'east']


df_res = pd.DataFrame()
#
# create dataframe structure

lst_df = []  # list of the results
# only north
for tally, direction in zip(tallies, lst_direction):
    for this_file in files:
        filename = folder + this_file + tally
        df = pd.DataFrame()     # contains the flux
        df_err = pd.DataFrame() # contains the error in the flux (fraction)
        df_energy = []  # energy column
        energyMade = False
        # print(filename, cell)
        fhandle = open(filename, 'r')
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
        df[0] = lstFlux
        df_err[0] = lst_d_Flux
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

        # print(df_energy)
        df['energy'] = df_energy  # energy and F4 tally response
        df_err['energy'] = df_energy  # energy and F4 tally relative error
        # df.set_index('energy', inplace=True)
        lst_df.append(df)

    series = (np.abs(lst_df[0][0]-lst_df[1][0]) / lst_df[0][0]) * 100.0

    df_res['relDiff_%s_percent' % direction] = series

df_res['energy_MeV'] = df_energy
df_res.set_index('energy_MeV', inplace=True)

df_res.to_csv("df_res.csv")



# # plot flux for one position
# x = df['energy'].values
# y = df[0].values
# # print(df)
# y_err = (df_err[0].values) * y

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# # x
# minor_locator = AutoMinorLocator(2)
# ax1.xaxis.set_minor_locator(minor_locator)
# # y
# # minor_locator = AutoMinorLocator(2)
# # ax1.yaxis.set_minor_locator(minor_locator)
# ax1.errorbar(x, y, yerr=y_err, color='black', fmt="-o", markersize=2, linewidth=0.25,  ecolor='red', elinewidth=2,capsize=5)
# fig.subplots_adjust(left=0.16)
# plt.xlabel(r'Tally bin center energy [MeV]', fontsize=14)
# plt.ylabel(r'F4 tally counts $\left[ \frac{\phi}{\textit{source particle}} \right]$', fontsize=14)
# plt.title(r'' + title, fontsize=16)
# # pyplot.xscale('log')
# plt.yscale('log')
# ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
# ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')
# # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e7))
# # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

# plt.xlim(0, 2.3)
# plt.ylim(1e-10, 1e-4)
# for tick in ax1.xaxis.get_major_ticks():
#     tick.label.set_fontsize(14)
# for tick in ax1.yaxis.get_major_ticks():
#     tick.label.set_fontsize(14)
# # ticks
# # y_ticks = np.arange(1.0e6, 10e6 + 1.0e6, 1.0e6)
# # y = np.arange(1, 10 + 1, 1)
# # plt.yticks(y_ticks, y)
# # plt.legend(loc='best', title=r"\textbf{Legend}")
# # plt.savefig( filename + '.png', dpi=600)
# plt.show()
# # plt.close()