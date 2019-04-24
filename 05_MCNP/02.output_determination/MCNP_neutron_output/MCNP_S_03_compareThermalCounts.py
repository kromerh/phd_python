"""
Compares the thermal counts (E < 0.1 MeV) to the total counts
"""

import re
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
"""
1.) Load the dataframe counts output from S_01_neutron_output
"""
inputFilePrefix = 'CurrentTarget' # insert the prefix of the MCNP input file
pathOutputDirectory = '//fs03//LTH_Neutimag//hkromer//10_Experiments//02_MCNP//'

# IDrun = 99 # insert the ID of the MCNP run
IDruns = np.arange(126, 135, 1) # insert the ID of the MCNP run
# IDruns = np.arange(99, 108, 1) # insert the ID of the MCNP run

df_count = pd.DataFrame()  # dataframe that contains col: IDrun, Edeut, N, S, W, E and the values is the neutron output per 100
                     # µSv/hr at 50 cm

lstDirection = ['N', 'S', 'W', 'E']
df_result = pd.DataFrame()
df_result['direction'] = ' '
df_result['distance'] =  ' '
df_result['ctsThermal'] = 0
df_result['ctsTotal'] = 0
df_result['IDrun'] = 0
df_result['ratio'] = 0
# print(df_result)
for IDrun in IDruns:
    run = inputFilePrefix + str(IDrun)

    thisMode = '_normal'

    # read tally files in the directory
    path_MCNP_inputFolder = pathOutputDirectory + run + '//' + run + thisMode + '//'
    print('Processing: ', str(path_MCNP_inputFolder))

    filename = path_MCNP_inputFolder + '/df_counts.csv'
    df_count = pd.read_csv(filename, encoding='utf-8')  # load DataFrame

    df_total = df_count.iloc[-1]  # total number of counts
    # print(df_total)

    # get only thermal counts, E < 0.1 MeV
    df_count = df_count.iloc[:-1]
    df_count['energy'] = pd.to_numeric(df_count['energy'], downcast='float')
    # df_count = df_count.set_index('energy')
    df_thermal = df_count.where(df_count['energy']< 0.1)
    df_thermal = df_thermal.dropna(axis=0, how='all')


    # cols = ['direction', 'distance', 'ctsThermal', 'ctsTotal']

    lstDir = []
    lstDist = []
    for col in df_count.columns[1:]:
        s = col.split()
        dir = s[0]
        dist = s[1]
        lstDir.append(dir)
        lstDist.append(dist)
    df_this_result = pd.DataFrame()
    df_this_result['direction'] = lstDir
    df_this_result['distance'] = lstDist
    df_this_result['ctsThermal'] = df_thermal.loc[:, 'W 20':].sum().values
    df_this_result['ctsTotal'] = df_total.iloc[1:].values
    df_this_result['IDrun'] = IDrun
    df_this_result['ratio'] = df_this_result['ctsThermal'] / df_this_result['ctsTotal']
    # print(df_this_result)
    # print(df_total.iloc[1:].values)

    # deuterium energy
    filename = glob.glob(path_MCNP_inputFolder + '/df_neutron_output_for_Edeut_*')
    if len(filename) > 1:
        print('****ERROR! More than one neutron output filename in directory... Exit.')
        break
    t = re.findall(r'Edeut_(\d+)', filename[0])
    Edeut = t[0]

    fig = plt.figure(figsize=(8, 5))
    # fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    minor_locator = AutoMinorLocator(2)
    ax1.xaxis.set_minor_locator(minor_locator)
    minor_locator = AutoMinorLocator(5)
    ax1.yaxis.set_minor_locator(minor_locator)
    # plt.yscale('log')
    colors = ['blue', 'red', 'green', 'olive']
    markers = ['o', 'x', 'd', 's']
    for dir in lstDirection:
        idx = lstDirection.index(dir)
        this_dir_df = df_this_result[ df_this_result['direction'] == dir ]
        ax1.plot(this_dir_df['distance'], this_dir_df['ratio'] * 100.0, label=dir, marker=markers[idx], color=colors[idx])
    plt.xlabel('Distance to source [cm]')
    plt.ylabel('Ratio thermal counts / total counts [%]')
    plt.title('Thermal counts / total counts for ' + str(Edeut) + ' keV' )
    ax1.grid(b=True, which='major', linestyle='-')
    ax1.grid(b=True, which='minor', linestyle='--')
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5e6))
    # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # plt.xticks(d)
    ax1.axhline(linewidth=1, color='black')
    plt.legend(loc='best')
    # plt.show()
    plt.tight_layout()
    plt.savefig(pathOutputDirectory + 'thermal_ratio/ratio_thermal_counts_' + str(Edeut) + '_keV.pdf')
    plt.savefig(pathOutputDirectory + 'thermal_ratio/ratio_thermal_counts_' + str(Edeut) + '_keV.png', dpi=1200)
    plt.close(fig)
    # dataframe for the results
    # break
    df_result= df_result.append(df_this_result, ignore_index=True)

df_result = df_result.set_index('IDrun')

filename = pathOutputDirectory + '/thermal_ratio/df_thermal_count_ratio_for_' + str(IDruns[0]) + '_to_' + str(IDruns[-1]) + '.csv'  # contains the flux*response --> counts column and total row
df_result.to_csv(filename, index=True, encoding='utf-8')  # save DataFrame

