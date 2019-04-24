import re
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
"""
1.) Load the dataframe output from S_01_neutron_output
"""
inputFilePrefix = 'CurrentTarget' # insert the prefix of the MCNP input file
pathOutputDirectory = '//fs03//LTH_Neutimag//hkromer//10_Experiments//02_MCNP//'

# IDrun = 99 # insert the ID of the MCNP run
# IDruns = np.arange(108, 117, 1) # insert the ID of the MCNP run
IDruns = np.arange(136, 145, 1) # insert the ID of the MCNP run

df = pd.DataFrame()  # dataframe that contains col: IDrun, Edeut, N, S, W, E and the values is the neutron output per 100
                     # µSv/hr at 50 cm

df['IDrun'] = IDruns
df['Edeut'] = 0
lstDirection = ['N', 'S', 'W', 'E']
for dir in lstDirection:
    df[dir] = 0  # neutron output per 100 µSv/h at 50cm

df = df.set_index('IDrun')
for IDrun in IDruns:
    run = inputFilePrefix + str(IDrun)

    thisMode = '_normal'

    # read tally files in the directory
    path_MCNP_inputFolder = pathOutputDirectory + run + '//' + run + thisMode + '//'
    print('Processing: ', str(path_MCNP_inputFolder))


    # load dataframe
    df_out = pd.DataFrame()
    filename = glob.glob(path_MCNP_inputFolder + '/df_neutron_output_for_Edeut_*')
    if len(filename) > 1:
        print('****ERROR! More than one neutron output filename in directory... Exit.')
        break
    t = re.findall(r'Edeut_(\d+)', filename[0])
    Edeut = t[0]
    df['Edeut'].loc[IDrun] = Edeut
    df_out = pd.read_csv(filename[0], encoding='utf-8')  # save DataFrame
    # print(df_out)
    df.loc[IDrun, 'N':'E'] = df_out.loc[:, 'N':'E'][ df_out['distance'] == 50 ].values[0]
    # break

print(df)

fig = plt.figure(figsize=(8, 5))
# fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
minor_locator = AutoMinorLocator(5)
ax1.yaxis.set_minor_locator(minor_locator)
# plt.yscale('log')
ax1.plot(df['Edeut'], df['N'], label = 'North', marker='o', color='blue')
ax1.plot(df['Edeut'], df['S'], label = 'South', marker='x', color='red')
ax1.plot(df['Edeut'], df['W'], label = 'West', marker='d', color='green')
ax1.plot(df['Edeut'], df['E'], label = 'East', marker='s', color='olive')
plt.xlabel('Deuterium Energy [keV]')
plt.ylabel('Neutron output per 100 µSv/hr [1/s]')
plt.title('LB6411 50 cm distance from source ('  + str(IDruns[0]) + ' to ' + str(IDruns[-1]) + ')')
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, which='minor', linestyle='--')
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5e6))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
# plt.xticks(d)
ax1.axhline(linewidth=1, color='black')
plt.legend(loc='best')
# plt.show()
plt.tight_layout()
plt.savefig(pathOutputDirectory + '/neutron_output/neutron_output_' + str(IDruns[0]) + '_to_' + str(IDruns[-1]) + '_keV.pdf')
plt.savefig(pathOutputDirectory + '/neutron_output/neutron_output_' + str(IDruns[0]) + '_to_' + str(IDruns[-1]) + '_keV.png', dpi=1200)
plt.close(fig)