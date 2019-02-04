import pandas as pd
import numpy as np
import os, sys, glob
import re
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
pd.set_option("display.max_columns",101)

# file that contains the processed data
master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_20180807/'
fname = f'{master_folder}/df_FWHM_mean_for_time_80_hrs.csv'

df = pd.read_csv(fname, index_col=0)

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(1, 1, 1)

X = df['FWHM_mean']
Y = df['diameter']
ax1.scatter(X, Y, marker='o', color='darkblue')
# X = df['FWHM_mean']
# Y = df['diameter']
# ax1.scatter(X, Y, marker='o', color='red')

ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	
# grid
ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')

ax1.set_xlabel(r'FWHM of LSF [mm]')
ax1.set_ylabel(r'FWHM of source in MCNP [mm]')
# plt.title('Diameter vs FWHM for case {}'.format(case))
# ylims = ax1.get_ylim()
# ax1.set_ylim(0, 1e7)
# fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
plt.xlim(0,3)
plt.ylim(0,7)
plt.tight_layout()
plt.show()
# plt.savefig('{}case_{:.0f}_for_time_{:.0f}_h.png'.format(savefolder, case, tot_meas_time), dpi=100)
# plt.clf()
# plt.close('all')


print(df.head())