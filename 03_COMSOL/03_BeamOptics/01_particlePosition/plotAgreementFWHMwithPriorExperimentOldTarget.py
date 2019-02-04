import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import itertools
from scipy.stats import kde
from scipy import optimize
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot, transforms

remote_path = '//fs03/LTH_Neutimag/hkromer/'  # PSI

# remote_path = '/home/hkromer/01_PhD/'  # local laptop


# FWHM_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data_BIDIR/2D_histogram_lastTimestep/FWHMs/'.format(remote_path)
FWHM_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data_TD/2D_histogram_lastTimestep/FWHMs/'.format(remote_path)
runtype = re.findall(r'particle_data_(.+)/2D', FWHM_path)[0]

# import the fwhm_x and fwhm_y
files = os.listdir(FWHM_path)
files = [f for f in files if f.endswith('.csv')]  # select only csv

df = pd.DataFrame(columns = ['ID', 'fwhm_x', 'fwhm_y'])
for file in files:
	df_t = pd.read_csv('{}/{}'.format(FWHM_path,file))

	ID = re.findall(r'(\d*)_df', file)  # take the ID from the filename
	fwhm_x = df_t['fwhm_x'].unique()[0]
	fwhm_y = df_t['fwhm_y'].unique()[0]
	df_out = pd.DataFrame()
	df_out['ID'] = ID
	df_out['fwhm_x'] = fwhm_x
	df_out['fwhm_y'] = fwhm_y

	df = df.append(df_out)

# Output directory for the fwhm data
directory = '{}/plot_fwhms'.format(FWHM_path)
if not os.path.exists(directory):
	os.makedirs(directory)
# print(df)

f, ax = plt.subplots()

Yx = df['fwhm_x'].values
Yy = df['fwhm_y'].values
X = df['ID'].values
plt.title('Comparison simulation FWHM for old target \n {}'.format(runtype))
# my_width=0.01
for ii in range(0,len(X)):
	ax.plot([X[ii], X[ii]], [Yx[ii], Yy[ii]], linewidth=2)
	ax.scatter([X[ii], X[ii]], [Yx[ii], Yy[ii]])

xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.plot(xlim, [2.17, 2.17], color='red', linestyle='dashed', label='Adams et al')

plt.xlabel('Run')
plt.ylabel('FWHM [mm]')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
# plt.show()

filename =  '{}/fwhm'.format(directory)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
# plt.close('all')

# plt.show()	
df.to_csv('{}/df_fwhms.csv'.format(directory))

