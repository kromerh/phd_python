import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

remote_path = '//fs03/LTH_Neutimag/hkromer/'

project_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/\
01_OldTarget/IGUN_geometry/2018-09-21_comsol/\
longer_target/'.format(remote_path)

# electric field along the x axis
file_long = f'{project_path}01.longTarget.es.normE.xaxis.csv'
file_normal = f'{remote_path}02_Simulations/06_COMSOL/03_BeamOptics/\
01_OldTarget/IGUN_geometry/2018-09-18_comsolGeometry/02.define_release_time/\
particleData/01.es.normE.xaxis.csv'  # short target

df_long = pd.read_csv(f'{file_long}', header=None, skiprows=9,
							index_col=None, delimiter=',')
df_long.columns = ['x', 'y', 'z', 'E_long']
df_normal = pd.read_csv(f'{file_normal}', header=None, skiprows=9,
							index_col=None, delimiter=',')
df_normal.columns = ['x', 'y', 'z', 'E_normal']


X = np.linspace(0, 85, 200)  # query points
df = pd.DataFrame()
df['x'] = X

E_interp_long = interp1d(df_long.x, df_long.E_long, fill_value='extrapolate')
E_interp_normal = interp1d(df_normal.x, df_normal.E_normal, fill_value='extrapolate')
df['E_normal'] = E_interp_normal(X)
df['E_long'] = E_interp_long(X)

df['E_diff'] = np.abs(np.abs(df['E_long'] - df['E_normal']) / df['E_normal'] )
	# print(this_df.head())


f, ax = plt.subplots(figsize=(8, 6))
f.suptitle('Comparison long and shorter target \n Electric field along x axis')

# plot
this_plot0, = ax.plot(X, 100*df['E_diff'])
ax.set_xlabel('x position [mm]')
ax.set_ylabel('rel. diff. electric field long - normal sized target[%]')

ax.grid(True)



# plt.ylim(0,3)
directory = f'{project_path}/plots/'
if not os.path.exists(directory):
	os.makedirs(directory)

print(directory)
filename =  f'{directory}es.normE.xaxis.comparisonLongAndShorterTarget'
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
