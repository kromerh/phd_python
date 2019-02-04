import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

remote_path = '//fs03/LTH_Neutimag/hkromer/'

files_along_x = '{}02_Simulations/06_COMSOL/03_BeamOptics/\
01_OldTarget/IGUN_geometry/2018-09-18_comsolGeometry/\
01.mesh_refinement/'.format(remote_path)
# files_along_x = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
# 2018-09-18_comsolGeometry/01.mesh_refinement/'

# Do not forget to comment out the reference df!

files = os.listdir(files_along_x)
# files = [f for f in files if f.endswith('.csv')]
files = [f for f in files if f.endswith('.csv')]
# files = [f for f in files if f.startswith('V')]
# print(files)

# Do this only for the 01 files
starts_with = '01'
files = [f for f in files if f.startswith(f'{starts_with}.es.')]

# files = [f for f in files if f.startswith('V')]
# print(files)


# load the two files (normal and coarse) into a DataFrame
# normal file
for f in files:
	if 'coarse' in f:
		file_2 = f
	else:
		file_1 = f
df_1 = pd.read_csv(f'{files_along_x}{file_1}', header=None, skiprows=9,
						index_col=None, delimiter=',')
colname = ['x', 'y', 'z', 'V_n_1', 'V_n_0.9', 'V_n_0.5', 'V_n_0.25']
df_1.columns = colname
# print(df_1.head())

# coarse file
df_2 = pd.read_csv(f'{files_along_x}{file_2}', header=None, skiprows=9,
						index_col=None, delimiter=',')
colname = ['x', 'y', 'z', 'V_n_30', 'V_n_10', 'V_n_5', 'V_n_4',
			'V_n_3', 'V_n_1']
df_2.columns = colname
# print(df_2.head())
df_2 = df_2[ ['x', 'y', 'z', 'V_n_10', 'V_n_1'] ]
# check if the min and the max of the two datasets are the same
print(f'Minimum x of coarse file: {np.min(df_2.x)}')
print(f'Minimum x of normal file: {np.min(df_1.x)}')

print(f'Maximum x of coarse file: {np.max(df_2.x)}')
print(f'Maximum x of normal file: {np.max(df_1.x)}')


V_ref_interp_1 = interp1d(df_1['x'], df_1['V_n_1'], fill_value='extrapolate')
V_ref_interp_2 = interp1d(df_2['x'], df_2['V_n_1'], fill_value='extrapolate')


# PLOT ACCELERATOR COLUMN

my_plots = []

X = np.linspace(0, 85, 200)  # query points
V_ref_1 = V_ref_interp_1(X)
V_ref_2 = V_ref_interp_2(X)


f, ax = plt.subplots(figsize=(8, 6))
f.suptitle('Mesh refinement air in room \n Electric field along x axis')

# relate ndofs to the mesh refinement number
file_ndofs = '01.ndofs.refine_air.txt'
df_ndofs = pd.read_csv(f'{files_along_x}{file_ndofs}', header=None,
						index_col=None, delimiter='\s+')
df_ndofs.columns = ['n','time','ndofs']
# print(df_ndofs.head())
# sys.exit()
def plotDifferences(df_x, col, V_ref):

	# continue only for columns with a "V" and do not for the reference run
	if ('V' in col.name) & (col.name != 'V_n_1'):
		# print(col.name)
		# voltage interpolation
		V_interp = interp1d(df_x, col, fill_value='extrapolate')

		this_V = V_interp(X)

		diff_V = np.abs(np.abs(this_V - V_ref) / V_ref)

		#
		#
		# # get label
		lbl = col.name
		n_mesh = float(re.findall(r'(\d\.*\d*)', lbl)[0])
		ndofs = df_ndofs.ndofs[ df_ndofs.n == n_mesh ].values[0]
		ndofs = int(ndofs)

		print(lbl, n_mesh, ndofs)
		#
		# plot
		this_plot0, = ax.plot(X, 100*diff_V, label=ndofs)
		ax.set_xlabel('x position [mm]')
		ax.set_ylabel('rel. diff. electric field [%]')
		ax.set_ylim(0,5)
		ax.grid(True)


		my_plots.append(this_plot0)


df_1.apply(lambda x: plotDifferences(df_1['x'], x, V_ref_1))
df_2.apply(lambda x: plotDifferences(df_2['x'], x, V_ref_2))

# sys.exit()
# legend
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
# print(labels)
# print(handles)
labels = [labels[3], labels[0], labels[1], labels[2]]
handles = [handles[3], handles[0], handles[1], handles[2]]
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels, title='ndofs')
plt.ylim(0,3)
directory = f'{files_along_x}/plots/'
if not os.path.exists(directory):
	os.makedirs(directory)

filename =  f'{directory}{file_1}'
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
