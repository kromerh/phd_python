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
starts_with = '02'
files = [f for f in files if f.startswith(f'{starts_with}.es.')]

# files = [f for f in files if f.startswith('V')]
# print(files)


# load all the files, the n_mesh number is in the filename
df_main = pd.DataFrame()  # main dataframe that contains all columns
X = np.linspace(0, 85, 200)  # query points
df_main['x'] = X
for f in files:
	df_f = pd.read_csv(f'{files_along_x}{f}', header=None, skiprows=9,
							index_col=None, delimiter=',')
	n_mesh = float(re.findall(r'n(\d\.*\d*)', f)[0])
	colname = [f'x', 'y', 'z', f'V_n_{n_mesh}']
	df_f.columns = colname
	# if n_mesh == 1.0:
	# 	V_ref_interp = interp1d(df_f[f'x_n_{n_mesh}'], df_f['V_n_1.0'], fill_value='extrapolate')
		# print(f'yes{n_mesh}')


	V_interp = interp1d(df_f.x, df_f[f'V_n_{n_mesh}'], fill_value='extrapolate')
	this_V = V_interp(X)
	df_main[f'V_n_{n_mesh}'] = this_V
	# print(this_df.head())

print(df_main.head())

# PLOT ACCELERATOR COLUMN

my_plots = []

# df_main.to_csv(f'{files_along_x}df_main.csv')

f, ax = plt.subplots(figsize=(8, 6))
f.suptitle('Mesh refinement in vacuum chamber \n Electric field along x axis')

# relate ndofs to the mesh refinement number
file_ndofs = '02.ndofs.refine_vacuum.txt'
df_ndofs = pd.read_csv(f'{files_along_x}{file_ndofs}', header=None,
						index_col=None, delimiter='\s+')
df_ndofs.columns = ['n','time','ndofs']
# print(df_ndofs.head())
# sys.exit()
def plotDifferences(df_x, col, V_ref):

	# continue only for columns with a "V" and do not for the reference run
	if ('V' in col.name) & (col.name != 'V_n_1.0'):

		# voltage interpolation

		diff_V = np.abs(np.abs(col - V_ref) / V_ref)

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

		ax.grid(True)


		my_plots.append(this_plot0)


df_main.apply(lambda x: plotDifferences(df_main['x'], x, df_main['V_n_1.0']))

# sys.exit()
# legend
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
print(labels)
# print(handles)
labels = [labels[0], labels[4], labels[5], labels[3], labels[1], labels[2]]
handles = [handles[0], handles[4], handles[5], handles[3], handles[1], handles[2]]
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles, labels, title='ndofs')
# plt.ylim(0,3)
directory = f'{files_along_x}/plots/'
if not os.path.exists(directory):
	os.makedirs(directory)

outfile = re.findall(r'(02.+_vacuum)',files[0])[0]

filename =  f'{directory}{outfile}'
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
