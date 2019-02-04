import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

# PSI computer
# remote_path = '//fs03/LTH_Neutimag/hkromer/'

# local computer
# remote_path = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
# 03.new_chamber/07.mesh_refinement.right_alignment/meshRefinement/'
remote_path = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
02.current_chamber/01.setup_CPT/lineData/'

# files_along_x = '{}/02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/\
# IGUN_geometry\2018-09-19_comsolGeometry\mesh_refinement'.format(remote_path)

files_along_x = f'{remote_path}'

# Do not forget to comment out the reference df!

files = os.listdir(files_along_x)
# files = [f for f in files if f.endswith('.csv')]
files = [f for f in files if f.endswith('.csv')]
# print(files)
# sys.exit()

# process files
# (1): load x value and value of electric field (es.normE) as well as potential field (V)
df = pd.DataFrame()
for file in files:
	print('Processing file: {}'.format(file))
	if len(df) < 1:
		df = pd.read_csv('{}/{}'.format(files_along_x, file),
			header=None, skiprows=9)
		# print(df)
		colname = ['x', 'y', 'z', 'ElField']  # mm, mm, mm, mm, kV/mm
		df.columns = colname
		fname = re.findall(r'(.+).csv', file)[0]
		run = re.findall(r'\.(\d\d\d)\.', file)[0]
		# print(run)
		# hmax = re.findall(r'hmax(\d*)', file)[0]
		hmax = run
		df['ID'] = fname
		df['run'] = run
		df['hmax'] = int(hmax)
		df = df.sort_values(by=['x'])
		df = df.reset_index()
	else:
		this_df = pd.read_csv('{}/{}'.format(files_along_x, file),
			header=None, skiprows=9)
		colname = ['x', 'y', 'z', 'ElField']  # mm, mm, mm, kV/mm
		this_df.columns = colname
		fname = re.findall(r'(.+).csv', file)[0]
		run = re.findall(r'\.(\d\d\d)\.', file)[0]
		hmax = run
		# hmax = re.findall(r'hmax(\d*)', file)[0]
		this_df['ID'] = fname
		this_df['run'] = run
		this_df['hmax'] = int(hmax)
		# print(df.head())
		# print(this_df.head())
		this_df = this_df.sort_values(by=['x'])
		this_df = this_df.reset_index()
		df = df.append(this_df, sort=True)


df = df.sort_values(by=['ID'])
# asser that no nonzero entries in the df
assert len(df[df.isnull().any(axis=1)]) == 0



# check if the min and max of the x are the same
def minMaxX(df):
	my_ID = df.ID.unique()[0]
	print('Processing ID: {}'.format(my_ID))
	print(np.max(df['x']), np.min(df['x']))


# df.groupby('ID').apply(lambda x: minMaxX(x))


# group by ID and compute the differences in the profile
# reference dataframe with the finest mesh

# comment one of them !
# ref_df = df[ df['ID'] == '101_mr_0.5' ]   # new target
ref_df = df[df['hmax'] == 1]  # old target

E_ref_interp = interp1d(ref_df['x'], ref_df['ElField'],
	fill_value='extrapolate')


# PLOT ACCELERATOR COLUMN

my_plots = []

X = np.linspace(0, 95, 400)  # query points
E_ref = E_ref_interp(X)

f, ax = plt.subplots(figsize=(6, 8))
f.suptitle('Mesh refinement in accelerator column')


def plotDifferences(df):
	my_ID = df.ID.unique()[0]
	hmax = df.hmax.unique()[0]
	if my_ID != ref_df.ID.unique()[0]:

		# el field interpolation
		E_interp = interp1d(df['x'], df['ElField'], fill_value='extrapolate')

		this_E = E_interp(X)

		diff_E = np.abs(np.abs(this_E - E_ref) / E_ref)

		# get label
		lbl = hmax

		this_plot1, = ax.plot(X, 100 * diff_E, label=lbl)
		ax.set_xlabel('x position [mm]')
		ax.set_ylim(0, 50)
		ax.set_ylabel('rel. diff. electric field [%]')
		ax.grid(True)

		my_plots.append(this_plot1)


# print(df)
df.groupby('ID').apply(lambda x: plotDifferences(x))

# legend
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
# print(labels)
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles[0:], labels[0:], title='hmax')

filename = '{}/mr_accelerator_column'.format(files_along_x)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')


# PLOT ION EXTRACTION REGION

X = np.linspace(0, 11, 200)  # query points

E_ref = E_ref_interp(X)

f, ax = plt.subplots(figsize=(6, 8))
f.suptitle('Mesh refinement in extraction region column')


def plotDifferences(df):
	my_ID = df.ID.unique()[0]
	hmax = df.hmax.unique()[0]
	if my_ID != ref_df.ID.unique()[0]:
		# el field interpolation
		E_interp = interp1d(df['x'], df['ElField'], fill_value='extrapolate')

		this_E = E_interp(X)

		diff_E = np.abs(np.abs(this_E - E_ref) / E_ref)

		# get label
		lbl = hmax

		ax.plot(X, 100 * diff_E, label=lbl)
		ax.set_xlabel('x position [mm]')
		ax.set_ylim(0, 50)
		ax.set_ylabel('rel. diff. electric field [%]')
		ax.grid(True)


df.groupby('ID').apply(lambda x: plotDifferences(x))


# legend
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax.legend(handles[0:], labels[0:], title='hmax')

# plt.show()
filename = '{}/mr_extraction_region'.format(files_along_x)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
