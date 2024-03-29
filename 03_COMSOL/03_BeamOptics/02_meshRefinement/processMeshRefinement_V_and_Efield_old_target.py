import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

remote_path = '//fs03/LTH_Neutimag/hkromer/'

files_along_x = '{}/02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget\IGUN_geometry\2018-09-19_comsolGeometry\mesh_refinement'.format(remote_path)

# Do not forget to comment out the reference df!

files = os.listdir(files_along_x)
# files = [f for f in files if f.endswith('.csv')]
files = [f for f in files if f.endswith('.txt')]
files = [f for f in files if f.startswith('V')]
# print(files)
# process files
# (1): load x value and value of electric field (es.normE) as well as potential field (V)
df = pd.DataFrame()
for file in files:
	print('Processing file: {}'.format(file))
	if len(df) < 1:
		df = pd.read_csv('{}/{}'.format(files_along_x,file), header=None, skiprows=9, delimiter=r'\s+')
		# print(df)
		colname = ['x', 'y', 'z', 'V', 'ElField']  # mm, mm, mm, V/m, V
		df.columns = colname
		fname = re.findall(r'(.+).txt', file)[0]
		df['ID'] = fname
		df = df.sort_values(by=['x'])
		df = df.reset_index()
	else:
		this_df = pd.read_csv('{}/{}'.format(files_along_x,file), header=None, skiprows=9, delimiter=r'\s+')
		colname = ['x', 'y', 'z', 'V', 'ElField']  # mm, mm, mm, V/m, V
		this_df.columns = colname
		fname = re.findall(r'(.+).txt', file)[0]
		this_df['ID'] = fname
		# print(df.head())
		# print(this_df.head())
		this_df = this_df.sort_values(by=['x'])
		this_df = this_df.reset_index()
		df = df.append(this_df)


df = df.sort_values(by=['ID'])

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
ref_df = df[ df['ID'] == 'V_along_x_MR_0.9' ]   # old target


V_ref_interp = interp1d(ref_df['x'], ref_df['V'], fill_value='extrapolate')
E_ref_interp = interp1d(ref_df['x'], ref_df['ElField'], fill_value='extrapolate')


# PLOT ACCELERATOR COLUMN

my_plots = []

X = np.linspace(0, 85, 200)  # query points
V_ref = V_ref_interp(X)
E_ref = E_ref_interp(X)

f, axarr = plt.subplots(2, figsize=(6,8))
f.suptitle('Mesh refinement in accelerator column')
def plotDifferences(df):
	my_ID = df.ID.unique()[0]
	if my_ID != ref_df.ID.unique()[0]:

		# voltage interpolation
		V_interp = interp1d(df['x'], df['V'], fill_value='extrapolate')
		# el field interpolation
		E_interp = interp1d(df['x'], df['ElField'], fill_value='extrapolate')


		this_V = V_interp(X)
		this_E = E_interp(X)

		diff_V = np.abs(np.abs(this_V-V_ref)/V_ref)
		diff_E = np.abs(np.abs(this_E-E_ref)/E_ref)

		# get label
		lbl = re.findall(r'_MR_(.+)$', my_ID)[0]

		# plot
		this_plot0, = axarr[0].plot(X, 100*diff_V, label=lbl)
		axarr[0].set_xlabel('x position [mm]')
		axarr[0].set_ylabel('rel. diff. potential field [%]')
		axarr[0].set_ylim(0,5)
		axarr[0].grid(True)

		this_plot1, = axarr[1].plot(X, 100*diff_E, label=lbl)
		axarr[1].set_xlabel('x position [mm]')
		axarr[1].set_ylim(0,5)
		axarr[1].set_ylabel('rel. diff. electric field [%]')
		axarr[1].grid(True)

		my_plots.append(this_plot0)

df.groupby('ID').apply(lambda x: plotDifferences(x))

# legend
handles, labels = axarr[0].get_legend_handles_labels()
# sort both labels and handles by labels
# print(labels)
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
axarr[0].legend(handles, labels)

filename =  '{}/mr_accelerator_column'.format(files_along_x)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')








# PLOT ION EXTRACTION REGION

X = np.linspace(-1, 2, 200)  # query points
V_ref = V_ref_interp(X)
E_ref = E_ref_interp(X)

f, axarr = plt.subplots(2, figsize=(6,8))
f.suptitle('Mesh refinement in extraction region column')
def plotDifferences(df):
	my_ID = df.ID.unique()[0]
	if my_ID != ref_df.ID.unique()[0]:

		# voltage interpolation
		V_interp = interp1d(df['x'], df['V'], fill_value='extrapolate')
		# el field interpolation
		E_interp = interp1d(df['x'], df['ElField'], fill_value='extrapolate')


		this_V = V_interp(X)
		this_E = E_interp(X)

		diff_V = np.abs(np.abs(this_V-V_ref)/V_ref)
		diff_E = np.abs(np.abs(this_E-E_ref)/E_ref)

		# get label
		lbl = re.findall(r'_MR_(.+)$', my_ID)[0]

		# plot
		axarr[0].plot(X, 100*diff_V, label=lbl)
		axarr[0].set_xlabel('x position [mm]')
		axarr[0].set_ylabel('rel. diff. potential field [%]')
		axarr[0].set_ylim(0,5)
		# axarr[0].legend(loc='best')
		axarr[0].grid(True)

		axarr[1].plot(X, 100*diff_E, label=lbl)
		axarr[1].set_xlabel('x position [mm]')
		axarr[1].set_ylim(0,5)
		axarr[1].set_ylabel('rel. diff. electric field [%]')
		axarr[1].grid(True)

df.groupby('ID').apply(lambda x: plotDifferences(x))


# legend
handles, labels = axarr[0].get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
axarr[0].legend(handles, labels)


# plt.show()
filename =  '{}/mr_extraction_region'.format(files_along_x)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
