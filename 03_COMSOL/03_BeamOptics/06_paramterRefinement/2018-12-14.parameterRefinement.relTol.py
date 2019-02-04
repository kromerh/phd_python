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
remote_path = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
03.new_chamber/12.relTol/lineData/'

df_LUT = pd.read_csv(f'{remote_path}/LUT.run_vs_relTol.txt',
							delimiter=r'\s*', header=None)
df_LUT.columns = ['id', 'relTol']

# print(df_LUT)
# sys.exit()

files_along_x = f'{remote_path}'

# Do not forget to comment out the reference df!

files = os.listdir(files_along_x)
# files = [f for f in files if f.endswith('.csv')]
files = [f for f in files if f.endswith('.csv')]
# print(files)
# sys.exit()


df_main = pd.DataFrame()
for file in files:
	print('Processing file: {}'.format(file))

	df = pd.read_csv('{}/{}'.format(files_along_x, file),
		header=None, skiprows=9)
	# print(df)
	colname = ['x', 'y', 'z', 'ElField']  # mm, mm, mm, mm, kV/mm
	df.columns = colname
	fname = re.findall(r'(.+).csv', file)[0]
	run = re.findall(r'(\d\d\d).line', file)[0]
	# hmax = re.findall(r'hmax(\d*)', file)[0]
	df['ID'] = fname
	df['run'] = int(run)
	# df['hmax'] = int(hmax)
	df = df.sort_values(by=['x'])
	df = df.reset_index()

	df_main = df_main.append(df)


df_main = df_main.sort_values(by=['run'])

# print(df_main)
# sys.exit()

# check if the min and max of the x are the same
def minMaxX(df):
	my_ID = df.run.unique()[0]
	print('Processing ID (max x, min x): {}'.format(my_ID))
	print(np.max(df['x']), np.min(df['x']))


# df_main.groupby('run').apply(lambda x: minMaxX(x))
# sys.exit()


# comment one of them !
ref_df = df_main[df_main['run'] == 101]  # reference is 1e-5
# print(ref_df)
# sys.exit()

E_ref_interp = interp1d(ref_df['x'], ref_df['ElField'],
	fill_value='extrapolate')


# PLOT ACCELERATOR COLUMN

my_plots = []

def moving_average(a, n):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


X_orig = np.linspace(0, 85, 200)  # query points
E_ref = E_ref_interp(X_orig)
N_avg = 10
X = moving_average(X_orig, n=N_avg)
# print(len(X), X)
# sys.exit()


f, ax = plt.subplots(figsize=(8, 8))
f.suptitle('Relative tolerance refinement in accelerator column')


def myplotDifferences(df):
	my_ID = df.run.unique()[0]
	relTol = df_LUT[df_LUT['id'] == my_ID].relTol.values[0]
	id = int(df_LUT[df_LUT['id'] == my_ID].id.values[0])

	# print(my_ID, hmax)

	# print(hmax)
	if my_ID != ref_df.run.unique()[0]:
		if my_ID != ref_df.run.unique()[0]:
			if my_ID == 2:  # mesh to be chosen for the final runs
				my_linestyle = '-'
				my_lw = 2.0
			else:
				my_linestyle = '--'
				my_lw = 1.0

		# el field interpolation
		E_interp = interp1d(df['x'], df['ElField'], fill_value='extrapolate')

		this_E = E_interp(X_orig)

		Y = np.abs(np.abs(this_E - E_ref) / E_ref)
		Y[np.isinf(Y)] = 0
		Y[np.isnan(Y)] = 0
		# print(my_ID, Y)

		# print(Y)
		# get label
		lbl = relTol

		# print('before', len(Y), len(X))

		Y = 100 * moving_average(Y, n=N_avg)
		# print('after', len(Y), len(X))

		ax.plot(X, Y, label=lbl, linestyle=my_linestyle, linewidth=my_lw)

		ax.set_xlabel('x position [mm]')
		ax.set_ylim(0, 1)
		ax.set_ylabel('rel. diff. electric field [%]')
		ax.grid(True)

		# my_plots.append(this_plot1)


# print(df)
df_main.groupby('run').apply(lambda x: myplotDifferences(x))
# sys.exit()
# legend
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
# print(labels)
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
# ax.legend(handles[1:], labels[1:], title='hmax')
ax.legend(handles, labels, title='relTol')


filename = '{}/relTol_accelerator_column'.format(files_along_x)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')

# sys.exit()
# PLOT ION EXTRACTION REGION

X_orig = np.linspace(0, 11, 200)  # query points

E_ref = E_ref_interp(X_orig)
N_avg = 10
X = moving_average(X_orig, n=N_avg)

f, ax = plt.subplots(figsize=(8, 8))
f.suptitle('Mesh refinement in extraction region column')


def plotDifferences_extr(df):
	my_ID = df.run.unique()[0]
	# print(my_ID)
	relTol = df_LUT[df_LUT['id'] == my_ID].relTol.values[0]
	id = int(df_LUT[df_LUT['id'] == my_ID].id.values[0])


	# print(hmax)
	if my_ID != ref_df.run.unique()[0]:
		if my_ID == 2:  # mesh to be chosen for the final runs
			my_linestyle = '-'
			my_lw = 2.0
		else:
			my_linestyle = '--'
			my_lw = 1.0

		# el field interpolation
		E_interp = interp1d(df['x'], df['ElField'], fill_value='extrapolate')

		this_E = E_interp(X_orig)

		Y = np.abs(np.abs(this_E - E_ref) / E_ref)
		Y[np.isinf(Y)] = 0
		Y[np.isnan(Y)] = 0

		# get label
		lbl = relTol
		Y = 100 * moving_average(Y, n=N_avg)

		ax.plot(X, Y, label=lbl, linestyle=my_linestyle, linewidth=my_lw)
		ax.set_xlabel('x position [mm]')
		ax.set_ylim(0, 1)
		ax.set_ylabel('rel. diff. electric field [%]')
		ax.grid(True)


df_main.groupby('run').apply(lambda x: plotDifferences_extr(x))


# legend
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
print(labels, handles)
ax.legend(handles, labels, title='relTol')


# plt.show()
filename = '{}/relTol_extraction_region'.format(files_along_x)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
