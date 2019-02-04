import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import kde
from scipy import optimize
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot, transforms

pd.set_option("display.max_columns", 300)

foldername = '13.nIter'
# remote_path = '//fs03/LTH_Neutimag/hkromer/'
# Local
remote_path = f'/Users/hkromer/02_PhD/02_Data/01_COMSOL/\
01_IonOptics/03.new_chamber/{foldername}/particleData/'

COMSOL_data_file_path = remote_path

COMSOL_files = os.listdir(COMSOL_data_file_path)
# COMSOL_files = [f for f in COMSOL_files if 'particleData' in f]
COMSOL_files = [f for f in COMSOL_files if f.endswith('.csv')]
COMSOL_files = [f'{COMSOL_data_file_path}{f}' for f in COMSOL_files]

df_output = pd.DataFrame()

f, ax = plt.subplots(figsize=(8, 8))
f.suptitle('Beam radius versus timestep')
for COMSOL_data_file in COMSOL_files:

	runfile = re.findall(r'[^/]+(?=/$|$)', COMSOL_data_file)[0]
	print(f'Doing file {runfile}')
	# import the particle data file
	df = pd.read_csv(COMSOL_data_file, skiprows=8, header=None)
	df_FWHM = pd.DataFrame()  # this is the output dataframe that contains the FWHM
	# find column headers
	c = []
	with open(COMSOL_data_file, 'r') as myfile:
		for line in myfile:
			if 'Index' in line:
				l = line.rstrip().split(',')
				c.append(l)

	myfile.close()
	cols = c[0]
	# print(c)
	# get the time stepping
	# extract t=... from the cols
	my_cols = []
	for ii in range(0,len(cols)):
		col = cols[ii]

		t0 = re.findall(r'(t=.*)', col)
		if len(t0) > 0:
			my_cols.append(t0[0])
		else:
			my_cols.append('noTimestamp')

	# timeStep = my_cols[4]
	time_cols = (pd.Series(item for item in my_cols)).unique()[1:] # drop the timestamp

	#set column header of df
	cols[0] = 'particleindex'
	df.columns = cols

	# go through the time steps and compute the max qr
	lst_max_qr = []
	for my_time in time_cols:
		this_df = df.filter(regex=my_time, axis=1)
		qy = this_df.iloc[:,1]
		qz = this_df.iloc[:,2]

		qr = np.sqrt(qy**2+qz**2)
		max_qr = np.max(qr)
		lst_max_qr.append(max_qr)

	df_qr = pd.DataFrame({'time': time_cols, 'max_qr': lst_max_qr,
							'runfile': runfile})
	X = df_qr.time.values
	X = [ii for ii in range(0, len(X))]
	Y = df_qr.max_qr.values
	lbl = re.findall(r'\.(\d\d\d)\.', runfile)[0]
	ax.plot(X, Y, label=lbl)
	ax.set_xlabel('Timestep')
	# ax.set_ylim(0, 5)
	ax.set_ylabel('Maximum qr [mm]')
	ax.grid(True)


print(df_qr)
# legend
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
# print(labels, handles)
ax.legend(handles, labels, title='run')


# plt.show()
filename = '{}/nIter_max_qr_vs_timestep'.format(remote_path)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
