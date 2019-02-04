import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import itertools


# Create a function which returns a Gaussian (normal) distribution.
def gauss(x, *p):
	a, b, c, d = p
	y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) + d

	return y

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# compute the FWHM
def getFWHM(x, *popt):
	X = np.linspace(np.min(x), np.max(x), 1000)  # expand x-space
	# print(X)
	Y = gauss(X, *popt)

	# maximum
	Y_max = np.max(Y)
	Y_arg_max = np.argmax(Y) # maximum Y value position
	X_max = X[Y_arg_max] # maximum X value

	# take only the positive range from the maximum on
	Y = Y[Y_arg_max-1:]
	X = X[Y_arg_max-1:]
	Y_FWHM = find_nearest(Y, Y_max/2)
	Y_FWHM_pos = np.argwhere(Y == Y_FWHM)
	X_FWHM = X[Y_FWHM_pos]
	# print('Y_max={}'.format(Y_max))
	# print('X_max={}'.format(X_max))
	# print('Y_FWHM={}'.format(Y_FWHM))
	# print('X_FWHM={}'.format(X_FWHM))

	return X_FWHM

remote_path = '//fs03/LTH_Neutimag/hkromer/'

particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/particle_data'.format(remote_path)

files = os.listdir(particle_data_path)
files = [f for f in files if f.endswith('.csv')]

# process files
# (1): load x value and value of electric field (es.normE) as well as potential field (V)
df = pd.DataFrame()

# works only for 6 csv files
f, axarr = plt.subplots(3,2, figsize=(7,8.5))
jj = 0
kk = 0
for file in files:
	# read the file into dataframe
	df = pd.read_csv('{}/{}'.format(particle_data_path,file), skiprows=7)
	cols = df.columns.tolist()  # columns as list
	# print(df.columns)

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
	

	# check which particles have arrived at the target
	# get the latest timestamp
	df_last = df.filter(regex=time_cols[-1], axis=1)

	# length: total number of particles
	n_total = len(df_last)

	# only those particles that have made it to the target: 10 mm in +x direction
	# for dist in [1,5,10,80]:
	dist = 10
	df_arrived = df_last[ df_last.iloc[:,0] > dist ]
	n_arrived = len(df_arrived)
	# percent of those that have arrived
	perc_arrived = round((n_arrived/n_total)*100.0,2)

	print('{}% of the initial {} particles have arrived at the target (x > {} mm).'.format(perc_arrived, n_total, dist))

	# index of the particle that have arrived at the target
	idx_arrived = df_arrived.index.tolist()


	# select only the last timestep

	this_df = df.filter(regex=time_cols[-1], axis=1)

	# compute beam radius

	this_df = this_df.iloc[idx_arrived,:]

	qy = this_df.iloc[:,1]
	qz = this_df.iloc[:,2]
	qr = np.sqrt(qy**2+qz**2)
	

	
	# print(jj, kk)
	axarr[jj][kk].set_title('{}'.format(file))

	# plot qr (beam radius)
	num_bins = 100
	n, bins, patches = axarr[jj][kk].hist(qr.values, num_bins, facecolor='g', alpha=0.75, edgecolor='black', normed=True)



	# add FWHM

	# initialize x from the histogram bins
	x = np.zeros(np.shape(bins)[0]-1)
	# get the center values for each bin
	for ii in range(0,len(bins)-1):
		x[ii] = (bins[ii]+bins[ii+1])/2


	y = n  # from histogram
	# Use curve_fit to fit the gauss function to our data. Use the
	# unperturbed p_initial as our initial guess.
	
	m = [1e-2, 5e-2, 1e-1, 5e-1, 0, 1, 5, 1e1]

	for mm in list(itertools.product(m,m,m,m)):
		p_initial = [mm[0],mm[1],mm[2],mm[3]]
		popt, pcov = curve_fit(gauss, x, y, p0=p_initial, maxfev=100000)
		y_fit = gauss(x, *popt)
		X_FWHM = getFWHM(x, *popt)[0]
		if (popt[0] < 10) and (popt[0] > 1e-3) and (popt[1] < 10) and (len(X_FWHM) == 1) and (X_FWHM[0] < np.max(x)):
			# print(X_FWHM)
			X_FWHM = X_FWHM[0]
			break

	print('X_FWHM={}'.format(X_FWHM))
	print('Parameter fit for gauss: {}'.format(popt))
	# Generate y-data based on the fit.
	
	
	axarr[jj][kk].plot(x, y_fit, color='red')

	axarr[jj][kk].set_xlabel('Beam radius [mm]')
	axarr[jj][kk].set_ylabel('Counts')
	axarr[jj][kk].grid(True)
	axarr[jj][kk].set_ylim(0,1.2)
	axarr[jj][kk].set_xlim(0,3.0)

	txt = 'FWHM = {} mm'.format(round(X_FWHM,2))
	axarr[jj][kk].text(2.0, 0.5, txt, fontsize=10, color='red')




	if (kk == 0):
		kk = kk + 1
	elif kk == 1:
		kk = 0
		jj = jj + 1
	


	# print('Processing file: {}'.format(file))
	# if len(df) < 1:
	# 	df = pd.read_csv('{}/{}'.format(files_along_x,file), header=None, skiprows=9)
	# 	colname = ['x', 'y', 'z', 'ElField', 'V']  # mm, mm, mm, V/m, V
	# 	df.columns = colname
	# 	fname = re.findall(r'(.+).csv', file)[0]
	# 	df['ID'] = fname
	# 	df = df.sort_values(by=['x'])
	# 	df = df.reset_index()
	# else:
	# 	this_df = pd.read_csv('{}/{}'.format(files_along_x,file), header=None, skiprows=9)
	# 	colname = ['x', 'y', 'z', 'ElField', 'V']  # mm, mm, mm, V/m, V
	# 	this_df.columns = colname
	# 	fname = re.findall(r'(.+).csv', file)[0]
	# 	this_df['ID'] = fname
	# 	# print(df.head())
	# 	# print(this_df.head())
	# 	this_df = this_df.sort_values(by=['x'])
	# 	this_df = this_df.reset_index()
	# 	df = df.append(this_df)

plt.tight_layout()
# plt.show()
filename =  '{}/mr_beamRadius_histograms'.format(particle_data_path)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')