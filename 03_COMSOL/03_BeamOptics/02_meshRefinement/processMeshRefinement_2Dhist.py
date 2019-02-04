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

remote_path = '//fs03/LTH_Neutimag/hkromer/'  # PSI
# remote_path = '/home/hkromer/01_PhD/'  # local laptop

# particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/2018-07-11/CPT_MR_timeDependent/CPT_01'.format(remote_path)
# particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/2018-07-11/CPT_MR_timeDependent/CPT_03'.format(remote_path)
# particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/2018-07-12/CPT_TD_04/'.format(remote_path)
# particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/2018-07-12/CPT_TD_05/'.format(remote_path)
particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/2018-07-11/CPT_MR_BIDIR/'.format(remote_path)




files = os.listdir(particle_data_path)
files = [f for f in files if f.endswith('.csv')]
files = [f for f in files if not f.startswith('df')]
files = ['{}/{}'.format(particle_data_path, f) for f in files]

# bins for the histogram
bins = np.linspace(0,3,100)
df_hist = pd.DataFrame()
df_FWHM = pd.DataFrame()

for file in files:

	print('Processing file {}'.format(file))
	# import data
	df = pd.DataFrame()
	df = pd.read_csv(file, skiprows=8, header=None)

	# find column headers
	c = []
	with open(file, 'r') as myfile:
		for line in myfile:
			if 'Index' in line:
				l = line.rstrip().split(',')
				c.append(l)

	myfile.close()
	
	cols = []
	for item in c[0]:

		t0 = re.findall(r'mesh_refinement', item)
		if len(t0) < 1:
			cols.append(item)


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


	# histogram
	n, bins = np.histogram(qr.values, bins=bins, density=True)
	n = np.append(n, -1)

	x = np.zeros(np.shape(bins)[0]-1)
	# get the center values for each bin
	# print(bins)
	for ii in range(0,len(bins)-1):
		# print(bins[ii])
		x[ii] = (bins[ii]+bins[ii+1])/2
	y = n[0:-1]

	# FWHM
	
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





	this_df_hist = pd.DataFrame()
	ID = re.findall(r'qz_(.+).csv', file)[0]
	this_df_hist['bins'] = bins
	this_df_hist['n'] = n
	
	this_df_hist['ID'] = ID

	mr = ID
	run = re.findall(r'\/MR_(.+)_qx', file)[0]
	df_t = pd.DataFrame([[run, X_FWHM, float(mr)]], columns = ['run', 'FWHM', 'mr'])
	# print(df_t)
	df_FWHM = df_FWHM.append(df_t, ignore_index=True)

	# # plot histogram
	# f, ax = plt.subplots()

	# # my_width=0.01
	# ax.hist(qr.values, bins=100, facecolor='g', alpha=0.75, edgecolor='black', normed=True)
	# ax.plot(x, y_fit, color='red')
	# plt.ylim(0,1.2)
	# plt.xlabel('Bin')
	# plt.ylabel('Count')
	# plt.grid(True)
	# # plt.legend(loc='best')
	# plt.tight_layout()
	# # plt.show()
	# filename =  '{}/histogram_{}_MR'.format(particle_data_path, ID)
	# # plt.savefig(filename + '.eps', dpi=1200)
	# # plt.savefig(filename + '.svg', dpi=1200)
	# plt.savefig(filename + '.png', dpi=600)
	# plt.close('all')
	# # plt.show()


	# 2D Histogram
	directory = '{}/2D_histogram_MR_{}'.format(particle_data_path,ID)
	if not os.path.exists(directory):
		os.makedirs(directory)
	nn = 0
	times = [1,2,3,4,5]
	times = ['t={}E-7'.format(t) for t in times]
	times = ['t=0'] + times
	print(times)
	# for mytime in times:
	for mytime in time_cols:
		print('Plotting time {} s'.format(mytime))
		# select only the last timestep
		this_df = df.filter(regex=mytime, axis=1)

		# compute beam radius

		this_df = this_df.iloc[idx_arrived,:]
		qx = this_df.iloc[:,0]
		median_qx = np.median(qx)
		qy = this_df.iloc[:,1]
		qz = this_df.iloc[:,2]
		qr = np.sqrt(qy**2+qz**2)

		# # histogram2d
		# # qy qz from earlier
		# H, xedges, yedges = np.histogram2d(qy,qz, bins=(80,80), normed=True)
		# f, ax = plt.subplots()
		# f.suptitle('2D Histogram time {} s'.format(mytime))
		# X, Y = np.meshgrid(xedges, yedges)
		# pcm=ax.pcolormesh(X,Y,H, cmap='jet', vmin=0, vmax=1)
		# plt.colorbar(pcm,ax=ax)
		# ax.set_aspect('equal')
		# ax.set_xlabel('y [mm]')
		# ax.set_ylabel('z [mm]')
		# plt.xlim(-3,3)
		# plt.ylim(-3,3)
		# # # np.savetxt('{}_H'.format(directory), H)
		# # # sys.exit()
		# plt.show()

		f, ax = plt.subplots()
		f.suptitle('2D Histogram time {} s \n median x position {} mm'.format(mytime, round(median_qx,2)))
		x = qy
		y = qz
		edge1 = -4
		edge2 = 4
		numedge = 150
		xedges, yedges = np.linspace(edge1, edge2, numedge), np.linspace(edge1, edge2, numedge)
		hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges), normed=True)

		xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
		yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)

		c = hist[xidx, yidx]
		p = ax.scatter(x, y, c=c, cmap='jet',s=20, edgecolors='black')
		plt.colorbar(p, ax=ax)
		# plt.show()

		
		# plt.tight_layout()
		# plt.show()
		filename =  '{}/{}_2Dhistogram_{}_MR_time_{}'.format(directory, nn, ID, mytime)
		# plt.savefig(filename + '.eps', dpi=1200)
		# plt.savefig(filename + '.svg', dpi=1200)
		plt.savefig(filename + '.png', dpi=600)
		plt.close('all')
		# plt.show()
		nn = nn + 1


df_FWHM = df_FWHM.sort_values(by=['mr'])
print(df_FWHM)
# df_FWHM.to_csv('{}/df_FWHM.csv'.format(particle_data_path))