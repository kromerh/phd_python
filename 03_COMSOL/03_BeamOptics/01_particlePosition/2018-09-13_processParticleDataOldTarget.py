import pandas as pd
pd.set_option("display.max_columns",300)
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


# single file processing
# remote_path = '//fs03/LTH_Neutimag/hkromer/'

# COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data_flat_meniscus/particle_data/'.format(remote_path)
COMSOL_data_file_path = 'E:/COMSOL/03_BeamOptics/01_COMSOL_files/01_old_target_reproduce_results/IGUN_geometry/2018-09-13_comsol/particle_data/'


# all csv files in directory
csv_files = os.listdir(COMSOL_data_file_path)
csv_files = [f for f in csv_files if f.endswith('.csv')]
csv_files = [f'{COMSOL_data_file_path}/{f}' for f in csv_files]
# print(csv_files)
# sys.exit()

# only one file
# COMSOL_data_file = '{}2018-09-13_01_particleData.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
# COMSOL_data_file = '{}2018-09-13_01_BIDIR_current_noParticleInteractions_particleData.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
# COMSOL_data_file = '{}2018-09-13_01_BIDIR_current_particleData.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop

# COMSOL_data_file = '{}2018-09-13_01_BIDIR_particleData.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
# COMSOL_data_file = '{}2018-09-13_01_current_particleData.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
# COMSOL_data_file = '{}2018-09-13_01_current_noParticleInteractions_particleData.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop


for COMSOL_data_file in csv_files:
	# import the particle data file
	df = pd.read_csv(COMSOL_data_file, skiprows=8, header=None)

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

	# check which particles have arrived at the target
	# get the latest timestamp
	df_last = df.filter(regex=time_cols[-1], axis=1)

	# length: total number of particles
	n_total = len(df_last)

	# only those particles that have made it to the target: 10 mm in +x direction
	# for dist in [1,5,10,80]:
	dist_min = 10
	dist_max = 90
	df_arrived = df_last[ df_last.iloc[:,0] > dist_min ]
	df_arrived = df_last[ df_last.iloc[:,0] < dist_max ]
	n_arrived = len(df_arrived)
	# percent of those that have arrived
	perc_arrived = round((n_arrived/n_total)*100.0,2)

	print('{}% of the initial {} particles have arrived at the target (x > {} mm and x < {} mm).'.format(perc_arrived, n_total, dist_min, dist_max))
	# print(df.head())
	# print(sys.exit())
	# index of the particle that have arrived at the target
	idx_arrived = df_arrived.index.tolist()


	# select only the last timestep
	this_df = df.filter(regex=time_cols[-1], axis=1)

	# compute beam radius

	this_df = this_df.iloc[idx_arrived,:]

	qy = this_df.iloc[:,1]
	qz = this_df.iloc[:,2]

	fname = re.findall(r'/(2018-09-13_.+).csv',COMSOL_data_file)[0]
	# print(fname)
	directory = '{}/{}_2D_histogram_lastTimestep'.format(COMSOL_data_file_path,fname)
	if not os.path.exists(directory):
		os.makedirs(directory)


	time_cols = time_cols[::5]
	df_times = pd.DataFrame()
	df_times['time'] = time_cols
	df_times.to_csv(f'{directory}/df_times.csv')
	# mytime = time_cols[-1]
	n = 0
	df_median_qx = pd.DataFrame()
	for mytime in time_cols:

		print('Creating plot for time time {} s'.format(mytime))
		# select only the last timestep
		this_df = df.filter(regex=mytime, axis=1)

		# compute beam radius

		this_df = this_df.iloc[idx_arrived,:]
		qx = this_df.iloc[:,0]
		median_qx = np.median(qx)
		df_median_qx = df_median_qx.append(pd.Series([mytime,median_qx], index = ['time','median_qx']),ignore_index=True)
		# print(df_median_qx)
		qy = this_df.iloc[:,1]
		qz = this_df.iloc[:,2]
		qr = np.sqrt(qy**2+qz**2)

		nbins = 250
		x = qy
		y = qz
		data = np.vstack([qy, qz])
		k = kde.gaussian_kde(data)
		# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
		xi, yi = np.mgrid[-3:3:nbins*1j, -3:3:nbins*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		# scale between 0 and 1
		zi = (zi - np.min(zi))/(np.max(zi)-np.min(zi))
		# print(zi)


		f = plt.figure(1, figsize=(7, 7))
		# plt.title('KDE Gaussian on target for run \n {}'.format(type_file))
		nullfmt = NullFormatter()         # no labels

		# definitions for the axes
		left, width = 0.05, 0.65
		bottom, height = 0.05, 0.65
		bottom_h = left_h = left + width + 0.02

		rect_scatter = [left, bottom, width, height]
		rect_histx = [left, bottom_h, width, 0.2]
		rect_histy = [left_h, bottom, 0.2, height]

		axScatter = plt.axes(rect_scatter)

		axHistx = plt.axes(rect_histx)
		axHisty = plt.axes(rect_histy)

		# no labels
		axHistx.xaxis.set_major_formatter(nullfmt)
		axHisty.yaxis.set_major_formatter(nullfmt)
		axHistx.grid(True)
		axHisty.grid(True)


		p = axScatter.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.jet)

		# contours = axScatter.contour(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Blues, levels=[my_lvl])
		# plt.clabel(contours, inline=True, fontsize=8)
		axScatter.set_facecolor('#000080ff')
		plt.colorbar(p)

		# compute FWHM for all x and y histograms
		# select the largest FWHM
		lim = 3
		axScatter.set_xlim((-lim, lim))
		axScatter.set_ylim((-lim, lim))

		# compute FWHM for all points parallel to the x and y axis
		qry_eval = np.linspace(-lim,lim,100)
		eval_x = [k.evaluate([x,0])[0] for x in qry_eval] 
		eval_y = [k.evaluate([0,y])[0] for y in qry_eval]

		# # print(kint)
		# # 
		df_res = pd.DataFrame()
		df_res['qry_eval'] = qry_eval
		df_res['eval_x'] = eval_x
		df_res['eval_y'] = eval_y
		# # df_res['kint_1'] = kint
		# df_res['type_file'] = type_file
		# # df_res['contour_level'] = my_lvl
		# fwhm_x = calculateFWHM(qry_eval,df_res['eval_x'])
		# fwhm_y = calculateFWHM(qry_eval,df_res['eval_y'])


		# first of all, the base transformation of the data points is needed
		base = pyplot.gca().transData
		rot = transforms.Affine2D().rotate_deg(270)
		axScatter.plot([-lim, lim], [0, 0], color='black', linestyle='dashed')
		axScatter.plot([0, 0], [-lim, lim], color='black')

		axHistx.plot(df_res['qry_eval'].values, df_res['eval_x'], c='black', linestyle='dashed')
		axHisty.plot(df_res['qry_eval'].values, df_res['eval_y'].values[::-1], c='black', transform= rot + base)

		axHistx.set_xlim(axScatter.get_xlim())
		axHisty.set_ylim(axScatter.get_ylim())
		plt.xlabel('y [mm]')
		plt.ylabel('z [mm]')
		# f.tight_layout()
		# df_x_FWHM, df_y_FWHM = getLargestFWHM(k, lim)

		# df_x_FWHM.to_csv('{}/df_x_FWHM.csv'.format(directory))
		# df_y_FWHM.to_csv('{}/df_y_FWHM.csv'.format(directory))

		# plt.show()

		filename =  '{}/2D_histogram_timestep_{}'.format(directory,n)
		n = n + 1
		# print(nn)
		# plt.savefig(filename + '.eps', dpi=1200)
		# plt.savefig(filename + '.svg', dpi=1200)
		plt.savefig(filename + '.png', dpi=600)
		plt.close('all')
		# plt.show()
				
	df_median_qx.to_csv(f'{directory}/df_median_qx.csv')
