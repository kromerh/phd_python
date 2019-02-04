import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt


remote_path = '//fs03/LTH_Neutimag/hkromer/'

# create a csv file for each timestamp


# This only works for single COMSOL solutions, no parameters!


# first column: particle index
# second column: qx
# third column: qy
# fourth column: qz
def createCSVforEachTimestep(COMSOL_data_file, output_path):
	# read the file into dataframe
	df = pd.read_csv(COMSOL_data_file, skiprows=7)
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
	# print(idx_arrived)
	
	# dataframe with the radii
	df_qr = pd.DataFrame()
	for ii in range(0,len(time_cols)):
		# get the subset of columns
		this_df = df.filter(regex=time_cols[ii], axis=1)
		# select only those particles that have arrived at the target	
		this_df = this_df.iloc[idx_arrived,:]

		qy = this_df.iloc[:,1]
		qz = this_df.iloc[:,2]
		qr = np.sqrt(qy**2+qz**2)
		df_qr[time_cols[ii]] = qr



	# 	# create parent directory if not exists
	folder = re.findall(r'/(\w+).csv$',COMSOL_data_file)[0]
	directory = '{}{}'.format(COMSOL_data_file_path, folder)
	if not os.path.exists(directory):
		os.makedirs(directory)

	# calculate column mean
	df_qr_mean = df_qr.mean(axis = 0)

	# calculate column std
	df_qr_std = df_qr.std(axis = 0)

	# calculate column median
	df_qr_median = df_qr.median(axis = 0)

	# calculate column 95th percentile
	df_qr_95perc = pd.DataFrame(index=df_qr_mean.index)
	df_qr_95perc['95perc'] = np.percentile(df_qr, 95, axis = 0)
	
	# print(df_qr_95perc)

	df_qx = pd.DataFrame()
	for ii in range(0,len(time_cols)):
		# get the subset of columns
		this_df = df.filter(regex=time_cols[ii], axis=1)
		# select only those particles that have arrived at the target	
		this_df = this_df.iloc[idx_arrived,:]

		df_qx[time_cols[ii]] = this_df.iloc[:,0]

	# calculate x position median
	df_qx_median = df_qx.median(axis = 0)

	# export mean, std and median dataframes
	# UNIT IS MM!
	df_qr_mean.to_csv('{}/qr_mean.csv'.format(directory))
	df_qr_std.to_csv('{}/qr_std.csv'.format(directory))
	df_qr_median.to_csv('{}/qr_median.csv'.format(directory))

	df_qr_95perc.to_csv('{}/qr_95perc.csv'.format(directory))

	df_qx_median.to_csv('{}/qx_median.csv'.format(directory))
	


	# plot beam radius
	ii_time = 10  # which time in the timecol
	for ii_time in [5,10,12,30,50,80,-1]:
	# for ii_time in [50,-1]:
		# plot a histogram
		f, axarr = plt.subplots(2, figsize=(6,8))

		print('Plot for time: {}s.'.format(time_cols[ii_time]))

		f.suptitle('Beam radius and beam diameter for time: {} s \n {}'.format(time_cols[ii_time], folder))

		# get the df for the qx position
		this_df = df.filter(regex=time_cols[ii_time], axis=1)
		this_df = this_df.iloc[idx_arrived,:]

		# plot qr (beam radius)
		num_bins = 100
		axarr[0].hist(df_qr.iloc[:,ii_time].values, num_bins, facecolor='g', alpha=0.75, edgecolor='black', normed=True)
		axarr[0].set_xlabel('Beam radius [mm]')
		axarr[0].set_ylabel('Counts')
		axarr[0].grid(True)

		# average and std qr, 90th percentile
		av = np.mean(df_qr.iloc[:,ii_time].values)
		std = np.std(df_qr.iloc[:,ii_time].values)
		ylim = axarr[0].get_ylim()
		axarr[0].plot([av, av], [0, ylim[1]], color='red')
		axarr[0].plot([av+std, av+std], [0, ylim[1]], color='orange')
		axarr[0].plot([av-std, av-std], [0, ylim[1]], color='orange')
		perc90 = np.percentile(df_qr.iloc[:,ii_time].values, 95)
		axarr[0].plot([perc90, perc90], [0, ylim[1]], '--', color='blue')


		# plot qx position
		num_bins = 1000
		# print(this_df.iloc[:,0])
		axarr[1].hist(this_df.iloc[:,0].values, num_bins, facecolor='r', alpha=0.75, edgecolor='black', normed=True)
		axarr[1].set_xlabel('X position [mm]')
		axarr[1].set_ylabel('Counts')
		# plt.title('Histogram of IQ')
		# (?# plt.text(60, .025, r'$\mu=100,\ \sigma=15$'))
		# plt.axis([40, 160, 0, 0.03])
		plt.grid(True)

		# median and std qx
		av = np.median(this_df.iloc[:,0].values)
		std = np.std(this_df.iloc[:,0].values)
		ylim = axarr[1].get_ylim()
		axarr[1].plot([av, av], [0, ylim[1]], color='red')
		axarr[1].plot([av+std, av+std], [0, ylim[1]], color='orange')
		axarr[1].plot([av-std, av-std], [0, ylim[1]], color='orange')

		# plt.tight_layout()
		# plt.show()
		jj = np.where(time_cols==time_cols[ii_time])[0][0]
		# print(np.min(this_df.iloc[:,0].values))
		# print(np.max(this_df.iloc[:,0].values))
		filename =  '{}/{}_histogram_time_{}'.format(directory,jj,time_cols[ii_time])
		# plt.savefig(filename + '.eps', dpi=1200)
		# plt.savefig(filename + '.svg', dpi=1200)
		plt.savefig(filename + '.png', dpi=600)
		plt.close('all')


	# # export all timestamps
	# for ii in range(0,len(time_cols)):
	# 	# get the subset of columns
	# 	this_df = df.filter(regex=time_cols[ii], axis=1)







	
	# print(timeStep)





COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data_bidirectionally/'.format(remote_path)

# # single file processing
# COMSOL_data_file = '{}010a1_10000_qx_qy_qz_hmax2.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
# createCSVforEachTimestep(COMSOL_data_file, COMSOL_data_file_path)


# loop processing
files = os.listdir(COMSOL_data_file_path)
files = [f for f in files if f.endswith('.csv')]
for f in files:
	COMSOL_data_file = '{}{}'.format(COMSOL_data_file_path, f)  # change this to each filename, i.e. loop
	print('Processing file: {}'.format(f))
	createCSVforEachTimestep(COMSOL_data_file, COMSOL_data_file_path)