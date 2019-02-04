import pandas as pd
import numpy as np
import os
import re 

remote_path = '//fs03/LTH_Neutimag/hkromer/'

# create a csv file for each timestamp
# first column: particle index
# second column: qx
# third column: qy
# fourth column: qz
def createCSVforEachTimestep(COMSOL_data_file, output_path):
	# read the file into dataframe
	df = pd.read_csv(COMSOL_data_file, skiprows=7)
	cols = df.columns.tolist()  # columns as list
	
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
	
	# export all timestamps
	for ii in range(0,len(time_cols)):
		# get the subset of columns
		this_df = df.filter(regex=time_cols[ii], axis=1)

		# create parent directory if not exists
		folder = re.findall(r'/(\w+).csv$',COMSOL_data_file)[0]
		directory = '{}{}'.format(COMSOL_data_file_path, folder)
		if not os.path.exists(directory):
			os.makedirs(directory)

		# export timestamp
		fname = '{}_{}.csv'.format(ii,time_cols[ii])
		this_df.to_csv('{}/{}'.format(directory,fname))



	
	# print(timeStep)





COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data/'.format(remote_path)
# COMSOL_data_file = '{}011a1_3000_qx_qy_qz.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop

# loop
files = os.listdir(COMSOL_data_file_path)
files = [f for f in files if f.endswith('.csv')]

for f in files:
	COMSOL_data_file = '{}{}'.format(COMSOL_data_file_path, f)  # change this to each filename, i.e. loop
	print('Processing file: {}'.format(f))
	createCSVforEachTimestep(COMSOL_data_file, COMSOL_data_file_path)