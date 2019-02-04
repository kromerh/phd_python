import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt


remote_path = '//fs03/LTH_Neutimag/hkromer/'

COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data/'.format(remote_path)

# folders in the directory
folders = os.listdir(COMSOL_data_file_path)
folders = [f for f in folders if os.path.isdir('{}/{}'.format(COMSOL_data_file_path,f))]
df = pd.DataFrame()




# go through all folders, accessing the csv files
# only works if the timestamp is the same!

for f in folders:

	this_folder = '{}/{}/'.format(COMSOL_data_file_path,f)
	files = os.listdir(this_folder)
	files = [f for f in files if f.endswith('.csv')]
	# print(this_folder)
	# combine into one dataframe
	ID = re.findall(r'/(\w+)_qx',this_folder)[-1]
	for this_file in files:
		if len(df) < 1:
			df = pd.read_csv('{}/{}'.format(this_folder,this_file), index_col=0, header=None)
			
			colname = re.findall(r'(.+).csv', this_file)[0]
			colname = ID + '_' + colname
			
			df.columns=[colname]
		else:
			my_df = pd.read_csv('{}/{}'.format(this_folder,this_file), index_col=0, header=None)
			colname = re.findall(r'(.+).csv', this_file)[0]
			colname = ID + '_' + colname
			my_df.columns=[colname]
			df = df.join(my_df)

# plot 95th percentile over distance

# filter for the 95perc
this_df = df.filter(regex='95perc', axis=1)
this_df = this_df.filter(regex='15000', axis=1)
this_df = this_df.apply(pd.to_numeric, errors='ignore')

# plt.figure()
this_df.plot(figsize=(9.5,4))
plt.xlabel('Time [s]')
plt.ylabel('95th percentile [mm]')
plt.grid(True)
plt.tight_layout()
filename =  '{}/95percentile'.format(COMSOL_data_file_path)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
