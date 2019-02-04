import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt


remote_path = '//fs03/LTH_Neutimag/hkromer/'

COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data_bidirectionally/'.format(remote_path)

# folders in the directory
folders = os.listdir(COMSOL_data_file_path)
folders = [f for f in folders if os.path.isdir('{}/{}'.format(COMSOL_data_file_path,f))]
df = pd.DataFrame()




# go through all folders, accessing the csv files
for f in folders:
	this_df = pd.DataFrame()  # dataframe that contains the results of this file

	this_folder = '{}/{}/'.format(COMSOL_data_file_path,f)
	files = os.listdir(this_folder)
	files = [f for f in files if f.endswith('.csv')]
	# print(this_folder)
	# combine into one dataframe
	ID = re.findall(r'/(\w+)_qx',this_folder)[-1]
	for this_file in files:
		if len(this_df) < 1:
			this_df = pd.read_csv('{}/{}'.format(this_folder,this_file), index_col=0, header=None)
			
			colname = re.findall(r'(.+).csv', this_file)[0]
			this_df.columns=[colname]
			this_df['ID'] = ID
		else:
			my_df = pd.read_csv('{}/{}'.format(this_folder,this_file), index_col=0, header=None)
			colname = re.findall(r'(.+).csv', this_file)[0]
			my_df.columns=[colname]
			this_df = this_df.join(my_df)

	df = df.append(this_df)

# 95th percentile plot
f = plt.figure(figsize=(10,5))

ax1 = f.add_subplot(1, 1, 1)
# f.suptitle('Beam radius and beam diameter for time: {} s \n {}'.format(time_cols[ii_time], folder))

def plotRadiusVSdistance95perc(df):
	ID = df['ID'].unique()[0]
	# plot only the 15000 particle contribution
	# t0 = re.findall(r'_15000', ID)
	# if len(t0) > 0:

	X = df.qx_median.values
	# Y = df.qr_mean.values + df.qr_std.values
	Y = 2.0 * df.qr_95perc.values.astype(float)
	# print(Y)
	ax1.plot(X, Y, label = ID)	

df.groupby('ID').apply(lambda x: plotRadiusVSdistance95perc(x))

plt.legend(loc='best')
plt.ylabel('Diameter from 95th percentile [mm]')
plt.xlabel('Distance from plasma aperture [mm]')
plt.grid(True)
plt.tight_layout()
filename =  '{}/Diameter_95percentile'.format(COMSOL_data_file_path)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')

# mean + std
f = plt.figure(figsize=(10,5))

ax1 = f.add_subplot(1, 1, 1)
# f.suptitle('Beam radius and beam diameter for time: {} s \n {}'.format(time_cols[ii_time], folder))

def plotRadiusVSdistanceMeanAndStd(df):
	ID = df['ID'].unique()[0]
	# plot only the 15000 particle contribution
	# t0 = re.findall(r'_15000', ID))
	# if len(t0) > 0:

	X = df.qx_median.values
	Y = 2.0 * df.qr_mean.values + np.abs(df.qr_std.values)
	# Y = 2.0 * df.qr_95perc.values.astype(float)
	# print(Y)
	ax1.plot(X, Y, label = ID)	

df.groupby('ID').apply(lambda x: plotRadiusVSdistanceMeanAndStd(x))

plt.legend(loc='best')
plt.ylabel('Diameter from mean plus std [mm]')
plt.xlabel('Distance from plasma aperture [mm]')
plt.grid(True)
plt.tight_layout()
filename =  '{}/Diameter_meanAndStd'.format(COMSOL_data_file_path)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')



# 95th percentile plot
f = plt.figure(figsize=(10,5))

ax1 = f.add_subplot(1, 1, 1)
# f.suptitle('Beam radius and beam diameter for time: {} s \n {}'.format(time_cols[ii_time], folder))

def plotRadiusVSdistance95perc(df):
	ID = df['ID'].unique()[0]
	# plot only the 15000 particle contribution
	# t0 = re.findall(r'_15000', ID)
	# if len(t0) > 0:
	df = df.reset_index()
	X = df.index.values
	Y = 2.0 * df.qr_95perc.values.astype(float)
	# Y = 2.0 * df.qr_95perc.values.astype(float)
	# print(Y)
	ax1.plot(X, Y, label = ID)	


df.groupby('ID').apply(lambda x: plotRadiusVSdistance95perc(x))

plt.legend(loc='best')
plt.ylabel('Diameter from 95th percentile [mm]')
plt.xlabel('Simulation time [-]')
plt.grid(True)
plt.tight_layout()
filename =  '{}/Diameter_95percentile_time'.format(COMSOL_data_file_path)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')





# mean + std over time
f = plt.figure(figsize=(10,5))

ax1 = f.add_subplot(1, 1, 1)
# f.suptitle('Beam radius and beam diameter for time: {} s \n {}'.format(time_cols[ii_time], folder))

def plotRadiusVSdistanceMeanAndStd_time(df):
	ID = df['ID'].unique()[0]
	# plot only the 15000 particle contribution
	# t0 = re.findall(r'_15000', ID)
	# if len(t0) > 0:
	df = df.reset_index()
	X = df.index.values
	Y = 2.0 * df.qr_mean.values + np.abs(df.qr_std.values)
	# Y = 2.0 * df.qr_95perc.values.astype(float)
	# print(Y)
	ax1.plot(X, Y, label = ID)	

df.groupby('ID').apply(lambda x: plotRadiusVSdistanceMeanAndStd_time(x))

plt.legend(loc='best')

plt.ylabel('Diameter from mean plus std [mm]')
plt.xlabel('Simulation time [-]')

plt.grid(True)
plt.tight_layout()
filename =  '{}/Diameter_meanAndStd_time'.format(COMSOL_data_file_path)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')





# FWHM over time
f = plt.figure(figsize=(10,5))

ax1 = f.add_subplot(1, 1, 1)
# f.suptitle('Beam radius and beam diameter for time: {} s \n {}'.format(time_cols[ii_time], folder))
# print(df.columns)
def plotRadiusVSdistanceFWHM_time(df):
	ID = df['ID'].unique()[0]
	# plot only the 15000 particle contribution
	# t0 = re.findall(r'_15000', ID)
	# if len(t0) > 0:
	df = df.reset_index()
	X = df.index.values
	Y = 2.0 * df.qr_FWHM.values.astype(float)
	# Y = 2.0 * df.qr_95perc.values.astype(float)
	# print(Y)
	ax1.plot(X, Y, label = ID)	

df.groupby('ID').apply(lambda x: plotRadiusVSdistanceFWHM_time(x))

plt.legend(loc='best')

plt.ylabel('Diameter from FWHM [mm]')
plt.xlabel('Simulation time [-]')

plt.grid(True)
plt.tight_layout()
filename =  '{}/Diameter_FWHM_time'.format(COMSOL_data_file_path)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
