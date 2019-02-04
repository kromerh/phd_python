import pandas as pd
import numpy as np
import os, sys, glob
import re
from scipy.interpolate import interp1d

sys.path.append('//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_useful_functions/')
from fun_MCNP_read_tally_data import *

# -----------------------------------------------------------------------------------------------------------------
# Neutron detection setting
# -----------------------------------------------------------------------------------------------------------------

# cell track data from the out file
detector_cell = 90  # detector cell

# -----------------------------------------------------------------------------------------------------------------
# Folder settings, MCNP must be finished, ptrac must be analyzed (df exist)
# -----------------------------------------------------------------------------------------------------------------




master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180904/'

# master_folder ='D:/neutron_emitting_spotsize/'
# case which this run corresponds to and master folder where the case will be in is given by master script
# lst_cases = np.arange(0,179+1,1)  # be careful to update this in the excel file!
# lst_cases = np.arange(0,24+1,1)  # be careful to update this in the excel file!
lst_cases = [0]

# -----------------------------------------------------------------------------------------------------------------
# Cross Sections
# -----------------------------------------------------------------------------------------------------------------
# BC-400 datasheet
# densities of hydrogen and carbon
C_H = 5.23e22  # atoms/cm3
C_C = 4.74e22  # atoms/cm3

# import cross sections
df_xs_H = pd.read_csv('//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/0_inputFile/cross_section_MeV_barns_hydrogen.csv', header=None, names=['energy_MeV', 'xs_H_barn'])
df_xs_C = pd.read_csv('//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/0_inputFile/cross_section_MeV_barns_carbon.csv', header=None, names=['energy_MeV', 'xs_C_barn'])
df_xs_H['xs_cm2'] = df_xs_H['xs_H_barn']*1e-24
df_xs_C['xs_cm2'] = df_xs_C['xs_C_barn']*1e-24

df_xs_H['XS_1_per_cm'] = df_xs_H['xs_cm2'] * C_H
df_xs_C['XS_1_per_cm'] = df_xs_C['xs_cm2'] * C_C

# interpolation functions
interp_XS_H = interp1d(df_xs_H['energy_MeV'], df_xs_H['XS_1_per_cm'], kind='cubic')
interp_XS_C = interp1d(df_xs_C['energy_MeV'], df_xs_C['XS_1_per_cm'], kind='cubic')




for case in lst_cases:
	print('Case {}'.format(case))
	# template input file that will be modified
	# input_file = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/0_inputFile/000_case_{}'.format(case)

	# case folder
	case_folder = master_folder + 'case_{}/'.format(case)


	# -----------------------------------------------------------------------------------------------------------------
	# Get list of directories
	# -----------------------------------------------------------------------------------------------------------------
	# list of all the directories, not files
	lst_runs = [filename for filename in os.listdir(case_folder) if os.path.isdir(os.path.join(case_folder,filename))]
	lst_runs = ['{}{}/'.format(case_folder, this_folder) for this_folder in lst_runs]

	# for run in lst_runs:
		# print(len(os.listdir(run + 'segmented_ptrac/')))
	# -----------------------------------------------------------------------------------------------------------------
	# Analyze the df
	# -----------------------------------------------------------------------------------------------------------------
	df_result = pd.DataFrame(columns = ['case','radius','x_pos','run','nps','entering_tracks', 'population', 'F4_tot', 'R_tot'])
	
	ii = 0 # to know how many there were
	for folder_run in lst_runs:
		print(folder_run)
		print('Run {} out of {}'.format(ii, len(lst_runs)-1))
		# path to ptrac file
		

		# run
		_ = re.findall(r'.*/([^/]+)/', folder_run)
		this_run = _[0]
		
		# entering_tracks: analyze out file
		file_content = []
		lineNr = 0
		start_cell_tracks = 0
		end_cell_tracks = 0
		find_string = 'tracks     population   collisions   collisions     number        flux        average'
		with open('{}out'.format(folder_run), 'r') as file:
			for line in file:
				file_content.append(line.split())
				_ = re.findall(r'nps\s(\d\S+)',line)
				if len(_) > 0:
					nps = _[0]
				this_regex = r'(' + find_string + r')'
				_ = re.findall(this_regex, line)
				if len(_) > 0:
					start_cell_tracks = lineNr
				this_regex = r'(total)'
				_ = re.findall(this_regex, line)
				if (len(_) > 0) and (start_cell_tracks > 0) and (end_cell_tracks == 0):
					end_cell_tracks = lineNr
					
				
				lineNr = lineNr + 1
			
			file.close()
		track_data = file_content[start_cell_tracks:end_cell_tracks-1]  # select only the track data
		# print(track_data)
		for item in track_data:
			# print(item)
			if len(item) > 0:
				if item[1].isdigit():
					if int(item[1]) == detector_cell:  # detector cell
						entering_tracks = int(item[2])
						population = int(item[3])

		# append to the dataframe
		_ = re.findall(r'rad(.+)_', this_run)
		radius = _[0]
		_ = re.findall(r'x(.+)', this_run)
		x_pos = _[0]
		
		# print(df_result)
		# sys.exit()
		
		# compute total interaction rate / density
		this_df = fun_MCNP_read_tally_data('{}out'.format(folder_run), detector_cell)
		this_df['XS_C'] = interp_XS_C(this_df['binCenterEnergy_MeV'])
		this_df['XS_H'] = interp_XS_H(this_df['binCenterEnergy_MeV'])

		# get the volume
		# go one path up
		this_case_folder = re.findall(r"(.+/)rad", folder_run)[0]
		# open file
		with open('{}case_{}_properties.txt'.format(this_case_folder, case), 'r') as file:
			for line in file:
				line = line.rstrip()
				# print(line)
				_ = re.findall(r'WxHxD: (.+) cm', line)
				if len(_) > 0:
					str_dim = _[0]
			file.close()

		str_dim = str_dim.split('x')
		fl_dim = [float(s) for s in str_dim]
		
		V = fl_dim[0] * fl_dim[1] * fl_dim[2]
		print(fl_dim, V)

		# this_df['R'] = (this_df['XS_H'] / (this_df['XS_H'] + this_df['XS_C'])) * this_df['tally'] * V	
		this_df['R'] = (this_df['XS_H']) * this_df['tally'] * V	

		this_df.to_csv('{}tally_result.csv'.format(folder_run))

		F4_tot = np.sum(this_df['tally'])
		R_tot = np.sum(this_df['R'])
		df_result = df_result.append(pd.Series([case,radius,x_pos,this_run,nps,entering_tracks,population, F4_tot, R_tot], index=df_result.columns), ignore_index=True)
		ii = ii + 1




	df_result.to_csv('{}case_{}_df_result.csv'.format(case_folder,case))
	lst_radius = (df_result['radius'].unique())



	# make df for each radius
	for radius in lst_radius:
		(df_result.loc[ df_result['radius'] == radius ]).to_csv('{}case_{}_df_result_radius_{}cm.csv'.format(case_folder,case,radius))

	print('***** Processing finished.')



