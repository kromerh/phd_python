import pandas as pd
import numpy as np
import os, sys, glob
import re
from scipy.interpolate import interp1d
# import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy
from scipy import stats
from scipy.stats import poisson
from scipy import optimize
from scipy.misc import derivative
import itertools
sys.path.append('//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/neutron_emitting_spot/')
# sys.path.append('/home/hippo/Documents/01_PhD/01_Python/neutron_emitting_spot/')
from fun_penumbra_investigation_relative_yield import *



# workflow:
# 	1. import the results (edge position vs countrate in the detector)
#	2. fit logistic fermi function to the ESF --> obtain FWHM as p[2]*3.53
#	3. select +/- 1 FWHM plus 3 points of this fermi function and interpolate the fitted function
#	4. select 30 measurement points along that fitted function (use interpolation)
#	5. sample from poisson 1000 points for each of these positions (this means from the fitted data!)
#	6. analog to 2, re-compute logistic fermi function to ESF --> obtain 1000 FWHMs as p[2]*3.53 
#	7. least squares linear regression for y = diameter in MCNP, x = FWHM from the fit to the ESF. 
#      Then find uncertainty for each FWHM point x, translated to an uncertainty in y

# ----------------------------------------------------------------------------------------------------
# 	1. import the results (edge position vs countrate in the detector). convert units to mm, load cross sections and compute countrate
# ----------------------------------------------------------------------------------------------------
# PSI network
# which folder to modify
#invs = ['05','06','07']

master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180808/'

# make folders if not exist
lst_folderstomake = ['ESF_fit_csv', 'ESF_fit_plots', 'linregressfit_source_diam_VS_FWHM', 'noised_ESF_fit_csv', 'noised_ESF_fit_plot', 'plots_errMCNP_vs_meas_time'] 
for folder in lst_folderstomake:
	directory = '{}{}/'.format(master_folder, folder)
	if not os.path.exists(directory):
		os.makedirs(directory)

# local
# master_folder = '/home/hippo/Documents/01_PhD/01_Python/neutron_emitting_spot/_investigation_02/'

df_MCNP_results = pd.read_csv('{}df_results.csv'.format(master_folder), index_col=0)

S0 = 3e7  # total neutron yield 3*10**7 neutrons/s

deutEnergy = 100
emission_angle = 2
f = fun_fraction_of_emissions_in_angle(deutEnergy, # deuterium ion energy in keV
emission_angle  # MCNP emission angle (full cone angle, not the half one!)
)

# energy and cutoff energy
E_neutron = 2.8
E_cutoff = 0.7

df_MCNP_results['diameter'] = np.multiply(np.multiply(df_MCNP_results['radius'], 2), 10)  # diameter of the MCNP source in mm

df_MCNP_results['f'] = f  # fraction of neutrons in emission angle
df_MCNP_results['Cps'] = np.multiply(np.multiply(df_MCNP_results['R_tot'],f),S0)  # neutrons per second in the detector (100% efficiency)
df_MCNP_results['Cps_cutoff'] = df_MCNP_results['Cps'] * (1 - (E_cutoff/E_neutron))  # neutrons per second in the detector (assuming 0.7 MeV cutoff energy)

df_MCNP_results['x_pos'] = df_MCNP_results['x_pos']*10.0  # convert to mm!

# # ----------------------------------------------------------------------------------------------------
# # 	2. fit logistic fermi function to the ESF --> obtain FWHM as p[2]*3.53
# # ----------------------------------------------------------------------------------------------------

# def fun_calculate_FWHM_of_ESF(
# 	# calculate the FWHM of the LSF fitted to the ESF
# 	# return FWHM, Y_log_fermi_func, p1, r_squared
# 	X,  # X values from the ESF (in mm) 
# 	Y,  # Y values from the ESF (can be anything)
# 	p0 # initial guess for the fit
# 	):

# 	# log_fermi_func_zboray
# 	def fitfunc(p, x):
# 		z = np.exp( -( (x-p[1])/(p[2]) ) )
# 		return (p[0] / ( 1 + z )) + p[3]
# 	def errfunc(p, x, y):
# 		return fitfunc(p, x) - y # Distance to the fit function


# 	# m = np.mean(Y[0:5])
# 	# p0 = [5e+03, 5e-2, radius, m] # Initial guess for the parameters
# 	p1, success = optimize.leastsq(errfunc, p0[:], args=(X, Y))

# 	# r-squared
# 	residuals = Y - fitfunc(p1, X)
# 	ss_residuals = np.sum(residuals**2)   # residual sum of squares
# 	ss_tot =  np.sum((Y-np.mean(Y))**2) # total sum of squares
# 	r_squared = 1 - (ss_residuals / ss_tot)
# 	FWHM = 3.53*p1[2]
	
# 	Y_log_fermi_func = fitfunc(p1, X)
	
# 	# return the FWHM from the 3.53c (logistic fit) and 
# 	# Y_log_fermi_func: is the Y values for the plot of the log fermi function
	
# 	return FWHM, Y_log_fermi_func, p1, r_squared

# def find_nearest(array,value):
# 	idx = (np.abs(array-value)).argmin()
# 	return idx

# def fit_FWHM(df,Y_colname):
# 	# returns the FWHM for a given X, Y
# 	# X is the edge position
# 	# Y is the countrate or the counts in the detector

# 	case = df['case'].unique()[0]
# 	diameter = df['diameter'].unique()[0]
# 	# if (case == 167) & (diameter == 3.0):  # do only one diameter
# 	if case > -1:  # always valid
			
		
		

# 		print('Processing case {}, diameter {} mm.'.format(case, diameter))

# 		X = df['x_pos'].values
# 		Y = df[Y_colname].values


# 		# find the optimal p0 as initial guess for the fit, then use that to return the FWHM
# 		# m = np.mean(Y[0:5])
# 		# m = [1e6, 1e7, 1e8, 1e9, 1e10]
# 		m = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, np.max(Y)]
# 		a = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
# 		b = [1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-3, 5e-3, 1e-4]
# 		c = [0.5, 2, 4, 6, 8, 10, 15, 20, 30]
# 		for kk in list(itertools.product(a,b,c,m)):
			
# 		# for kk in list(itertools.product(a,b,c)):
# 			p0 = [-kk[0], kk[1], kk[2]/3.53, kk[3]] # Initial guess for the parameters	
# 			FWHM_0, Y_log_fermi_func_0, p1_0, r_squared_0 = fun_calculate_FWHM_of_ESF(X, Y, p0)
# 			FWHM_0 = np.abs(FWHM_0)
# 			diff_fit =np.abs(Y_log_fermi_func_0[0]-Y_log_fermi_func_0[-1])
# 			diff_Y = np.abs(Y[0]-Y[-1])
# 			if (np.abs(FWHM_0) > 1e-1) & (np.abs(FWHM_0) < 2*diameter) & (np.divide(np.abs(diff_fit-diff_Y), diff_Y) < 0.5 ):  # computed FWHM larger than 0

# 				print(p1_0)
# 				break


# 		if FWHM_0 > 100:
# 			print('FWHM {} LARGER THAN 100! Exit.'.format(FWHM_0))
# 			sys.exit()
# 		print('FWHM_0: {} '.format(FWHM_0))
# 		print('p1_0: {}'.format(p1_0))
# 		###################################################
# 		# fit again with 0 +/- 1 FWHM and three points more!
# 		###################################################
		
# 		idx_X_pos = find_nearest(X,FWHM_0)
# 		# print(FWHM, X[idx_X_pos-1], X[idx_X_pos], X[idx_X_pos+1])

# 		idx_X_neg = find_nearest(X,-FWHM_0)
# 		# print(-FWHM, X[idx_X_neg-1], X[idx_X_neg], X[idx_X_neg+1])
# 		# plt.scatter(X,Y,color='blue')
# 		# plt.scatter(X[idx_X_pos], Y[idx_X_pos], color='red')
# 		# plt.scatter(X[idx_X_neg], Y[idx_X_neg], color='green')
# 		# plt.show()
# 		offset_points = 10
# 		X = X[idx_X_neg-offset_points:idx_X_pos+offset_points]
# 		Y = Y[idx_X_neg-offset_points:idx_X_pos+offset_points]
# 		print(idx_X_neg, idx_X_pos)
		

# 		FWHM, Y_log_fermi_func, p1, r_squared = fun_calculate_FWHM_of_ESF(X, Y, p1_0)
# 		# print(p1)
# 		# a = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
# 		# b = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
# 		# c = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
# 		# d = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
	 
# 		# for kk in list(itertools.product(a,b,c,d)):
# 			# p0 = [kk[0], kk[1], kk[2], kk[3]] # Initial guess for the parameters			
# 			# FWHM, Y_log_fermi_func, p1, r_squared = fun_calculate_FWHM_of_ESF(X, Y, p1_0)
					
# 			# if (p1[1] > 1e-3) & (p1[1] < 100):  # computed FWHM larger than 0
# 				# print(p1)
# 				# break


# 		print('Plotting fit to ESF for case {}, diameter {} mm.'.format(case, diameter))


# 		fig, ax = plt.subplots(1)	
# 		ax.plot(X, Y_log_fermi_func, color='darkblue', label='ESF fit')
		
# 		ax.scatter(X, Y, edgecolor='darkorange', label='MCNP data', color='darkorange')
# 		# print(X,Y)
# 		# ax.scatter([X[idx_X_neg],X[idx_X_pos]], [Y[idx_X_neg],Y[idx_X_pos]], edgecolor='red', color='red', marker='x', s=120)
			
# 			# FWHM, Y_log_fermi_func, p1, r_squared = compute_FWHM(X, df_poisson.iloc[:,ii].values,rad)
			
# 		ax.tick_params('x', colors='black', labelsize=12)	
# 		ax.tick_params('y', colors='black', labelsize=12)	
# 		# grid
# 		ax.grid(b=True, which='major', linestyle='-')#, color='gray')
# 		ax.grid(b=True, which='minor', linestyle='--')#, color='gray')
# 		plt.title('Case: {:.0f}, Diameter: {:.2f} mm'.format(case, diameter))
# 		plt.legend(loc='best')
# 		plt.xlabel('Edge position [mm]')
# 		plt.ylabel('Countrate for tot. Neutron Yield: {:.1e} n/s'.format(S0))
# 		plt.tight_layout()
# 		# plt.show()
# 		plt.savefig('{}ESF_fit_plots/diameter_{:.2f}_mm_case_{:.0f}_ESF_fit.png'.format(master_folder, diameter, case), dpi=100)
# 		plt.clf()
# 		plt.close('all')


# 		# print into file! 
# 		fname = '{}ESF_fit_csv/case_{}_diameter_{}_mm_ESF_fit.csv'.format(master_folder,case,diameter)
# 		this_df = pd.DataFrame()
# 		this_df['x_pos_mm'] = X
# 		this_df[Y_colname] = Y
# 		this_df['Y_log_fermi_func'] = Y_log_fermi_func
# 		this_df.to_csv(fname)

# 		return pd.Series(p1_0)
# 		# FWHM: 3.53*c of fitted LSF
# 		# X: edge positions for the fitted Y_log_fermi_function
# 		# Y_log_fermi_func values of the fitted function
# 		# p1: fitted parameters
# 		# r_squared: 1-(residual sum of squares / total sum of squares)

# Y_colname = 'Cps_cutoff'
# df_MCNP_results.groupby(['case', 'diameter']).apply(lambda x: fit_FWHM(x,Y_colname)).reset_index()
# fname = '{}df_MCNP_results.csv'.format(master_folder)
# df_MCNP_results.to_csv(fname)
# # sys.exit()

# # plot the fit!
# def plot_fit_to_ESF(df, Y_colname):
# 	case = df['case'].unique()[0]
	
# 	diameter = df['diameter'].unique()[0]

# 	print('Plotting fit to ESF for case {}, diameter {} mm.'.format(case, diameter))

# 	fname = '{}/ESF_fit_csv/case_{}_diameter_{}_mm_ESF_fit.csv'.format(master_folder,case,diameter)
# 	this_df = pd.read_csv(fname, index_col=0)

# 	X = this_df['x_pos_mm'].values
# 	Y_log_fermi_func = this_df['Y_log_fermi_func'].values

# 	Y = this_df[Y_colname].values

# 	fig, ax = plt.subplots(1)	
# 	ax.plot(X, Y_log_fermi_func, color='darkblue', label='ESF fit')
	
# 	ax.scatter(X, Y, edgecolor='darkorange', label='MCNP data', color='darkorange')
		
# 		# FWHM, Y_log_fermi_func, p1, r_squared = compute_FWHM(X, df_poisson.iloc[:,ii].values,rad)
		
# 	ax.tick_params('x', colors='black', labelsize=12)	
# 	ax.tick_params('y', colors='black', labelsize=12)	
# 	# grid
# 	ax.grid(b=True, which='major', linestyle='-')#, color='gray')
# 	ax.grid(b=True, which='minor', linestyle='--')#, color='gray')
# 	plt.title('Case: {:.0f}, Diameter: {:.2f} mm'.format(case, diameter))
# 	plt.legend(loc='best')
# 	plt.xlabel('Edge position [mm]')
# 	plt.ylabel('Countrate for tot. Neutron Yield: {:.1e} n/s'.format(S0))
# 	plt.tight_layout()
# 	# plt.show()
# 	plt.savefig('{}ESF_fit_plots/diameter_{:.2f}_mm_case_{:.0f}_ESF_fit.png'.format(master_folder, diameter, case), dpi=100)
# 	plt.clf()
# 	plt.close('all')

# Y_colname = 'Cps_cutoff'
# df_MCNP_results.groupby(['case', 'diameter']).apply(lambda x: plot_fit_to_ESF(x, Y_colname)).reset_index()


# # ----------------------------------------------------------------------------------------------------
# # 	3. select +/- 1 FWHM of this fermi function and interpolate the fitted function
# #	4. select 30 measurement points along that fitted function (use interpolation)
# #	5. sample from poisson 1000 points for each of these positions
# # ----------------------------------------------------------------------------------------------------

# # +/-1 FWHM already taken in the last step

# def interpolate_and_resample(df, Y_colname, num_measurements, tot_meas_time, num_samples):
# 	case = df['case'].unique()[0]
	
# 	diameter = df['diameter'].unique()[0]

# 	fname = '{}ESF_fit_csv/case_{}_diameter_{}_mm_ESF_fit.csv'.format(master_folder,case,diameter)

# 	df_ESF = pd.read_csv(fname, index_col=0)

# 	X = df_ESF['x_pos_mm'].values
# 	Y = df_ESF['Y_log_fermi_func'].values
# 	Y_mcnp = df_ESF[Y_colname].values
	
# 	# interpolate
# 	interp_ESF = interp1d(X, Y, kind='cubic')  # angle is in radian
# 	qry_X = np.linspace(np.min(X), np.max(X), num=num_measurements, endpoint=True)
# 	qry_Y = interp_ESF(qry_X)

# 	# take measurement time into account
# 	# time per position
# 	time_per_pos = (tot_meas_time*60*60) / num_measurements  # convert from hour to second and then divide by number of measurements
	
# 	qry_Y = np.multiply(qry_Y, time_per_pos) 
# 	Y_mcnp = np.multiply(Y_mcnp, time_per_pos)
# 	Y = np.multiply(Y, time_per_pos)

# 	# PLOT
# 	# print('Plotting interpolated fit to ESF for case {}, diameter {} mm.'.format(case, diameter))
# 	# fig, ax = plt.subplots(1)	
# 	# ax.plot(X, Y, color='black', label='ESF fit')
	
# 	# ax.scatter(X, Y_mcnp, edgecolor='black', label='MCNP data', color='darkorange', s=70)
# 	# ax.scatter(qry_X, qry_Y, marker='x', edgecolor='red', label='Interpolated fit', color='red', s=25)
		
# 	# # FWHM, Y_log_fermi_func, p1, r_squared = compute_FWHM(X, df_poisson.iloc[:,ii].values,rad)
		
# 	# ax.tick_params('x', colors='black', labelsize=12)	
# 	# ax.tick_params('y', colors='black', labelsize=12)	
# 	# # grid
# 	# ax.grid(b=True, which='major', linestyle='-')#, color='gray')
# 	# ax.grid(b=True, which='minor', linestyle='--')#, color='gray')
# 	# plt.title('Case: {:.0f}, Diameter: {:.2f} mm, tot. meas time {:.0f} h'.format(case, diameter, time_per_pos))
# 	# plt.legend(loc='best')
# 	# plt.xlabel('Edge position [mm]')
# 	# plt.ylabel('Counts for tot. Neutron Yield: {:.1e} n/s'.format(S0))
# 	# plt.tight_layout()
# 	# plt.show()
# 	# # plt.savefig('./ESF_fit/interpolate_fitted_ESF_diameter_{:.2f}_mm_case_{:.0f}_ESF_fit.png'.format(diameter, case), dpi=300)
# 	# # plt.clf()
# 	# # plt.close('all')

# 	# add poisson noise to the fitted data(!). Noise is added to the FIT, not to the MCNP data
# 	this_df = pd.DataFrame()
# 	this_df['x_pos_mm'] = qry_X
# 	this_df['counts'] = qry_Y

	
# 	df_poisson = this_df['counts'].apply(lambda x: pd.Series(np.random.poisson(x, size=num_samples)))
# 	df_poisson['x_pos_mm'] = this_df['x_pos_mm']
# 	df_poisson['counts'] = this_df['counts']

# 	# PLOT
# 	print('Plotting interpolated fit and poisson noise to ESF for case {}, diameter {} mm.'.format(case, diameter))
# 	fig, ax = plt.subplots(1)	
# 	ax.plot(X, Y, color='black', label='ESF fit')
	
# 	# ax.scatter(X, Y_mcnp, edgecolor='black', label='MCNP data', color='darkorange', s=70)
# 	ax.scatter(qry_X, qry_Y, marker='x', edgecolor='red', label='Interpolated fit', color='red', s=25)
# 	for ii in range(0, 3):
# 		ax.scatter(df_poisson['x_pos_mm'].values, df_poisson.iloc[:,ii].values, marker='o', label='Poisson noise', s=25)
		
# 	# FWHM, Y_log_fermi_func, p1, r_squared = compute_FWHM(X, df_poisson.iloc[:,ii].values,rad)
		
# 	ax.tick_params('x', colors='black', labelsize=12)	
# 	ax.tick_params('y', colors='black', labelsize=12)	
# 	# grid
# 	ax.grid(b=True, which='major', linestyle='-')#, color='gray')
# 	ax.grid(b=True, which='minor', linestyle='--')#, color='gray')
# 	plt.title('Case: {:.0f}, Diameter: {:.2f} mm, meas time per pos. {:.0f} s'.format(case, diameter, time_per_pos))
# 	plt.legend(loc='best')
# 	plt.xlabel('Edge position [mm]')
# 	plt.ylabel('Counts for tot. Neutron Yield: {:.1e} n/s'.format(S0))
# 	plt.tight_layout()
# 	# plt.show()
# 	plt.savefig('{}noised_ESF_fit_plot/noised_interpolatd_fitted_ESF_diameter_{:.2f}_mm_case_{:.0f}_ESF_fit.png'.format(master_folder, diameter, case), dpi=100)
# 	plt.clf()
# 	plt.close('all')

# 	fname = '{}noised_ESF_fit_csv/case_{}_diameter_{}_mm_tot_time_{}_hrs.csv'.format(master_folder,case,diameter, tot_meas_time)
# 	df_poisson.to_csv(fname)




# # import the fitted data
# Y_colname = 'Cps_cutoff'
# num_measurements = 30  # number of measurements (edge positions) for one experimental setup
# tot_meas_time = 80  # total measurement time in hours
# num_samples = 2  # number of poisson samples

# df_MCNP_results.groupby(['case', 'diameter']).apply(lambda x: interpolate_and_resample(x, Y_colname, num_measurements, tot_meas_time, num_samples)).reset_index()


# # ----------------------------------------------------------------------------------------------------
# #	6. analog to 2, re-compute logistic fermi function to ESF --> obtain 1000 FWHMs as p[2]*3.53 
# # ----------------------------------------------------------------------------------------------------

# def fit_FWHM_to_poisson(df, tot_meas_time):

# 	case = df['case'].unique()[0]
# 	diameter = df['diameter'].unique()[0]
# 	if case > -1:

# 	# if (case == 149) & (diameter == 1.0):
		

# 		fname = '{}noised_ESF_fit_csv/case_{}_diameter_{}_mm_tot_time_{}_hrs.csv'.format(master_folder,case,diameter, tot_meas_time)
		
# 		df_ESF = pd.read_csv(fname, index_col=0)

# 		print('Processing FWHM of poisson noised fit case {}, diameter {} mm.'.format(case, diameter))

# 		X = df_ESF['x_pos_mm'].values
# 		Y = df_ESF['counts'].values


# 		# find the optimal p0 as initial guess for the fit, then use that to return the FWHM
# 		# m = np.mean(Y[0:5])
# 		# m = [1e6, 1e7, 1e8, 1e9, 1e10]
# 		m = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, np.max(Y)]
# 		a = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
# 		b = [1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-3, 5e-3, 1e-4]
# 		c = [0.5, 2, 4, 6, 8, 10, 15, 20, 30]
# 		for kk in list(itertools.product(a,b,c,m)):
			
# 		# for kk in list(itertools.product(a,b,c)):
# 			p0 = [-kk[0], kk[1], kk[2]/3.53, kk[3]] # Initial guess for the parameters	
# 			FWHM_0, Y_log_fermi_func_0, p1_0, r_squared_0 = fun_calculate_FWHM_of_ESF(X, Y, p0)
# 			FWHM_0 = np.abs(FWHM_0)
# 			diff_fit =np.abs(Y_log_fermi_func_0[0]-Y_log_fermi_func_0[-1])
# 			diff_Y = np.abs(Y[0]-Y[-1])
# 			if (np.abs(FWHM_0) > 1e-1) & (np.abs(FWHM_0) < 2*diameter) & (np.divide(np.abs(diff_fit-diff_Y), diff_Y) < 0.5 ):  # computed FWHM larger than 0

# 				print(p1_0)
# 				break


# 		if FWHM_0 > 100:
# 			print('FWHM {} LARGER THAN 100! Exit.'.format(FWHM_0))
# 			# sys.exit()
# 			FWHM_0 = diameter
# 		print('FWHM_0: {} '.format(FWHM_0))
# 		print('p1_0: {}'.format(p1_0))
		
# 		# fit for the poisson noise
# 		cols = df_ESF.columns
# 		cols = [col for col in cols if re.search(r'\d', col)]

# 		df_ESF = df_ESF[cols]

# 		for ii in range(0,len(df_ESF.columns)):
# 			Y = df_ESF.iloc[:,ii].values
# 			FWHM, Y_log_fermi_func, p1, r_squared = fun_calculate_FWHM_of_ESF(X, Y, p1_0)
# 			FWHM = np.abs(FWHM)
# 			if ii == 0:
# 				s1 = pd.Series([FWHM], index=['FWHM_{:.0f}'.format(ii)])
# 			else:
# 				s1 = s1.append(pd.Series([FWHM], index=['FWHM_{:.0f}'.format(ii)]))

# 		s1 = s1.append(pd.Series([FWHM_0], index=['FWHM_orig']))
# 		# n, bins, patches = plt.hist(s1, 100, facecolor='g', alpha=0.75)
# 		# plt.plot([FWHM_0, FWHM_0], [0, np.max(n)], color='red')
# 		# plt.xlabel('FWHM')
# 		# plt.ylabel('Counted')
# 		# plt.grid(True)
# 		# # plt.show()
# 		# plt.savefig('./FWHM_on_poisson/noised_interpolated_fitted_ESF_diameter_{:.2f}_mm_case_{:.0f}_ESF_fit.png'.format(diameter, case), dpi=300)
# 		# plt.clf()
# 		# plt.close('all')
		

# 		# # a.set_index('idx', inplace=True)


		

# 		return s1


# # # change only if needs recomputation for some reason
# # # tot_meas_time =   # total measurement time in hours

# df_poisson = df_MCNP_results.groupby(['case', 'diameter']).apply(lambda x: fit_FWHM_to_poisson(x, tot_meas_time)).reset_index()
# fname = '{}FWHM_on_poisson_tot_time_{}_hrs.csv'.format(master_folder,tot_meas_time)
# df_poisson.to_csv(fname)

# # sys.exit()

# # ----------------------------------------------------------------------------------------------------
# #	7. least squares linear regression for y = diameter in MCNP, x = FWHM from the fit to the ESF. 
# #      Then find uncertainty for each FWHM point x, translated to an uncertainty in y
# # ----------------------------------------------------------------------------------------------------
# fname = '{}FWHM_on_poisson_tot_time_{}_hrs.csv'.format(master_folder,tot_meas_time)
# df_poisson = pd.read_csv(fname, index_col=0)



# def my_linreg(X,Y):

# 	# log_fermi_func_zboray
# 	def fitfunc(p, x):
# 		return p[0]*x+p[1]
# 	def errfunc(p, x, y):
# 		return fitfunc(p, x) - y # Distance to the fit function

	
# 	# initial guess y=m*x+c
# 	p0 = [0,0]
# 	dy = Y[1]-Y[0]
# 	dx = X[1]-X[0]
# 	p0[0] = dy/dx
# 	p0[1] = Y[1]-p0[0]*X[1]

	
# 	p1, success = optimize.leastsq(errfunc, p0[:], args=(X, Y))

# 	# r-squared
# 	residuals = Y - fitfunc(p1, X)
# 	ss_residuals = np.sum(residuals**2)   # residual sum of squares
# 	ss_tot =  np.sum((Y-np.mean(Y))**2) # total sum of squares
# 	r_squared = 1 - (ss_residuals / ss_tot)
	
	
# 	# return the FWHM from the 3.53c (logistic fit) and 
# 	# Y_log_fermi_func: is the Y values for the plot of the log fermi function
	
# 	return p1[0], p1[1], r_squared

# def get_mean_and_std_from_FWHM(df, cols_0):
# 	diam = df['diameter'].unique()[0]
# 	case = df['case'].unique()[0]

# 	print('Computing mean and std from FWHM for case: {:.0f}, diameter: {:.2f} mm'.format(case, diam))
# 	this_df = df[cols_0]
# 	m = this_df.apply(lambda x: np.mean(x), axis = 1).values[0]
# 	std = this_df.apply(lambda x: np.std(x), axis = 1).values[0]
# 	stdRel = std/m
# 	FWHM_0 = df['FWHM_orig'].values[0]
# 	diff_FWHM = np.abs(FWHM_0-m)
# 	# print([m, std])
# 	# compare FWHM_mean and FWHM_0, where FWHM_0 is from the original fit to the ESF
# 	s = pd.Series([m, std, stdRel, FWHM_0, diff_FWHM], index=['FWHM_mean', 'std', 'stdRel', 'FWHM_orig', 'diff_FWHM_orig'])


# 	return s







# # FWHM column names
# cols = df_poisson.columns
# cols_0 = [col for col in cols if re.search(r'\d', col)]

# # change meastime only if absolutely necessary
# # tot_meas_time = 3  # hrs
# df_FWHM_mean = df_poisson.groupby(['case', 'diameter']).apply(lambda x: get_mean_and_std_from_FWHM(x, cols_0)).reset_index()
# df_FWHM_mean.to_csv('{}df_FWHM_mean_for_time_{}_hrs.csv'.format(master_folder, tot_meas_time))
	
# # check if the difference is more than 2, 5 % for the FWHM_mean and original
# df_FWHM_mean_large_diffs = df_FWHM_mean[ df_FWHM_mean['diff_FWHM_orig'] > 0.02 ]
# print('FWHM mean and original larger than 2% for {} rows (case, diameter):'.format(len(df_FWHM_mean_large_diffs)))
# df_FWHM_mean_large_diffs = df_FWHM_mean[ df_FWHM_mean['diff_FWHM_orig'] > 0.05 ]
# print('FWHM mean and original larger than 5% for {} rows (case, diameter):'.format(len(df_FWHM_mean_large_diffs)))





# def plot_diameter_vs_FWHM(case, diameter, FWHM, FWHM_error, slope, intercept, savefolder, r_squared, tot_meas_time):
# 	# -------------------------------------------------------------------
# 	# plot
# 	# -------------------------------------------------------------------
# 	# plt.rc('text', usetex=True)
# 	# plt.rc('font', weight='bold')
# 	# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# 	# matplotlib.rcParams['mathtext.rm'] = 'Arial'
# 	# matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
# 	# matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
# 	# matplotlib.rcParams['mathtext.tt'] = 'Arial'
# 	# matplotlib.rcParams['mathtext.cal'] = 'Arial'
# 	# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


# 	fig = plt.figure(figsize=(8*0.60,5*0.60))

# 	####################
# 	# axis 1
# 	####################
# 	ax1 = fig.add_subplot(1, 1, 1)
# 	# ax1.scatter(FWHM, diameter, marker='o', color='darkblue')
# 	ax1.errorbar(FWHM, diameter, xerr=FWHM_error, marker='o', color='darkblue', capsize=5)
# 	plt.plot(FWHM, intercept + slope*FWHM, 'r')
# 	ax1.tick_params('x', colors='black', labelsize=12)	
# 	ax1.tick_params('y', colors='black', labelsize=12)	
# 	# grid
# 	ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
# 	ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')
# 	ax1.text(0.5,5,r'slope: {:.3f}, r-squared: {:.3f}'.format(slope, r_squared), color='r')
# 	ax1.set_xlabel(r'FWHM [mm]')
# 	ax1.set_ylabel(r'Diameter from MCNP [mm]')
# 	plt.title('Diameter vs FWHM for case {}'.format(case))
# 	# ylims = ax1.get_ylim()
# 	# ax1.set_ylim(0, 1e7)
# 	# fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
# 	plt.xlim(0,6)
# 	plt.ylim(0,6)
# 	plt.tight_layout()
# 	# plt.show()
# 	plt.savefig('{}case_{:.0f}_for_time_{:.0f}_h.png'.format(savefolder, case, tot_meas_time), dpi=100)
# 	plt.clf()
# 	plt.close('all')



# def fit_source_diam_VS_FWHM(df, tot_meas_time):

# 	diameter = df['diameter'].values
# 	FWHM = df['FWHM_orig'].values
# 	FWHM_error = df['std'].values
# 	case = df['case'].unique()[0]
# 	# print('Doing case {}'.format(case))
	
# 	savefolder = '{}/linregressfit_source_diam_VS_FWHM/'.format(master_folder)
# 	# linear least-squares regression
# 	slope, intercept, r_value, p_value, std_err = stats.linregress(FWHM, diameter)
# 	r_squared = r_value**2

# 	# slope, intercept, r_squared = my_linreg(FWHM, diameter)
	
# 	print('Case: {:.0f}, Slope: {:.3f}, r-squared is: {:.3f}'.format(case, slope, r_squared))
# 	_p = plot_diameter_vs_FWHM(case, diameter, FWHM, FWHM_error, slope, intercept, savefolder, r_squared, tot_meas_time)
	

# 	s = pd.Series([slope, intercept, r_squared], index = ['slope', 'intercept', 'r_squared'])
# 	# print(s)
# 	return s





# df_linregress = df_FWHM_mean.groupby(['case']).apply(lambda x: fit_source_diam_VS_FWHM(x, tot_meas_time)).reset_index()
# df_linregress.to_csv('{}df_linregress_for_time_{}_hrs.csv'.format(master_folder, tot_meas_time))

# # sys.exit()






# def get_error_in_MCNP_diameter(df, df_FWHM_mean):
# # takes the df_linregress. Computes the diameter and std in the diameter from the fitted curve in df_linregress
# 	case = df['case'].unique()[0]
# 	# print(case)
	
# 	this_df = df_FWHM_mean[ df_FWHM_mean['case'] == case ]
# 	diameter = this_df['diameter'].values
	
	
# 	slope = df['slope'].values
# 	intercept = df['intercept'].values

# 	X = this_df['FWHM_mean'].values
# 	dX = this_df['std'].values
# 	# print(slope, dX)
# 	# compute uncertainty in Y
# 	Y = X*slope + intercept
# 	Y0 = (X+dX)*slope + intercept
# 	Y1 = (X-dX)*slope + intercept
# 	dY = np.abs(Y1-Y0)
# 	# print(diameter)
# 	df_out = pd.DataFrame()

# 	# what is the actual MCNP diameter
# 	df_out['diameter'] = diameter
# 	# error in the MCNP diameter, that would be defined from the slope, intercept
# 	df_out['error_diameter'] = dY
# 	# df_out['case'] = case
# 	# print(df_out)
	
# 	return df_out


# df_error_MCNP_diameter = df_linregress.groupby(['case']).apply(lambda x: get_error_in_MCNP_diameter(x, df_FWHM_mean)).reset_index()
# df_error_MCNP_diameter.to_csv('{}df_error_MCNP_diameter_for_time_{}_h.csv'.format(master_folder, tot_meas_time))