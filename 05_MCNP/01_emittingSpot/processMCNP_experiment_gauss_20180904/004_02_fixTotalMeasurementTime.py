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
from scipy.optimize import curve_fit
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

master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180904/'

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

# ----------------------------------------------------------------------------------------------------
# 	2. fit logistic fermi function to the ESF --> obtain FWHM as p[2]*3.53
# ----------------------------------------------------------------------------------------------------

def fun_calculate_FWHM_of_ESF(
	# calculate the FWHM of the LSF fitted to the ESF
	# return FWHM, Y_log_fermi_func, p1, r_squared
	X,  # X values from the ESF (in mm) 
	Y,  # Y values from the ESF (can be anything)
	p0 # initial guess for the fit
	):

	# log_fermi_func_zboray
	def fitfunc(p, x):
		z = np.exp( -( (x-p[1])/(p[2]) ) )
		return (p[0] / ( 1 + z )) + p[3]
	def errfunc(p, x, y):
		return fitfunc(p, x) - y # Distance to the fit function


	# m = np.mean(Y[0:5])
	# p0 = [5e+03, 5e-2, radius, m] # Initial guess for the parameters
	p1, success = optimize.leastsq(errfunc, p0[:], args=(X, Y))

	# r-squared
	residuals = Y - fitfunc(p1, X)
	ss_residuals = np.sum(residuals**2)   # residual sum of squares
	ss_tot =  np.sum((Y-np.mean(Y))**2) # total sum of squares
	r_squared = 1 - (ss_residuals / ss_tot)
	FWHM = 3.53 * p1[2]
	
	Y_log_fermi_func = fitfunc(p1, X)
	
	# return the FWHM from the 3.53c (logistic fit) and 
	# Y_log_fermi_func: is the Y values for the plot of the log fermi function
	
	return FWHM, Y_log_fermi_func, p1, r_squared

def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx

def fit_FWHM(df,Y_colname):
	# returns the FWHM for a given X, Y
	# X is the edge position
	# Y is the countrate or the counts in the detector

	case = df['case'].unique()[0]
	diameter = df['diameter'].unique()[0]
	# if (case == 167) & (diameter == 3.0):  # do only one diameter
	if case > -1:  # always valid
		
		

		print('Processing case {}, diameter {} mm.'.format(case, diameter))

		X = df['x_pos'].values
		Y = df[Y_colname].values


		def fitfunc(x, *p):
			a, b, c, d = p
			z = np.exp( -( (x-b)/(c) ) )
			# z = np.exp( -( (x-p[1])/(p[2]) ) )
			return (a / ( 1 + z )) + d
			# return (p[0] / ( 1 + z )) + p[3]


		# for mm in list(itertools.product(m,m,m,m)):
		# 	p0 = [mm[0],mm[1],mm[2],mm[3]]
			# FWHM, Y_log_fermi_func, p1, r_squared = fun_calculate_FWHM_of_ESF(x, y, p0)
		p0 = [1, 1e-3, 1e-1, 1e3]
		popt, pcov = curve_fit(fitfunc, X, Y, p0=p0, maxfev=1000000)
		# print(popt)
		FWHM_0 = np.abs(3.53*popt[2])  # 3.53 * c in fermi function


		# if FWHM_0 > 100:
		# 	print('FWHM {} LARGER THAN 100! Exit.'.format(FWHM_0))
		# 	sys.exit()
		# print('FWHM_0: {} '.format(FWHM_0))
		p1_0 = popt
		print('p1_0: {} '.format(p1_0))

		###################################################
		# fit again with 0 +/- 1 FWHM and three points more!
		###################################################
		
		# idx_X_pos = find_nearest(X,FWHM_0)


		# idx_X_neg = find_nearest(X,-FWHM_0)


		# print('{:.1f}'.format(diameter))
		# if '{:.1f}'.format(diameter) == '3.5':
		# 	plt.scatter(X,Y)
		# 	plt.show()
		# offset_points = 10
		# i_max = idx_X_pos+offset_points
		# i_min = idx_X_neg-offset_points
		# if (i_min > 0) & (i_max <= len(X)):
		# 	X = X[i_min:i_max]
		# 	Y = Y[i_min:i_max]a<Y
		# 	print(idx_X_neg, idx_X_pos)



		p0 = [1, 1e-3, 1e-1, 1e3]
		popt, pcov = curve_fit(fitfunc, X, Y, p0=p0, maxfev=1000000)
		# print(popt)
		FWHM = np.abs(3.53*popt[2])  # 3.53 * c in fermi function
		print('FWHM: {} '.format(FWHM))
		
		
			# r-squared
		residuals = Y - fitfunc(X, *popt)
		ss_residuals = np.sum(residuals**2)   # residual sum of squares
		ss_tot =  np.sum((Y-np.mean(Y))**2) # total sum of squares
		r_squared = 1 - (ss_residuals / ss_tot)
		# print(p1)
		# a = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
		# b = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
		# c = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
		# d = [-1e6, -1e5, -1e4, -1e3, -1e2, -1e1,-1e-1, -1e-2, -1e-3, -1e-4, 0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
	 
		# for kk in list(itertools.product(a,b,c,d)):
			# p0 = [kk[0], kk[1], kk[2], kk[3]] # Initial guess for the parameters			
			# FWHM, Y_log_fermi_func, p1, r_squared = fun_calculate_FWHM_of_ESF(X, Y, p1_0)
					
			# if (p1[1] > 1e-3) & (p1[1] < 100):  # computed FWHM larger than 0
				# print(p1)
				# break


		print('Plotting fit to ESF for case {}, diameter {} mm.'.format(case, diameter))


		fig, ax = plt.subplots(1)
		X_fit = np.sort(X)
		# print(X_fit)	
		Y_log_fermi_func = fitfunc(X_fit, *popt)
		ax.plot(X_fit, Y_log_fermi_func, color='darkblue', label='ESF fit')
		
		ax.scatter(X, Y, edgecolor='darkorange', label='MCNP data', color='darkorange')
		# print(X,Y)
		# ax.scatter([X[idx_X_neg],X[idx_X_pos]], [Y[idx_X_neg],Y[idx_X_pos]], edgecolor='red', color='red', marker='x', s=120)
			
			# FWHM, Y_log_fermi_func, p1, r_squared = compute_FWHM(X, df_poisson.iloc[:,ii].values,rad)
			
		ax.tick_params('x', colors='black', labelsize=12)	
		ax.tick_params('y', colors='black', labelsize=12)	
		# grid
		ax.grid(b=True, which='major', linestyle='-')#, color='gray')
		ax.grid(b=True, which='minor', linestyle='--')#, color='gray')
		plt.title('Case: {:.0f}, Diameter: {:.2f} mm'.format(case, diameter))
		plt.legend(loc='best')
		plt.xlabel('Edge position [mm]')
		plt.ylabel('Countrate for tot. Neutron Yield: {:.1e} n/s'.format(S0))
		plt.tight_layout()
		# plt.show()
		plt.savefig('{}ESF_fit_plots/diameter_{:.2f}_mm_case_{:.0f}_ESF_fit.png'.format(master_folder, diameter, case), dpi=100)
		plt.clf()
		plt.close('all')


		# print into file! 
		fname = '{}ESF_fit_csv/case_{}_diameter_{}_mm_ESF_fit.csv'.format(master_folder,case,diameter)
		this_df = pd.DataFrame()
		this_df['x_pos_mm'] = X
		this_df[Y_colname] = Y
		this_df['Y_log_fermi_func'] = Y_log_fermi_func
		this_df.to_csv(fname)

		return pd.Series(np.abs(3.53*popt[2]), index = ['FWHM'])
		# FWHM: 3.53*c of fitted LSF
		# X: edge positions for the fitted Y_log_fermi_function
		# Y_log_fermi_func values of the fitted function
		# p1: fitted parameters
		# r_squared: 1-(residual sum of squares / total sum of squares)

Y_colname = 'Cps_cutoff'
a = df_MCNP_results.groupby(['case', 'diameter']).apply(lambda x: fit_FWHM(x, Y_colname)).reset_index()
fname = '{}df_MCNP_FWHM.csv'.format(master_folder)
a.to_csv(fname)

fname = '{}df_MCNP_results.csv'.format(master_folder)
df_MCNP_results.to_csv(fname)
# sys.exit()
