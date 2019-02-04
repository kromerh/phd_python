from scipy.optimize import curve_fit
import itertools
import datetime
import sys
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import time
import re
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator
pd.set_option("display.max_columns", 100)


# import csv file
my_date = '2018-07-30'
fname = f'//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{my_date}_measurement/{my_date}_ESF.csv'
df = pd.read_csv(fname, index_col=0)


# sort by edge_pos_mm
df = df.sort_values(by=['edge_pos_mm'])
df.reset_index(inplace=True, drop=True)
# print(df.head())
# print(df.iloc[32:35,:])

# normalize countrate by highest countrate
df['norm_cps'] = df['norm_cps']/df['norm_cps'].max()
# print(df['norm_cps'])
# zero around the center
# mean at high end (edge out of beam)
thresh_out = 17  # edge completely removed
thresh_in = 12  # edge completely in
mu_out = np.mean(df.norm_cps[ df.edge_pos_mm > thresh_out ])
mu_in = np.mean(df.norm_cps[ df.edge_pos_mm < thresh_in ])
print(f'Mean of counts when edge is out: {mu_out}')
print(f'Mean of counts when edge is in: {mu_in}')

# linear fit around the middle region
center_estimated = 14.5  # edge pos center estimated in mm
pts_fit = 1  # mm left and right of center to take for fit
x_fit_range1 = center_estimated + pts_fit
x_fit_range2 = center_estimated - pts_fit
def fitfunc(p, x):
	return p[0] * x + p[1]
def errfunc(p, x, y):
	return fitfunc(p, x) - y # Distance to the fit function

p0 = [1, 1] # Initial guess for the parameters
X_f = df.edge_pos_mm[ (df.edge_pos_mm <= x_fit_range1) & (df.edge_pos_mm >= x_fit_range2) ].values
Y_f = df.norm_cps[ (df.edge_pos_mm <= x_fit_range1) & (df.edge_pos_mm >= x_fit_range2) ].values
# print(df.norm_cps)
# print(X_f, Y_f)
p1, success = optimize.leastsq(errfunc, p0[:], args=(X_f, Y_f))
X_fit = np.arange(12,16.5+0.01,0.01)
Y_fit = fitfunc(p1,X_fit)

# plt.scatter(df.edge_pos_mm, df.norm_cps)
# plt.plot([df.edge_pos_mm.min(), df.edge_pos_mm.max()], [mu_in, mu_in], c='red', label='mu edge in or out')
# plt.plot([df.edge_pos_mm.min(), df.edge_pos_mm.max()], [mu_out, mu_out], c='red')

# plt.plot(X_fit, Y_fit, c='green', label='linear fit')
# plt.xlabel('edge position [mm]')
# plt.ylabel('countrate normalized')
# plt.grid(True)
# plt.legend(loc='best')
# # plt.show()
# filename =  f'//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{my_date}_measurement/{my_date}_ESF_fit_to_find_center'
# plt.savefig(filename + '.png', dpi=600)
# plt.close('all')


# find where the linear fit intersects the mean curved (edge fully in or fully out curve)
def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx

near_mu = np.array([])  # first entry: edge fully in, second entry: edge fully out in mm edge position
for mu in [mu_in, mu_out]:
	idx = find_nearest(Y_fit, mu)
	near_mu = np.append(near_mu,X_fit[idx])

# print(near_mu)
# center is in between the two
center = np.mean(near_mu)
print(near_mu, center)



# fit logistic function to the ESF --> LSF
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
	FWHM = 3.53*p1[2]
	
	Y_log_fermi_func = fitfunc(p1, X)
	
	# return the FWHM from the 3.53c (logistic fit) and 
	# Y_log_fermi_func: is the Y values for the plot of the log fermi function
	
	return FWHM, Y_log_fermi_func, p1, r_squared

df['edge_pos_mm_centered'] = df['edge_pos_mm'] - center
m = [1e-2, 5e-2, 1e-1, 5e-1, 0, 1, 5, 1e1]
x = df['edge_pos_mm_centered']
y = df['norm_cps']
# print(x,y)
def fitfunc(x, *p):
	a, b, c, d = p
	z = np.exp( -( (x-b)/(c) ) )
	# z = np.exp( -( (x-p[1])/(p[2]) ) )
	return (a / ( 1 + z )) + d
	# return (p[0] / ( 1 + z )) + p[3]


# for mm in list(itertools.product(m,m,m,m)):
# 	p0 = [mm[0],mm[1],mm[2],mm[3]]
	# FWHM, Y_log_fermi_func, p1, r_squared = fun_calculate_FWHM_of_ESF(x, y, p0)
p0 = [1,1,1,1]
popt, pcov = curve_fit(fitfunc, x, y, p0=p0, maxfev=1000000)
# print(popt)
FWHM = 3.53*popt[2]  # 3.53 * c in fermi function
print(FWHM)
y_fit = fitfunc(x, *popt)


plt.scatter(df.edge_pos_mm_centered, df.norm_cps, label='measurement')
plt.plot(x,y_fit, color='red', label='logistic fit')
plt.xlabel('Edge position [mm]')
plt.ylabel('Countrate normalized')
plt.grid(True)
plt.legend(loc='best')
# plt.show()
filename =  f'//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{my_date}_measurement/{my_date}_ESF_fit_LSF'
plt.savefig(filename + '.png', dpi=600)
plt.close('all')