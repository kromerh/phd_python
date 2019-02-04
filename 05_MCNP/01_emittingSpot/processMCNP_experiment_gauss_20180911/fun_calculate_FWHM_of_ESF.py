import pandas as pd
import numpy as np
import os, sys, glob
import re
from scipy.interpolate import interp1d
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy
from scipy import optimize
from scipy.misc import derivative


def fun_calculate_FWHM_of_ESF(
	X,  # X values from the ESF (in mm) 
	Y,  # Y values from the ESF (can be anything)
	p0  #  initial guess
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