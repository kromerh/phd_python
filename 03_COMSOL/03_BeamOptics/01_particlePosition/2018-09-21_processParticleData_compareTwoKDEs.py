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

# FILE ONE (Working directory)
# # single file processing
# id_file_1 = 'Longer target file'
# remote_path = '//fs03/LTH_Neutimag/hkromer/'
# COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/\
# 01_OldTarget/IGUN_geometry/2018-09-21_comsol/longer_target/\
# plots/2D_histograms_lastTimestep/'.format(remote_path)
# particle_file_1 = '{}01.longTarget.particleData.csv_df_histData.csv'\
# .format(COMSOL_data_file_path)  # change this to each filename, i.e. loop

id_file_1 = '02'
remote_path = '//fs03/LTH_Neutimag/hkromer/'


COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/\
01_OldTarget/IGUN_geometry/2018-09-18_comsolGeometry/02.define_release_time/\
particleData/plots/2D_histograms_lastTimestep/'.format(remote_path)
particle_file_1 = '{}02.particleData.csv_df_histData.csv'\
.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop

# COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/\
# 01_OldTarget/IGUN_geometry/2018-09-24_comsol/define_current/\
# particleData/plots/2D_histograms_lastTimestep/'.format(remote_path)
# particle_file_1 = '{}02.particleData.csv_df_histData.csv'\
# .format(COMSOL_data_file_path)  # change this to each filename, i.e. loop

df_1 = pd.read_csv(particle_file_1, index_col=0)
# print(df_1.head())


# FILE TWO

# set the file directly
# id_file_2 = 'Normal file 01'
# particle_file_2 = f'{remote_path}/02_Simulations/06_COMSOL/03_BeamOptics/\
# 01_OldTarget/IGUN_geometry/2018-09-18_comsolGeometry/02.define_release_time/\
# particleData/plots/2D_histograms_lastTimestep/\
# 01.particleData.csv_df_histData.csv'

# or recursively
files = os.listdir(COMSOL_data_file_path)
files = [f for f in files if f.endswith('df_histData.csv')]
files = [f for f in files if '02.particleData' not in f]
files = [f'{COMSOL_data_file_path}/{f}' for f in files if '01' not in f]

directory = f'{COMSOL_data_file_path}/differences.relTo02/'
if not os.path.exists(directory):
	os.makedirs(directory)

for particle_file_2 in files:
	fname = re.findall(r'[^/]+(?=/$|$)', particle_file_2)[0]
	id_file_2 = re.findall(r'(.+).csv', fname)[0]
	print(f'Doing file: {fname}')
	df_2 = pd.read_csv(particle_file_2, index_col=0)
	# print(df_2.head())

	# create the two KDEs

	def return_kde_data(qy, qz, nbins, lim):

		x = qy
		y = qz
		data = np.vstack([qy, qz])
		k = kde.gaussian_kde(data)
		# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
		xi, yi = np.mgrid[-3:3:nbins*1j, -3:3:nbins*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))


		# compute FWHM for all points parallel to the x and y axis
		qry_eval = np.linspace(-lim,lim,100)
		eval_x = [k.evaluate([x,0])[0] for x in qry_eval]
		eval_y = [k.evaluate([0,y])[0] for y in qry_eval]


		# fit FWHM
		# Create a function which returns a Gaussian (normal) distribution.
		def gauss(p, x):
			a, b, c, d = p
			y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) + d
			return y
		def errfunc(p, x, y):
			return gauss(p, x) - y # Distance to the fit function

		p0 = [1, 1, 1, 1] # Initial guess for the parameters


		# fit for parallel to x axis
		X_f = qry_eval
		Y_f = eval_x
		# print(df.norm_cps)
		# print(X_f, Y_f)
		p1, success = optimize.leastsq(errfunc, p0[:], args=(X_f, Y_f), maxfev=100000)
		Y_fit_x = gauss(p1,X_f)


		# fit for parallel to y axis

		Y_f = eval_y
		# print(df.norm_cps)
		# print(X_f, Y_f)
		p1, success = optimize.leastsq(errfunc, p0[:], args=(X_f, Y_f))
		Y_fit_y = gauss(p1,X_f)

		return xi, yi, zi, Y_fit_x, Y_fit_y, X_f

	nbins = 200
	lim = 3
	xi_1, yi_1, zi_1, Y_fit_x_1, Y_fit_y_1, X_f = return_kde_data(df_1.qy, df_1.qz, nbins, lim)
	# print(np.shape(xi))
	# print(np.shape(yi))
	# print(np.shape(zi))

	xi_2, yi_2, zi_2, Y_fit_x_2, Y_fit_y_2, X_f = return_kde_data(df_2.qy, df_2.qz, nbins, lim)
	# print(np.shape(xi))
	# print(np.shape(yi))
	# print(np.shape(zi))

	# calculate the differences in zi (relative differences)
	zi_diff = (zi_2 - zi_1)
	# print(np.mean(zi_diff))
	# print(np.max(zi_diff))
	# print(np.min(zi_diff))
	fwhm_curve_diff_x = np.abs(np.abs(Y_fit_x_2 - Y_fit_x_1) / Y_fit_x_2)
	fwhm_curve_diff_y = np.abs(np.abs(Y_fit_y_2 - Y_fit_y_1) / Y_fit_y_2)

	# plot


	f, ax = plt.subplots(figsize=(7, 7))
	plt.title(f'Difference in KDE: {id_file_2} - {id_file_1}')
	nullfmt = NullFormatter()         # no labels
	p = ax.pcolormesh(xi_1, yi_1, zi_diff.reshape(xi_1.shape), shading='gouraud', cmap=plt.cm.jet)
	ax.set_facecolor('#000080ff')
	plt.colorbar(p)

	ax.set_xlim((-lim, lim))
	ax.set_ylim((-lim, lim))
	filename = f'{directory}/{id_file_2}'
	plt.savefig(filename + '.png', dpi=600)
	plt.close('all')
	plt.close()

	# FWHM_x and FWHM_y
	f, axarr = plt.subplots(2, figsize=(7, 7))
	plt.title(f'Difference in KDE: {id_file_2} - {id_file_1}')
	axarr[0].plot(X_f, Y_fit_x_1, label=f'{id_file_1}')
	axarr[0].plot(X_f, Y_fit_x_2, label=f'{id_file_2}')
	axarr[0].plot(X_f, fwhm_curve_diff_x, label=f'diff')
	axarr[0].set_ylim(-0.05,0.5)
	axarr[0].legend(loc='best')

	axarr[1].plot(X_f, Y_fit_y_1, label=f'{id_file_1}')
	axarr[1].plot(X_f, Y_fit_y_2, label=f'{id_file_2}')
	axarr[1].plot(X_f, fwhm_curve_diff_y, label=f'relDiff')
	axarr[1].set_ylim(-0.05,0.5)
	axarr[1].legend(loc='best')

	my_directory = f'{COMSOL_data_file_path}/differences.relTo02/fwhm/'
	if not os.path.exists(my_directory):
		os.makedirs(my_directory)
	print(id_file_2)
	my_id = re.findall(r'(\d\d.+particleData)', id_file_2)[0]
	filename = f'{my_directory}/{my_id}.fwhm'

	plt.savefig(filename + '.png', dpi=600)

	plt.close()
	plt.close('all')
