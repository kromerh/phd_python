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


# # single file processing
remote_path = '//fs03/LTH_Neutimag/hkromer/'
datefile = '2018-08-14_comsol'

COMSOL_data_file_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/02_current_target/particle_data/{}/particle_data/'.format(remote_path, datefile)

# # sweep_HV
COMSOL_data_file = '{}disc_uniform_Lplasma_0.4_sweep_HV.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
sweep_variable = 'V_HV'

# sweep_extrHV
# COMSOL_data_file = '{}disc_uniform_Lplasma_0.4_sweep_extrHV.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
# sweep_variable = 'V_extraction'

# # sweep_Ekin
# COMSOL_data_file = '{}disc_uniform_Lplasma_0.4_sweep_Ekin.csv'.format(COMSOL_data_file_path)  # change this to each filename, i.e. loop
# sweep_variable = 'E_kin


sweep_type = re.findall(r'_sweep_([\w]+).csv$',COMSOL_data_file)
fname = re.findall(r'/([\w.]+).csv',COMSOL_data_file)[0]

# import the particle data file
df_main = pd.read_csv(COMSOL_data_file, skiprows=8, header=None)

# find column headers
c = []
with open(COMSOL_data_file, 'r') as myfile:
	for line in myfile:
		if 'Index' in line:
			l = re.sub(r'(=\d)(,)', r'\1 @', line)  # t=0
			l = re.sub(r'(=\d\w\-\d)(,)', r'\1 @', l)  # t=1E-1
			l = re.sub(r'(=\d\.\d\w\-\d)(,)', r'\1 @', l) # t=1.1E-1
			l = re.sub(r'(=\d\.\d\d\w\-\d)(,)', r'\1 @', l) # t=1.15E-1
			l = l.rstrip().split(',')
			c.append(l)

myfile.close()
cols = c[0]

# print(time_cols)
#set column header of df
cols[0] = 'particleindex'
df_main.columns = cols

# get all the sweep values
s_cols = ','.join(cols)  # join into string
sweep_vals = re.findall(sweep_variable + r'=([^,]+),', s_cols)
sweep_vals = list(set(sweep_vals))  # get only unique values in the list

# df_main.to_csv('{}/{}_cleaned.csv'.format(COMSOL_data_file_path,fname))
# print(sweep_vals)

# sys.exit()

df_output = pd.DataFrame()  # contains the sweep_variable, FWHM_x, FWHM_y

# filter by the sweep parameter
for kk in range(0,len(sweep_vals)):
	print('Processing {} sweep value {}'.format(sweep_variable, sweep_vals[kk]))

	df = df_main.filter(regex=sweep_vals[kk], axis = 1)
	cols = df.columns

	df_FWHM = pd.DataFrame()  # this is the output dataframe that contains the FWHM
	df_FWHM[sweep_variable] = sweep_vals[kk]

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
	dist_min = 10
	dist_max = 90
	df_arrived = df_last[ df_last.iloc[:,0] > dist_min ]
	df_arrived = df_last[ df_last.iloc[:,0] < dist_max ]
	n_arrived = len(df_arrived)
	# percent of those that have arrived
	perc_arrived = round((n_arrived/n_total)*100.0,2)

	print(f'{perc_arrived}% of the initial {n_total} particles have arrived at the target (x > {dist_min} mm and x < {dist_max} mm).')
	# print(df.head())
	# print(sys.exit())
	# index of the particle that have arrived at the target
	idx_arrived = df_arrived.index.tolist()


	# select only the last timestep
	this_df = df.filter(regex=time_cols[-1], axis=1)

	# compute beam radius

	this_df = this_df.iloc[idx_arrived,:]

	qy = this_df.iloc[:,1]
	qz = this_df.iloc[:,2]

	
	# print(fname)
	directory = '{}/{}_2D_histogram_lastTimestep'.format(COMSOL_data_file_path,fname)
	if not os.path.exists(directory):
		os.makedirs(directory)




	mytime = time_cols[-1]
	print('Creating plot for time time {} s'.format(mytime))
	# select only the last timestep
	this_df = df.filter(regex=mytime, axis=1)

	# compute beam radius

	this_df = this_df.iloc[idx_arrived,:]
	qx = this_df.iloc[:,0]
	median_qx = np.median(qx)
	qy = this_df.iloc[:,1]
	qz = this_df.iloc[:,2]
	qr = np.sqrt(qy**2+qz**2)

	# nbins = 500
	nbins = 250
	x = qy
	y = qz
	data = np.vstack([qy, qz])
	k = kde.gaussian_kde(data)
	# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
	xi, yi = np.mgrid[-3:3:nbins*1j, -3:3:nbins*1j]
	zi = k(np.vstack([xi.flatten(), yi.flatten()]))
	# scale between 0 and 1
	zi = (zi - np.min(zi))/(np.max(zi)-np.min(zi))
	# print(zi)


	f = plt.figure(1, figsize=(9.5, 9.5))
	# plt.title('KDE Gaussian on target for run \n {}'.format(type_file))
	nullfmt = NullFormatter()         # no labels

	# definitions for the axes
	left, width = 0.05, 0.65
	bottom, height = 0.05, 0.65
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	axScatter = plt.axes(rect_scatter)

	axHistx = plt.axes(rect_histx)
	axHisty = plt.axes(rect_histy)

	# no labels
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)
	axHistx.grid(True)
	axHisty.grid(True)


	p = axScatter.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.jet)

	# contours = axScatter.contour(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Blues, levels=[my_lvl])
	# plt.clabel(contours, inline=True, fontsize=8)
	axScatter.set_facecolor('#000080ff')
	plt.colorbar(p)

	# compute FWHM for all x and y histograms
	# select the largest FWHM
	lim = 3
	axScatter.set_xlim((-lim, lim))
	axScatter.set_ylim((-lim, lim))

	# compute FWHM for all points parallel to the x and y axis
	qry_eval = np.linspace(-lim,lim,100)
	eval_x = [k.evaluate([x,0])[0] for x in qry_eval] 
	eval_y = [k.evaluate([0,y])[0] for y in qry_eval]

	# print(kint)
	df_res = pd.DataFrame()
	df_res['qry_eval'] = qry_eval
	df_res['eval_x'] = eval_x
	df_res['eval_y'] = eval_y

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
	Y_fit = gauss(p1,X_f)
	# save df to csv
	df_FWHM_x = pd.DataFrame(X_f, columns=['X_fit'])
	df_FWHM_x['Y_fit'] = Y_fit
	df_FWHM_x['sigma'] = p1[2]  # sigma in gaussian
	df_FWHM_x['FWHM'] = 2.08 * p1[2] * np.sqrt(2 * np.log(2))  # FWHM
	# fname = f'{master_folder}/df_FWHM_x.csv'
	# df_FWHM_x.to_csv(fname)
	df_FWHM['FWHM_x'] = df_FWHM_x['FWHM'].unique()


	# fit for parallel to y axis
	X_f = qry_eval
	Y_f = eval_y
	# print(df.norm_cps)
	# print(X_f, Y_f)
	p1, success = optimize.leastsq(errfunc, p0[:], args=(X_f, Y_f))
	Y_fit = gauss(p1,X_f)
	# save df to csv
	df_FWHM_y = pd.DataFrame(X_f, columns=['X_fit'])
	df_FWHM_y['Y_fit'] = Y_fit
	df_FWHM_y['sigma'] = p1[2]  # sigma in gaussian
	df_FWHM_y['FWHM'] = 2.08 * p1[2] * np.sqrt(2 * np.log(2))  # FWHM
	# fname = f'{master_folder}/df_FWHM_y.csv'
	# df_FWHM_y.to_csv(fname)
	df_FWHM['FWHM_y'] = df_FWHM_y['FWHM'].unique()

	df_FWHM['FWHM'] = (np.abs(df_FWHM['FWHM_y']) + np.abs(df_FWHM['FWHM_x']) ) / 2.0
	# first of all, the base transformation of the data points is needed
	base = pyplot.gca().transData
	rot = transforms.Affine2D().rotate_deg(270)
	axScatter.plot([-lim, lim], [0, 0], color='black', linestyle='dashed')
	axScatter.plot([0, 0], [-lim, lim], color='black')

	axHistx.plot(df_res['qry_eval'].values, df_res['eval_x'], c='black', linestyle='dashed')
	axHistx.plot(df_FWHM_x['X_fit'], df_FWHM_x['Y_fit'], c='red', linestyle='dotted')
	axHisty.plot(df_res['qry_eval'].values, df_res['eval_y'].values[::-1], c='black', transform= rot + base)
	axHisty.plot(df_FWHM_y['X_fit'], df_FWHM_y['Y_fit'].values[::-1], c='red', linestyle='dotted', transform= rot + base)

	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())
	# f.tight_layout()
	# df_x_FWHM, df_y_FWHM = getLargestFWHM(k, lim)

	# df_x_FWHM.to_csv('{}/df_x_FWHM.csv'.format(directory))
	# df_y_FWHM.to_csv('{}/df_y_FWHM.csv'.format(directory))

	# plt.show()
	axScatter.set_xlabel('y [mm]')
	axScatter.set_ylabel('z [mm]')
	filename =  '{}/2D_histogram_sweep_value_{}'.format(directory,sweep_vals[kk])

	# print(nn)
	# plt.savefig(filename + '.eps', dpi=1200)
	# plt.savefig(filename + '.svg', dpi=1200)
	plt.savefig(filename + '.png', dpi=600)
	plt.close('all')
	df.to_csv('{}.csv'.format(filename))

	df_output = df_output.append(df_FWHM)
	# plt.show()
			

fname = f'{COMSOL_data_file_path}/df_res_{sweep_variable}.csv'
df_output.to_csv(fname)