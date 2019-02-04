import pandas as pd
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


# Create a function which returns a Gaussian (normal) distribution.
def gauss(x, *p):
	a, b, c, d = p
	y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) + d
	return y

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# compute the FWHM
def extractFWHMvalue(x, *popt):
	X = np.linspace(np.min(x), np.max(x), 1000)  # expand x-space
	# print(X)
	Y = gauss(X, *popt)

	# maximum
	Y_max = np.max(Y)
	Y_arg_max = np.argmax(Y) # maximum Y value position
	X_max = X[Y_arg_max] # maximum X value

	# take only the positive range from the maximum on
	Y1 = Y[Y_arg_max-1:]
	X1 = X[Y_arg_max-1:]
	Y_FWHM1 = find_nearest(Y1, Y_max/2)
	Y_FWHM_pos1 = np.argwhere(Y1 == Y_FWHM1)
	X_FWHM1 = np.abs(X1[Y_FWHM_pos1][0])

	# take only the range until the maximum
	Y2 = Y[0:Y_arg_max]
	X2 = X[0:Y_arg_max]
	if len(Y2) > 0:
		Y_FWHM2 = find_nearest(Y2, Y_max/2)	
		Y_FWHM_pos2 = np.argwhere(Y2 == Y_FWHM2)
		X_FWHM2 = np.abs(X2[Y_FWHM_pos2][0])

		X_FWHM = X_FWHM1+X_FWHM2
		
		return X_FWHM
	else:
		return 2*X_FWHM1

def calculateFWHM(x,s):
	y = s.values.astype(float)
	# print('Doing value: {}'.format(s.name))
	m = [1e-2, 5e-2, 1e-1, 5e-1, 0, 1, 5, 1e1]

	for mm in list(itertools.product(m,m,m,m)):
		p_initial = [mm[0],mm[1],mm[2],mm[3]]
		popt, pcov = curve_fit(gauss, x, y, p0=p_initial, maxfev=1000000)
		y_fit = gauss(x, *popt)
		X_FWHM = extractFWHMvalue(x, *popt)[0]
		if X_FWHM == -1:
			return X_FWHM
		# print(X_FWHM)
		if (popt[0] < 10) and (popt[0] > 1e-3) and (popt[1] < 10) and (X_FWHM < np.max(x)):
			# print(X_FWHM)
			
			break

	return X_FWHM

def getLargestFWHM(k, lim):
	# k: scipy.gaussian_kde
	# lim: evaluation limits (mus tbe symmetrical)
	# evaluate along x (qy coordinate)
	qry_eval = np.linspace(-lim,lim,100)

	
	# create a dataframe, columns are the different parallel lines
	df_x = pd.DataFrame()  # parallel to x
	df_y = pd.DataFrame()  # parallel to y
	df_x_FWHM = pd.DataFrame()  # values of the FWHM parallel to x
	df_y_FWHM = pd.DataFrame()  # values of the FWHM parallel to x
	# the column names are the y (in the case of parallel to x) and x (in the case of parallel to y) values of the points
	# column cells are the values from the evaluation of the gaussian kde
	for e in qry_eval:
		df_x[e] = [k.evaluate([x,e])[0] for x in qry_eval] 
		df_y[e] = [k.evaluate([e,y])[0] for y in qry_eval] 

	
	df_tmp = df_x.apply(lambda x: calculateFWHM(qry_eval,x))
	df_x_FWHM = df_x_FWHM.append(df_tmp, ignore_index=True)

	df_tmp = df_y.apply(lambda x: calculateFWHM(qry_eval,x))
	df_y_FWHM = df_y_FWHM.append(df_tmp, ignore_index=True)
	
	return df_x_FWHM, df_y_FWHM
	# evaluate along y (qz coordinate)
	
	


	# return x1, y1, fwhm1, x2, y2, fwhm2  #x1 y1 is parallel to the x axis maximum fwhm curve, x2, y2 is parallel to y axis maximum fwhm curve

remote_path = '//fs03/LTH_Neutimag/hkromer/'  # PSI
# remote_path = '/home/hkromer/01_PhD/'  # local laptop


particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data_BIDIR/'.format(remote_path)
# particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/particle_data_TD/'.format(remote_path)

folders = os.listdir(particle_data_path)
folders = [f for f in folders if os.path.isdir('{}/{}'.format(particle_data_path,f))]

df_out = pd.DataFrame()
nn = 0
my_lvl = 0.5
for this_folder in folders:
	print('Doing now folder: {}'.format(this_folder))

	files = os.listdir('{}/{}'.format(particle_data_path, this_folder))
	files = [f for f in files if f.endswith('.csv')]
	files = [f for f in files if not f.startswith('df')]


	# bins for the histogram
	bins = np.linspace(0,3,100)
	
	for file in files:
		df_res = pd.DataFrame()
		print('Processing file {}'.format(file))
		type_file = re.findall(r'(.+).csv', file)[0]
		file = '{}/{}/{}'.format(particle_data_path,this_folder, file) 
		# import data
		df = pd.DataFrame()
		df = pd.read_csv(file, skiprows=8, header=None)

		# find column headers
		c = []
		with open(file, 'r') as myfile:
			for line in myfile:
				if 'Index' in line:
					l = line.rstrip().split(',')
					c.append(l)

		myfile.close()
		
		cols = []
		for item in c[0]:

			t0 = re.findall(r'mesh_refinement', item)
			if len(t0) < 1:
				cols.append(item)


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

		#set column header of df
		cols[0] = 'particleindex'
		df.columns = cols
		
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

		print('{}% of the initial {} particles have arrived at the target (x > {} mm and x < {} mm).'.format(perc_arrived, n_total, dist_min, dist_max))

		# index of the particle that have arrived at the target
		idx_arrived = df_arrived.index.tolist()


		# select only the last timestep
		this_df = df.filter(regex=time_cols[-1], axis=1)

		# compute beam radius

		this_df = this_df.iloc[idx_arrived,:]

		qy = this_df.iloc[:,1]
		qz = this_df.iloc[:,2]

		Lplasma = re.findall(r'Lplasma_(.+).csv', file)[0]


		# # plot histogram
		# f, ax = plt.subplots()

		# # my_width=0.01
		# ax.hist(qr.values, bins=100, facecolor='g', alpha=0.75, edgecolor='black', normed=True)
		# ax.plot(x, y_fit, color='red')
		# plt.ylim(0,1.2)
		# plt.xlabel('Bin')
		# plt.ylabel('Count')
		# plt.grid(True)
		# # plt.legend(loc='best')
		# plt.tight_layout()
		# # plt.show()
		# filename =  '{}/histogram_{}_MR'.format(particle_data_path, ID)
		# # plt.savefig(filename + '.eps', dpi=1200)
		# # plt.savefig(filename + '.svg', dpi=1200)
		# plt.savefig(filename + '.png', dpi=600)
		# plt.close('all')
		# # plt.show()

		# 2D Histogram plots
		directory = '{}/2D_histogram_lastTimestep'.format(particle_data_path)
		if not os.path.exists(directory):
			os.makedirs(directory)

		# Results for the FWHM
		FWHM_directory = '{}/FWHMs'.format(directory)
		if not os.path.exists(FWHM_directory):
			os.makedirs(FWHM_directory)

		# times = [1,2,3,4,5]
		# times = ['t={}E-7'.format(t) for t in times]
		# times = ['t=0'] + times
		# print(times)
		# for mytime in times:
	
	
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

		nbins = 50
		x = qy
		y = qz
		data = np.vstack([qy, qz])
		k = kde.gaussian_kde(data)
		xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		# scale between 0 and 1
		zi = (zi - np.min(zi))/(np.max(zi)-np.min(zi))
		# print(zi)


		f = plt.figure(1, figsize=(7, 7))
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

		
		# ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r
		# ax.set_title('Contour')
		# normi = mpl.colors.Normalize(vmin=0, vmax=1)
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
		# 
		df_res['qry_eval'] = qry_eval
		df_res['eval_x'] = eval_x
		df_res['eval_y'] = eval_y
		# df_res['kint_1'] = kint
		df_res['type_file'] = type_file
		# df_res['contour_level'] = my_lvl
		fwhm_x = calculateFWHM(qry_eval,df_res['eval_x'])
		fwhm_y = calculateFWHM(qry_eval,df_res['eval_y'])

		df_res['fwhm_x'] = fwhm_x
		df_res['fwhm_y'] = fwhm_y
		df_res.to_csv('{}/{}_df_res_file_{}.csv'.format(FWHM_directory,nn,type_file))
		# first of all, the base transformation of the data points is needed
		base = pyplot.gca().transData
		rot = transforms.Affine2D().rotate_deg(270)
		axScatter.plot([-lim, lim], [0, 0], color='black', linestyle='dashed')
		axScatter.plot([0, 0], [-lim, lim], color='black')

		axHistx.plot(df_res['qry_eval'].values, df_res['eval_x'], c='black', linestyle='dashed')
		axHisty.plot(df_res['qry_eval'].values, df_res['eval_y'].values[::-1], c='black', transform= rot + base)

		axHistx.set_xlim(axScatter.get_xlim())
		axHisty.set_ylim(axScatter.get_ylim())
		# f.tight_layout()
		# df_x_FWHM, df_y_FWHM = getLargestFWHM(k, lim)

		# df_x_FWHM.to_csv('{}/df_x_FWHM.csv'.format(directory))
		# df_y_FWHM.to_csv('{}/df_y_FWHM.csv'.format(directory))

		# plt.show()

		# sys.exit()
		df_out = df_out.append(df_res)
	
		filename =  '{}/{}_2D_histogram_Lplasma_{}_file_{}'.format(directory, nn, type_file)
		nn = nn + 1
		# print(nn)
		# plt.savefig(filename + '.eps', dpi=1200)
		# plt.savefig(filename + '.svg', dpi=1200)
		plt.savefig(filename + '.png', dpi=600)
		plt.close('all')
		# plt.show()
		

df_out.to_csv('{}/df_out.csv'.format(directory))