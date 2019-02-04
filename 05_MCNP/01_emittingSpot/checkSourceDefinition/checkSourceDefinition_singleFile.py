import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from scipy.interpolate import interp1d
import itertools
from scipy.stats import kde
from scipy import optimize
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot, transforms
pd.set_option("display.max_columns",101)

# load the ptrac file
master_folder = 'C:/Users/kromer_h/Downloads/ex/'
fname = f'{master_folder}/ptrac'

# # create a pd dataframe
# df = pd.DataFrame()

# # loop through the ptrac file line by line. Find birth values X Y Z and check u v w for sanity
# particle_id = 0
# lines = []  # these are lines that contain the info on the starting particle data
# ii = 0
# with open(fname, 'r') as file:
# 	for line in file:
# 		l = line.rstrip().split()
# 		# print(l)
# 		if len(l) < 2:
# 			# print(ii)
# 			continue
# 		if l[1] == '1000':  # birth event
# 			# print(ii)
# 			if int(l[0]) == particle_id + 1:  # l[1] does not have to be a 1000 event
# 				lines.append(ii + 3)
# 				particle_id = particle_id + 1

# 		ii = ii + 1
# 	file.close()

# # sanity check: It must be 1e5 particles because that is how many were born with MCNP
# nps = 1e4
# print(f'NPS is {nps}, found {len(lines)} birth events in ptrac file.')

# # loop again through the file, this time take the X Y Z and u v w from the line
# ii = 0
# jj = 0
# # kk = 0  # just for testing and early breaking of the loop
# with open(fname, 'r') as file:
# 	for line in file:
# 		l = line.rstrip().split()

# 		if jj >= len(lines):
# 			print('jj > len(lines)')
# 			break
# 		if ii == lines[jj]:
# 			s = pd.Series([l[0], l[1], l[2], l[3], l[4], l[5]], index=['X', 'Y', 'Z', 'u', 'v', 'w'])
# 			df = df.append(s, ignore_index=True)

# 			jj = jj + 1
		
# 			# kk = kk + 1
# 		# if kk == 100:
# 		# 	df.to_csv('C:/Users/kromer_h/Downloads/_experiment2018-08-07/df_ptrac.csv')
# 		# 	break			
# 		if ii%10000 == 0:  # every 1e4 lines
# 			print(f'Reading ptrac file line {ii}')
# 			fname = f'{master_folder}/df_ptrac.csv'
# 			df.to_csv(fname)
# 		ii = ii + 1
# 	file.close()

# fname = f'{master_folder}/df_ptrac.csv'
# df.to_csv(fname)



fname = f'{master_folder}/df_ptrac.csv'

df = pd.read_csv(fname, index_col=0)
# print(df.head())

nbins = 50
x = df['X'].values.astype(float)
y = df['Y'].values.astype(float)
data = np.vstack([x, y])
k = kde.gaussian_kde(data)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
# scale between 0 and 1
zi = (zi - np.min(zi))/(np.max(zi)-np.min(zi))
# print(zi)


f = plt.figure(1, figsize=(8, 8))
# plt.title('KDE Gaussian on target for run \n {}'.format(type_file))
nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.08, 0.66
bottom, height = 0.08, 0.66
bottom_h = left_h = left + width + 0.04

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
lim = 1
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
p1, success = optimize.leastsq(errfunc, p0[:], args=(X_f, Y_f))
Y_fit = gauss(p1,X_f)
# save df to csv
df_FWHM_x = pd.DataFrame(X_f, columns=['X_fit'])
df_FWHM_x['Y_fit'] = Y_fit
df_FWHM_x['sigma'] = p1[2]  # sigma in gaussian
df_FWHM_x['FWHM'] = 2.08 * p1[2] * np.sqrt(2 * np.log(2))  # FWHM
fname = f'{master_folder}/df_FWHM_x.csv'
df_FWHM_x.to_csv(fname)

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
fname = f'{master_folder}/df_FWHM_y.csv'
df_FWHM_y.to_csv(fname)

df_res.to_csv(f'{master_folder}/df_res.csv')
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




# plt.show()
filename = f'{master_folder}/plotKDE'
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')