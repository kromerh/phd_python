import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import acos
from math import sqrt
from math import pi
import seaborn as sns

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner
def kreisbogen(alpha, r):
	b = np.pi * r * (alpha/180)
	return b

# datafile from COMSOL
path_to_data = "E://COMSOL/T_vs_bpower/220W.txt"

# Mesh refinement fluid
# 300W, 180 rpm, 0.05 kg/s, 3 mm Cu, 5 µm Ti
# Refinement factor	Maximum temperature [degC]		Number of elements
# ref_fac		maxT_degC	numEl

# import data
df = pd.read_csv(path_to_data, delimiter=r"\s+", skiprows=7)
print(df)

# df.set_index('y', inplace=True)
# Set up the figure
plt.contourf(df['x'], df['y'], df['Color'], 20, cmap='RdGy')
plt.colorbar();
plt.show()


# sns.heatmap(df)
# plt.show()
# df.to_csv('df.csv')

# convert y and z to s, which is the length along the arc
# radius
# r = 20  # mm
# def compute_arc(row):
# 	a = [row]	
# ang = angle_clockwise([1,1],[1,0])
# b = kreisbogen(ang, r)
# print(ang, 8*b, 2*np.pi*r)

# def find_nearest(array,value):
#     idx = (np.abs(array-value)).argmin()
#     return array[idx]

# x_unique = df['x'].unique()
# for x in x_unique:
# 	print(x)
# 	this_df = df[df['x'] == x]
# 	print(this_df.iloc[(this_df['y']-20).abs().argsort()[:1]])

# # -------------------------------------------------------------------
# # plot
# # -------------------------------------------------------------------
# plt.rc('text', usetex=True)
# plt.rc('font', weight='bold')
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Arial'
# matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
# matplotlib.rcParams['mathtext.tt'] = 'Arial'
# matplotlib.rcParams['mathtext.cal'] = 'Arial'
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

# fig = plt.figure(figsize=(8*0.60,5*0.60))

# ####################
# # axis 1
# ####################
# ax1 = fig.add_subplot(1, 1, 1)

# # plot
# ax1.plot(df['ref_fac'], df['maxT_degC'], 's-',color="darkblue", linewidth=1)
# # axes label
# ax1.set_ylabel(r'\textbf{Maximum target temperature [$^{\circ}$C]}', fontsize=12, labelpad=10)
# ax1.set_xlabel(r'\textbf{Mesh refinement}', fontsize=12, labelpad=2)
# plt.ylim(150,155)
# # ticks
# ax1.xaxis.set_ticks(df['ref_fac'].values)
# # ax1.yaxis.set_ticks([170, 175, 180, 185])
# # minor ticks x
# minor_locator = AutoMinorLocator(2)
# ax1.xaxis.set_minor_locator(minor_locator)
# # minor ticks y
# minor_locator = AutoMinorLocator(2)
# ax1.yaxis.set_minor_locator(minor_locator)
# # tick font size
# ax1.tick_params('x', colors='black', labelsize=12)	
# ax1.tick_params('y', colors='black', labelsize=12)	
# # grid
# ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
# ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')

# # ####################
# # # other axis
# # ####################
# # ax2 = ax1.twinx()
# # # plot
# # ax2.plot(df['vol_flow_rate_lpmin'], df['Re_number'], '--', marker='D', color='darkred', linewidth=2)

# # ax2.yaxis.set_ticks([1000,2000,4000,6000])
# # #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
# # # Use the pyplot interface to change just one subplot...
# # # cur_axes = plt.gca()
# # # plt.yticks([0, 1.4e7], [r"\textbf{0}", r"\textbf{1.4e7}"])
# # # ax2.spines['top'].set_visible(False)


# fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)

# #y label coordinates
# # ax1.yaxis.set_label_coords(-0.11,0.5)
# # plt.savefig('maximum_target_temperature_vs_coolant_flow_rate.eps', dpi=1200)
# # plt.savefig('maximum_target_temperature_vs_coolant_flow_rate.svg', dpi=1200)
# # plt.savefig('maximum_target_temperature_vs_coolant_flow_rate.png', dpi=1200)
# plt.show()


