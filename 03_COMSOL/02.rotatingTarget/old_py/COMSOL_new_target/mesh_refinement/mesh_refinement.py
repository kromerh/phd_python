import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator



# datafile from COMSOL
path_to_data = "//fs03/LTH_Neutimag/hkromer//01_Software/19_Python/COMSOL_new_target/mesh_refinement/fluid_results.txt"

# Mesh refinement fluid
# 300W, 180 rpm, 0.05 kg/s, 3 mm Cu, 5 µm Ti
# Refinement factor	Maximum temperature [degC]		Number of elements
# ref_fac		maxT_degC	numEl

# import data
df = pd.read_csv(path_to_data, delimiter=r"\t+", skiprows=3)


df.to_csv('df.csv')

# -------------------------------------------------------------------
# plot
# -------------------------------------------------------------------
plt.rc('text', usetex=True)
plt.rc('font', weight='bold')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
matplotlib.rcParams['mathtext.tt'] = 'Arial'
matplotlib.rcParams['mathtext.cal'] = 'Arial'
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

fig = plt.figure(figsize=(8*0.60,5*0.60))

####################
# axis 1
####################
ax1 = fig.add_subplot(1, 1, 1)

# plot
ax1.plot(df['ref_fac'], df['maxT_degC'], 's-',color="darkblue", linewidth=1)
# axes label
ax1.set_ylabel(r'\textbf{Maximum target temperature [$^{\circ}$C]}', fontsize=12, labelpad=10)
ax1.set_xlabel(r'\textbf{Mesh refinement}', fontsize=12, labelpad=2)
plt.ylim(150,155)
# ticks
ax1.xaxis.set_ticks(df['ref_fac'].values)
# ax1.yaxis.set_ticks([170, 175, 180, 185])
# minor ticks x
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
# minor ticks y
minor_locator = AutoMinorLocator(2)
ax1.yaxis.set_minor_locator(minor_locator)
# tick font size
ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	
# grid
ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')

# ####################
# # other axis
# ####################
# ax2 = ax1.twinx()
# # plot
# ax2.plot(df['vol_flow_rate_lpmin'], df['Re_number'], '--', marker='D', color='darkred', linewidth=2)

# ax2.yaxis.set_ticks([1000,2000,4000,6000])
# #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
# # Use the pyplot interface to change just one subplot...
# # cur_axes = plt.gca()
# # plt.yticks([0, 1.4e7], [r"\textbf{0}", r"\textbf{1.4e7}"])
# # ax2.spines['top'].set_visible(False)


fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)

#y label coordinates
# ax1.yaxis.set_label_coords(-0.11,0.5)
# plt.savefig('maximum_target_temperature_vs_coolant_flow_rate.eps', dpi=1200)
# plt.savefig('maximum_target_temperature_vs_coolant_flow_rate.svg', dpi=1200)
# plt.savefig('maximum_target_temperature_vs_coolant_flow_rate.png', dpi=1200)
plt.show()


