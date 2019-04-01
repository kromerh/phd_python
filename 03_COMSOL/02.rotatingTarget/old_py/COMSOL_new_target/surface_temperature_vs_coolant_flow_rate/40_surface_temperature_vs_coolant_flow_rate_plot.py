import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator



# datafile from COMSOL
path_to_data = "//fs03/LTH_Neutimag/hkromer/02_Simulations/03_MATLAB/2_target_temperature/run40_lowRe_pureCu/COMSOL_run40_lowRe_massFlowRate.m"

# % Model:              run40_lam_7_3_m_1.mph
# % Version:            COMSOL 5.2.0.220
# % Date:               Dec 19 2016, 13:10
# % Table:              Table 1 - Volume Maximum 1 (T)
# % p_beam = 200 W
# % v_rotation = 25 rpm
# % d_Cu = 3 mm, material CuOFE
# % d_Ti = 5 µm
# % d_water = 2.675 mm
# %volumetric flow rate l/min	% mass flow rate (kg/s)  %  maximum surface temperature (K)  % Reynolds number
# vol_flow_rate_lpmin	mass_flow_rate_kgps	max_surf_temp_K	Re_number

# import data
df = pd.read_csv(path_to_data, delimiter="\t", skiprows=10)

# convert Temperature to degC
df['max_surf_temp_C'] = df['max_surf_temp_K'] - 273.15


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

fig = plt.figure(figsize=(8*0.70,5*0.75))

####################
# axis 1
####################
ax1 = fig.add_subplot(1, 1, 1)

# plot
ax1.plot(df['vol_flow_rate_lpmin'], df['max_surf_temp_C'], 's-',color="darkblue", linewidth=1)
# axes label
ax1.set_ylabel(r'\textbf{Maximum target temperature [$^{\circ}$C]}', fontsize=12)
ax1.set_xlabel(r'\textbf{Volumetric flow rate [l/min]}', fontsize=12, labelpad=2)
# ticks

# ax1.xaxis.set_ticks(df['vol_flow_rate_lpmin'].values)
ax1.xaxis.set_ticks(np.arange(2,24,2))
ax1.yaxis.set_ticks(np.arange(100,260+40,40))
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
plt.gca().fill_between([0,25],
                        240, 260,
                        facecolor='red',
                        alpha=0.25)
plt.ylim(90,260)
plt.xlim(2,22)
ax1.text(4.25, 245,r"\textbf{T$_{outgas}$}", fontsize=12, color='red')
fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)

#y label coordinates
ax1.yaxis.set_label_coords(-0.11,0.5)
plt.savefig('Figure_6_maximum_target_temperature_vs_coolant_flow_rate.eps', dpi=1200)
plt.savefig('Figure_6_maximum_target_temperature_vs_coolant_flow_rate.svg', dpi=1200)
plt.savefig('Figure_6_maximum_target_temperature_vs_coolant_flow_rate.png', dpi=1200)
plt.show()


