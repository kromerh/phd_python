import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# path to the temperature data from COMSOL
data_folder = '//fs03/LTH_Neutimag/hkromer/01_Software/19_Python/COMSOL/Target_Temperature/data/'
# folder to store the plots
plot_folder = '//fs03/LTH_Neutimag/hkromer/01_Software/19_Python/COMSOL/Target_Temperature/plots/'

# Current target
# Model:              runCR_lam_current_output.mph
# Version:            COMSOL 5.2.0.220
# Date:               May 15 2017, 17:54
# Table:              Table 1: T Volume - Volume Maximum 1 (T)
# v_rotation (rpm)       p_beam (W)               Temperature (degC)
df_CR = pd.read_csv(data_folder + 'COMSOL_Tmax_CR_lam.m', skiprows=5, delim_whitespace=True, header=None)
df_CR.rename(columns = {0:'rpm', 1:'p', 2:'T_max'}, inplace=True)

# unique beam power values
rpm = df_CR['rpm'].unique()
# print(rpm)


# New target
# % Model:              run40_lam_final_2_output.mph
# % Version:            COMSOL 5.2.0.220
# % Date:               Jan 18 2017, 12:34
# % Table:              Table 2 - Volume Maximum 1 (T)
# % v_rotation (rpm)       p_beam (W)               Temperature (K)
df_40 = pd.read_csv(data_folder + 'COMSOL_Tmax_40_lam.m', skiprows=5, delim_whitespace=True, header=None)
df_40.rename(columns = {0:'rpm', 1:'p', 2:'T_max'}, inplace=True)

# convert T_max from K to °C
df_40['T_max'] = df_40['T_max'] - 273.15

# unique beam power values
rpm_40 = df_40['rpm'].unique()

# calculate beam power density
r_beam = 0.1  # beam radius in cm
A_beam = np.pi * (r_beam)**2# beam area on target, assumption to be a disc
df_40['p_dens_kWpcm2'] = (df_40['p'] / A_beam) / 1000

df_40.to_csv('pbeam.csv')

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


ax1 = fig.add_subplot(1, 1, 1)
colors = ['black', 'green', 'blue', 'orange']
markers = ['o', 's', 'd', 'p']
# colors_40 = ['green', 'blue', 'magenta', 'orange']
colors_40 = ['darkblue', 'g', 'orange', 'red']
markers_40 = ['o', 's', 'd', 'x', 'p']

# # current target
# plot1 = []
# for this_rpm, c, m in zip(rpm, colors, markers):
#     this_df = df_CR[ df_CR['rpm']  == this_rpm ]
#     _, = ax1.plot(this_df['p'], this_df['T_max'], '--', marker=m, label=r""+str(this_rpm)+" rpm", color=c,  mfc='white', markersize=10)
#     plot1.append(_)


# New target
plot2 = []
for this_rpm, c, m in zip(rpm_40, colors_40, markers_40):
    this_df = df_40[ df_40['rpm']  == this_rpm ]
    # print(df_40['p'])
    _, = ax1.plot(this_df['p_dens_kWpcm2'], this_df['T_max'], '-', marker=m, label=r""+str(this_rpm)+" rpm", color=c, markersize=10)
    plot2.append(_)

# Shrink current axis by 20%
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Add vertical line (outgassing)
# ax1.plot([-10,400],[200,200], linestyle='dotted',color='red',linewidth=2)
# ax1.text(200, 210, r'\textbf{T}$_{\textrm{\textbf{outgas}}}$', fontsize=10, color='red')

# add separate legends
# legend1 = plt.legend(handles=plot1, bbox_to_anchor=[1, 0.8], loc='center left', title=r"\textbf{Current target}")
legend2 = plt.legend(handles=plot2, loc='upper left',fontsize=10)
legend2.set_title(r"Target rotational velocity", prop = {'size': 10})
# ax1.add_artist(legend1)
ax1.add_artist(legend2)
# fig.subplots_adjust(bottom=0.15)
plt.xlabel(r'\textbf{Beam power density [kW/cm$^2$]}', fontsize=12, labelpad=10)
plt.ylabel(r'\textbf{Maximum target temperature [$^\circ$C]}', fontsize=12, labelpad=10)
# plt.title(r'\textbf{Estimated Maximum Surface Temperature}', fontsize=16, y = 1.04)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, which='minor', linestyle='--')
# ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e4))
# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
# ax1.xaxis.set_ticks(this_df['p_dens_kWpcm2'])
plt.ylim(0,250)
plt.xlim(-0.1,10)
# ticks
# y_ticks = np.arange(4.5e5,7.5e5+0.5e5, 0.5e5)
# y = np.arange(4.5, 7.5+0.5, 0.5)
# plt.yticks(y_ticks, y)
# x
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
# y
minor_locator = AutoMinorLocator(2)
ax1.yaxis.set_minor_locator(minor_locator)
# tick font size
ax1.tick_params('x', colors='black', labelsize=12)  
ax1.tick_params('y', colors='black', labelsize=12)  
# plt.legend(loc='best', title=r"\textbf{Legend}")
# plt.xticks(df_orig['distance'])
# ax1.axes.xaxis.set_ticklabels([])
# plt.gca().fill_between([-10, 400],
#                        [200, 200],
#                        [400, 400],
#                        facecolor='red',
#                        alpha=0.25)
# plt.gca().fill_between(df_south['distance'],
#                        df_south['total_neutron_output_from_BS2_cps'], df_south['total_neutron_output'],
#                        facecolor='red',
#                        alpha=0.25)
# plt.gca().fill_between(df_east['distance'],
#                        df_east['total_neutron_output_from_BS2_cps'], df_east['total_neutron_output'],
#                        facecolor='olive',
#                        alpha=0.25)
# plt.savefig(plot_folder + 'comparison_CR_and_40_Tmax_vs_p.png', dpi=2400)
# plt.savefig(plot_folder + 'comparison_CR_and_40_Tmax_vs_p.pdf', dpi=600)
fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
plt.savefig('maximum_target_temperature_vs_beam_power_density.eps', dpi=1200)
plt.savefig('maximum_target_temperature_vs_beam_power_density.svg', dpi=1200)
plt.savefig('maximum_target_temperature_vs_beam_power_density.png', dpi=1200)
plt.show()
# plt.close()