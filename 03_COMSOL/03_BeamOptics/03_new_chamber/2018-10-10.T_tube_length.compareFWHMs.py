import os
import pandas as pd
import re
import matplotlib.pyplot as plt

project_folder = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
2018-10-03_comsol/new_ion_source/T_tube_length/plots/\
2D_histograms_lastTimestep/'

folders = os.listdir(project_folder)
folders = [f for f in folders if 'sweep_L_tube' in f]
folders = [f'{project_folder}{f}/' for f in folders]

# loop through the folders and collect the FWHMs
df = pd.DataFrame()
for folder in folders:
    this_df = pd.read_csv(f'{folder}/df_FWHMs.csv', index_col=0)
    df = df.append(this_df)


def extract_L_tube(id_col):
    L_tube = re.findall(r'\.ID_(\d+)mm', id_col)[0]
    return L_tube


df['D_tube'] = df['id'].apply(lambda x: extract_L_tube(x))

print(df)
# plot
fig, axarr = plt.subplots(2, figsize=(8, 8), sharex=True)
# top figure
df = df.sort_values(by=['D_tube'])


def plot_FWHM_vs_distance_target_aperture(df):
    L_tube = df['D_tube'].unique()[0]
    df = df.sort_values(by=['D_target_to_aperture'])
    X = df['D_target_to_aperture'].values.astype(float)
    FWHM_x = df['FWHM_x'].values.astype(float)
    FWHM_y = df['FWHM_y'].values.astype(float)
    axarr[0].plot(X, FWHM_x)
    axarr[0].scatter(X, FWHM_x)
    axarr[0].set_ylabel('FWHM_x [mm]')
    axarr[1].plot(X, FWHM_y)
    axarr[1].scatter(X, FWHM_y, label=L_tube)
    axarr[1].set_ylabel('FWHM_y [mm]')


df.groupby('D_tube', as_index=True).\
    apply(lambda x: plot_FWHM_vs_distance_target_aperture(x))
plt.xlabel('Distance target to aperture [mm]')
plt.legend(loc='best', title='D_tube [mm]')
axarr[0].set_title('New ion source, grounded chamber, -100 kV, no suppression')

axarr[0].grid(True)
axarr[1].grid(True)

plt.savefig(f'{project_folder}plots_FWHMs.png', dpi=600)
plt.close('all')
