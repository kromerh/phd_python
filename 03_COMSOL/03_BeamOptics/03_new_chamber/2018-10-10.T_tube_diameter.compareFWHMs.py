import pandas as pd
import matplotlib.pyplot as plt


project_folder = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
2018-10-03_comsol/new_ion_source/T_tube_diameter/plots/\
2D_histograms_lastTimestep/01.sweep_ID_T_tube.particleData/'

df = pd.read_csv(f'{project_folder}df_FWHMs.csv', index_col=0)
df = df.sort_values(by=['ID_T_tube'])

fig, ax = plt.subplots(figsize=(8, 8))
X = df.ID_T_tube.values.astype(float)
FWHM_x = df.FWHM_x.values.astype(float)
FWHM_y = df.FWHM_y.values.astype(float)
ax.plot(X, FWHM_x, label='FWHM_x')
ax.scatter(X, FWHM_x)
ax.plot(X, FWHM_y, label='FWHM_y')
ax.scatter(X, FWHM_y)
plt.legend(loc='best')
plt.grid()
plt.title('New ion source, grounded chamber, -100 kV, no suppression')
plt.xlabel('ID accelerator tube (grounded) [mm]')
plt.ylabel('FWHM [mm]')
plt.savefig(f'{project_folder}plot_FWHMs.png', dpi=600)
plt.close('all')
