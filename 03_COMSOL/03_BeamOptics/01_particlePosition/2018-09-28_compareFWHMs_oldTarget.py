import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# directory where the FWHM file is located

directory = '//fs03/LTH_Neutimag/hkromer/02_Simulations/06_COMSOL/\
03_BeamOptics/01_OldTarget/IGUN_geometry/2018-09-18_comsolGeometry/\
02.define_release_time/particleData/plots/2D_histograms_lastTimestep/'

# directory = '//fs03/LTH_Neutimag/hkromer/02_Simulations/06_COMSOL/\
# 03_BeamOptics/01_OldTarget/IGUN_geometry/2018-09-24_comsol/\
# define_current/particleData/plots/2D_histograms_lastTimestep/'

fname_fwhm = f'{directory}df_FWHMs.csv'

df_fwhm = pd.read_csv(fname_fwhm, index_col=0)

print(df_fwhm.head())

# plot the fwhms in two separate plots for TD and BIDIR
f, axarr = plt.subplots(2, figsize=(7, 7), sharex=True)
# TD
def plot_TD(df):
    # print(df)
    df = df.sort_values(by=['id'])
    X = df.id.values
    Y = [df.FWHM_x.values, df.FWHM_y.values]
    if df.run_type.unique()[0] == 'TD':
        p1,=axarr[0].plot(X, Y[0], marker='o', color='darkorange')
        p2,=axarr[0].plot(X, Y[1], marker='s', color='darkblue')
        axarr[0].set_title('TD')
        # axarr[0].legend([p1,p2], ['x-direction', 'y-direction'])
        axarr[0].grid()
    else:
        p3,=axarr[1].plot(X, Y[0], marker='o', label='x-direction', color='darkorange')
        p4,=axarr[1].plot(X, Y[1], marker='s', label='y-direction', color='darkblue')
        axarr[1].legend([p3,p4], ['x-direction', 'y-direction'])
        axarr[1].set_title('BIDIR')
        axarr[1].grid()




df_fwhm.groupby('run_type').apply(lambda x: plot_TD(x))
plt.xlabel('ID')
plt.xticks(np.arange(1,15,1))
plt.grid()
f.text(0.04, 0.5, 'FWHM [mm]', va='center', rotation='vertical')

figname =  f'{directory}FWHM_plots'
plt.savefig(figname + '.png', dpi=600)
plt.close('f')
