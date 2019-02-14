import pandas as pd
import os
import re
import matplotlib.pyplot as plt

project_folder = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
suppression_electrode/sweep_bias/particleData/plots/\
2D_histograms_lastTimestep/'

folders = [f'{project_folder}{f}' for f in os.listdir(project_folder)]
folders = [f for f in folders if not f.endswith('.png')]
df = pd.DataFrame()
for folder in folders:
    this_df = pd.read_csv(f'{folder}/df_FWHMs.csv')

    df = df.append(this_df)


def extract_D_in_and_hole(df):
    id_col = df['id']
    df['D_in'] = re.findall(r'D_in(\d\d)', id_col)[0]
    df['hole'] = re.findall(r'D_in\d\d\.(\d)mm', id_col)[0]
    return df


df = df.apply(lambda x: extract_D_in_and_hole(x), axis=1)

print(df)

# Plot maximum electric field versus hole diameter for each bias voltage and
# for each diameter

diameters = df.D_in.unique()
V_biases = df.V_bias.unique()
holes = df.hole.unique()


def plot_FWHM_vs_hole(df):
    V_bias = df['V_bias'].unique()[0]
    # print(df)
    print(f'Creating plot for bias voltage {V_bias}')
    fig, axarr = plt.subplots(2, figsize=(8, 8))
    for diameter in diameters:
        this_df = df[df.D_in == diameter]
        this_df = this_df.sort_values(by=['hole'])
        X = this_df['hole'].values.astype(float)
        FWHM_x = this_df['FWHM_x'].values.astype(float)
        FWHM_y = this_df['FWHM_y'].values.astype(float)
        axarr[0].plot(X, FWHM_x)
        axarr[0].scatter(X, FWHM_x)
        axarr[0].set_ylabel('FWHM_x [mm]')
        axarr[1].plot(X, FWHM_y)
        axarr[1].scatter(X, FWHM_y, label=diameter)
        axarr[1].set_ylabel('FWHM_y [mm]')
    figname = f'{project_folder}/plot.FWHM_vs_hole.V_bias{V_bias}_V.png'
    plt.legend(loc='best', title='D_in mm')
    plt.xlabel('Extraction hole [mm]')
    axarr[0].grid(True)
    axarr[1].grid(True)
    axarr[0].set_title('1.2 mm diameter extraction, RF ion source, -100 kV')
    axarr[0].set_ylabel('FWHM_x [mm]')
    axarr[1].set_ylabel('FWHM_y [mm]')
    plt.savefig(figname, dpi=600)
    plt.close('all')


df.groupby('V_bias',
        as_index=True).apply(lambda x: plot_FWHM_vs_hole(x))


def plot_FWHM_vs_diameter(df):
    V_bias = df['V_bias'].unique()[0]
    # print(df)
    print(f'Creating plot for bias voltage {V_bias}')
    fig, axarr = plt.subplots(2, figsize=(8, 8))
    for hole in holes:
        this_df = df[df.hole == hole]
        this_df = this_df.sort_values(by=['D_in'])
        # print(this_df)
        X = this_df['D_in'].values.astype(float)
        FWHM_x = this_df['FWHM_x'].values.astype(float)
        FWHM_y = this_df['FWHM_y'].values.astype(float)
        axarr[0].plot(X, FWHM_x)
        axarr[0].scatter(X, FWHM_x)
        axarr[0].set_ylabel('FWHM_x [mm]')
        axarr[1].plot(X, FWHM_y)
        axarr[1].scatter(X, FWHM_y, label=hole)
        axarr[1].set_ylabel('FWHM_y [mm]')
    figname = f'{project_folder}/plot.FWHM_vs_diameter.V_bias{V_bias}_V.png'
    plt.legend(loc='best', title='D_in mm')
    plt.xlabel('Suppression electrode inner diameter [mm]')
    axarr[0].grid(True)
    axarr[1].grid(True)
    axarr[0].set_title('1.2 mm diameter extraction, RF ion source, -100 kV')
    axarr[0].set_ylabel('FWHM_x [mm]')
    axarr[1].set_ylabel('FWHM_y [mm]')
    plt.savefig(figname, dpi=600)
    plt.close('all')


df.groupby('V_bias',
        as_index=True).apply(lambda x: plot_FWHM_vs_diameter(x))
