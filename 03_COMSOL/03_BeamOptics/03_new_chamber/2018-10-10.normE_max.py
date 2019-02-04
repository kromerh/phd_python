import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt

remote_path = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
suppression_electrode/sweep_bias/es_normE.max/'

files = os.listdir(remote_path)
files = [f for f in files if f.endswith('.csv')]
df_main = pd.DataFrame()

for file in files:
    print(f'Doing file {file}')
    filepath = f'{remote_path}{file}'
    df = pd.read_csv(filepath, header=None, skiprows=5, index_col=None)
    # print(df)
    df.columns = ['V_bias', 't', 'es.NormE', 'x', 'y', 'z']

    D_in = re.findall(r'D_in(\d\d)\.', file)[0]
    hole = re.findall(r'D_in\d\d\.(\d)mm', file)[0]
    df = df.drop('t', axis=1)
    df['D_in'] = D_in
    df['hole'] = hole

    df_main = df_main.append(df)


# print(df_main)

# Plot maximum electric field versus hole diameter for each bias voltage and
# for each diameter

diameters = df_main.D_in.unique()
V_biases = df_main.V_bias.unique()
holes = df_main.hole.unique()


def plot_esNormMax_vs_hole(df):
    V_bias = df['V_bias'].unique()[0]
    # print(df)
    print(f'Creating plot for bias voltage {V_bias}')
    fig, ax = plt.subplots(figsize=(8, 8))
    for diameter in diameters:
        this_df = df[df.D_in == diameter]
        X = this_df['hole'].values
        Y = this_df['es.NormE'].values
        ax.plot(X, Y, label=f'{diameter}')
        ax.scatter(X, Y)
    figname = f'{remote_path}plot.esNormE_vs_hole.V_bias{V_bias}_V.png'
    plt.legend(loc='best', title='D_in mm')
    plt.xlabel('Extraction hole [mm]')
    plt.ylabel('Maximum electric field [kV/mm]')
    plt.savefig(figname, dpi=600)
    plt.close('all')


def plot_esNormMax_vs_diameter(df):
    V_bias = df['V_bias'].unique()[0]
    # print(df)
    print(f'Creating plot for bias voltage {V_bias}')
    fig, ax = plt.subplots(figsize=(8, 8))
    for hole in holes:
        this_df = df[df.hole == hole]
        this_df = this_df.sort_values(by=['D_in'])
        # print(this_df)
        X = this_df['D_in'].values.astype(float)
        # print(X)
        Y = this_df['es.NormE'].values.astype(float)
        ax.plot(X, Y, label=f'{hole}')
        ax.scatter(X, Y)
    figname = f'{remote_path}plot.esNormE_vs_diameter.V_bias{V_bias}_V.png'
    plt.legend(loc='best', title='Hole mm')
    plt.xticks(X, X)
    plt.xlabel('Suppression electrode inner diameter [mm]')
    plt.ylabel('Maximum electric field [kV/mm]')
    plt.savefig(figname, dpi=600)
    plt.close('all')


df_main.groupby('V_bias',
                as_index=True).apply(lambda x: plot_esNormMax_vs_hole(x))

df_main.groupby('V_bias',
                as_index=True).apply(lambda x: plot_esNormMax_vs_diameter(x))
