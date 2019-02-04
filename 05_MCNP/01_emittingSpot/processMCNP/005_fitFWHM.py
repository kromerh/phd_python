import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import itertools
from scipy.stats import kde
from scipy import optimize
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot, transforms
pd.set_option("display.max_columns",101)


# import df_MCNP_results.csv
master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180808/'
fname = f'{master_folder}/df_MCNP_results.csv'
df = pd.read_csv(fname, index_col=0)

def plotCps(df):
	f, ax = plt.subplots()
	ax.scatter(df['x_pos'], df['Cps_cutoff'])
	plt.title(f'Diameter: {df.diameter.unique()[0]}')
	plt.show()


df.groupby('diameter').apply(lambda x: plotCps(x))