import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns 
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

path_to_data = "E://COMSOL/T_vs_bpower/220W.txt"

# Load data from CSV
df = pd.read_csv(path_to_data, delimiter=r"\s+", skiprows=7)
x = df['x']
y = df['y']
z = df['Color']

N = int(len(z)**.5)
z = z.reshape(N, N)
plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
        cmap=cm.hot, norm=LogNorm())
plt.colorbar()
plt.show()