import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns 
import matplotlib.cm as cm

path_to_data = "E://COMSOL/T_vs_bpower/220W.txt"

# Load data from CSV
df = pd.read_csv(path_to_data, delimiter=r"\s+", skiprows=7)
X_dat = df['x']
Y_dat = df['y']
Z_dat = df['Color']

# Convert from pandas dataframes to numpy arrays
X, Y, Z, = np.array([]), np.array([]), np.array([])
for i in range(len(X_dat)):
        X = np.append(X,X_dat[i])
        Y = np.append(Y,Y_dat[i])
        Z = np.append(Z,Z_dat[i])

# create x-y points to be used in heatmap
xi = np.linspace(X.min(),X.max(),1000)
yi = np.linspace(Y.min(),Y.max(),1000)

# Z is a matrix of x-y values
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')

# I control the range of my colorbar by removing data 
# outside of my range of interest
zmin = 20
zmax = np.max(df['Color'])+10
zi[(zi<zmin) | (zi>zmax)] = None

# Create the contour plot
levels = np.arange(0,125,0.1)
# CS = plt.contourf(xi, yi, zi, 100, cmap=plt.cm.rainbow, antialiased=True)
CS = plt.pcolormesh(xi, yi, zi)

# CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
                  # vmax=zmax, vmin=zmin)
plt.colorbar()  
plt.show()