import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def cart2pol_rho(x):
    rho = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return rho

def cart2pol_phi(x):
    phi = np.arctan2(x[1], x[0])
    phi = (phi + 720) % 360
    # phi = x[1] + x[0]
    return phi

fname = 'E:/COMSOL/run40/run40_lam_final_01_out_surface_temp_01.csv'
# fluid flow is in +x direction
header = ['x','y','z','T']

df = pd.read_csv(fname, skiprows=9, header=None)
df.columns = header
# df = df[ (df['x'] > 95.0) & (df['x'] < 105.0) ]
# print(df.drop_duplicates)
# df = df[ (df['Temperature'] > 20.0) ]
# print(df.iloc[:,1:3])
# df['rho'] = df.iloc[:,1:3].apply(cart2pol_rho, axis = 1)

df = df.sort_values(['x'], ascending=False)


x_unique = df.x.unique().tolist()

df['phi'] = df.iloc[:,1:3].apply(cart2pol_phi, axis = 1)
df = df[['x', 'phi', 'T']]

df = df.reset_index(drop=True)

df_intpol = pd.DataFrame()
intpol_x = np.arange(0, 360+1, 1)  # interpolation points
for x in x_unique:
    df_t = df[ (df['x'] == x) ]  # dataframe for that x value
    df_intpol[x] = np.interp(intpol_x, df_t['phi'], df_t['T'])

df_intpol.to_csv('E:/COMSOL/run40/run40_lam_final_01_out_surface_temp_01_toPlot.csv', header=True, index=False)








