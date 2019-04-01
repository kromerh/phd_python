import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fname = 'E:/COMSOL/run40/run40_lam_final_01_out_surface_temp_01_toPlot.csv'
df = pd.read_csv(fname)


print(len(df))
df_rev = df.iloc[::-1]  # reverse DF
df_rev = df_rev.iloc[1:]
df = df.append(df_rev)  # add the reversed DF
df = df.iloc[180:541]
print(len(df))

df = df.reset_index(drop=True)

print(df)

fig = plt.figure()
ax = fig.gca(projection='3d')

for column in df:
    if float(column) > 95 and float(column) < 105:
        # print(df[column])
        # print(column)
        # x = np.ones(len(y))
        col = np.ones(len(df.index)) * float(column)
        x = col  # height
        y = df.index # phi
        z = df[column]  # temperature



        # ax.plot(x, y, z)
        ax.plot(y, x, z)


# ax.set_ylim3d(20,300)
# # ax.set_zlim3d(0,1000)
ax.set_ylabel('Height')
ax.set_xlabel('Phi')
ax.set_zlabel('Temperature')

# for ii in np.arange(0,360,1):
        # ax.view_init(elev=ii, azim=237)
        # plt.savefig("movie%d.png" % ii)
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.scatter(df.index, df[df.columns[230]])
plt.show()