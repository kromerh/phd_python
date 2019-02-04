import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import sys



# PSI computer
# remote_path = '//fs03/LTH_Neutimag/hkromer/'

# local computer
remote_path = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
2018-11-26_comsol/mesh_refinement/'

file = f'{remote_path}/hmesh_vs_ndof.txt'

df = pd.read_csv(file, delimiter=r'\s+')

print(df)

# total electric energy global

# relative differences
X = df['ndof'].values.astype(float)
ref_W = df['total_electric_energy'].values[0].astype(float)
print(ref_W)
W = 100 * (np.abs(df['total_electric_energy'].values.astype(float)-ref_W)/ref_W)

f, ax = plt.subplots()

ax.plot(X, W)
ax.scatter(X, W)

plt.grid(True)
plt.ylabel('Relative difference in total electric energy [%]')
plt.xlabel('d.o.f.')
locs, labels = plt.xticks()

plt.xticks([2000000, 3000000, 4000000, 5000000, 6000000], ['2 000 000',
    '3 000 000', '4 000 000', '5 000 000', '6 000 000'])

plt.tight_layout()
# plt.show()
filename = '{}/global_mr_totElEn_vs_dof'.format(remote_path)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')


# total electric energy accelerator column

# relative differences
X = df['ndof'].values.astype(float)
ref_W = df['electric_energy_acc_col'].values[0].astype(float)
print(ref_W)
W = 100 * (np.abs(df['electric_energy_acc_col'].values.astype(float)-ref_W)/ref_W)

f, ax = plt.subplots()

ax.plot(X, W)
ax.scatter(X, W)

plt.grid(True)
plt.ylabel('Relative difference in total electric energy acc col [%]')
plt.xlabel('d.o.f.')
locs, labels = plt.xticks()

plt.xticks([2000000, 3000000, 4000000, 5000000, 6000000], ['2 000 000',
    '3 000 000', '4 000 000', '5 000 000', '6 000 000'])

plt.tight_layout()
# plt.show()
filename = '{}/accCol_mr_totElEn_vs_dof'.format(remote_path)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')
