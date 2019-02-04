import re
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

dir = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/02_NumCoresTest/'

folders = os.listdir(dir)
folders = [f'{dir}{f}/' for f in folders]

df = pd.DataFrame()

for thisf in folders:
    run = re.findall(r'(run_\d\d)', thisf)[0]

    files = os.listdir(thisf)
    files = [f for f in files if f.startswith('lsf')]
    files = [f'{thisf}{f}' for f in files]

    for file_lsf in files:
        c = []
        with open(file_lsf, 'r') as h_file:
            for line in h_file:
                c.append(line.rstrip().split())
            h_file.close()
        if len(c) < 100:
            print(file_lsf)
            continue
        c = c[-10:-1]
        for this_c in c:
            if this_c[0] == 'Total':
                time_total = this_c[2]  # time in seconds
                # print(this_c)
            if this_c[0] == 'Saving':
                fname = this_c[2]  # file name
                # print(this_c)
        num_cores = re.findall(r'ncores_(\d+)_', fname)[0]
        this_df = pd.DataFrame()

        this_df['run'] = [run]
        this_df['time_total'] = [time_total]
        this_df['num_cores'] = [num_cores]
        lsf_ID = re.findall(r'(lsf.+)', file_lsf)[0]
        this_df['lsf'] = [lsf_ID]

        df = df.append(this_df)

df['time_total'] = df['time_total'].astype(float)
print(df)
print(df.groupby(['num_cores'])['time_total'].mean())

# print((9761 + 7301 + 8624) / 3)  # test for num_cores 4
