import re
import numpy as np
import pandas as pd
import glob
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
"""
1.) Load the tally outputs. This is flux per source particle flux(E). Take the center bin energy as the energy for that bin.

2.) Use LB6411 manual to get the response in the LB6411 for that energy, R(E) 
Get M(E) = R(E) * flux(E)
M(E) is the counts in LB6411 per source particle for each energy

3.)  Get M as the total of M(E) which is the total counts in LB6411 per source particle

4.) Assume 1 µSv/hr --> 0.79 counts in the LB6411 --> counts / M yields the number of source particles 
"""



"""
1.) Load the tally outputs. This is flux per source particle flux(E). Take the center bin energy as the energy for that bin.
"""
inputFilePrefix = 'CurrentTarget' # insert the prefix of the MCNP input file
pathOutputDirectory = '//fs03//LTH_Neutimag//hkromer//10_Experiments//02_MCNP//'

# IDrun = 99 # insert the ID of the MCNP run
# IDruns = np.arange(237, 243+1, 1) # insert the ID of the MCNP run
IDruns = [241]

for IDrun in IDruns:
    run = inputFilePrefix + str(IDrun)

    thisMode = '_normal'

    # read tally files in the directory
    path_MCNP_inputFolder = pathOutputDirectory + run + '//' + run + thisMode + '//'
    print('Processing: ', str(path_MCNP_inputFolder))

    df_energy = []  # energy column
    energyMade = False
    df = pd.DataFrame()     # contains the flux
    df_err = pd.DataFrame() # contains the error in the flux (fraction)
    # create dataframe structure
    for filename in glob.glob(path_MCNP_inputFolder + "tallyOutputInCell*"):
        cell = filename[-2:]
        # print(filename, cell)
        fhandle = open(filename, 'r')
        lstFlux = []
        lst_d_Flux = []
        df_col = []  # column of the dataframe
        ll = 0  # line index
        for line in fhandle:
            if ll == 0:  # check which direction the tally is
                lLine = line.split()  # split the line into list
                x = float(lLine[0])
                y = float(lLine[1])
                z = float(lLine[2])
                if x > 0.0:  # west direction
                    df_col.append('W ' + '%.0f' % x)
                elif x < 0.0:  # east direction
                    x = abs(x)
                    df_col.append('E ' + '%.0f' % x)
                elif z > 0.0:  # north direction
                    df_col.append('N ' + '%.0f' % z)
                elif z < 0.0:  # south direction
                    z = abs(z)
                    df_col.append('S ' + '%.0f' % z)
            if ll > 0:
                line = line.rstrip().split()
                flux = line[1]
                d_flux = line[2]
                lstFlux.append(float(flux))
                lst_d_Flux.append(float(d_flux))
            if ll > 0 and energyMade == False:
                energy = line[0]
                df_energy.append(float(energy))
            ll = ll + 1
        df[df_col[0]] = lstFlux
        df_err[df_col[0]] = lst_d_Flux
        energyMade = True
        fhandle.close()


    # print(df_cols)
    df_energy = np.array(df_energy)
    df_energy_orig = np.copy(df_energy)
    e_diff = np.diff(df_energy)
    for ii in range(0, len(df_energy_orig)):
        if ii == 0:  # first element
            df_energy[ii] = -1
        else:
            this = df_energy_orig[ii]
            prev = df_energy_orig[ii - 1]
            res = prev + abs(this - prev)/2
            df_energy[ii] = res


    df['energy'] = df_energy
    df_err['energy'] = df_energy



    """
    2.) Use LB6411 manual to get the response in the LB6411 for that energy, R(E) 
    Get M(E) = R(E) * flux(E)
    M(E) is the counts in LB6411 per source particle for each energy
    """
    # LB6411 manual
    lstLB6411_E = []  # float
    lstLB6411_Rphi = []  # float
    fhandle = open('LB6411_energy_Rphi.txt', 'r')
    for line in fhandle:
        lstLine = line.split()
        lstLB6411_E.append(float(lstLine[0]))
        lstLB6411_Rphi.append(float(lstLine[1]))
    fhandle.close()

    # inter = np.interp([2.5], lstLB6411_E, lstLB6411_Rphi)[0]

    # response from the manual
    R = df['energy'].apply(lambda x: np.interp([x], lstLB6411_E, lstLB6411_Rphi)[0])

    df['R'] = R
    df = df.set_index('energy')




    # # plot flux for one position
    # x = df.index
    # y = df['E 195'].values
    #
    # y_err = (df_err['E 195'].values) * y
    #
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # # x
    # minor_locator = AutoMinorLocator(2)
    # ax1.xaxis.set_minor_locator(minor_locator)
    # # y
    # # minor_locator = AutoMinorLocator(2)
    # # ax1.yaxis.set_minor_locator(minor_locator)
    # ax1.errorbar(x, y, yerr=y_err, color='black', fmt="-o", markersize=2, linewidth=0.25,  ecolor='red', elinewidth=2)
    # fig.subplots_adjust(left=0.16)
    # plt.xlabel(r'Tally bin center energy [MeV]', fontsize=14)
    # plt.ylabel(r'F4 tally counts $\left[ \frac{\phi}{\textit{source particle}} \right]$', fontsize=14)
    # plt.title(r'\textbf{Tally at 195 cm East}', fontsize=16)
    # # pyplot.xscale('log')
    # plt.yscale('log')
    # ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
    # ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')
    # # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e7))
    # # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    #
    # # other axis
    # ax2 = ax1.twinx()
    # ax2.plot(lstLB6411_E, lstLB6411_Rphi, color='b', ls="-", marker='x', markersize=2, linewidth=0.25)
    # ax2.set_ylabel('LB6411 response [a.u.]', color='b', fontsize=14)
    # ax2.tick_params('y', colors='b')
    #
    # plt.xlim(0, 3.0)
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # # ticks
    # # y_ticks = np.arange(1.0e6, 10e6 + 1.0e6, 1.0e6)
    # # y = np.arange(1, 10 + 1, 1)
    # # plt.yticks(y_ticks, y)
    # # plt.legend(loc='best', title=r"\textbf{Legend}")
    # plt.savefig( path_MCNP_inputFolder + '/flux_tally_E195.png', dpi=600)
    # # plt.show()
    # plt.close()
    #
    # # plot flux for one position
    # x = df.index
    # y = df['N 20'].values
    #
    # y_err = (df_err['N 20'].values) * y
    #
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # # x
    # minor_locator = AutoMinorLocator(2)
    # ax1.xaxis.set_minor_locator(minor_locator)
    # # y
    # # minor_locator = AutoMinorLocator(2)
    # # ax1.yaxis.set_minor_locator(minor_locator)
    # ax1.errorbar(x, y, yerr=y_err, color='black', fmt="-o", markersize=2, linewidth=0.25, ecolor='red', elinewidth=2)
    # fig.subplots_adjust(left=0.14)
    # plt.xlabel(r'Tally bin center energy [MeV]', fontsize=14)
    # plt.ylabel(r'F4 tally counts $\left[ \frac{\phi}{\textit{source particle}} \right]$', fontsize=14)
    # plt.title(r'\textbf{Tally at 20 cm North}', fontsize=16)
    # # pyplot.xscale('log')
    # plt.yscale('log')
    # ax1.grid(b=True, which='major', linestyle='-')  # , color='gray')
    # ax1.grid(b=True, which='minor', linestyle='--')  # , color='gray')
    # # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e7))
    # # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    #
    # # other axis
    # ax2 = ax1.twinx()
    # ax2.plot(lstLB6411_E, lstLB6411_Rphi, color='b', ls="-", marker='x', markersize=2, linewidth=0.25)
    # ax2.set_ylabel('LB6411 response [a.u.]', color='b', fontsize=14)
    # ax2.tick_params('y', colors='b')
    # plt.xlim(0, 3.0)
    # # for tick in ax1.xaxis.get_major_ticks():
    # #    tick.label.set_fontsize(14)
    # # for tick in ax1.yaxis.get_major_ticks():
    # #    tick.label.set_fontsize(14)
    # # for tick in ax2.yaxis.get_major_ticks():
    # #    tick.label.set_fontsize(14)
    # # ticks
    # x_ticks = [0, 3.0]
    # plt.yticks(x_ticks, x_ticks)
    # # plt.legend(loc='best', title=r"\textbf{Legend}")
    # plt.savefig(path_MCNP_inputFolder + '/flux_tally_N20_pres.png', dpi=600)
    # # plt.show()
    # plt.close()
    #
    # # plot flux for one position
    # x = df.index
    # plotTally = 'S 255'
    # y = df[plotTally].values
    #
    # y_err = (df_err[plotTally].values) * y
    #
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # # x
    # minor_locator = AutoMinorLocator(2)
    # ax1.xaxis.set_minor_locator(minor_locator)
    # # y
    # # minor_locator = AutoMinorLocator(2)
    # # ax1.yaxis.set_minor_locator(minor_locator)
    # ax1.errorbar(x, y, yerr=y_err, color='black', fmt="-o", markersize=2, linewidth=0.25, ecolor='red', elinewidth=2)
    # fig.subplots_adjust(left=0.14)
    # plt.xlabel(r'Tally bin center energy [MeV]', fontsize=14)
    # plt.ylabel(r'F4 tally counts $\left[ \frac{\phi}{\textit{source particle}} \right]$', fontsize=14)
    # plt.title(r'\textbf{Tally %s }' % plotTally, fontsize=16)
    # # pyplot.xscale('log')
    # plt.yscale('log')
    # ax1.grid(b=True, which='major', linestyle='-')  # , color='gray')
    # ax1.grid(b=True, which='minor', linestyle='--')  # , color='gray')
    # # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e7))
    # # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    #
    # # other axis
    # ax2 = ax1.twinx()
    # ax2.plot(lstLB6411_E, lstLB6411_Rphi, color='b', ls="-", marker='x', markersize=2, linewidth=0.25)
    # ax2.set_ylabel('LB6411 response [a.u.]', color='b', fontsize=14)
    # ax2.tick_params('y', colors='b')
    # plt.xlim(0, 3.0)
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # # ticks
    # # y_ticks = np.arange(1.0e6, 10e6 + 1.0e6, 1.0e6)
    # # y = np.arange(1, 10 + 1, 1)
    # # plt.yticks(y_ticks, y)
    # # plt.legend(loc='best', title=r"\textbf{Legend}")
    # plt.savefig(path_MCNP_inputFolder + '/flux_tally_'+ plotTally + '.png', dpi=600)
    # # plt.show()
    # plt.close()


    filename = path_MCNP_inputFolder + '/df_flux.csv'  # contains the raw flux data and the response column from LB6411
    df.to_csv(filename, index=True, encoding='utf-8')  # save DataFrame

    # print(df['N 40'])
    # print(df['R'])
    # multiply each colum with the R column
    cols = list(df.columns.values)
    cols = cols[:-1]# -1 is the R col
    df = df[cols].multiply(df['R'], axis="index")  # -1 is the R col
    # print(df['N 40'])


    """
    3.)  Get M as the total of M(E) which is the total counts in LB6411 per source particle
    """

    df.loc['Total'] = df.sum()

    filename = path_MCNP_inputFolder + '/df_counts.csv'  # contains the flux*response --> counts column and total row
    df.to_csv(filename, index=True, encoding='utf-8')  # save DataFrame

    """
    4.) Assume 1 µSv/hr --> 0.79 counts in the LB6411 --> counts / M yields the number of source particles 
    """

    df_dir = pd.DataFrame()  # total countrate per source particle per µSv/hr
    df_dir['position'] = cols

    df_dir['direction'] = df_dir['position'].apply(lambda x: x.split()[0])

    df_dir['distance'] = df_dir['position'].apply(lambda x: float(x.split()[1]))

    tot = df.iloc[-1].tolist()
    # print(tot)
    df_dir['ctPerS'] = tot

    # df_dir = df_dir.set_index('direction')

    filename = path_MCNP_inputFolder + '/df_total_countrate_per_nps.csv'
    df_dir.to_csv(filename, index=True, encoding='utf-8')  # save DataFrame
    # print(df_dir)



    dose = 100  # assumed dose of 100 µSv/hr
    d = np.arange(20, 240, 5)  # distance source to detector in cm
    cps_per_dose = 0.79  # counts per dose, from the LB6411 manual
    cps = cps_per_dose * dose  # counts in the LB6411

    df_out = pd.DataFrame()  # output dataframe
    df_out['distance'] = d  # position of the LB6411
    lstDirection = ['N', 'S', 'W', 'E']  # directions where the tallies/detectors are in
    for col in lstDirection:
        df_out[col] = 0  # Total neutron output for 100 µSv/hr for each direction

    # interpolate total countrate per
    for thisDir in lstDirection:
        this_df = df_dir[ df_dir['direction'] == thisDir ]
        x = this_df['distance']  # distance source to detector in MCNP
        y = this_df['ctPerS']  # counts per source particle in MCNP
        inter = np.interp(d, x, y)  # interpolate results: counts per source particle if LB6411 where at that distance
        neutron_output = cps / inter  # neutron source output

        df_out[thisDir] = neutron_output

    df_out = df_out.set_index('distance')

    # find the deuterium energy
    fname = path_MCNP_inputFolder + '/out'
    with open(fname, 'r') as file:
        for line in file:
            t = re.findall(r'deuterium energy (\d+)', line)
            if len(t) < 1: continue
            deut_energy = t[0]
            break
        file.close()

    filename = path_MCNP_inputFolder + '/df_neutron_output_for_Edeut_' + str(deut_energy) + '.csv'
    df_out.to_csv(filename, index=True, encoding='utf-8')  # save DataFrame

    fig = plt.figure(figsize=(8, 5))
    # fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.rc('text', usetex=True)
    plt.rc('font', weight='bold')
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Arial'
    matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
    matplotlib.rcParams['mathtext.tt'] = 'Arial'
    matplotlib.rcParams['mathtext.cal'] = 'Arial'
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    # plt.yscale('log')
    ax1.plot(df_out.index, df_out['N'], label=r'North', ls='-', color='blue')
    ax1.plot(df_out.index, df_out['S'], label=r'South', ls='--', color='red')
    ax1.plot(df_out.index, df_out['W'], label=r'West', ls='-.', color='green')
    ax1.plot(df_out.index, df_out['E'], label=r'East', ls='dotted', color='olive')
    plt.xlabel(r"LB6411 distance from source [cm]", fontsize=14)
    plt.ylabel(r"Neutron output [n/s] per  100 $ \frac{\mu Sv}{h}$", fontsize=14)
    plt.title(r"Neutron output for deuterium energy " + str(deut_energy) + r" keV", fontsize=16)

    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.125e7))
    plt.xlim(50,80)
    plt.ylim(0,0.5*1e7)    
    # y_ticks = np.arange(1.0e6, 26e6 + 2.0e6, 2.0e6)
    # y = np.arange(2, 26 + 2, 2)
    # d = np.arange(50, 85, 5)  # distance source to detector in cm
    ax1.xaxis.set_ticks(np.arange(50, 85, 5))
    # ax1.yaxis.set_ticks(np.arange(0,0.5e7+0.1e7,0.1e7))

    # plt.yticks(y_ticks, y)

    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # # minor ticks x
    minor_locator = AutoMinorLocator(2)
    ax1.xaxis.set_minor_locator(minor_locator)
    # # minor ticks y
    # minor_locator = AutoMinorLocator(5)
    # ax1.yaxis.set_minor_locator(minor_locator)
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
        # plt.show()
    ax1.axhline(linewidth=1, color='black')
    plt.legend(loc='best')
    ax1.tick_params('x', colors='black', labelsize=12)  
    ax1.tick_params('y', colors='black', labelsize=12)  
    ax1.grid(b=True, which='major', linestyle='-')
    ax1.grid(b=True, which='minor', linestyle='--')
    plt.tight_layout()
    # plt.savefig(path_MCNP_inputFolder + 'neutron_output' + str(deut_energy) + '_keV.pdf')
    plt.savefig(path_MCNP_inputFolder + 'neutron_output' + str(deut_energy) + '_keV.png', dpi=600)
    # plt.savefig('test.png', dpi=600)
    # plt.savefig(path_MCNP_inputFolder + 'neutron_output_100keV.png', dpi=600)
    # plt.savefig(pathOutputDirectory + '/neutron_output/neutron_output_'+ str(IDrun) + '_' + str(deut_energy) + '_keV.pdf')
    plt.savefig(pathOutputDirectory + '/neutron_output/neutron_output_' + str(IDrun) + '_' +  str(deut_energy) + '_keV.png', dpi=600)
    # plt.show()
    plt.close(fig)

    # plt only for a certain range
    # x1 = 30
    # x2 = 100
    # df_out_zoom = df_out[ (df_out.index >= x1) & (df_out.index <= x2) ]
    #
    # fig = plt.figure(figsize=(8, 5))
    # # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # minor_locator = AutoMinorLocator(2)
    # ax1.xaxis.set_minor_locator(minor_locator)
    # minor_locator = AutoMinorLocator(5)
    # ax1.yaxis.set_minor_locator(minor_locator)
    # # plt.yscale('log')
    # ax1.plot(df_out_zoom.index, df_out_zoom['N'], label = 'North', color='blue')
    # ax1.plot(df_out_zoom.index, df_out_zoom['S'], label = 'South', color='red')
    # ax1.plot(df_out_zoom.index, df_out_zoom['W'], label = 'West', color='green')
    # ax1.plot(df_out_zoom.index, df_out_zoom['E'], label = 'East', color='olive')
    # plt.xlabel('Distance from source [cm]')
    # plt.ylabel('Neutron output per 100 muSv/hr')
    # plt.title('Deuterium energy ' + str(deut_energy) + ' keV')
    # ax1.grid(b=True, which='major', linestyle='-')
    # ax1.grid(b=True, which='minor', linestyle='--')
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e7))
    # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # d = np.arange(x1-10, x2+10, 10)  # distance source to detector in cm
    # plt.xticks(d)
    # ax1.axhline(linewidth=1, color='black')
    # plt.legend(loc='best')
    # # plt.show()
    # # plt.savefig(path_MCNP_inputFolder + '/neutron_output' + str(deut_energy) + '_keV_' + str(x1) + '_to_' + str(x2) + '.pdf')
    # plt.savefig(path_MCNP_inputFolder + '/neutron_output' + str(deut_energy) + '_keV_' + str(x1) + '_to_' + str(x2) + '.png', dpi=600)
    # plt.close(fig)
    #
    # print('*done. going to next ...')

print('Done with all runs ', IDruns)