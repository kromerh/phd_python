import pymysql
import time
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine
from scipy.interpolate import interp1d

# correct the dose that the arduino reads. This is done using the dose_lookup_table which relates the pi dose with the displayed dose.
df_LT = pd.read_csv("/home/pi/Documents/dose_lookup_table.txt", delimiter="\t")

# interpolation function
interp_dose = interp1d(df_LT['dose_pi'], df_LT['dose_display'], fill_value='extrapolate')

# correct the pressure that the arduino reads. This is done using the dose_lookup_table which relates the pi dose with the displayed dose.
df_LT_pressure = pd.read_csv("/home/pi/Documents/LUT_pressure_ion_source.txt", delimiter="\t")

# interpolation function
interp_pressure_IS = interp1d(pd.to_numeric(df_LT_pressure['pressure_IS_pi']).values, pd.to_numeric(df_LT_pressure['pressure_IS_display']).values, fill_value='extrapolate')

"""
Copies data from live tables HBox_Uno and BBox to the storage tables data_dose, data_pressure, data_HV.
Then it truncates the live tables.
Script is executed every night at 3 am on twofast-RPi3-0 (DB, HBox_Uno)
"""

# convert ms to second date
def to_the_second(ts):
    return pd.to_datetime((round(ts.value, -9)))

""" 
PSI database
"""
db = pymysql.connect(host="twofast-RPi3-0",  # your host
                     user="writer",  # username
                     passwd="heiko",  # password
                     db="NG_twofast_DB")  # name of the database


"""
local database

db = pymysql.connect(host="localhost",  # your host
                     user="root",  # username
                     passwd="axfbj1122!",  # password
                     db="NG_twofast_DB")  # name of the database
"""

# Create a Cursor object to execute queries.
cur = db.cursor()

try:
    """
    BBox - dose read by the LB6411, counts in the white sphere, counts in the second bonner sphere and the duty factor.
    Note that if the second bonner sphere is not connected to the readout unit, the attachInterrupt has to be disabled
    in the Arduino Due! 
    0: 'ID', 1: 'date', 2: 'dose_voltage', 3: 'counts_WS', 4: 'counts_BS', 5: 'counts_DF'
    """

    ############################################################################################################
    # HBox_Uno
    ############################################################################################################
    cur.execute("""SELECT * FROM HBox_Uno""")
    rows = cur.fetchall()
    df_HBox_Uno = pd.DataFrame([[ij for ij in i] for i in rows])
    df_HBox_Uno.rename(columns={0: 'ID', 1: 'time', 2: 'dose_voltage', 3: 'HV_current', 4: 'HV_voltage'},
              inplace=True)

    if len(df_HBox_Uno) > 0:
        df_HBox_Uno = df_HBox_Uno.set_index(['ID'])
        df_HBox_Uno['dose'] = df_HBox_Uno['dose_voltage'] * 3000 / 5.5
        df_HBox_Uno['dose_corrected'] = interp_dose(df_HBox_Uno['dose'])

        df_dose = df_HBox_Uno[['time', 'dose', 'dose_voltage', 'dose_corrected']]  # dose corrected is with the lookup table!
        df_HV = df_HBox_Uno[['time', 'HV_voltage', 'HV_current']] 
    else:
        df_dose = pd.DataFrame()
        df_HV = pd.DataFrame()
        
    ############################################################################################################
    # BBox
    ############################################################################################################
    cur.execute("""SELECT * FROM BBox""")
    rows = cur.fetchall()
    df_BBox = pd.DataFrame([[ij for ij in i] for i in rows])
    df_BBox.rename(columns={0: 'ID', 1: 'time', 2: 'voltage_IS', 3: 'voltage_VC'}, inplace=True)

    if len(df_BBox) > 0:
        df_BBox = df_BBox.set_index(['ID'])
        df_BBox['pressure_IS'] = 10 ** (1.667 * df_BBox['voltage_IS'] - 11.33)
        df_BBox['pressure_VC'] = 10 ** (1.667 * df_BBox['voltage_VC'] - 11.33)
        df_BBox['pressure_IS_corrected'] = interp_pressure_IS(df_BBox['pressure_IS'])
        # df_BBox['pressure_VC_corrected'] = interp_pressure_IS(df_BBox['pressure_VC'])

        # df_pressure = df_BBox[['time', 'pressure_IS', 'pressure_VC', 'voltage_IS', 'voltage_VC', 'pressure_IS_corrected', 'pressure_VC_corrected']]
        df_pressure = df_BBox[['time', 'pressure_IS', 'pressure_VC', 'voltage_IS', 'voltage_VC', 'pressure_IS_corrected']]
    else:
        df_pressure = pd.DataFrame()

    ############################################################################################################
    # HBox_Due
    ############################################################################################################
    cur.execute("""SELECT * FROM HBox_Due""")
    rows = cur.fetchall()
    df_HBox_Due = pd.DataFrame([[ij for ij in i] for i in rows])
    df_HBox_Due.rename(columns={0: 'ID', 1: 'time', 2: 'counts_WS', 3: 'counts_BS'}, inplace=True)
    
    if len(df_HBox_Due) > 0:
        df_HBox_Due = df_HBox_Due.set_index(['ID'])
        df_counts = df_HBox_Due[['time', 'counts_WS', 'counts_BS']]
    else:
        df_counts = pd.DataFrame()

    ############################################################################################################
    # DistanceSensor
    ############################################################################################################
    cur.execute("""SELECT * FROM distanceSensor""")
    rows = cur.fetchall()
    df_distanceSensor = pd.DataFrame([[ij for ij in i] for i in rows])
    df_distanceSensor.rename(columns={0: 'ID', 1: 'time', 2: 'rpm_peaks', 3: 'avg_peaks', 4: 'avg_bottoms'}, inplace=True)
    
    if len(df_distanceSensor) > 0:
        df_distanceSensor = df_distanceSensor.set_index(['ID'])
        df_distanceSensor = df_distanceSensor[['time', 'rpm_peaks', 'avg_peaks', 'avg_bottoms']]
    else:
        df_distanceSensor = pd.DataFrame()

    ############################################################################################################
    # referenceDetectors
    ############################################################################################################
    cur.execute("""SELECT * FROM referenceDetectors""")
    rows = cur.fetchall()
    df_refDets = pd.DataFrame([[ij for ij in i] for i in rows])
    df_refDets.rename(columns={0: 'ID', 1: 'time', 2: 'ard_time', 3: 'counts_D1', 4: 'counts_D2', 5: 'counts_D3', 6: 'counts_D4'}, inplace=True)
    
    if len(df_refDets) > 0:
        df_refDets = df_refDets.set_index(['ID'])
        df_refDets = df_refDets[['time', 'ard_time', 'counts_D1', 'counts_D2', 'counts_D3', 'counts_D4']]
    else:
        df_refDets = pd.DataFrame()


    cnx = create_engine("mysql+pymysql://writer:heiko@twofast-RPi3-0:3306/NG_twofast_DB", echo=False)
    # # cnx = create_engine("mysql+pymysql://writer:root@localhost:3306/NG_twofast_DB", echo=False)
    for df, myName in zip([df_dose, df_HV, df_pressure, df_counts, df_distanceSensor, df_refDets], 
        ['data_dose', 'data_HV', 'data_pressure', 'data_counts','data_distanceSensor','data_referenceDetectors']):
        print(myName)
        if len(df) > 0:
            df = df.replace(np.inf, 0)
            df = df.replace(np.nan, 0)
            df.to_sql(name=myName, con=cnx, if_exists = 'append', index=False)
            # df.to_csv(myName + '.csv')


    # # drop live tables
    cur.execute("""TRUNCATE TABLE HBox_Uno""")
    cur.execute("""TRUNCATE TABLE HBox_Due""")
    cur.execute("""TRUNCATE TABLE BBox""")
    cur.execute("""TRUNCATE TABLE distanceSensor""")
    cur.execute("""TRUNCATE TABLE referenceDetectors""")
    cur.execute("""TRUNCATE TABLE chillers""")  # this table is not stored
    

except:
    cur.rollback()

cur.close()

