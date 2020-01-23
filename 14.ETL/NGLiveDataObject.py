import numpy as np
import pandas as pd
import pymysql
import sqlalchemy as sql
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy.interpolate import interp1d

class NGLiveDataObject():

    def __init__(self, host, database, user, password):
        self.password = password
        self.host = host
        self.database = database
        self.user = user

    def get_from_database(self, query):
        """
        Connects to the storage database and retrieves the data for the given date.
        INPUT:
            - query: SQL query as string
        OUTPUT:
            - pandas dataframe
        """
        connect_string = 'mysql+pymysql://%(user)s:%(pw)s@%(host)s/%(db)s'% {
            "user": self.user,
            "pw": self.password,
            "host": self.host,
            "db": self.database}

        sql_engine = sql.create_engine(connect_string)

        df = pd.read_sql(query, sql_engine)

        # convert to datetime
        df['time'] = pd.to_datetime(df['time'])
        # set index as time
        df.set_index('time', drop=True, inplace=True)
        return df

    def correct_pressure(self, lookup_table_path, data_pressure):
        """
        Applies the correction to the pressure and computes pressure from voltage measured.
        INPUT:
            - lookup_table_path: path to the lookup table
            - data_pressure pandas dataframe with columns time, id, voltage_VC, voltage_IS
        OUTPUT:
            - pandas dataframe with corrected pressure
        """
        # correct the pressure that the arduino reads. This is done using the dose_lookup_table which relates the pi dose with the displayed dose.
        df_LT_pressure = pd.read_csv(lookup_table_path, delimiter="\t")
        interp_pressure_IS = interp1d(pd.to_numeric(df_LT_pressure['pressure_IS_pi']).values, pd.to_numeric(df_LT_pressure['pressure_IS_display']).values, fill_value='extrapolate')
        data_pressure['pressure_IS'] = 10 ** (1.667 * data_pressure['voltage_IS'] - 11.33)
        data_pressure['pressure_VC'] = 10 ** (1.667 * data_pressure['voltage_VC'] - 11.33)
        data_pressure['pressure_IS_corrected'] = interp_pressure_IS(data_pressure['pressure_IS'])

        return data_pressure

    def resample_10_seconds(self, df, cols):
        """
        Takes the a grouped df (grouped by day) and resamples the columns cols in 10s
        OUTPUT:
            - dataframe
        """

        d = {}
        range_start = df.iloc[0].name
        range_end = df.iloc[-1].name
        # resample 24 hours in steps of 10 s
        s = pd.date_range(start=range_start, end=range_end, freq='10S')
        df_out = pd.DataFrame(pd.Series(s, name='time')).set_index('time')
        for col in cols:
            d[col] = df[col].resample('10S').mean()

        this_d = pd.DataFrame(d)

        df_out = df_out.merge(this_d, left_on=df_out.index, right_on=this_d.index, how='outer')
        df_out.set_index('key_0', drop=True, inplace=True)

        return df_out

# read password and user to database
credentials_file = '/Users/hkromer/02_PhD/01.github/dash_NG/credentials.pw'
credentials = pd.read_csv(credentials_file, header=0)
user = credentials['username'].values[0]
pw = credentials['password'].values[0]
host="twofast-RPi3-0"  # your host
user=user  # username
passwd=pw  # password
db="NG_twofast_DB" # name of the database