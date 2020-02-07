import numpy as np
import pandas as pd
import pymysql
import sqlalchemy as sql


class NGDataObject():

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



    def resample_10_seconds(self, df, cols, day):
        """
        Takes the a grouped df (grouped by day) and resamples the columns cols in 10s
        OUTPUT:
            - dataframe
        """

        d = {}
        range_start = f'{day} 00:00:00'
        range_end = f'{day} 23:59:00'
        # resample 24 hours in steps of 10 s
        s = pd.date_range(start=range_start, end=range_end, freq='10S')
        df_out = pd.DataFrame(pd.Series(s, name='time')).set_index('time')
        for col in cols:
            d[col] = df[col].resample('10S').mean()

        this_d = pd.DataFrame(d)

        df_out = df_out.merge(this_d, left_on=df_out.index, right_on=this_d.index, how='outer')
        df_out = df_out.set_index('key_0')
        return df_out
