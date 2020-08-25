import re
import sys
import sqlalchemy as sql
import datetime
import time
from time import sleep
from scipy.interpolate import interp1d
import pandas as pd

# Settings for the simulator
# path
PATH_CREDENTIALS = '/Users/hkromer/02_PhD/01.github/FNL_Neutron_Generator_Control/credentials.pw'

# readout frequency
FREQUENCY = 1 # sleep time in second


# values to save
VOLTAGE_IS = 2 # V
VOLTAGE_VC = 0 # V

VERBOSE = True


# read password and user to database

credentials = pd.read_csv(PATH_CREDENTIALS, header=0)
user = credentials['username'].values[0]
pw = credentials['password'].values[0]
host="localhost"  # your host
user=user # username
passwd=pw  # password
db="FNL" # name of the database
connect_string = 'mysql+pymysql://%(user)s:%(pw)s@%(host)s:3306/%(db)s'% {"user": user, "pw": pw, "host": host, "db": db}
sql_engine = sql.create_engine(connect_string)




def get_experiment_id(sql_engine, verbose=False):
    query = f"SELECT experiment_id FROM experiment_control;"
    df = pd.read_sql(query, sql_engine)

    experiment_id = df['experiment_id'].values[0]

    if verbose: print(f"Experiment id is {experiment_id}")

    return experiment_id


def saveDB(experiment_id, voltage_IS, voltage_VC, verbose=False):
    # Create a Cursor object to execute queries.
    query = f"""INSERT INTO live_pressure (experiment_id, voltage_IS, voltage_VC) VALUES (\"{experiment_id}\", \"{voltage_IS}\", \"{voltage_VC}\");"""
    sql_engine.execute(sql.text(query))

    if verbose: print(query)


while True:
    try:
        experiment_id = get_experiment_id(sql_engine, VERBOSE)

        voltage_IS = float(VOLTAGE_IS)
        voltage_VC = float(VOLTAGE_VC)

        saveDB(experiment_id, voltage_IS, voltage_VC, VERBOSE)

        sleep(FREQUENCY)
    except KeyboardInterrupt:
        print('Ctrl + C. Exiting. Flushing serial connection.')
        sys.exit(1)


