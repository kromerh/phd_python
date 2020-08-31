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
VOLTAGE_FLOW_1 = 20 # mV
VOLTAGE_FLOW_2 = 80 # mV



VERBOSE = True

ALTERNATE_FREQ = 10 # how many seconds to switch from one value to the next

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




def get_experiment_id_and_flow(sql_engine, verbose=False):
    query = f"SELECT * FROM experiment_control;"
    df = pd.read_sql(query, sql_engine)

    experiment_id = df['experiment_id'].values[0]
    d2flow_set = df['d2flow_set'].values[0]

    if verbose: print(f"Experiment id is {experiment_id}")

    return experiment_id, d2flow_set





def saveDB(experiment_id, voltage_flow, voltage_flow_set, verbose=False):
    # Create a Cursor object to execute queries.
    query = f"""INSERT INTO live_d2flow (experiment_id, voltage_flow, voltage_flow_set) VALUES (\"{experiment_id}\", \"{voltage_flow}\", \"{voltage_flow_set}\");"""
    sql_engine.execute(sql.text(query))

    if verbose: print(query)

cnt = 0
VOLTAGE_FLOW = VOLTAGE_FLOW_1

alternate = True
while True:
    try:
        if cnt == ALTERNATE_FREQ:
            cnt = 0
            if alternate:
                VOLTAGE_FLOW = VOLTAGE_FLOW_2
                alternate = False
            else:
                VOLTAGE_FLOW = VOLTAGE_FLOW_1
                alternate = True

        experiment_id, flow_set = get_experiment_id_and_flow(sql_engine, VERBOSE)

        voltage_flow = float(VOLTAGE_FLOW)
        voltage_flow_set = float(flow_set)

        saveDB(experiment_id, voltage_flow, voltage_flow_set, VERBOSE)

        sleep(FREQUENCY)
        cnt += 1
    except KeyboardInterrupt:
        print('Ctrl + C. Exiting. Flushing serial connection.')
        sys.exit(1)


