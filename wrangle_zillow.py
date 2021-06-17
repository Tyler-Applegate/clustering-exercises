# this is my wrangle module for zillow clustering exercises

import pandas as pd
import numpy as np
import os
from env import host, user, password

# general framework / template
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my env file to create a connection url to access
    the Codeup database.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# now let's get some data...
def get_db_data():
    '''
    This function reads data from the Codeup db into a pandas DataFrame.
    '''
    sql_query = 'inset sql query here; TEST in SQL Ace 1st!'
    return pd.read_sql(sql_query, get_connection('db_name'))