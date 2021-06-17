# This is my classification acquire module

import pandas as pd
import numpy as np
import os
from env import host, user, password

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# general framework / template
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my env file to create a connection url to access
    the Codeup database.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
    return df.set_index('customer_id')

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

def summarize(df):
    '''
    This function will take in a single argument (pandas DF)
    and output to console various statistics on said DF, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observe null values
    '''
    print('----------------------------------------------------')
    print('DataFrame Head')
    print(df.head(3))
    print('----------------------------------------------------')
    print('DataFrame Info')
    print(df.info())
    print('----------------------------------------------------')
    print('DataFrame Description')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('----------------------------------------------------')
    print('DataFrame Value Counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('----------------------------------------------------')
    print('Nulls in DataFrae by Column: ')
    print(nulls_by_col(df))
    print('----------------------------------------------------')
    print('Nulls in DataFrame by Rows: ')
    print(nulls_by_row(df))
    print('----------------------------------------------------')
    df.hist()
    plt.tight_layout()
    return plt.show()