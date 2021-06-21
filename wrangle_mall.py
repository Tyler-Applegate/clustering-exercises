# this is my wrangle module for the mall customers clustering exercises

import pandas as pd
import numpy as np
import os
from env import host, user, password

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
            print('--------------------------------------------')
            print('')
        else:
            print(df[col].value_counts(bins=10, sort=False))
            print('--------------------------------------------')
            print('')
    print('----------------------------------------------------')
    print('Nulls in DataFrame by Column: ')
    print(nulls_by_col(df))
    print('----------------------------------------------------')
    print('Nulls in DataFrame by Rows: ')
    print(nulls_by_row(df))
    print('----------------------------------------------------')
    df.hist()
    plt.tight_layout()
    return plt.show()

def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df

def get_lower_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x:max([x - lower_bound, 0]))

def add_lower_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_lower_outliers'] = get_lower_outliers(df[col], k)
    return df

def outlier_describe(df):
    outlier_cols = [col for col in df.columns if col.endswith('_outliers')]
    for col in outlier_cols:
        print(col, ': ')
        subset = df[col][df[col] > 0]
        print(subset.describe())

def drop_outliers(df, col_list, k=1.5):
    '''
    This function takes in a dataframe and removes outliers that are k * the IQR
    '''
    
    for col in col_list:

        q_25, q_75 = df[col].quantile([0.25, 0.75])
        q_iqr = q_75 - q_25
        q_upper = q_75 + (k * q_iqr)
        q_lower = q_25 - (k * q_iqr)
        df = df[df[col] > q_lower]
        df = df[df[col] < q_upper]
#       these bottome two lines are only necessary if previous functions have been callled
#       if this function is run BEFORE add_upper/lower, then these columns need to be commented out
        outlier_cols = [col for col in df.columns if col.endswith('_outliers')]
        df = df.drop(columns=outlier_cols)
        
    return df     

def mall_encoder(df, col):
    df = pd.get_dummies(df, columns=col, drop_first=True)
    return df
        
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=1221)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=1221)
    return train, validate, test

def min_max_scale(train, validate, test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    train_scaled_array = scaler.transform(train[numeric_cols])
    validate_scaled_array = scaler.transform(validate[numeric_cols])
    test_scaled_array = scaler.transform(test[numeric_cols])

    # convert arrays to dataframes
    train_scaled = pd.DataFrame(train_scaled_array, columns=numeric_cols).set_index(
        [train.index.values]
    )

    validate_scaled = pd.DataFrame(
        validate_scaled_array, columns=numeric_cols
    ).set_index([validate.index.values])

    test_scaled = pd.DataFrame(test_scaled_array, columns=numeric_cols).set_index(
        [test.index.values]
    )

    return train_scaled, validate_scaled, test_scaled

def wrangle_mall_data():
    col_list = ['age', 'annual_income', 'spending_score']
    # let's acquire our data...
    df = get_mallcustomer_data()
    # summarize the data
    print(summarize(df))
    # add upper outlier columns
    df = add_upper_outlier_columns(df)
    # add lower outlier columns
    df = add_lower_outlier_columns(df)
    # describe the outliers
    print(outlier_describe(df))
    # drop outliers
    df = drop_outliers(df, col_list)
    # split the data
    train, validate, test = split_data(df)
    # drop missing values from train
    train = train.dropna()
    # scale the data
    train_scaled, \
    validate_scaled, \
    test_scaled = min_max_scale(train, validate, test, col_list)
    print(f'          train shape: {train.shape}')
    print(f'       validate shape: {validate.shape}')
    print(f'           test shape: {test.shape}')
    print(f'   train_scaled shape: {train_scaled.shape}')
    print(f'validate_scaled shape: {validate_scaled.shape}')
    print(f'    test_scaled shape: {test_scaled.shape}')
    return train, validate, test, train_scaled, validate_scaled, test_scaled
    