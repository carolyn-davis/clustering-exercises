#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:23:01 2021

@author: carolyndavis
"""

# IMPORTS UTILIZED FOR IMPORT OF ZILLOW DATA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env import host, user, password
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer





def get_db_url(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# =============================================================================
# 
# =============================================================================

def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query =  """
SELECT prop.*,
       pred.logerror,
       pred.transactiondate,
       air.airconditioningdesc,
       arch.architecturalstyledesc,
       build.buildingclassdesc,
       heat.heatingorsystemdesc,
       landuse.propertylandusedesc,
       story.storydesc,
       construct.typeconstructiondesc
FROM   properties_2017 prop
       INNER JOIN (SELECT parcelid,
                   Max(transactiondate) transactiondate
                   FROM   predictions_2017
                   GROUP  BY parcelid) pred
               USING (parcelid)
               			JOIN predictions_2017 as pred USING (parcelid, transactiondate)
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
       LEFT JOIN storytype story USING (storytypeid)
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
WHERE  prop.latitude IS NOT NULL
       AND prop.longitude IS NOT NULL
"""
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    
    return df
#
# =============================================================================
# 
# =============================================================================

def get_zillow_data():
    '''
    This function reads i zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
        
    return df

zillow_df = get_zillow_data()


zillow_df.columns.value_counts()


zillow_df.info()



cols_to_remove = ['parcelid',
         'calculatedbathnbr',
         'fullbathcnt',
         'heatingorsystemtypeid',
         'propertycountylandusecode',
         'propertylandusetypeid',
         'propertyzoningdesc',
         'censustractandblock',
         'propertylandusedesc', 'airconditioningtypeid',
         'architecturalstyletypeid', 'basementsqft', 'calculatedbathnbr',
         'decktypeid', 'finishedfloor1squarefeet', 'finishedsquarefeet12',
         'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50',
         'finishedsquarefeet6', 'fireplacecnt', 'garagecarcnt', 'garagetotalsqft',
         'hashottuborspa', 'latitude', 'longitude', 'poolsizesum',
         'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'roomcnt', 'typeconstructiontypeid',
         'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'numberofstories', 'fireplaceflag',
         'taxdelinquencyflag', 'taxdelinquencyyear', 'airconditioningdesc', 'architecturalstyledesc',
         'buildingclassdesc', 'heatingorsystemdesc', 'propertylandusedesc', 'storydesc', 'typeconstructiondesc'
         ]
#buildingclasstypeid, structuretaxvaluedollarcnt, assessmentyear, 

def remove_columns(df, cols_to_remove):
	#remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df

zillow_df = remove_columns(zillow_df, cols_to_remove)
# def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
# 	#function that will drop rows or columns based on the percent of values that are missing:\
# 	#handle_missing_values(df, prop_required_column, prop_required_row
#     threshold = int(round(prop_required_column*len(df.index),0))
#     df = df.dropna(axis=1, thresh=threshold)
#     threshold = int(round(prop_required_row*len(df.columns),0))
#     df.dropna(axis=0, thresh=threshold, inplace=True)
#     return df



# zillow_df = handle_missing_values(zillow_df)

# =============================================================================
#                     DATA SUMMARY
# =============================================================================
#Rename the Columns:
zillow_df.columns

#Look at the data for values 
zillow_df.info()
#fips is a unique county idenfier code

zillow_df.shape


zillow_df.describe().T




zillow_df.isnull().sum()



def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
	#function that will drop rows or columns based on the percent of values that are missing:\
	#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def remove_columns(df, cols_to_remove):
	#remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df

def wrangle_zillow():
    if os.path.isfile('zillow_cached.csv') == False:
        df = new_zillow_data(sql)
        df.to_csv('zillow_cached.csv',index = False)
    else:
        df = pd.read_csv('zillow_cached.csv')

    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]

    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)

    # Add column for counties
    df['county'] = df['fips'].apply(
        lambda x: 'Los Angeles' if x == 6037\
        else 'Orange' if x == 6059\
        else 'Ventura')

    # drop unnecessary columns
    dropcols = ['typeconstructiontypeid', 'storytypeid', 'propertylandusetypeid',
       'heatingorsystemtypeid', 'buildingclasstypeid',
       'architecturalstyletypeid', 'airconditioningtypeid', 'id', 'parcelid',
       'basementsqft','buildingqualitytypeid',
       'calculatedbathnbr', 'decktypeid', 'finishedfloor1squarefeet','finishedsquarefeet12',
       'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50',
       'finishedsquarefeet6','fireplacecnt', 'fullbathcnt',
       'garagecarcnt', 'garagetotalsqft', 'hashottuborspa','lotsizesquarefeet', 'poolcnt', 'poolsizesum',
       'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertycountylandusecode', 'propertyzoningdesc',
       'rawcensustractandblock', 'regionidcity', 'regionidcounty',
       'regionidneighborhood', 'regionidzip', 'roomcnt', 'threequarterbathnbr',
       'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
       'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
       'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
       'censustractandblock', 'airconditioningdesc', 'architecturalstyledesc',
       'buildingclassdesc', 'heatingorsystemdesc', 'id', 'propertylandusedesc', 'storydesc', 'typeconstructiondesc']

    df = remove_columns(df, dropcols)

    # replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)

    # assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)

    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 8000]

    # Just to be sure we caught all nulls, drop them here
    
    df = df.dropna()

    return df


zillow_df = wrangle_zillow()





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
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe)
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3).to_markdown())
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe().to_markdown())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('============================================')




summarize(zillow_df)

zillow_df.columns
zillow_df = zillow_df.rename(columns= {})

###### Too Many Columns:
    









train, validate, test = train_validate_test_split(zillow_df)
# =============================================================================
# 1.) Ask at least 5 questions about the data, keeping in mind that your
#     target variable is logerror. e.g. Is logerror significantly different
#     for properties in LA County vs Orange County vs Ventura County?
# =============================================================================











    
    