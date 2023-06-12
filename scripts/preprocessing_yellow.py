import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import timedelta



def data_missing_value_imputation(df):
    def value_imputation(df, column_name, fill_value):
        '''Function to fill missing rows with specific value'''
        df[column_name] = df[column_name].fillna(fill_value)
        return df
    df = (df
        ## missing value imputation with values out of range of standard values that we can later recognize
        .pipe(value_imputation, column_name='RatecodeID', fill_value=0)
        ## gonna assume that it had to be stored in car if there was a problem with server connection
        .pipe(value_imputation, column_name='store_and_fwd_flag', fill_value='Y')
        ## most of the time there there is a value of 2.5 so im gonna use it
        .pipe(value_imputation, column_name='congestion_surcharge', fill_value=2.5)
        ## im gonna assume there is not airport fee, but mayby better idea is to do it by identifying starting district
        .pipe(value_imputation, column_name='airport_fee', fill_value=0)
        ## passengers are problematic, for now assigning -1 value for problematics cases
        .pipe(value_imputation, column_name='passenger_count', fill_value=-1)
    )
    
    if sum(df.isna().sum()) > 0:
        print(df.isna().sum())
        raise Exception('Column with missing values found after value imputation')

    return df


def data_integrity_filtering(df, start_date, end_date):
    '''
    Performs filtering to remove observations with nonsense data, explaination in query
    '''
    df = df[
        ## only valid vendors and zones
        (df['VendorID'].isin([1, 2]) & df['DOLocationID'].between(1, 263) & df['PULocationID'].between(1, 263))
        ## so there are cases with negative, or 0 passengers im not sure if the nyc taxi 
        # can work as delivery service that can transport other stuff than people so im gonna remove these cases
        # also i found information that https://www.nyc.gov/site/tlc/passengers/passenger-frequently-asked-questions.page
        # there can be up to 4-5 passengers, and sometimes a child (so up to 6) can be transproted, also keeping imputed -1 indicator
        & ((df['passenger_count'].between(1, 6)) | (df['passenger_count'] == -1))
        ## making sure that variables that have exactly specified values in documentation take those values
        & (df['RatecodeID'].between(0, 6) & df['store_and_fwd_flag'].isin(['Y', 'N']) & df['payment_type'].between(1, 6) 
        & df['airport_fee'].isin([0, 1.25]) & df['mta_tax'].isin([0, 0.5])) & (df['congestion_surcharge'].isin([0, 2.5])) 
        ## values that should always be non negative
        & ((df['fare_amount'] >= 0) & (df['extra'] >= 0) & (df['tip_amount'] >= 0) & (df['tolls_amount'] >= 0) & (df['total_amount'] >= 0) )
        ## valid dates
        & ((df['tpep_dropoff_datetime'] >= f'{start_date}-01') & (df['tpep_dropoff_datetime'] <= f'{end_date}-01'))
        & ((df['tpep_pickup_datetime'] >= f'{start_date}-01') & (df['tpep_pickup_datetime'] <= f'{end_date}-01'))
        ## removing problematic cases where drop off happend before pickup or the trip took 0 seconds,
        & (df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime'])
    ]
    return df


def data_feature_engineering(df, loc_borough_map):
    """Feature engineering associated with extractin time and localizatin related infromation"""
    df = (df
        ## adding simple information about traveling time in minutes
        .assign(trip_time_min=lambda x: (x['tpep_dropoff_datetime']-x['tpep_pickup_datetime'])/timedelta(minutes=1))
        # borough mapping
        .assign(PU_Borough=lambda x: x['PULocationID'].map(loc_borough_map))
        .assign(DO_Borough=lambda x: x['DOLocationID'].map(loc_borough_map))
        .astype({'PU_Borough': 'category', 'DO_Borough': 'category'})
    )
    return df


def data_infered_filtering(df):
    '''This function is used to perform filtering on a dataframe by infromation infered from EDA'''
    df = df[
        # 99.8% of trips below 25 miles, there were higher values such as 1000miles with is most likely an error // len(df[df['trip_distance'] <= 25])/len(df)
        (df['trip_distance'] <= 25) 
        # 99.8% of trips took less than 80 minutes, other entries were much greater so i also consider them as an error // len(df[df['trip_time_min'] <= 80])/len(df)
        & (df['trip_time_min'] <= 80) 
        ## now filtering based on scatter relation inspection
        # There is quiet alot of short trips with big distances which is not possible to do in such a bussy city, im giving a limit of 3 miles in 2 minutes
        & ( (df['trip_time_min'] >= 2)  | ((df['trip_time_min'] <= 2) & (df['trip_distance'] <= 3)) ) 
        # Now trips that took way too long given that there was zero traveling, im being extra careful as i think its possible to have long (in time) 
        # trips in a traffic or simply that a taxi was waiting for a passanger
        & ( (df['trip_distance'] >= .5)  | ((df['trip_distance'] <= .5) & (df['trip_time_min'] <= 30)) )
        # there is very little trips with cost higher than 100, im kinda expexting that some of the highest values are like fine for destroying something?
        & (df['total_amount'] < 125)
    ]
    return df


def define_start_end_date(year, month):
    ## preparation for various months
    if month < 9:
        start_date, end_date = f'{year}-0{month}', f'{year}-0{month+1}'
    elif month == 9:
        start_date, end_date = f'{year}-0{month}', f'{year}-10'
    elif month == 10 or month == 11:
        start_date, end_date = f'{year}-{month}', f'{year}-{month+1}'
    elif month == 12:
        start_date, end_date = f'{year}-{month}', f'{year+1}-01'
    return start_date, end_date


def define_zone_mapping(df_path):
    ## dict mapping zone - borough
    df_zones = pd.read_csv(f'{df_path}/taxi_zones/taxi+_zone_lookup.csv').iloc[:-2, :]
    zones_mapping = df_zones[['LocationID', 'Borough']].to_dict()
    keys = list(zones_mapping.keys())
    loc_borough_map = {zones_mapping[keys[0]][i]: zones_mapping[keys[1]][i]  for i in range(len(zones_mapping['LocationID']))}
    return loc_borough_map


def clean_yellow_taxi_df(df_path, year, month):
    ## mapping constants
    start_date, end_date = define_start_end_date(year, month)
    loc_borough_map = define_zone_mapping(df_path)

    ## processing data
    df = pd.read_parquet(f'{df_path}/yellow_tripdata_{start_date}.parquet')
    df = data_missing_value_imputation(df)
    df = data_integrity_filtering(df, start_date, end_date)
    df = data_feature_engineering(df, loc_borough_map)
    df = data_infered_filtering(df)

    return df