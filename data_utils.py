from datetime import datetime, date
import numpy as np
import pandas as pd
import os
import re

date_anomalies = np.array([])


def load_raw_data(file_path, csv_copy=False, csv_title=None):
    print('file: ', file_path)

    df_raw = pd.read_csv(file_path,
                         index_col=False)

    df = df_raw[['Time',
                 'SiteNo',
                 'Water level in feet relative to NAVD88',
                 'Depth to Water Below Land Surface in ft.',
                 ]]

    # TODO: Need to add: precipitation, temperature, and coordinates,
    df = df.rename(columns={'Time': 'Date'})
    df = df.rename(columns={'SiteNo': 'well_id'})
    df = df.rename(columns={'Water level in feet relative to NAVD88': 'wl'})
    df = df.rename(columns={'Depth to Water Below Land Surface in ft.': 'dtw'})

    if csv_copy:
        create_csv(df, csv_title)
    return df


def create_csv(df, csv_title):
    df.to_csv(csv_title, index=False)


def clean_dates(df):
    df['Date'] = df.Date.astype(str)
    df['Date'] = df['Date'].apply(clean_dates_one)
    df = clean_dates_two()
    df.to_csv('data_fixed_dates.csv', index=False)


def clean_dates_one(one_date):
    if re.search('\d{4}-\d{2}-\d{2}', one_date):
        pos = re.search('\d{4}-\d{2}-\d{2}', one_date).end()

        return one_date[:pos]

    else:
        np.append(date_anomalies, one_date)
        return one_date


def clean_dates_two():
    for x in date_anomalies:
        df = df[df.Date != x]
    return df


def load_data(file):
    dateparser = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv(file,
                     index_col=False,
                     parse_dates=['Date'],
                     date_parser=dateparser)

    df['wl'] = df.wl.astype(float)
    df['dtw'] = df.dtw.astype(float)
    return df


def pivot_dataframe(df, all_data):
    if all_data:
        df = df.pivot_table(index='Date', columns=["well_id"],
                            # values=['dtw', 'wl', 'P_ft', 'Temp_mean_F'])
                            values=['dtw', 'wl']) \
            .reorder_levels([1, 0], axis=1) \
            .sort_index(axis=1)
    else:
        df = df.pivot_table(index='Date', columns=["Well_ID"], values=['WL_elev_ft']) \
            .reorder_levels([1, 0], axis=1) \
            .sort_index(axis=1)
    return df


def remove_data(df):
    df.dropna(axis='columns', thresh=1090, inplace=True)
    df.dropna(axis='index', thresh=1, inplace=True)
    return df
