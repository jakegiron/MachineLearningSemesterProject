import pandas as pd
from datetime import datetime
from sklearn.neighbors import KDTree


def load_data():
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    raw_data_df = pd.read_csv('oldData/mesilla_aquifer_data.csv',
                              index_col=False,
                              parse_dates=['Date'],
                              date_parser=dateparse)
    raw_data_df.columns = ['id', 'date', 'wl', 'precip', 'mean_temp']
    raw_data_df['date'] = raw_data_df['date'].apply(lambda x: x.strftime('%Y-%m-01'))
    return raw_data_df


def combine_duplicate_month_entries(df):
    for i in range(len(df) - 1):
        df.update(df, overwrite=True)

        if df.loc[i].id == df.loc[i + 1].id:
            if df.loc[i].date == df.loc[i + 1].date:
                avg_wl = sum([df.loc[i].wl, df.loc[i + 1].wl]) / 2
                sum_precipitation_to_ft = (df.loc[i].precip + df.loc[i + 1].precip) / 12
                avg_temp = sum([df.loc[i].mean_temp, df.loc[i + 1].mean_temp]) / 2

                df.loc[df.index == i + 1, 'wl'] = avg_wl
                df.loc[df.index == i + 1, 'precip'] = sum_precipitation_to_ft
                df.loc[df.index == i + 1, 'mean_temp'] = avg_temp

                df = df.drop(i)
    return df


def pivot_table_and_clean_dates(raw):
    df = raw.pivot_table(index='date', columns=["id"], values=['wl', 'precip', 'mean_temp']) \
        .reorder_levels([1, 0], axis=1) \
        .sort_index(axis=1)
    #  Don't ever delete this!!!
    df.index = pd.to_datetime(df.index).date
    idx = pd.date_range('1985-01-01', '2014-12-01')
    mask = idx.day == 1
    idx = idx[mask].date
    df = df.reindex(idx)
    df = df.dropna(axis='columns', thresh=276)
    return df


def clean_values(df):
    df = df.iloc[:-60, :]
    df = df.interpolate(method='linear')
    return df


def save_clean_data(df):
    dft = df.stack(level=0).reset_index()
    dft.columns = ['date', 'id', 'mean_temp', 'precip', 'wl']
    dft['date'] = pd.to_datetime(dft['date'])
    dft['id'] = dft['id'].astype(str)
    dft = dft.reindex(columns=['id', 'date', 'wl', 'mean_temp', 'precip'])
    dft.to_csv('clean_waterlevel_data.csv', index=False)


def see_well_df_info(df_elong, thesh):
    dft_piv = pivot_table(df_elong)
    dft_piv = clean_dates(dft_piv, thesh)

    dft = dft_piv.stack(level=0).reset_index()
    dft.columns = ['date', 'id', 'mean_temp', 'precip', 'wl']
    dft['date'] = pd.to_datetime(dft['date'])
    dft['id'] = dft['id'].astype(str)
    print('thresh: ', thesh)
    print('data shape: ', dft_piv.shape)
    print('unique wells:', len(dft.id.unique()))
    print('Nan count:', dft_piv.isna().sum().sum())
    print('\n')


def pivot_table(raw):
    df_r = raw.pivot_table(index='date', columns=["id"], values=['wl', 'precip', 'mean_temp']) \
        .reorder_levels([1, 0], axis=1) \
        .sort_index(axis=1)
    return df_r


def clean_dates(df_r, thresh):
    df_r.index = pd.to_datetime(df_r.index).date
    idx = pd.date_range('1985-01-01', '2014-12-01')
    mask = idx.day == 1
    idx = idx[mask].date
    df_r = df_r.reindex(idx)
    df_r = df_r.dropna(axis='columns', thresh=thresh)
    return df_r


def get_nearest_data(well_coor, target_well, df):
    all_pts = well_coor[['x', 'y']].values
    target_pts = well_coor.loc[target_well].values
    tree = KDTree(all_pts)
    d, i = tree.query([(target_pts[0], target_pts[1])], k=3)
    well_names = well_coor.index.values[i[0, 0]], well_coor.index.values[i[0, 1]], well_coor.index.values[i[0, 2]]

    df_1 = df.loc[:, [well_names[0], well_names[1], well_names[2]]]

    df_2 = df_1.drop((well_names[1], 'precip'), axis=1)
    df_2 = df_2.drop((well_names[1], 'mean_temp'), axis=1)
    df_2 = df_2.drop((well_names[2], 'precip'), axis=1)
    df_2 = df_2.drop((well_names[2], 'mean_temp'), axis=1)
    df_2['trend'] = df
    return df_2, well_names
