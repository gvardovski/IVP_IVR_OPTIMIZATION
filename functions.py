import pandas as pd
import os

def get_time_interval(config):
    start_date = pd.to_datetime(config['Backtesting_dates']['start']).normalize()
    end_date = pd.to_datetime(config['Backtesting_dates']['end']).normalize()

    return start_date, end_date

def make_path(config):
    start_date, end_date = get_time_interval(config)
    file_path = config['Data_flname'].split('.')[0]
    split_do = file_path.split('[')[0]
    split_po = file_path.split(']', 1)[1]
    split_po = split_po.split(']', 1)[1]
    file_path = f"{split_do}[{start_date.date()}][{end_date.date()}]{split_po}"
    
    return file_path

def make_wdir(file_path, config):
    wdir = file_path.split("_")[0]
    start_date, end_date = get_time_interval(config)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    wdir = f"{wdir}/[{start_str}_{end_str}]"
    os.makedirs(wdir, exist_ok=True)

    return wdir

def take_years(df):
    years = []
    START_YEAR = df.index[0].year
    END_YEAR = df.index[-1].year

    while START_YEAR <= END_YEAR:
        years.append(f"{START_YEAR}-01-01")
        START_YEAR += 1
    years.append(f"2026-01-01")

    return years