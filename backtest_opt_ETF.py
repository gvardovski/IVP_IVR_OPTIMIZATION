import pandas as pd
import vectorbt as vbt
import numpy as np
import yaml
from makemetricpng import create_hitmap
from savetopdf import save_backtesting_results_to_pdf
from functions import get_time_interval, make_path, take_years

def processdata(config):
    df = pd.read_csv(config['Data_flname'], parse_dates=['Time'], index_col='Time')
    df.drop(columns=['Volume'], inplace=True)
    df.dropna(inplace=True)

    roll = df['IV'].rolling(window=252, min_periods=120)
    df['IVR'] = (df['IV'] - roll.min()) / (roll.max() - roll.min()) * 100

    def calc_percentile(x):
        current_iv = x.iloc[-1]
        return (x < current_iv).mean() * 100

    df['IVP %'] = roll.apply(calc_percentile, raw=False)
    df['IVP/IVR blend'] = df['IVR'] * 0.5 + df['IVP %'] * 0.5
    df.dropna(inplace=True)
    start_date, end_date = get_time_interval(config)
    df = df.loc[start_date:end_date]
    return df

def backtest(df, blend):

    index_arr = df.index.to_numpy()
    close_arr = df['Close'].to_numpy()
    blend_arr = df['IVP/IVR blend'].to_numpy()
    entry_arr = np.full(len(index_arr), False)
    exit_arr = np.full(len(index_arr), False)
    price_arr = np.full(len(index_arr), np.nan)

    trade_is_open = False
    ent_str = blend
    ent_end = blend + 15
    ext_str = blend + 30
    ext_end = blend + 45

    for i in range(0, len(index_arr)):
        momentum_ok = abs(close_arr[i] - close_arr[i-5]) / close_arr[i] > 0.01
        if not trade_is_open and (ent_str <= blend_arr[i] <= ent_end) and momentum_ok:
            trade_is_open = True
            entry_arr[i] = True
            price_arr[i] = close_arr[i]
        elif trade_is_open and (ext_str <= blend_arr[i] <= ext_end):
            trade_is_open = False
            exit_arr[i] = True
            price_arr[i] = close_arr[i]

    pf = vbt.Portfolio.from_signals(
    entries = entry_arr,
    exits = exit_arr,
    price = price_arr,
    open = df["Open"],
    close = close_arr,
    size = config['Trade']['size'],
    size_type = config['Trade']['size_type'],
    init_cash = config['Initial_cash'],
    freq = '1d'
    )

    return pf

if __name__ == "__main__":

    instruments = [
    # {"Token": "ABNY", "Path": "NASDAQ:ABNY_[2024-06-25][2025-12-26]_FMP.csv"},
    # {"Token": "AMDY", "Path": "NASDAQ:AMDY_[2023-09-19][2025-12-26]_FMP.csv"},
    # {"Token": "AMZY", "Path": "NASDAQ:AMZY_[2023-07-25][2025-12-26]_FMP.csv"},
    # {"Token": "APLY","Path": "NASDAQ:APLY_[2023-04-18][2025-12-26]_FMP.csv"},
    # !!!{"Token": "BABO", "Path": "NASDAQ:BABO_[2024-08-07][2025-12-26]_FMP.csv"},
    # !!!{"Token": "BRKC", "Path": "NASDAQ:BRKC_[2025-06-05][2025-12-26]_FMP.csv"},
    # {"Token": "CONY", "Path": "NASDAQ:CONY_[2023-08-15][2025-12-26]_FMP.csv"},
    # !!!{"Token": "CRCO", "Path": "NASDAQ:CRCO_[2025-09-29][2025-12-26]_FMP.csv"},
    # !!!{"Token": "CVNY", "Path": "NASDAQ:CVNY_[2025-01-30][2025-12-26]_FMP.csv"},
    # {"Token": "DISO", "Path": "NASDAQ:DISO_[2023-08-25][2025-12-26]_FMP.csv"},
    # !!!{"Token": "DRAY", "Path": "NASDAQ:DRAY_[2025-07-15][2025-12-26]_FMP.csv"},
    # {"Token": "FBY",  "Path": "NASDAQ:FBY_[2023-07-28][2025-12-26]_FMP.csv"},
    # {"Token": "GDXY", "Path": "NASDAQ:GDXY_[2024-05-21][2025-12-26]_FMP.csv"},
    # !!!{"Token": "GMEY", "Path": "NASDAQ:GMEY_[2025-09-09][2025-12-26]_FMP.csv"},
    # {"Token": "GOOY", "Path": "NASDAQ:GOOY_[2023-07-28][2025-12-26]_FMP.csv"},
    # !!!{"Token": "HIYY", "Path": "NASDAQ:HIYY_[2025-09-22][2025-12-26]_FMP.csv"},
    # !!!{"Token": "HOOY", "Path": "NASDAQ:HOOY_[2025-05-08][2025-12-26]_FMP.csv"},
    # {"Token": "JPMO", "Path": "NASDAQ:JPMO_[2023-09-11][2025-12-26]_FMP.csv"},
    # !!!{"Token": "MARO", "Path": "NASDAQ:MARO_[2024-12-10][2025-12-26]_FMP.csv"},
    # {"Token": "MRNY", "Path": "NASDAQ:MRNY_[2023-09-25][2025-12-26]_FMP.csv"},
    # {"Token": "MSFO", "Path": "NASDAQ:MSFO_[2023-08-25][2025-12-26]_FMP.csv"},
    # {"Token": "MSTY", "Path": "NASDAQ:MSTY_[2024-02-22][2025-12-26]_FMP.csv"},
    # {"Token": "NFLY", "Path": "NASDAQ:NFLY_[2023-08-08][2025-12-26]_FMP.csv"},
    # {"Token": "NVDY", "Path": "NASDAQ:NVDY_[2023-05-11][2025-12-26]_FMP.csv"},
    # !!!{"Token": "PLTY", "Path": "NASDAQ:PLTY_[2024-10-08][2025-12-26]_FMP.csv"},
    # {"Token": "PYPY",  "Path": "NASDAQ:PYPY_[2023-09-26][2025-12-26]_FMP.csv"},
    # !!!{"Token": "RBLY", "Path": "NASDAQ:RBLY_[2025-07-29][2025-12-26]_FMP.csv"},
    # !!!{"Token": "RDYY", "Path": "NASDAQ:RDYY_[2025-08-25][2025-12-26]_FMP.csv"},
    # !!!{"Token": "SMCY", "Path": "NASDAQ:SMCY_[2024-08-20][2025-12-26]_FMP.csv"},
    # {"Token": "SNOY", "Path": "NASDAQ:SNOY_[2024-06-11][2025-12-26]_FMP.csv"},
    # {"Token": "TSLY", "Path": "NASDAQ:TSLY_[2022-11-23][2025-12-26]_FMP.csv"},
    # {"Token": "TSMY", "Path": "NASDAQ:TSMY_[2024-08-21][2025-12-26]_FMP.csv"},
    # {"Token": "XOMO", "Path": "NASDAQ:XOMO_[2023-08-31][2025-12-26]_FMP.csv"},
    # !!!{"Token": "YBIT", "Path": "NASDAQ:YBIT_[2024-04-23][2025-12-26]_FMP.csv"}
    ]

    for ins in instruments:

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        config['Data_flname'] = f"data/{ins['Path']}"

        df_day = processdata(config)
        if df_day.empty:
            print(f"No data loaded for {ins['Token']}, skipping instrument")
            continue
        years = take_years(df_day)

        for i in range(len(years), 2, -1): 
            config['Backtesting_dates']['start'] = years[i - 2]
            config['Backtesting_dates']['end'] = years[i - 1]
            file_path = make_path(config)

            df_copy = df_day.copy()
            df_copy = df_copy.loc[years[i - 2]:years[i - 1]]

            if df_copy.empty:
                continue

            results = []

            for blend in range(0, 60, 5):
                pf = backtest(df_copy, blend=blend)
                config['IVP/IVR blend']['start'] = f"{blend}-{blend + 15}"
                config['IVP/IVR blend']['end'] = f"{blend + 30}-{blend + 45}"
                save_backtesting_results_to_pdf(pf, f"{file_path}", config)

                stats = pf.stats()

                results.append({
                'Blend Enter Start': blend,
                'Blend Enter End': blend + 15,
                'Blend Exit Start': blend + 30,
                'Blend Exit End': blend + 45,
                'Total Return': stats['Total Return [%]'],
                'Sharpe': stats['Sharpe Ratio'],
                })

            results_df = pd.DataFrame(results)
            create_hitmap(results_df, metric_name='Total Return', file_path=file_path, config=config)