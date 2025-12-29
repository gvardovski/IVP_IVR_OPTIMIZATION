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

    roll = df['IV'].rolling(window=120, min_periods=60)
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
    {"Token": "AAPL", "Path": "NASDAQ-AAPL_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "ABNB", "Path": "NASDAQ-ABNB_[2020-12-10][2025-12-24]_FMP.csv"},
    {"Token": "AMD",  "Path": "NASDAQ-AMD_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "AMZN", "Path": "NASDAQ-AMZN_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "BABA", "Path": "NASDAQ-BABA_[2014-09-19][2025-12-24]_FMP.csv"},
    {"Token": "COIN", "Path": "NASDAQ-COIN_[2021-04-14][2025-12-24]_FMP.csv"},
    # !!!{"Token": "CRCL", "Path": "NASDAQ-CRCL_[2025-06-04][2025-12-24]_FMP.csv"},
    {"Token": "CVNA", "Path": "NASDAQ-CVNA_[2017-04-28][2025-12-24]_FMP.csv"},
    {"Token": "DIS",  "Path": "NASDAQ-DIS_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "DKNG", "Path": "NASDAQ-DKNG_[2019-07-25][2025-12-24]_FMP.csv"},
    {"Token": "GDX",  "Path": "NASDAQ-GDX_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "GME",  "Path": "NASDAQ-GME_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "GOOGL","Path": "NASDAQ-GOOGL_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "HIMS", "Path": "NASDAQ-HIMS_[2019-09-13][2025-12-24]_FMP.csv"},
    {"Token": "HOOD", "Path": "NASDAQ-HOOD_[2021-07-29][2025-12-24]_FMP.csv"},
    {"Token": "IBIT", "Path": "NASDAQ-IBIT_[2024-01-11][2025-12-24]_FMP.csv"},
    {"Token": "JPM",  "Path": "NASDAQ-JPM_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "MARA", "Path": "NASDAQ-MARA_[2012-05-04][2025-12-24]_FMP.csv"},
    {"Token": "META", "Path": "NASDAQ-META_[2012-05-18][2025-12-24]_FMP.csv"},
    {"Token": "MRNA", "Path": "NASDAQ-MRNA_[2018-12-07][2025-12-24]_FMP.csv"},
    {"Token": "MSFT", "Path": "NASDAQ-MSFT_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "MSTR", "Path": "NASDAQ-MSTR_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "NFLX", "Path": "NASDAQ-NFLX_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "NVDA", "Path": "NASDAQ-NVDA_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "PLTR", "Path": "NASDAQ-PLTR_[2020-09-30][2025-12-24]_FMP.csv"},
    {"Token": "PYPL", "Path": "NASDAQ-PYPL_[2015-07-06][2025-12-24]_FMP.csv"},
    {"Token": "RBLX", "Path": "NASDAQ-RBLX_[2021-03-10][2025-12-24]_FMP.csv"},
    {"Token": "RDDT", "Path": "NASDAQ-RDDT_[2024-03-21][2025-12-24]_FMP.csv"},
    {"Token": "SMCI", "Path": "NASDAQ-SMCI_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "SMCI", "Path": "NASDAQ-SMCI_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "SNOW", "Path": "NASDAQ-SNOW_[2020-09-16][2025-12-24]_FMP.csv"},
    {"Token": "TSLA", "Path": "NASDAQ-TSLA_[2010-06-29][2025-12-24]_FMP.csv"},
    {"Token": "TSM",  "Path": "NASDAQ-TSM_[2010-01-04][2025-12-24]_FMP.csv"},
    {"Token": "XOM",  "Path": "NASDAQ-XOM_[2010-01-04][2025-12-24]_FMP.csv"}
    ]

    for ins in instruments:

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        config['Data_flname'] = f"data/{ins['Path']}"

        df_day = processdata(config)
        years = take_years(df_day)

        for i in range(len(years), 0, -2): 
            config['Backtesting_dates']['start'] = years[i - 3]
            config['Backtesting_dates']['end'] = years[i - 1]
            file_path = make_path(config)

            df_copy = df_day.copy()
            df_copy = df_copy.loc[years[i - 3]:years[i - 1]]

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