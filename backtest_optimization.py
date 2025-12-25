import pandas as pd
import vectorbt as vbt
import numpy as np
import yaml
from makemetricpng import create_hitmap
from savetopdf import save_backtesting_results_to_pdf

def get_time_interval(config):
    start_date = pd.to_datetime(config['Backtesting_dates']['start']).normalize()
    end_date = pd.to_datetime(config['Backtesting_dates']['end']).normalize()
    return start_date, end_date


def processdata(config):
    df = pd.read_csv(config['Data_flname'], parse_dates=['Time'], index_col='Time')
    df.drop(columns=['Volume'], inplace=True)
    df.dropna(inplace=True)

    roll = df['IV'].rolling(window=252, min_periods=252)
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

def make_path(config):
    start_date, end_date = get_time_interval(config)
    file_path = config['Data_flname'].split('.')[0]
    split_do = file_path.split('[')[0]
    split_po = file_path.split(']')[1]
    file_path = f"{split_do}[{start_date.date()}:{end_date.date()}]{split_po}"
    
    return file_path

if __name__ == "__main__":
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    df_day = processdata(config)
    file_path = make_path(config)

    # instruments = [
#     "ABNB", "AMD", "AMZN", "AAPL", "BABA", "BRK.B", "COIN", "CRCL", "CVNA", "DIS",
#     "DKNG", "META", "GDX", "GME", "GOOGL", "HIMS", "HOOD", "JPM", "MARA", "MRNA",
#     "MSFT", "MSTR", "NFLX", "NVDA", "PLTR", "PYPL", "RBLX", "RDDT", "SMCI", "SNOW",
#     "TSLA", "TSM", "XOM", "IBIT"
# ]

    results = []

    for blend in range(0, 60, 5):
        pf = backtest(df_day, blend=blend)
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
    create_hitmap(results_df, metric_name='Total Return', file_path=file_path)