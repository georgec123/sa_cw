import os
from cryptocmd import CmcScraper
import yfinance

prices_dir = os.path.join(os.path.dirname(__file__), 'prices')

coins = ['BTC', 'MAID', 'XMR', 'DOGE', 'LTC', 'DASH', 'XRP']


for coin in coins:
    scraper = CmcScraper(coin)
    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()
    file_path = os.path.join(prices_dir, f'./{coin}.csv')
    df.to_csv(file_path, index=False)
    print(f"Done {coin}")


# get eur data
euro = yfinance.Ticker('EURUSD=X').history(
    start='2014-01-01', end='2023-01-10')
euro.reset_index(inplace=True)
euro = euro.rename(columns={'Close': 'Price'})

euro.to_csv(os.path.join(prices_dir, 'EURUSD.csv'),
            index=False, header=True)
