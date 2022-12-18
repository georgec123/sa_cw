import yfinance as yf
import os
from cryptocmd import CmcScraper


prices_dir = os.path.join(os.path.dirname(__file__), 'prices')

coins = ['BTC', 'MAID', 'XMR', 'DOGE', 'LTC', 'DASH', 'XRP']


for coin in coins:
    scraper = CmcScraper(coin)
    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()
    file_path = os.path.join(prices_dir, f'./{coin}.csv')
    df.to_csv(file_path, index=False)
    print(f"Done {coin}")