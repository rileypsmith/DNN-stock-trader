"""
A script for collecting stock data from Yahoo finance.

@author: Riley Smith
Created: 08/30/2023
"""
from pathlib import Path
import time

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests
from tqdm import tqdm

BASE_URL = "https://query1.finance.yahoo.com/v7/finance/download/{}?period1=345427200&period2=1693353600&interval=1d&events=history&includeAdjustedClose=true"

def scrape_data(ticker, out_folder):
    """Scrape historical data from Yahoo Finance for the given stock ticker"""
    # Form a fake useragent for Chrome
    user_agent = UserAgent().chrome
    # Request the download data
    page = requests.get(BASE_URL.format(ticker), headers={'User-Agent': user_agent})
    # Save it to CSV
    with open(str(Path(out_folder, f'{ticker}.csv')), 'w+') as fp:
        fp.write(page.text)

def parse_sp500():
    """Parse a list of S&P 500 tickers from Wikipedia"""
    # Parse page of S&P companies into BeautifulSoup object
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table_body = soup.findAll('tbody')[0]
    tickers = []
    # For each row in the table, extract just the ticker and store it
    for row in table_body.findAll('tr')[1:]:
        local_ticker = row.findChildren('td')[0].text.strip()
        if '.' in local_ticker:
            continue
        tickers.append(local_ticker)
    return tickers

def make_dataset(out_folder, resume=True):
    """Get a list of S&P500 companies, scrape and save data for each one."""
    # Get list of tickers
    tickers = parse_sp500()
    # Sleep so you don't make too many requests in a row
    time.sleep(3)
    # Check which tickers you already have data for
    done_tickers = [f.stem for f in Path(out_folder).glob('*.csv')]
    for ticker in tqdm(tickers):
        # If you already have data, skip
        if resume and ticker in done_tickers:
            continue
        # Scrape and save data
        scrape_data(ticker, out_folder)
        time.sleep(3)
    return

if __name__ == '__main__':
    out_folder = 'data'
    make_dataset(out_folder)