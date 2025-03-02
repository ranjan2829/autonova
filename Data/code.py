import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

# Fetch S&P 500 index data (^GSPC)
end_date = datetime(2025, 2, 28)  # Approx end of Feb 2025
start_date = end_date - timedelta(days=365)  # 12 months back
spx = yf.download('^GSPC', start=start_date, end=end_date, interval='1mo')

# Process stock market data: monthly close, % change, volume
stock_data = []
if not spx.empty:
    for i, (date, row) in enumerate(spx.iterrows()):
        month = date.strftime('%b')
        value = round(row['Close'], 2)  # Closing price
        volume = round(row['Volume'] / 1e9, 1)  # Volume in billions
        # Calculate % change (0 for first month)
        change = round(((value - spx['Close'].iloc[i-1]) / spx['Close'].iloc[i-1]) * 100, 1) if i > 0 else 0.0
        stock_data.append({'month': month, 'value': value, 'change': change, 'volume': volume})
else:
    print("No S&P 500 data retrieved.")

# Fetch sector data using sector ETFs (approximation)
sectors = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial': 'XLF',
    'Consumer': 'XLY',
    'Energy': 'XLE'
}
sector_data = []
for name, ticker in sectors.items():
    etf = yf.download(ticker, start=start_date, end=end_date, interval='1mo')
    if not etf.empty and len(etf) > 1:  # Ensure we have at least 2 data points
        first_close = etf['Close'].iloc[0]  # First closing price
        last_close = etf['Close'].iloc[-1]  # Last closing price
        performance = round(((last_close - first_close) / first_close) * 100, 1)
        sector_data.append({'name': name, 'value': performance})
    else:
        print(f"No data retrieved for {name} ({ticker}).")
        sector_data.append({'name': name, 'value': 0.0})  # Default to 0 if no data

# Static market cap data (approximation, S&P 500 is mostly large-cap)
market_cap_data = [
    {'name': 'Large Cap', 'value': 85},
    {'name': 'Mid Cap', 'value': 14},
    {'name': 'Small Cap', 'value': 1},
    {'name': 'Micro Cap', 'value': 0}
]

# Combine all data
data = {
    'stockMarketData': stock_data,
    'sectorData': sector_data,
    'marketCapData': market_cap_data
}

# Save to JSON file
with open('sp500_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Data fetched and saved to sp500_data.json")