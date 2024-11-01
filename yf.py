import yfinance as yf
aapl = yf.Ticker("aapl")

print(aapl.dividends)
