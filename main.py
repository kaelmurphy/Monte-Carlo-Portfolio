import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Define Canadian stocks with the .TO suffix
stockList = ['BMO', 'TD', 'CNR', 'SHOP', 'RY', 'ENB']
stocks = [stock + '.TO' for stock in stockList]

# Define date range
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=10)

# Get mean returns and covariance matrix
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

print("Mean Returns:")
print(meanReturns)
print("\nCovariance Matrix:")
print(covMatrix)

# Generate random weights for portfolio and normalize them
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
print("\nPortfolio Weights:")
print(weights)

# Monte Carlo parameters
sims = 100  # number of simulations
days = 1000  # number of days to simulate

# Prepare simulation array
portSim = np.full(shape=(days, sims), fill_value=0.0)  # corrected variable name

initPort = 10000  # Initial portfolio value

# Monte Carlo Simulation
for m in range(sims):
    # Generate random daily returns for each stock
    Z = np.random.normal(size=(days, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanReturns.values + np.dot(Z, L.T)
    portSim[:, m] = np.cumprod(np.dot(dailyReturns, weights) + 1) * initPort

# Plot results
plt.plot(portSim)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Stock Portfolio Value Over Time')
plt.show()
