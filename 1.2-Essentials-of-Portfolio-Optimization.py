import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import PortfolioOptimizationKit as pok  # Ensure this custom toolkit is accessible

# Setting the style for the plots
sns.set_style("darkgrid")

# Define stock tickers and calculate the number of assets
tickers = ['AMZN', 'KO', 'MSFT']
n_assets = len(tickers)

# Initialize a DataFrame to store stock data
stocks = pd.DataFrame()

# Set the start and end dates for fetching historical data
start_date = "2011-01-01"
end_date = "2023-12-26"

# Loop through each stock to retrieve daily adjusted close prices over the specified period
for stock_name in tickers:
    ticker_data = yf.Ticker(stock_name)
    hist_data = ticker_data.history(start=start_date, end=end_date)
    stocks[stock_name] = hist_data['Close']

# Round the stock data for better readability
stocks = round(stocks, 2)

# Calculate daily returns using a portfolio optimization kit
daily_rets = pok.compute_returns(stocks)

# Annualize the daily returns assuming 252 trading days per year
ann_rets = pok.annualize_rets(daily_rets, 252)

# Compute mean, standard deviation, and covariance of daily returns
mean_rets = daily_rets.mean()
std_rets = daily_rets.std()
cov_rets = daily_rets.cov()

# Define parameters for portfolio simulation
periods_per_year = 252
num_portfolios = 4000
risk_free_rate = 0

# Generate portfolios with random weights
all_portfolios = []  # Initialize a list to collect all portfolio data

for i in range(num_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)

    portfolio_ret = pok.portfolio_return(weights, ann_rets)
    portfolio_vol = pok.portfolio_volatility(weights, cov_rets)
    portfolio_vol = pok.annualize_vol(portfolio_vol, periods_per_year)
    portfolio_spr = pok.sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)

    # Append the new portfolio data to the list
    all_portfolios.append({"return": portfolio_ret,
                           "volatility": portfolio_vol,
                           "sharpe ratio": portfolio_spr,
                           "w1": weights[0],
                           "w2": weights[1],
                           "w3": weights[2]})

# Convert the list of all portfolio data into a DataFrame
portfolios = pd.DataFrame(all_portfolios)

# Plotting Portfolios and Efficient Frontier
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
im = ax.scatter(portfolios["volatility"], portfolios["return"], c=portfolios["sharpe ratio"], s=20,
                edgecolor=None, cmap='RdYlBu')
ax.set_title("Portfolios and Efficient Frontier")
ax.set_ylabel("return (%)")
ax.set_xlabel("volatility (%)")
ax.grid()

df = pok.efficient_frontier(50, daily_rets, cov_rets, periods_per_year)
df.plot.line(x="volatility", y="return", style="--", color="tab:green", ax=ax, grid=True,
             label="Efficient frontier")

fig.colorbar(im, ax=ax)
plt.show()

# Function to get portfolio features
def get_portfolio_features(weights, rets, covmat, risk_free_rate, periods_per_year):
    """
    Calculate and print portfolio return, volatility, and Sharpe ratio.

    Parameters:
    - weights: Array of asset weights in the portfolio.
    - rets: Annualized returns for each asset.
    - covmat: Covariance matrix of asset returns.
    - risk_free_rate: Risk-free rate for Sharpe ratio calculation.
    - periods_per_year: Number of periods in a year (trading days).

    Returns:
    Tuple of (return, volatility, sharpe ratio) for the portfolio.
    """
    # Calculate portfolio volatility
    vol = pok.portfolio_volatility(weights, covmat)
    vol = pok.annualize_vol(vol, periods_per_year)

    # Calculate portfolio return
    ret = pok.portfolio_return(weights, rets)

    # Calculate portfolio Sharpe ratio
    shp = pok.sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)

    # Display the calculated metrics
    print("Portfolio return: {:.2f}%" .format(ret*100))
    print("Portfolio volatility: {:.2f}%" .format(vol*100))
    print("Portfolio Sharpe ratio: {:.2f}" .format(shp))

    return ret, vol, shp

# Optimizing for Minimum Volatility
print("Annual returns of individual assets:")
print(ann_rets)

# Find optimal weights for the portfolio with minimum volatility
optimal_weights = pok.minimize_volatility(ann_rets, cov_rets)
print("Optimal weights for Minimum Volatility Portfolio:")
print("  AMZN: {:.2f}%".format(optimal_weights[0]*100))
print("  KO:   {:.2f}%".format(optimal_weights[1]*100))
print("  MSFT: {:.2f}%".format(optimal_weights[2]*100))

# Calculate portfolio features for the portfolio with minimum volatility
ret, vol, shp = get_portfolio_features(optimal_weights, ann_rets, cov_rets, risk_free_rate, periods_per_year)

# Plotting the Efficient Frontier with Minimum Volatility Portfolio
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df.plot.line(x="volatility", y="return", style="--", color="coral", ax=ax, grid=True, label="Efficient frontier")
ax.scatter(vol, ret, marker="X", color='g', s=120, label="Minimum Volatility Portfolio")
ax.set_xlim([0.13, 0.33])
ax.legend()
ax.set_title("Minimum Volatility Portfolio on Efficient Frontier")
plt.show()

# Optimizing for Minimum Volatility for a Specific Return
target_return = 0.16

# Calculate optimal weights to minimize volatility for the given target return
optimal_weights = pok.minimize_volatility(ann_rets, cov_rets, target_return)
print("Optimal weights for Minimum Volatility Portfolio with Target Return:")
print("  AMZN: {:.2f}%".format(optimal_weights[0]*100))
print("  KO:   {:.2f}%".format(optimal_weights[1]*100))
print("  MSFT: {:.2f}%".format(optimal_weights[2]*100))

# Calculate and display the portfolio's return, volatility, and Sharpe ratio for the given target return
ret, vol, shp = get_portfolio_features(optimal_weights, ann_rets, cov_rets, risk_free_rate, periods_per_year)

# Plotting the Efficient Frontier with Minimum Volatility Portfolio for Target Return
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df.plot.line(x="volatility", y="return", style="--", color="coral", ax=ax, grid=True, label="Efficient frontier")
ax.scatter(vol, target_return, marker="X", color='g', s=120, label="Min. Volatility Portfolio")
ax.set_xlim([0.13, 0.33])
ax.legend()
ax.set_title("Minimum Volatility Portfolio for Given Return of 16%")
plt.show()

# Now, we will add the optimization for maximum Sharpe Ratio and for maximum Sharpe Ratio at a specified volatility
# based on the structure and formulas provided in the documentation.

# Optimizing Portfolios for Maximum Sharpe Ratio
optimal_weights_sharpe = pok.maximize_shape_ratio(ann_rets, cov_rets, risk_free_rate, periods_per_year)
print("Optimal weights for Maximum Sharpe Ratio Portfolio:")
print("  AMZN: {:.2f}%".format(optimal_weights_sharpe[0]*100))
print("  KO:   {:.2f}%".format(optimal_weights_sharpe[1]*100))
print("  MSFT: {:.2f}%".format(optimal_weights_sharpe[2]*100))

# Calculate the return, volatility, and Sharpe ratio of the optimal portfolio
ret_sharpe, vol_sharpe, shp_sharpe = get_portfolio_features(optimal_weights_sharpe, ann_rets, cov_rets, risk_free_rate, periods_per_year)

# Plotting the Efficient Frontier with Maximum Sharpe Ratio Portfolio
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df.plot.line(x="volatility", y="return", style="--", color="coral", ax=ax, grid=True, label="Efficient frontier")
ax.scatter(vol_sharpe, ret_sharpe, marker="X", color='r', s=120, label="Max Sharpe Ratio Portfolio")
ax.set_xlim([0.13, 0.33])
ax.legend()
ax.set_title("Maximum Sharpe Ratio Portfolio on Efficient Frontier")
plt.show()

# Optimizing Portfolios: Maximizing Sharpe Ratio at a Specified Volatility
target_volatility = 0.2

# Calculate optimal weights to maximize Sharpe ratio for the given target volatility
optimal_weights_sharpe_vol = pok.maximize_shape_ratio(ann_rets, cov_rets, risk_free_rate, periods_per_year, target_volatility)
print("Optimal weights for Maximum Sharpe Ratio Portfolio at Specified Volatility:")
print("  AMZN: {:.2f}%".format(optimal_weights_sharpe_vol[0]*100))
print("  KO:   {:.2f}%".format(optimal_weights_sharpe_vol[1]*100))
print("  MSFT: {:.2f}%".format(optimal_weights_sharpe_vol[2]*100))

# Retrieve and display the portfolio's return, volatility, and Sharpe ratio
ret_sharpe_vol, vol_sharpe_vol, shp_sharpe_vol = get_portfolio_features(optimal_weights_sharpe_vol, ann_rets, cov_rets, risk_free_rate, periods_per_year)

# Visualize the efficient frontier and indicate the portfolio with the highest Sharpe ratio at the specified volatility
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df.plot.line(x="volatility", y="return", style="--", color="coral", ax=ax, grid=True, label="Efficient frontier")
ax.scatter(vol_sharpe_vol, ret_sharpe_vol, marker="X", color='r', s=120, label="Highest Sharpe Ratio Portfolio")
ax.set_xlim([0.13, 0.33])
ax.legend()
ax.set_title(f"Maximum Sharpe Ratio Portfolio at {target_volatility*100}% Volatility")
plt.show()