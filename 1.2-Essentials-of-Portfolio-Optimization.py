# import pandas as pd  # For data manipulation
# import numpy as np  # For numerical operations
# import matplotlib.pyplot as plt  # For plotting graphs
# import scipy.stats  # For statistical functions
# import yfinance as yf  # For fetching financial data using yfinance instead of pandas_datareader
# from datetime import datetime  # For handling date and time objects
# from scipy.optimize import minimize  # For optimization functions
# import PortfolioOptimizationKit as pok  # Custom toolkit for portfolio optimization

# plt.style.use("seaborn")  # Setting the plot style

# # Define a function to calculate portfolio features such as return, volatility, and Sharpe ratio
# def get_portfolio_features(weights, rets, covmat, risk_free_rate, periods_per_year):
#     """
#     Calculate portfolio return, volatility, and Sharpe ratio.

#     Parameters:
#     - weights: Array of asset weights in the portfolio.
#     - rets: Annualized returns for each asset.
#     - covmat: Covariance matrix of asset returns.
#     - risk_free_rate: Risk-free rate for Sharpe ratio calculation.
#     - periods_per_year: Number of periods in a year (trading days).

#     Returns:
#     Tuple of (return, volatility, sharpe ratio) for the portfolio.
#     """
#     vol = pok.portfolio_volatility(weights, covmat)
#     vol = pok.annualize_vol(vol, periods_per_year)
#     ret = pok.portfolio_return(weights, rets)
#     shp = pok.sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
#     return ret, vol, shp

# # Modern Portfolio Theory (MPT) is a mathematical framework for constructing a portfolio of assets to maximize expected return for a given level of risk.
# # It's based on the principle of diversification, suggesting that a mixed variety of investments yields less risk than any single investment.

# # Efficient Frontiers in MPT: a graph showing the best possible return for a given level of risk.

# # Example with two assets to demonstrate portfolio volatility calculation.
# nret = 500  # Number of returns
# periods_per_year = 252  # Trading days in a year
# risk_free_rate = 0.0  # Risk-free rate for Sharpe ratio calculation

# mean_1 = 0.001019  # Mean return for asset 1
# mean_2 = 0.001249  # Mean return for asset 2
# vol_1  = 0.016317  # Volatility for asset 1
# vol_2  = 0.019129  # Volatility for asset 2

# rhos  = np.linspace(1, -1, num=6)  # Correlation range
# ncorr = len(rhos)

# nweig = 20
# w1 = np.linspace(0, 1, num=nweig)
# w2 = 1 - w1
# ww = pd.DataFrame([w1, w2]).T

# np.random.seed(1)

# fig, ax = plt.subplots(1, 6, figsize=(20, 4))    
# ax = ax.flatten()

# for k_rho, rho in enumerate(rhos):
#     portfolio = pd.DataFrame(columns=["return", "volatility", "sharpe ratio"])

#     cov_ij = rho * vol_1 * vol_2
#     cov_rets = pd.DataFrame([[vol_1**2, cov_ij], [cov_ij, vol_2**2]])
#     daily_rets = pd.DataFrame(np.random.multivariate_normal((mean_1, mean_2), cov_rets.values, nret))
    
#     for i in range(ww.shape[0]):
#         weights = ww.loc[i]

#         ann_rets = pok.annualize_rets(daily_rets, periods_per_year)
#         portfolio_ret = pok.portfolio_return(weights, ann_rets)
#         portfolio_vol = pok.portfolio_volatility(weights, cov_rets)
#         portfolio_vol = pok.annualize_vol(portfolio_vol, periods_per_year)
#         portfolio_spr = pok.sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)

#         new_row = pd.DataFrame([{"return": portfolio_ret, "volatility": portfolio_vol, "sharpe ratio": portfolio_spr}])
#         portfolio = pd.concat([portfolio, new_row], ignore_index=True)

#     im = ax[k_rho].scatter(portfolio["volatility"]*100, portfolio["return"]*100, c=w2, cmap='RdYlBu') 
#     ax[k_rho].grid()
#     ax[k_rho].set_title(f"Correlation: {np.round(rho,2)}", y=0.9, loc='left')
#     ax[k_rho].set_xlabel("Volatility (%)")
#     if k_rho == 0: ax[k_rho].set_ylabel("Return (%)") 
#     ax[k_rho].set_xlim([0, 32])
#     ax[k_rho].set_ylim([0, 95])

# fig.colorbar(im, ax=ax.ravel().tolist())
# plt.show()

# # Real-World Example: Analyzing U.S. Stocks for Portfolio Optimization
# tickers  = ['AMZN','KO','MSFT']
# n_assets = len(tickers)

# stocks = pd.DataFrame()

# start_date = "2011-01-01"
# end_date = "2023-01-01"

# for stock_name in tickers:
#     ticker_data = yf.Ticker(stock_name)  # Using yfinance to fetch data
#     hist_data = ticker_data.history(start=start_date, end=end_date)
#     stocks[stock_name] = hist_data['Close']

# stocks = round(stocks,2)
# daily_rets = pok.compute_returns(stocks)
# ann_rets = pok.annualize_rets(daily_rets, 252)

# mean_rets = daily_rets.mean()
# std_rets  = daily_rets.std()
# cov_rets  = daily_rets.cov()

# num_portfolios   = 4000
# portfolios       = pd.DataFrame(columns=["return","volatility","sharpe ratio","w1","w2","w3"])

# all_portfolios = []

# for i in range(num_portfolios):
#     weights = np.random.random(n_assets)
#     weights /= np.sum(weights)
    
#     portfolio_ret = pok.portfolio_return(weights, ann_rets)        
#     portfolio_vol = pok.portfolio_volatility(weights, cov_rets)
#     portfolio_vol = pok.annualize_vol(portfolio_vol, periods_per_year)
#     portfolio_spr = pok.sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)
    
#     all_portfolios.append({"return":portfolio_ret, 
#                            "volatility":portfolio_vol, 
#                            "sharpe ratio":portfolio_spr, 
#                            "w1": weights[0], 
#                            "w2": weights[1], 
#                            "w3": weights[2]})

# portfolios = pd.DataFrame(all_portfolios)

# fig, ax = plt.subplots(1,1, figsize=(10,6)) 

# im = ax.scatter(portfolios["volatility"], portfolios["return"], c=portfolios["sharpe ratio"], s=20, 
#                 edgecolor=None, cmap='RdYlBu')
# ax.set_title("Portfolios and Efficient Frontier")
# ax.set_ylabel("return (%)")
# ax.grid()

# df = pok.efficient_frontier(50, daily_rets, cov_rets, periods_per_year)
# df.plot.line(x="volatility", y="return", style="--", color="tab:green", ax=ax, grid=True, 
#              label="Efficient frontier")
# ax.set_xlim([0.125,0.33])
# ax.set_xlabel("volatility (%)")

# fig.colorbar(im, ax=ax)
# plt.show()

# # Maximizing the Sharpe Ratio Portfolio with a Non-Zero Risk-Free Asset
# risk_free_rate = 0.06
# optimal_weights = pok.maximize_shape_ratio(ann_rets, cov_rets, risk_free_rate, periods_per_year)
# print("Optimal weights for the maximum Sharpe Ratio portfolio:")
# print("  AMZN: {:.2f}%".format(optimal_weights[0]*100))
# print("  KO:   {:.2f}%".format(optimal_weights[1]*100))
# print("  MSFT: {:.2f}%".format(optimal_weights[2]*100))

# ret, vol, shp = get_portfolio_features(optimal_weights, ann_rets, cov_rets, risk_free_rate, periods_per_year)

# # Plotting the efficient frontier and the Capital Market Line
# df, ax = pok.efficient_frontier(40, daily_rets, cov_rets, periods_per_year, risk_free_rate=risk_free_rate, 
#                                 iplot=True, cml=True)
# ax.set_title("Maximum Sharpe Ratio Portfolio (SR={}) with Risk-Free Rate {}%".format(np.round(shp, 2), risk_free_rate*100))
# plt.show()

# # Adjusting the risk-free rate
# risk_free_rate = 0.05

# df, ax = pok.efficient_frontier(90, daily_rets, cov_rets, periods_per_year, risk_free_rate=risk_free_rate, 
#                                 iplot=True, hsr=True, cml=True, mvp=True, ewp=True)
# ax.set_title("Efficient Frontier with Maximum Sharpe Ratio Portfolio (SR={}) at Risk-Free Rate {}%".format(np.round(shp, 2), risk_free_rate*100))
# plt.show()

# print(df.tail())

import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For styling graphs
import scipy.stats  # For statistical functions
import yfinance as yf  # For fetching financial data using yfinance instead of pandas_datareader
from datetime import datetime  # For handling date and time objects
from scipy.optimize import minimize  # For optimization functions
import PortfolioOptimizationKit as pok  # Custom toolkit for portfolio optimization

sns.set(style="darkgrid")  # Setting the plot style using seaborn

# Define a function to calculate portfolio features such as return, volatility, and Sharpe ratio
def get_portfolio_features(weights, rets, covmat, risk_free_rate, periods_per_year):
    """
    Calculate portfolio return, volatility, and Sharpe ratio.

    Parameters:
    - weights: Array of asset weights in the portfolio.
    - rets: Annualized returns for each asset.
    - covmat: Covariance matrix of asset returns.
    - risk_free_rate: Risk-free rate for Sharpe ratio calculation.
    - periods_per_year: Number of periods in a year (trading days).

    Returns:
    Tuple of (return, volatility, sharpe ratio) for the portfolio.
    """
    vol = pok.portfolio_volatility(weights, covmat)
    vol = pok.annualize_vol(vol, periods_per_year)
    ret = pok.portfolio_return(weights, rets)
    shp = pok.sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
    return ret, vol, shp

# Modern Portfolio Theory (MPT) is a mathematical framework for constructing a portfolio of assets to maximize expected return for a given level of risk.
# It's based on the principle of diversification, suggesting that a mixed variety of investments yields less risk than any single investment.

# Efficient Frontiers in MPT: a graph showing the best possible return for a given level of risk.

# Example with two assets to demonstrate portfolio volatility calculation.
nret = 500  # Number of returns
periods_per_year = 252  # Trading days in a year
risk_free_rate = 0.0  # Risk-free rate for Sharpe ratio calculation

mean_1 = 0.001019  # Mean return for asset 1
mean_2 = 0.001249  # Mean return for asset 2
vol_1  = 0.016317  # Volatility for asset 1
vol_2  = 0.019129  # Volatility for asset 2

rhos  = np.linspace(1, -1, num=6)  # Correlation range
ncorr = len(rhos)

nweig = 20
w1 = np.linspace(0, 1, num=nweig)
w2 = 1 - w1
ww = pd.DataFrame([w1, w2]).T

np.random.seed(1)

fig, ax = plt.subplots(1, 6, figsize=(20, 4))    
ax = ax.flatten()

for k_rho, rho in enumerate(rhos):
    portfolio = pd.DataFrame(columns=["return", "volatility", "sharpe ratio"])

    cov_ij = rho * vol_1 * vol_2
    cov_rets = pd.DataFrame([[vol_1**2, cov_ij], [cov_ij, vol_2**2]])
    daily_rets = pd.DataFrame(np.random.multivariate_normal((mean_1, mean_2), cov_rets.values, nret))
    
    for i in range(ww.shape[0]):
        weights = ww.loc[i]

        ann_rets = pok.annualize_rets(daily_rets, periods_per_year)
        portfolio_ret = pok.portfolio_return(weights, ann_rets)
        portfolio_vol = pok.portfolio_volatility(weights, cov_rets)
        portfolio_vol = pok.annualize_vol(portfolio_vol, periods_per_year)
        portfolio_spr = pok.sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)

        new_row = pd.DataFrame([{"return": portfolio_ret, "volatility": portfolio_vol, "sharpe ratio": portfolio_spr}])
        portfolio = pd.concat([portfolio, new_row], ignore_index=True)

    im = ax[k_rho].scatter(portfolio["volatility"]*100, portfolio["return"]*100, c=w2, cmap='RdYlBu') 
    ax[k_rho].grid()
    ax[k_rho].set_title(f"Correlation: {np.round(rho,2)}", y=0.9, loc='left')
    ax[k_rho].set_xlabel("Volatility (%)")
    if k_rho == 0: ax[k_rho].set_ylabel("Return (%)") 
    ax[k_rho].set_xlim([0, 32])
    ax[k_rho].set_ylim([0, 95])

fig.colorbar(im, ax=ax.ravel().tolist())
plt.show()

# Real-World Example: Analyzing U.S. Stocks for Portfolio Optimization
tickers  = ['AMZN','KO','MSFT']
n_assets = len(tickers)

stocks = pd.DataFrame()

start_date = "2011-01-01"
end_date = "2023-01-01"

for stock_name in tickers:
    ticker_data = yf.Ticker(stock_name)  # Using yfinance to fetch data
    hist_data = ticker_data.history(start=start_date, end=end_date)
    stocks[stock_name] = hist_data['Close']

stocks = round(stocks,2)
daily_rets = pok.compute_returns(stocks)
ann_rets = pok.annualize_rets(daily_rets, 252)

mean_rets = daily_rets.mean()
std_rets  = daily_rets.std()
cov_rets  = daily_rets.cov()

num_portfolios   = 4000
portfolios       = pd.DataFrame(columns=["return","volatility","sharpe ratio","w1","w2","w3"])

all_portfolios = []

for i in range(num_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)
    
    portfolio_ret = pok.portfolio_return(weights, ann_rets)        
    portfolio_vol = pok.portfolio_volatility(weights, cov_rets)
    portfolio_vol = pok.annualize_vol(portfolio_vol, periods_per_year)
    portfolio_spr = pok.sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)
    
    all_portfolios.append({"return":portfolio_ret, 
                           "volatility":portfolio_vol, 
                           "sharpe ratio":portfolio_spr, 
                           "w1": weights[0], 
                           "w2": weights[1], 
                           "w3": weights[2]})

portfolios = pd.DataFrame(all_portfolios)

fig, ax = plt.subplots(1,1, figsize=(10,6)) 

im = ax.scatter(portfolios["volatility"], portfolios["return"], c=portfolios["sharpe ratio"], s=20, 
                edgecolor=None, cmap='RdYlBu')
ax.set_title("Portfolios and Efficient Frontier")
ax.set_ylabel("return (%)")
ax.grid()

df = pok.efficient_frontier(50, daily_rets, cov_rets, periods_per_year)
df.plot.line(x="volatility", y="return", style="--", color="tab:green", ax=ax, grid=True, 
             label="Efficient frontier")
ax.set_xlim([0.125,0.33])
ax.set_xlabel("volatility (%)")

fig.colorbar(im, ax=ax)
plt.show()

# Maximizing the Sharpe Ratio Portfolio with a Non-Zero Risk-Free Asset
risk_free_rate = 0.06
optimal_weights = pok.maximize_shape_ratio(ann_rets, cov_rets, risk_free_rate, periods_per_year)
print("Optimal weights for the maximum Sharpe Ratio portfolio:")
print("  AMZN: {:.2f}%".format(optimal_weights[0]*100))
print("  KO:   {:.2f}%".format(optimal_weights[1]*100))
print("  MSFT: {:.2f}%".format(optimal_weights[2]*100))

ret, vol, shp = get_portfolio_features(optimal_weights, ann_rets, cov_rets, risk_free_rate, periods_per_year)

# Plotting the efficient frontier and the Capital Market Line
df, ax = pok.efficient_frontier(40, daily_rets, cov_rets, periods_per_year, risk_free_rate=risk_free_rate, 
                                iplot=True, cml=True)
ax.set_title("Maximum Sharpe Ratio Portfolio (SR={}) with Risk-Free Rate {}%".format(np.round(shp, 2), risk_free_rate*100))
plt.show()

# Adjusting the risk-free rate
risk_free_rate = 0.05

df, ax = pok.efficient_frontier(90, daily_rets, cov_rets, periods_per_year, risk_free_rate=risk_free_rate, 
                                iplot=True, hsr=True, cml=True, mvp=True, ewp=True)
ax.set_title("Efficient Frontier with Maximum Sharpe Ratio Portfolio (SR={}) at Risk-Free Rate {}%".format(np.round(shp, 2), risk_free_rate*100))
plt.show()

print(df.tail())

