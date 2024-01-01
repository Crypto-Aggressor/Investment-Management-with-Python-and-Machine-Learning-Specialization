import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For styling graphs
import scipy.stats  # For statistical functions
import yfinance as yf  # For fetching financial data using yfinance instead of pandas_datareader
from datetime import datetime  # For handling date and time objects
from scipy.optimize import minimize  # For optimization functions
import PortfolioOptimizationKit as pok  # Custom toolkit for portfolio optimization
from tabulate import tabulate  # For pretty-printing dataframes

sns.set(style="darkgrid")  # Setting the plot style using seaborn

# Define the number of industries
nind = 30

# Retrieving industry returns, the number of firms per industry, and average industry sizes
ind_rets = pok.get_ind_file(filetype="rets", nind=nind)
ind_nfirms = pok.get_ind_file(filetype="nfirms", nind=nind)
ind_size = pok.get_ind_file(filetype="size", nind=nind)

# Calculate the market capitalization for each sector
ind_mkt_cap = ind_nfirms * ind_size
print(tabulate(ind_mkt_cap.head(3), headers='keys', tablefmt='github'))

# Summing across sectors to get the total market capitalization at each time point
total_mkt_cap = ind_mkt_cap.sum(axis=1)
print(tabulate(pd.DataFrame(total_mkt_cap.head()), headers='keys', tablefmt='github'))

# Calculating the weight of each industry's market cap in the total market capitalization
ind_cap_weights = ind_mkt_cap.divide(total_mkt_cap, axis=0)
print(tabulate(ind_cap_weights.head(3), headers='keys', tablefmt='github'))

# Creating plots to visualize the total market capitalization and the weights of selected sectors over time
fig, ax = plt.subplots(1, 2, figsize=(18, 4))
total_mkt_cap.plot(grid=True, ax=ax[0])
ax[0].set_title("Total market cap 1929-2018")

ind_cap_weights[["Steel", "Fin", "Telcm"]].plot(grid=True, ax=ax[1])
ax[1].set_title("Steel, Finance, and Telecommunication Market caps (%) 1929-2018")

# Calculating the total market return by weighting each sector's monthly returns
total_market_return = (ind_cap_weights * ind_rets).sum(axis=1)

# Assuming an initial investment and calculating the total market index
capital = 1000
total_market_index = capital * (1 + total_market_return).cumprod()

# Visualize the total market index and returns over time
fig, ax = plt.subplots(1, 2, figsize=(18, 4))
total_market_index.plot(grid=True, ax=ax[0])
ax[0].set_title("Total market cap-weighted index 1929-2018")

total_market_return.plot(grid=True, ax=ax[1])
ax[1].set_title("Total market cap-weighted returns 1929-2018")

plt.show()

# Analyzing Rolling Returns

# Plotting the total market cap-weighted index from 1990 onwards
total_market_index["1990":].plot(grid=True, figsize=(11,6), label="Total market cap-weighted index")

# Plotting moving averages over different window sizes to observe trends over time
total_market_index["1990":].rolling(window=60).mean().plot(grid=True, figsize=(11,6), label="60 months MA")  # 5 years MA
total_market_index["1990":].rolling(window=36).mean().plot(grid=True, figsize=(11,6), label="36 months MA")  # 3 years MA
total_market_index["1990":].rolling(window=12).mean().plot(grid=True, figsize=(11,6), label="12 months MA")  # 1 year MA

plt.legend()  # Adding legend to the plot
plt.show()  # Displaying the plot

# Calculating trailing 36 months compound returns of total market return
tmi_trail_36_rets = total_market_return.rolling(window=36).aggregate(pok.annualize_rets, periods_per_year=12)

# Plotting the original total market returns alongside the trailing 36 months compound returns
total_market_return.plot(grid=True, figsize=(12,5), label="Total market (monthly) return")
tmi_trail_36_rets.plot(grid=True, figsize=(12,5), label="Trailing 36 months total market compound return")
plt.legend()  # Adding legend to the plot
plt.show()  # Displaying the plot

# Calculating Rolling Correlations: Multi-Indices and Groupby

# Computing rolling correlations across industries
rets_trail_36_corr = ind_rets.rolling(window=36).corr()
rets_trail_36_corr.index.names = ["date","industry"]  # Naming the indices for clarity
print(tabulate(rets_trail_36_corr.tail(), headers='keys', tablefmt='github'))  # Displaying the last few entries

# Calculating the average of all correlation matrices for each date
ind_trail_36_corr = rets_trail_36_corr.groupby(level="date").apply(lambda corrmat: corrmat.values.mean())

# Plotting the trailing 36 months total market compound return and correlations
fig, ax1 = plt.subplots(1,1,figsize=(14,6))

tmi_trail_36_rets.plot(ax=ax1, color="blue", grid=True, label="Trailing 36 months total market compound return")
ax2 = ax1.twinx()  # Creating a second y-axis
ind_trail_36_corr.plot(ax=ax2, color="orange", grid=True, label="Trailing 36 months total market return correlations")

ax1.set_ylabel('trail 36mo returns')  # Labeling the y-axis
ax2.set_ylabel('trail 36mo corrs',rotation=-90)  # Labeling the second y-axis
ax1.legend(loc=2)  # Positioning the first legend
ax2.legend(loc=1)  # Positioning the second legend
plt.show()  # Displaying the plot

# Calculating the correlation between compounded returns and average correlations across industries
print(tabulate([[tmi_trail_36_rets.corr(ind_trail_36_corr)]], headers=['Correlation'], tablefmt='github'))

# Fetching industry and total market index returns
ind_return = pok.get_ind_file(filetype="rets", nind=nind)
tmi_return = pok.get_total_market_index_returns(nind=nind)

# Selecting industry returns from 2000 for three industries
risky_rets = ind_return["2000":][["Steel", "Fin", "Beer"]]

# Creating a DataFrame for safe asset returns with a fixed 3% annual return
safe_rets = pd.DataFrame().reindex_like(risky_rets)
safe_rets[:] = 0.03 / 12

# Setting initial values and parameters for the CPPI strategy
start_value = 1000  # Initial investment
account_value = start_value  # Current account value
floor = 0.8  # Floor as a percentage of the initial account value
floor_value = floor * account_value  # Absolute floor value
m = 3  # Multiplier, selected based on acceptable drop before breaching the floor

# Initializing DataFrames to track account history, cushion, and risky asset weights
account_history = pd.DataFrame().reindex_like(risky_rets)
cushion_history = pd.DataFrame().reindex_like(risky_rets)
risky_w_history = pd.DataFrame().reindex_like(risky_rets)

# Calculating the wealth growth for a 100% investment in risky assets
risky_wealth = start_value * (1 + risky_rets).cumprod()

# Implementing the CPPI strategy over time
for step in range(len(risky_rets.index)):
    cushion = (account_value - floor_value) / account_value  # Calculating the cushion as a percentage of current account value
    risky_w = m * cushion  # Determining allocation to risky asset
    risky_w = np.minimum(risky_w, 1)  # Ensuring weight is within [0,1]
    risky_w = np.maximum(risky_w, 0)
    safe_w = 1 - risky_w  # Determining allocation to safe asset
    risky_allocation = risky_w * account_value  # Value allocation to risky asset
    safe_allocation = safe_w * account_value  # Value allocation to safe asset
    account_value = risky_allocation * (1 + risky_rets.iloc[step]) + safe_allocation * (1 + safe_rets.iloc[step])  # Updating account value
    account_history.iloc[step] = account_value  # Recording data
    cushion_history.iloc[step] = cushion
    risky_w_history.iloc[step] = risky_w

# Calculating the returns from the CPPI strategy
cppi_rets = (account_history / account_history.shift(1) - 1).dropna()

# Plotting the account history to visualize the wealth evolution over time
ax = account_history.plot(figsize=(10,5), grid=True)
ax.set_ylabel("wealth ($)")
plt.show()

# Setting up the plot for comparative analysis between CPPI strategies and full investment in risky assets
fig, ax = plt.subplots(3, 2, figsize=(18, 15))
ax = ax.flatten()

# Comparative analysis for 'Beer'
account_history["Beer"].plot(ax=ax[0], grid=True, label="CPPI Beer")
risky_wealth["Beer"].plot(ax=ax[0], grid=True, label="Beer", style="k:")
ax[0].axhline(y=floor_value, color="r", linestyle="--", label="Fixed floor value")
ax[0].legend(fontsize=11)

# Plotting the allocation weights for 'Beer'
risky_w_history["Beer"].plot(ax=ax[1], grid=True, label="Risky weight in Beer")
ax[1].legend(fontsize=11)

# Comparative analysis for 'Fin'
account_history["Fin"].plot(ax=ax[2], grid=True, label="CPPI Fin")
risky_wealth["Fin"].plot(ax=ax[2], grid=True, label="Fin", style="k:")
ax[2].axhline(y=floor_value, color="r", linestyle="--", label="Fixed floor value")
ax[2].legend(fontsize=11)

# Plotting the allocation weights for 'Fin'
risky_w_history["Fin"].plot(ax=ax[3], grid=True, label="Risky weight in Fin")
ax[3].legend(fontsize=11)

# Comparative analysis for 'Steel'
account_history["Steel"].plot(ax=ax[4], grid=True, label="CPPI Steel")
risky_wealth["Steel"].plot(ax=ax[4], grid=True, label="Steel", style="k:")
ax[4].axhline(y=floor_value, color="r", linestyle="--", label="Fixed floor value")
ax[4].legend(fontsize=11)

# Plotting the allocation weights for 'Steel'
risky_w_history["Steel"].plot(ax=ax[5], grid=True, label="Risky weight in Steel")
ax[5].legend(fontsize=11)

plt.show()

# Displaying summary statistics for the pure risky asset investment and the CPPI strategy
print(tabulate(pok.summary_stats(risky_rets), headers='keys', tablefmt='github'))
print(tabulate(pok.summary_stats(cppi_rets), headers='keys', tablefmt='github'))

# Running the CPPI with a dynamic drawdown constraint using the toolkit's method
res = pok.cppi(risky_rets, start_value=1000, floor=0.8, drawdown=0.2, risk_free_rate=0.03, periods_per_year=12)

# Selecting a specific sector for analysis
sector = "Fin"

# Plotting the wealth progression of the CPPI strategy alongside the pure investment in the sector and the floor value
fig, ax = plt.subplots(1,2,figsize=(18,4))
ax = ax.flatten()

# Plotting the CPPI wealth and the risky wealth for the selected sector
res["CPPI wealth"][sector].plot(ax=ax[0], grid=True, label="CPPI "+sector)
res["Risky wealth"][sector].plot(ax=ax[0], grid=True, label=sector, style="k:")
res["Floor value"][sector].plot(ax=ax[0], grid=True, color="r", linestyle="--", label="Dynamic floor value")
ax[0].legend(fontsize=11)

# Plotting the allocation to the risky asset over time
res["Risky allocation"][sector].plot(ax=ax[1], grid=True, label="Risky weight in "+sector)
ax[1].legend(fontsize=11)

plt.show()

# Displaying the summary statistics for the sector's pure risky returns and the CPPI returns
print("Statistics for sector's pure risky returns:")
print(tabulate(pok.summary_stats(risky_rets[sector]), headers='keys', tablefmt='github'))

print("\nStatistics for CPPI returns:")
print(tabulate(pok.summary_stats(res["CPPI returns"][sector]), headers='keys', tablefmt='github'))

# Comparing CPPI strategies with different drawdown constraints
sector = "Fin"
drawdowns = [0.2, 0.4, 0.6]
    
fig, ax = plt.subplots(1,2,figsize=(18,4))
ax = ax.flatten()

# Plotting the risky wealth for comparison
res["Risky wealth"][sector].plot(ax=ax[0], grid=True, style="k:", label=sector)
ax[0].legend()

# Initializing a DataFrame to store summary statistics
summ = pd.DataFrame()

# Analyzing CPPI strategies with various drawdown constraints
for drawdown in drawdowns:
    # Running CPPI with the specified drawdown constraint
    res = pok.cppi(risky_rets, start_value=1000, floor=0.8, drawdown=drawdown, risk_free_rate=0.03, periods_per_year=12)    
    
    # Plotting the CPPI wealth and risky allocation for each drawdown scenario
    res["CPPI wealth"][sector].plot(ax=ax[0], grid=True, label="CPPI dd={}%, m={}".format(drawdown*100,round(res["m"],1)) )
    res["Risky allocation"][sector].plot(ax=ax[1], grid=True, label="dd={}%, m={}".format(drawdown*100,round(res["m"],1)) )
    
    # Appending summary statistics for each scenario
    summ = pd.concat([summ, pok.summary_stats(res["CPPI returns"][sector])], axis=0)
        
# Displaying the plots and summary statistics
ax[0].legend() 
ax[1].legend(fontsize=11)
ax[1].set_title("Risky weight", fontsize=11)
plt.show()

# Setting the index for the summary statistics DataFrame
summ.index = [["DD20%","DD40%","DD60%"]]
print("Summary statistics for various drawdown scenarios:")
print(tabulate(summ, headers='keys', tablefmt='github'))

# Simulating stock prices using Geometric Brownian Motion
prices_1, rets_1 = pok.simulate_gbm_from_returns(n_years=10, n_scenarios=10, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0)
prices_2, rets_2 = pok.simulate_gbm_from_prices(n_years=10, n_scenarios=10, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0)

# Plotting the generated prices
fig, ax = plt.subplots(1, 2, figsize=(20,5))
prices_1.plot(ax=ax[0], grid=True, title="Prices generated by compounding returns which follow a GBM")
prices_2.plot(ax=ax[1], grid=True, title="Prices generated by solving the GBM equation satisfied by log-returns")
plt.show()
