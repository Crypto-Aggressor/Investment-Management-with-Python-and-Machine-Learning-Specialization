##########################################################################################
##########################################################################################
print("# Fundamentals of Asset-Liability Management")
##########################################################################################
##########################################################################################

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


##########################################################################################
##########################################################################################
print("## Understanding the Time Value of Money")
##########################################################################################
print("### Grasping the Cumulative Present Value of Future Cash Flows")
##########################################################################################
##########################################################################################

# Calculating present value of 1 dollar due in 10 years with an annual interest rate of 3%
PV = pok.discount(10, 0.03)  # Assumes pok.discount returns a single numeric value.
print(tabulate(PV, headers='keys', tablefmt='github'))  # Print present value.

# Validating the calculation by computing the future value
FV = PV * (1 + 0.03) ** 10 # Calculate and convert future value to a DataFrame.
print(tabulate(FV, headers='keys', tablefmt='github'))  # Print future value.

# Considering two sets of liabilities (future cash flows) over the next three years
L = pd.DataFrame([[8000, 11000], [11000, 2000], [6000, 15000]], index=[1, 2, 3])
print(tabulate(L, headers='keys', tablefmt='github'))  # Print liabilities.

print(L.sum())

# Assuming the first liability has an annual rate of 5% and the second one 3%
r = [0.05, 0.03]
PV = pok.present_value(L, r)  # Calculate present value for each set of liabilities.
print(PV) # Print present values. 


##########################################################################################
##########################################################################################
print("### Understanding the Funding Ratio")
##########################################################################################
##########################################################################################

# Calculating the funding ratio for given asset values
asset = [20000, 27332]  # Define asset values.
FR = pok.funding_ratio(asset, L, r)  # Calculate funding ratio.
FR_df = pd.DataFrame({'Funding Ratio': FR})  # Convert funding ratio to DataFrame.
print(tabulate(FR_df, headers='keys', tablefmt='github'))  # Print funding ratio.

def show_funding_ratio(asset, L, r):
    fr = pok.funding_ratio(asset, L, r)  # Calculate funding ratio.
    print("Funding ratio: {:.3f}".format(float(fr)))  # Print funding ratio.

    # Set up a two-panel plot for visual representation.
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))

    # Ensure r and fr are of the same length for plotting.
    r_array = np.full_like(fr, r)  # Create an array of r repeated to match fr length.
    
    # Plot the funding ratio against interest rates.
    ax[0].scatter(r_array, fr)
    ax[0].set_xlabel("rates")
    ax[0].set_ylabel("funding ratio")
    ax[0].set_xlim([0.0, max(r_array)*1.1])  # Adjust the x-axis limit based on r values.
    ax[0].set_ylim([min(fr)*0.9, max(fr)*1.1])  # Adjust the y-axis limit based on fr values.
    ax[0].plot(r_array, fr, color="b", alpha=0.5)
    # Draw red dashed lines from the point to the axes.
    ax[0].vlines(x=r, ymin=0, ymax=fr, colors='r', linestyles='dashed')
    ax[0].hlines(y=fr, xmin=0, xmax=r, colors='r', linestyles='dashed')
    ax[0].grid()

    # Handle the case where asset is not iterable.
    if not hasattr(asset, '__iter__'):  # If asset is not an iterable
        asset = [asset]  # Convert it into a list

    # Plot the funding ratio against asset values.
    ax[1].scatter(asset, fr)
    ax[1].set_xlabel("assets")
    ax[1].set_ylabel("funding ratio")
    ax[1].set_xlim([min(asset)*0.9, max(asset)*1.1])  # Adjust the x-axis limit based on asset values.
    ax[1].set_ylim([min(fr)*0.9, max(fr)*1.1])  # Adjust the y-axis limit based on fr values.
    ax[1].plot(asset, fr, color="b", alpha=0.5)
    # Draw red dashed lines from the point to the axes.
    for a in asset:
        ax[1].vlines(x=a, ymin=0, ymax=fr, colors='r', linestyles='dashed')
    ax[1].hlines(y=fr, xmin=0, xmax=max(asset), colors='r', linestyles='dashed')
    ax[1].grid()

    plt.show()  # Display the plots.

# Demonstration of the funding ratio function
r = 0.02  # Define an interest rate.
asset = 24000  # Define an asset value.
L = pd.DataFrame([8000, 11000, 6000], index=[1, 2, 3])  # Define liabilities.
show_funding_ratio(asset, L, r)  # Call the function to display the funding ratio.


##########################################################################################
##########################################################################################
print("### Nominal Rate and Effective Annual Interest Rate")
##########################################################################################
print("#### Short-rate vs. Long-Rate (Annualized)")
##########################################################################################
##########################################################################################

# Define the nominal interest rate as 10% and the number of compounding periods as monthly (12 times a year)
nominal_rate = 0.1
periods_per_year = 12

# Calculate the rate for each period (month) and store it in a DataFrame for the first 10 periods
rets = pd.DataFrame([nominal_rate / periods_per_year for i in range(10)])
# Display the first three monthly rates to verify the calculations
print(rets.head(3))

# Calculate the annualized return
ann_ret = pok.annualize_rets(rets, periods_per_year)[0]
print("Annualized Return: ", ann_ret)

# Calculate the effective annual interest rate using the formula for discrete compounding
R = (1 + nominal_rate / periods_per_year) ** periods_per_year - 1
print("Effective Annual Interest Rate: ", R)


##########################################################################################
##########################################################################################
print("#### Continuous Compounding")
##########################################################################################
##########################################################################################

# Generate a range of N values from 1 to 12, totaling 30 points.
N = np.linspace(1,12,30)
# Define a range of nominal rates from 5% to 20%.
nom_rates = np.arange(0.05,0.2,0.01)

# Initialize a plot with specified size.
fig, ax = plt.subplots(1,1,figsize=(10,6))

# Iterate over each nominal rate.
for r in nom_rates:
    # Plot discrete compounding for each rate and N.
    ax.plot(N, (1 + r / N)**N - 1)
    # Plot the line for continuously compounded return for each rate.
    ax.axhline(y=np.exp(r) -  1, color="r", linestyle="-.", linewidth=0.7)
    # Set the y-label as the formula for discrete compounding.
    ax.set_ylabel("Discrete compounding: (1+r/N)^N - 1")
    # Set the x-label as 'payments (N)'.
    ax.set_xlabel("payments (N)")
# Enable grid for better readability.
plt.grid()
plt.show()



##########################################################################################
##########################################################################################
print("## CIR Model: Simulating Interest Rate Fluctuations")
##########################################################################################
print("### Applying the CIR Model to Price Zero-Coupon Bonds")
##########################################################################################
##########################################################################################

# Set the nominal rate.
r = 0.1

# Calculate discrete compounding rate with 12 periods per year.
R_disc = pok.compounding_rate(r, periods_per_year=12)
print("Discrete Compounding Rate: ", R_disc)

# Calculate continuous compounding rate.
R_cont = pok.compounding_rate(r)
print("Continuous Compounding Rate: ", R_cont)

# Convert back the continuous compounding rate to the nominal rate.
print("Nominal Rate from Continuous Compounding: ", pok.compounding_rate_inv(R_cont))

# Initial asset amount in millions of dollars
asset_0  = 0.75
# Total liability in millions of dollars
tot_liab = 1



##########################################################################################
##########################################################################################
print("## Liability Hedging")
##########################################################################################
##########################################################################################

# Nominal rate of the liability
mean_rate = 0.03
# Time horizon for the liability in years
n_years   = 10
# Number of different interest rate scenarios to simulate
n_scenarios = 10
# Number of periods per year for compounding
periods_per_year = 12


##########################################################################################
##########################################################################################
print("### Simulating Interest Rates and Zero-Coupon Bond Prices")
##########################################################################################
##########################################################################################

# Simulate interest rates using the CIR model and calculate corresponding ZCB prices
rates, zcb_price = pok.simulate_cir(n_years=n_years, n_scenarios=n_scenarios, 
                                    a=0.05, b=mean_rate, sigma=0.08, periods_per_year=periods_per_year)
print(rates.head())

# Assign the simulated ZCB prices as the liabilities
L = zcb_price
print(L.head())


##########################################################################################
##########################################################################################
print("#### Hedging with Zero-Coupon Bonds")
##########################################################################################
##########################################################################################

# Calculate the price of a ZCB maturing in 10 years with a rate equal to the mean rate
zcb = pd.DataFrame(data=[tot_liab], index=[n_years])
zcb_price_0 = pok.present_value(zcb, mean_rate)
print(zcb_price_0)

# Calculate the number of bonds that can be bought with the initial assets
n_bonds = float(asset_0 / zcb_price_0)
print(n_bonds)

# Calculate the future asset value of the zero-coupon bond investment
asset_value_of_zcb = n_bonds * zcb_price
print(asset_value_of_zcb.head())


##########################################################################################
##########################################################################################
print("#### Hedging by Holding Cash")
##########################################################################################
##########################################################################################

# Calculate the future asset value when holding cash, accounting for compounding interest
asset_value_in_cash = asset_0 * (1 + rates/periods_per_year).cumprod()
print(asset_value_in_cash.head())


##########################################################################################
##########################################################################################
print("#### Comparing the Two Investment Strategies")
##########################################################################################
##########################################################################################

# Plotting the future values of assets when invested in cash and zero-coupon bonds
fig, ax = plt.subplots(1,2,figsize=(20,5))

asset_value_in_cash.plot(ax=ax[0], grid=True, legend=False, color="indianred", title="Future value of asset put in cash")
asset_value_of_zcb.plot(ax=ax[1], grid=True, legend=False, color="indianred", title="Future value of asset put in ZCB")
ax[0].axhline(y=1.0, linestyle=":", color="black")
ax[1].axhline(y=1.0, linestyle=":", color="black")
ax[0].set_ylabel("millions $")
ax[1].set_ylabel("millions $")
if periods_per_year == 12:
    ax[0].set_xlabel("months ({:.0f} years)".format((len(asset_value_in_cash.index)-1)/periods_per_year))
    ax[1].set_xlabel("months ({:.0f} years)".format((len(asset_value_in_cash.index)-1)/periods_per_year))

plt.show()


# Calculate the funding ratios for both cash and zero-coupon bond investments
fr_cash = asset_value_in_cash / L
fr_zcb  = asset_value_of_zcb  / L

# Plotting the funding ratios and their percentage changes for both investments
fig, ax = plt.subplots(2,2,figsize=(20,8))

fr_cash.plot(ax=ax[0,0], grid=True, legend=False, color="indianred", 
             title="Funding ratios of investment in cash ({} scenarios)".format(n_scenarios))
fr_zcb.plot(ax=ax[0,1], grid=True, legend=False, color="indianred", 
            title="Funding ratios of investment in ZCB ({} scenarios)".format(n_scenarios))

ax[0,0].axhline(y=1.0, linestyle=":", color="black")
ax[0,1].axhline(y=1.0, linestyle=":", color="black")

fr_cash.pct_change().plot(ax=ax[1,0], grid=True, legend=False, color="indianred",
                          title="Pct changes in funding ratios of investment in cash ({} scenarios)".format(n_scenarios))
fr_zcb.pct_change().plot(ax=ax[1,1], grid=True, legend=False, color="indianred", 
                         title="Pct changes in funding ratios of investment in ZCB ({} scenarios)".format(n_scenarios))
plt.show()


# Simulate a larger number of scenarios
n_scenarios = 5000
rates, zcb_price = pok.simulate_cir(n_years=n_years, n_scenarios=n_scenarios, a=0.05, 
                                    b=mean_rate, sigma=0.08, periods_per_year=periods_per_year)
# Assign the simulated ZCB prices as liabilities
L = zcb_price
# Recalculate the ZCB and cash investments
zcb = pd.DataFrame(data=[tot_liab], index=[n_years])
zcb_price_0 = pok.present_value(zcb, mean_rate)
n_bonds = float(asset_0 / zcb_price_0)
asset_value_of_zcb = n_bonds * zcb_price
asset_value_in_cash = asset_0 * (1 + rates/periods_per_year).cumprod()

# Calculate terminal funding ratios
terminal_fr_zcb  = asset_value_of_zcb.iloc[-1]  / L.iloc[-1]
terminal_fr_cash = asset_value_in_cash.iloc[-1] / L.iloc[-1]

# Plotting histograms of terminal funding ratios for cash and zero-coupon bond investments
ax = terminal_fr_cash.plot.hist(label="(Terminal) Funding Ratio of investment in cash", bins=50, figsize=(12,5), color="orange", legend=True)
terminal_fr_zcb.plot.hist(ax=ax, grid=True, label="(Terminal) Funding Ratio of investment in ZCB", bins=50, legend=True, color="blue", secondary_y=True)
ax.axvline(x=1.0, linestyle=":", color="k")
ax.set_xlabel("funding ratios")
plt.show()



##########################################################################################
##########################################################################################
print("## Coupon-Bearing Bonds")
##########################################################################################
print("### Cash Flow from a Bond")
##########################################################################################
##########################################################################################

# Bond parameters
principal        = 100 
maturity         = 3
ytm              = 0.05
coupon_rate      = 0.03 
coupons_per_year = 2

# Calculating bond cash flows
cf = pok.bond_cash_flows(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year)
print(cf)


##########################################################################################
##########################################################################################
print("### Bond Price Calculation")
##########################################################################################
##########################################################################################

# Calculating the bond price given its parameters and YTM
bond_price = pok.bond_price(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year, ytm=ytm)
print(bond_price)

# Calculating the total sum paid by the bond if held until maturity 
tot_bond_paym = cf.sum()[0]
print(tot_bond_paym)

# Calculating the gain from investing in the bond
gain = -bond_price + tot_bond_paym
print(gain)

# Calculating the annual rate corresponding to the YTM
r = (tot_bond_paym / bond_price )**(1/maturity) - 1
print(r)


##########################################################################################
##########################################################################################
print("### Yield to Maturity and Bond Price Relationship")
##########################################################################################
##########################################################################################

# Calculating bond prices under different scenarios to illustrate the relationship between YTM and bond price
# Bond selling at a discount: bond price is smaller than face value
pok.bond_price(principal=100, maturity=3, coupon_rate=0.03, coupons_per_year=2, ytm=0.05)

# Bond selling at a premium: bond price is larger than face value
pok.bond_price(principal=100, maturity=3, coupon_rate=0.03, coupons_per_year=2, ytm=0.02)

# Bond selling at par: bond price is equal to face value
pok.bond_price(principal=100, maturity=3, coupon_rate=0.03, coupons_per_year=2, ytm=0.03)

# Plotting the relationship between YTM and bond price
coupon_rate = 0.04
principal = 100
ytm = np.linspace(0.01, 0.10, 20)
bond_prices = [pok.bond_price(maturity=3, principal=principal, coupon_rate=coupon_rate, coupons_per_year=2, ytm=r) for r in ytm]

# Visualizing the bond price as a function of YTM
ax = pd.DataFrame(bond_prices, index=ytm).plot(grid=True, title="Relation between bond price and YTM", figsize=(9,4), legend=False)
ax.axvline(x=coupon_rate, linestyle=":", color="black")
ax.axhline(y=principal, linestyle=":", color="black")
ax.set_xlabel("YTM")
ax.set_ylabel("Bond price (Face value)")
plt.show()


##########################################################################################
##########################################################################################
print("### Variations in Bond Price")
##########################################################################################
print("#### Observing Price Changes with Interest Rate Fluctuations")
##########################################################################################
##########################################################################################

# Simulation parameters
n_years          = 10
n_scenarios      = 10
b                = 0.03  # Long-term mean interest rate
periods_per_year = 2

# Simulating interest rates using the CIR model
rates, _ = pok.simulate_cir(n_years=n_years, n_scenarios=n_scenarios, a=0.02, b=b, sigma=0.02, periods_per_year=periods_per_year)
print(rates.tail())

# Bond characteristics
principal        = 100
maturity         = n_years
coupon_rate      = 0.04
coupons_per_year = periods_per_year

# Calculate bond prices based on the simulated interest rates
bond_prices = pok.bond_price(principal=principal, maturity=maturity, coupon_rate=coupon_rate, 
                             coupons_per_year=coupons_per_year, ytm=rates)

# Plotting the changes in interest rates and corresponding bond prices
fig, ax = plt.subplots(1,2,figsize=(20,5))
rates.plot(ax=ax[0], grid=True, legend=False) 
bond_prices.plot(ax=ax[1], grid=True, legend=False)
ax[0].set_xlabel("months")
ax[0].set_ylabel("interest rate (ytms)")
ax[1].set_xlabel("months")
ax[1].set_ylabel("bond price")
plt.show()

##########################################################################################
##########################################################################################
print("#### Calculating Total Return of a Coupon-Bearing Bond")
##########################################################################################
##########################################################################################

# Computing return by percentage changes in bond price
bond_rets = bond_prices.pct_change().dropna()

# Annualizing the returns
pok.annualize_rets(bond_rets, periods_per_year=periods_per_year)

# Setting a fixed yield to maturity
ytm = 0.035
# Calculating the bond price with the given YTM
b_price = pok.bond_price(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year, ytm=ytm) 

# Calculating total returns for the bond at the given price
b_ret = pok.bond_returns(principal=principal, bond_prices=b_price, coupon_rate=coupon_rate, 
                         coupons_per_year=coupons_per_year, periods_per_year=periods_per_year, maturity=maturity)

# Displaying the bond price and return
print("Bond price:  {:.6f}".format(b_price))
print("Bond return: {:.6f}".format(b_ret))



##########################################################################################
##########################################################################################
print("## Macaulay Duration")
##########################################################################################
print("### Calculating Macaulay Duration for a Bond")
##########################################################################################
##########################################################################################

# Bond parameters
principal        = 1000
maturity         = 3
ytm              = 0.06
coupon_rate      = 0.06
coupons_per_year = 2

cf = pok.bond_cash_flows(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year)
print(cf)

# Calculating Macaulay Duration using the YTM divided by the number of coupons per year
macd = pok.mac_duration(cf, discount_rate=ytm/coupons_per_year) 
macd = macd / coupons_per_year
print(macd)

##########################################################################################
##########################################################################################
print("### Alternative Approach: Normalizing Cash Flow Dates")
##########################################################################################
##########################################################################################

# Normalizing cash flows dates
cf = pok.bond_cash_flows(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year)
cf.index = cf.index / coupons_per_year
print(cf)

# Calculating Macaulay Duration using only the YTM as discount rate
pok.mac_duration(cf, discount_rate=ytm)

##########################################################################################
##########################################################################################
print("### Validating Zero-Coupon Bond Duration")
##########################################################################################
##########################################################################################

# Zero-Coupon Bond: only one cash flow at maturity
maturity = 3
cf = pd.DataFrame(data=[100], index=[maturity])
# Calculating Macaulay Duration for a zero-coupon bond, the rate is irrelevant
macd = pok.mac_duration(cf, discount_rate=0.05) # the rate does not impact the duration
print(macd)



##########################################################################################
##########################################################################################
print("## Liability Driven Investing (LDI)")
##########################################################################################
print("### Creating Duration-Matched Portfolios")
##########################################################################################
##########################################################################################

# Initial asset value
asset_value = 130000

# Interest rate and liabilities defined
interest_rate = 0.04

L = pd.DataFrame([100000, 100000], index=[10,12])
print(L)

# Calculating Macaulay duration of liabilities
macd_liab = pok.mac_duration(L, discount_rate=interest_rate)
print("Liability duration: {:.3f} years".format(macd_liab))

# Defining bond parameters
principal = 1000
maturity_short = 10
coupon_rate_short = 0.05 
coupons_per_year_short = 1
ytm_short = interest_rate

maturity_long = 20
coupon_rate_long = 0.05 
coupons_per_year_long = 1
ytm_long = interest_rate

# Calculating cashflows for short and long bonds
cf_short = pok.bond_cash_flows(principal=principal, maturity=maturity_short, coupon_rate=coupon_rate_short, coupons_per_year=coupons_per_year_short)
cf_long  = pok.bond_cash_flows(principal=principal, maturity=maturity_long, coupon_rate=coupon_rate_long, coupons_per_year=coupons_per_year_long)

# Calculating Macaulay durations for short and long bonds
macd_short = pok.mac_duration(cf_short, discount_rate=ytm_short /coupons_per_year_short) /coupons_per_year_short
macd_long  = pok.mac_duration(cf_long,  discount_rate=ytm_long  /coupons_per_year_long)  /coupons_per_year_long
print("(Short) bond duration: {:.3f} years".format(macd_short))
print("(Long) bond duration:  {:.3f} years".format(macd_long))

# Calculating weight for the short bond to match the portfolio duration with liability duration
w_short = (macd_liab - macd_long) / (macd_short - macd_long)
print(w_short)

# Determine prices for both short and long-term bonds
bondprice_short = pok.bond_price(principal=principal, maturity=maturity_short, coupon_rate=coupon_rate_short, 
                                 coupons_per_year=coupons_per_year_short, ytm=ytm_short)
bondprice_long  = pok.bond_price(principal=principal, maturity=maturity_long, coupon_rate=coupon_rate_long, 
                                 coupons_per_year=coupons_per_year_long, ytm=ytm_long)
print("Price of the short-term bond: {:.2f}".format(bondprice_short))
print("Price of the long-term bond: {:.2f}".format(bondprice_long))

# Calculate portfolio cashflows for short and long-term bonds
portfolio_cf_short = w_short     * asset_value / bondprice_short * cf_short
portfolio_cf_long  = (1-w_short) * asset_value / bondprice_long  * cf_long

# Combine cashflows from both bonds
portfolio_cf = pd.concat([portfolio_cf_short, portfolio_cf_long], axis=1).fillna(0)        
portfolio_cf.columns = ["Cashflow from short-term bond", "Cashflow from long-term bond"]

# Add the total cashflow of the portfolio
portfolio_cf["Total Portfolio Cashflow"] = portfolio_cf.sum(axis=1)
print(portfolio_cf)

# Convert the portfolio cashflow to a dataframe
portfolio_cf = pd.DataFrame(portfolio_cf["Total Portfolio Cashflow"])
portfolio_cf.columns = [0]

# Calculate Macaulay duration for the portfolio
macd_portfolio = pok.mac_duration(portfolio_cf, discount_rate=interest_rate)
print("Duration of the portfolio: {:.3f} years".format(macd_portfolio))

def funding_ratio(asset_value, liabilities, r):
    '''Computes the funding ratio between the present value of holding assets and the present 
    value of the liabilities given an interest rate r (or a list of)'''
    return pok.present_value(asset_value, r) / pok.present_value(liabilities, r)

# Series of cashflows for short and long-term bonds
short_bond_asset = asset_value / bondprice_short * cf_short
long_bond_asset  = asset_value / bondprice_long * cf_long

# Range of interest rates
rates = np.linspace(0, 0.1, 20)

# Calculate funding ratios for different investment strategies
funding_ratios = pd.DataFrame(
    {
        "Funding Ratio with Short-term Bond": [funding_ratio(short_bond_asset, L, r)[0] for r in rates],
        "Funding Ratio with Long-term Bond": [funding_ratio(long_bond_asset, L, r)[0] for r in rates],
        "Funding Ratio with Duration Matched Bonds": [funding_ratio(portfolio_cf, L, r)[0] for r in rates]
    }, index = rates
)

# Plotting funding ratios against interest rates
ax = funding_ratios.plot(grid=True, figsize=(10,5), title="Funding ratios with varying interest rates")
ax.set_xlabel("Interest rates")
ax.set_ylabel("Funding ratios")
ax.axhline(y=1, linestyle="--", c="k")
plt.show()


##########################################################################################
##########################################################################################
print("### Constructing Portfolios with Non-Matching Bond Durations")
##########################################################################################
##########################################################################################

def duration_match_weight(d1, d2, d_liab):
    w1 = (d_liab - d2) / (d1 - d2)
    w2 = 1 - w1
    return w1, w2

flat_yield = 0.05

# Defining future liabilities
L = pd.DataFrame([100000, 200000, 300000], index=[3,5,10])
print(L)

# Calculating the Macaulay duration for these liabilities
macd_L = pok.mac_duration(L, discount_rate=flat_yield)
print("Duration of liabilities: ", macd_L)

principal = 1000

# Details for Bond 1
maturity_b1         = 15
coupon_rate_b1      = 0.05
ytm_b1              = flat_yield
coupons_per_year_b1 = 2

# Details for Bond 2
maturity_b2         = 5
coupon_rate_b2      = 0.06
ytm_b2              = flat_yield
coupons_per_year_b2 = 4

# Calculate cash flows for both bonds and normalize dates
cf_b1 = pok.bond_cash_flows(principal=principal, maturity=maturity_b1, coupon_rate=coupon_rate_b1, coupons_per_year=coupons_per_year_b1)
cf_b2 = pok.bond_cash_flows(principal=principal, maturity=maturity_b2, coupon_rate=coupon_rate_b2, coupons_per_year=coupons_per_year_b2)
cf_b1.index = cf_b1.index / coupons_per_year_b1
cf_b2.index = cf_b2.index / coupons_per_year_b2

# Compute Macaulay durations for both bonds
macd_b1 = pok.mac_duration(cf_b1,discount_rate=ytm_b1) 
print("Duration of Bond 1: ", macd_b1)

macd_b2 = pok.mac_duration(cf_b2,discount_rate=ytm_b2)
print("Duration of Bond 2: ", macd_b2)

# Calculate the weights for the bonds to match the liability duration
w_b1, w_b2 = duration_match_weight(macd_b1, macd_b2, macd_L)
print("Weight in Bond 1: ", w_b1)
print("Weight in Bond 2: ", w_b2)

# Calculate prices for both bonds
bprice_b1 = pok.bond_price(principal=principal, maturity=maturity_b1, coupon_rate=coupon_rate_b1, 
                           coupons_per_year=coupons_per_year_b1, ytm=ytm_b1)
bprice_b2 = pok.bond_price(principal=principal, maturity=maturity_b2, coupon_rate=coupon_rate_b2, 
                           coupons_per_year=coupons_per_year_b2, ytm=ytm_b2)
print("Price of Bond 1: ", bprice_b1)
print("Price of Bond 2: ", bprice_b2)

# Compute portfolio cashflows from both bonds
portfolio_cf_b1 = w_b1 * asset_value / bprice_b1 * cf_b1
portfolio_cf_b2 = w_b2 * asset_value / bprice_b2 * cf_b2

# Combine cashflows from both bonds
portfolio_cf = pd.concat([portfolio_cf_b1, portfolio_cf_b2], axis=1).fillna(0)        
portfolio_cf.columns = ["Cashflow from Bond 1", "Cashflow from Bond 2"]

# Add the total cashflow of the portfolio
portfolio_cf["Total Portfolio Cashflow"] = portfolio_cf.sum(axis=1)
print(portfolio_cf)

# Convert the portfolio cashflow to a dataframe
portfolio_cf = pd.DataFrame(portfolio_cf["Total Portfolio Cashflow"].rename(0))

# Calculate Macaulay duration for the portfolio
macd_portfolio = pok.mac_duration(portfolio_cf, discount_rate=flat_yield)  
print("Duration of the portfolio: ", macd_portfolio)
print("Duration of the liabilities: ", macd_L)


##########################################################################################
##########################################################################################
print("### Integrating Performance-Seeking Portfolio (PSP) with Liability-Hedging Portfolio (LHP)")
##########################################################################################
print("#### Naive PSP/LHP Weighting Strategy")
##########################################################################################
##########################################################################################

# Bond details
principal = 100

# Short-term bond parameters
maturity_short         = 10
coupon_rate_short      = 0.028
coupons_per_year_short = 2

# Long-term bond parameters
maturity_long          = 20
coupon_rate_long       = 0.035
coupons_per_year_long  = 2

# Simulation parameters
n_scenarios      = 1000
n_years          = np.max([maturity_short, maturity_long])  # = maturity_long
mean_rate        = 0.03
periods_per_year = 2

# Simulating rates and zero-coupon bond prices
rates, zcb_price = pok.simulate_cir(n_years=n_years, n_scenarios=n_scenarios, a=0.05, b=mean_rate, 
                                    sigma=0.02, periods_per_year=periods_per_year)
print(rates.tail())

# Calculate bond prices for short and long-term bonds
l = int(coupons_per_year_short * n_years / periods_per_year)

bond_pr_short = pok.bond_price(principal=principal, maturity=maturity_short, coupon_rate=coupon_rate_short, 
                               coupons_per_year=coupons_per_year_short, ytm=rates.iloc[:l+1,:]) 

bond_pr_long = pok.bond_price(principal=principal, maturity=maturity_long, coupon_rate=coupon_rate_long, 
                              coupons_per_year=coupons_per_year_long, ytm=rates).iloc[:l+1,:]

# Calculate returns for both short and long-term bonds
bond_rets_short = pok.bond_returns(principal=principal, bond_prices=bond_pr_short, coupon_rate=coupon_rate_short, 
                                   coupons_per_year=coupons_per_year_short, periods_per_year=periods_per_year)

bond_rets_long = pok.bond_returns(principal=principal, bond_prices=bond_pr_long, coupon_rate=coupon_rate_long, 
                                   coupons_per_year=coupons_per_year_long, periods_per_year=periods_per_year)

# Define the weight for the short-term bond
w1 = 0.6

# Mix the returns of the two bonds to form the LHP
bond_rets = pok.ldi_mixer(bond_rets_short, bond_rets_long, allocator=pok.ldi_fixed_allocator, w1=w1)
print(bond_rets.head())

# Simulate stock prices and returns for the PSP
stock_price, stock_rets = pok.simulate_gbm_from_prices(n_years=maturity_short, n_scenarios=n_scenarios, 
                                                       mu=0.07, sigma=0.1, periods_per_year=2, start=100.0)


##########################################################################################
##########################################################################################
print("#### Implementing Fixed-Mixed Allocation in PSP/LHP Strategy")
##########################################################################################
##########################################################################################

# Define the fixed allocation weights for Stocks/Bonds
w1 = 0.7

# Combine the returns of the stocks and bonds using the fixed allocation
stock_bond_rets = pok.ldi_mixer(stock_rets, bond_rets, allocator=pok.ldi_fixed_allocator, w1=w1)
print(stock_bond_rets.head())

# Compute and print the stats summary of the PSP/LHP portfolio
stock_bond_rets_stats = pok.summary_stats(stock_bond_rets, risk_free_rate=0, periods_per_year=2)
print(stock_bond_rets_stats.tail())

# Print the mean statistics across all scenarios
print(stock_bond_rets_stats.mean())

# Define a floor value for risk assessment
floor = 0.8

# Calculate and print the summary stats of terminal parameters for different investment strategies
ldi_stats = pd.concat([
    pok.summary_stats_terminal(bond_rets, floor=floor, periods_per_year=periods_per_year, name="Bonds only"),
    pok.summary_stats_terminal(stock_rets, floor=floor, periods_per_year=periods_per_year, name="Stocks only"),
    pok.summary_stats_terminal(stock_bond_rets, floor=floor, periods_per_year=periods_per_year, name="70/30 Stocks/Bonds"),
], axis=1)
print(ldi_stats)

# Plotting histograms of terminal wealth for different investment strategies
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot( pok.terminal_wealth(bond_rets), bins=40, color="red", label="Bonds only", ax=ax[0]) 
sns.distplot( pok.terminal_wealth(stock_rets), bins=40, color="blue", label="Stocks only", ax=ax[1])
sns.distplot( pok.terminal_wealth(stock_bond_rets), bins=40, color="orange", label="70/30 Stocks/Bonds", ax=ax[1])
plt.suptitle("Terminal wealth histograms")
ax[0].axvline( x=pok.terminal_wealth(bond_rets).mean(), linestyle="-.", color="red", linewidth=1)
ax[1].axvline( x=pok.terminal_wealth(stock_rets).mean(), linestyle="-.", color="blue", linewidth=1)
ax[1].axvline( x=pok.terminal_wealth(stock_bond_rets).mean(), linestyle="-.", color="orange", linewidth=1)
ax[1].axvline( x=floor, linestyle="--", color="k")
ax[1].set_xlim(left=0.1) 
ax[0].legend(), ax[0].grid()
ax[1].legend(), ax[1].grid()
plt.show()


##########################################################################################
##########################################################################################
print("#### Glide Path Weight Allocation Strategy")
##########################################################################################
##########################################################################################

# Generate the glide path allocation between stocks and bonds from 80/20 to 20/80
print(pok.ldi_glidepath_allocator(stock_rets, bond_rets, start=0.8, end=0.2))

# Calculate the returns of the PSP/LHP strategy with a glide path allocation
stock_bond_rets_glide = pok.ldi_mixer(stock_rets, bond_rets, allocator=pok.ldi_glidepath_allocator, start=0.8, end=0.2)
print(stock_bond_rets_glide.head())

# Define a floor value for risk assessment
floor = 0.8

# Calculate and print the summary stats of terminal parameters for different investment strategies
ldi_stats = pd.concat([
    pok.summary_stats_terminal(bond_rets, floor=floor, periods_per_year=periods_per_year, name="Bonds only"),
    pok.summary_stats_terminal(stock_rets, floor=floor, periods_per_year=periods_per_year, name="Stocks only"),
    pok.summary_stats_terminal(stock_bond_rets, floor=floor, periods_per_year=periods_per_year, name="70/30 Stocks/Bonds"),
    pok.summary_stats_terminal(stock_bond_rets_glide, floor=floor, periods_per_year=periods_per_year, name="Glide 80/20 Stocks/Bonds"),
], axis=1)
print(ldi_stats)


##########################################################################################
##########################################################################################
print("#### Integrating Floor Considerations with Performance-Seeking and Liability-Hedging Portfolios")
##########################################################################################
##########################################################################################

# Parameters for simulating interest rates and zero-coupon bond prices
n_scenarios      = 1000
n_years          = 10
mean_rate        = 0.03
periods_per_year = 12 

# Simulating rates and zero-coupon bond prices
rates, zcb_price = pok.simulate_cir(n_years=n_years, n_scenarios=n_scenarios, a=0.05, b=mean_rate, 
                                    sigma=0.02, periods_per_year=periods_per_year)

# Computing zero-coupon bond returns and simulating stock prices
zcb_rets = zcb_price.pct_change().dropna()
stock_price, stock_rets = pok.simulate_gbm_from_prices(n_years=n_years, n_scenarios=n_scenarios, 
                                                       mu=0.07, sigma=0.15, periods_per_year=periods_per_year)

w1 = 0.7  # Allocation weight for stocks
stock_zcb_rets = pok.ldi_mixer(stock_rets, zcb_rets, allocator=pok.ldi_fixed_allocator, w1=w1)

floor = 0.8  # Predefined floor value
# Calculating and comparing summary stats of terminal parameters for ZCBs, stocks, and 70/30 stocks/ZCBs
ldi_stats = pd.concat([
    pok.summary_stats_terminal(zcb_rets, floor=floor, periods_per_year=periods_per_year, name="ZCB only"),
    pok.summary_stats_terminal(stock_rets, floor=floor, periods_per_year=periods_per_year, name="Stocks only"),
    pok.summary_stats_terminal(stock_zcb_rets, floor=floor, periods_per_year=periods_per_year, name="70/30 Stocks/ZCB"),
], axis=1).round(4)
print(ldi_stats)


##########################################################################################
##########################################################################################
print("#### Floor Allocator")
##########################################################################################
##########################################################################################

floor = 0.8  # Floor value

# Implementing the floor allocator with different multipliers (m = 1,3,5) to modulate PSP allocation
stock_zcb_floor_m1_rets = pok.ldi_mixer(stock_rets, zcb_rets, allocator=pok.ldi_floor_allocator, 
                                        zcb_price=zcb_price.loc[1:], floor=floor, m=1)
# Repeat for m=3 and m=5

# Comparing strategies with different multipliers
ldi_stats = pd.concat([
    ldi_stats,
    pok.summary_stats_terminal(stock_zcb_floor_m1_rets, floor=floor, periods_per_year=periods_per_year, name="Floor(0.8-1) Stocks/ZCB"),
    # Repeat for m=3 and m=5 strategies
], axis=1).round(4)
print(ldi_stats)


##########################################################################################
##########################################################################################
print("#### Drawdown Allocator")
##########################################################################################
##########################################################################################

# Implementing the drawdown allocator with a maximum drawdown constraint
maxdd = 0.2  # Maximum drawdown limit
stock_zcb_dd_02_rets = pok.ldi_mixer(stock_rets, zcb_rets, allocator=pok.ldi_drawdown_allocator, maxdd=maxdd)

# Comparing strategies with and without drawdown constraints
ldi_stats = pd.concat([
    ldi_stats,
    pok.summary_stats_terminal(stock_zcb_dd_02_rets, floor=1 - maxdd, periods_per_year=periods_per_year, name="DD(0.2) Stocks/ZCB"),
], axis=1).round(4)
print(ldi_stats)


##########################################################################################
##########################################################################################
print("#### Considering Cash as an Alternative LHP")
##########################################################################################
##########################################################################################

ann_cashrate = 0.02  # Annual cash rate
monthly_cashrets = (1 + ann_cashrate)**(1/12) - 1  # Monthly cash returns
cash_rets = pd.DataFrame(data=monthly_cashrets, index=stock_rets.index, columns=stock_rets.columns)

# Implementing the drawdown allocator with cash as the LHP
stock_cash_dd_02_rets = pok.ldi_mixer(stock_rets, cash_rets, allocator=pok.ldi_drawdown_allocator, maxdd=0.2)

# Comparing strategies with cash as LHP
ldi_stats = pd.concat([
    ldi_stats,
    pok.summary_stats_terminal(stock_cash_dd_02_rets, floor=1 - 0.2, periods_per_year=periods_per_year, name="DD(0.2) Stocks/Cash"),
], axis=1).round(4)
print(ldi_stats)

# Plotting histograms of the terminal wealths to understand the distribution and risk profiles of different strategies
# Compute terminal wealth for each investment strategy
tw_stock              = pok.terminal_wealth(stock_rets)
tw_stock_zcb          = pok.terminal_wealth(stock_zcb_rets)
tw_stock_zcb_floor_m1 = pok.terminal_wealth(stock_zcb_floor_m1_rets)
tw_stock_cash_dd_02   = pok.terminal_wealth(stock_cash_dd_02_rets)

# Create a figure and axis for the histogram plot
fig, ax = plt.subplots(1,1,figsize=(20,5))

# Plot histograms for terminal wealth of each strategy
sns.distplot(tw_stock, bins=40, color="red", label="Stocks only", ax=ax)
sns.distplot(tw_stock_zcb, bins=40, color="blue", label="70/30 Stocks/ZCB", ax=ax)
sns.distplot(tw_stock_zcb_floor_m1, bins=40, color="orange", label="Floor(0.8-1) Stocks/ZCB", ax=ax)
sns.distplot(tw_stock_cash_dd_02, bins=40, color="green", label="DD(0.2) Stocks/Cash", ax=ax)

# Add a title and labels
plt.suptitle("Terminal wealth histograms")

# Add vertical lines representing the mean terminal wealth for each strategy
ax.axvline(x=tw_stock.mean(), linestyle="-.", color="red", linewidth=1)
ax.axvline(x=tw_stock_zcb.mean(), linestyle="-.", color="blue", linewidth=1)
ax.axvline(x=tw_stock_zcb_floor_m1.mean(), linestyle="-.", color="orange", linewidth=1)
ax.axvline(x=tw_stock_cash_dd_02.mean(), linestyle="-.", color="green", linewidth=1)

# Add a vertical line representing the floor value
ax.axvline(x=floor, linestyle="--", color="k")

# Set the x-axis limit to focus on the relevant part of the distribution
ax.set_xlim(left=0.1)

# Add a legend and grid for better readability
ax.legend(), ax.grid()

# Display the plot
plt.show()


##########################################################################################
##########################################################################################
print("#### Real-World Application with Historical Data")
##########################################################################################
##########################################################################################

tmi_rets = pok.get_total_market_index_returns()["1990":]  # Total market index returns

# Computing drawdown and peaks for total market index
dd_tmi = pok.drawdown(tmi_rets)

# Constructing the LHP with cash returns
ann_cashrate = 0.03
monthly_cashrets = (1 + ann_cashrate)**(1/12) - 1
cash_rets = pd.DataFrame(data=monthly_cashrets, index=tmi_rets.index, columns=[0])  # Single scenario

# PSP/LHP strategy with Total Market/Cash
tmi_cash_dd_02_rets = pok.ldi_mixer(pd.DataFrame(tmi_rets), cash_rets, allocator=pok.ldi_drawdown_allocator, maxdd=0.2)

# Computing drawdowns and peaks for the PSP/LHP strategy
dd_psp_lhp = pok.drawdown(tmi_cash_dd_02_rets[0])

# Visualizing wealth and peaks for total market and PSP/LHP strategies
fig, ax = plt.subplots(1,1,figsize=(10,6))
dd_tmi["Wealth"].plot(ax=ax, grid=True, color="red", label="Total market")
dd_tmi["Peaks"].plot(ax=ax, grid=True, ls=":", color="red", label="Total market peaks")
dd_psp_lhp["Wealth"].plot(ax=ax, grid=True, color="blue", label="PSP/LHP DD 0.2")
dd_psp_lhp["Peaks"].plot(ax=ax, grid=True, ls=":", color="blue", label="PSP/LHP DD 0.2 peaks")
plt.legend()
plt.show()

# Computing and displaying summary stats for investments
invests = pd.concat([
    tmi_rets.rename("Tot. Market (PSP)"), 
    cash_rets[0].rename("Cash (LHP)"), 
    tmi_cash_dd_02_rets[0].rename("PSP/LHP(DD0.2)")
], axis=1)
print(pok.summary_stats(invests, risk_free_rate=0, periods_per_year=12))