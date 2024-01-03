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

# CIR Model: Simulating Interest Rate Fluctuations
print("CIR Model: Simulating Interest Rate Fluctuations")

# Applying the CIR Model to Price Zero-Coupon Bonds
print("Applying the CIR Model to Price Zero-Coupon Bonds")

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

# Liability Hedging
print("Liability Hedging")

# Nominal rate of the liability
mean_rate = 0.03
# Time horizon for the liability in years
n_years   = 10
# Number of different interest rate scenarios to simulate
n_scenarios = 10
# Number of periods per year for compounding
periods_per_year = 12

# Simulating Interest Rates and Zero-Coupon Bond Prices
print("Simulating Interest Rates and Zero-Coupon Bond Prices")

# Simulate interest rates using the CIR model and calculate corresponding ZCB prices
rates, zcb_price = pok.simulate_cir(n_years=n_years, n_scenarios=n_scenarios, 
                                    a=0.05, b=mean_rate, sigma=0.08, periods_per_year=periods_per_year)
print(rates.head())

# Assign the simulated ZCB prices as the liabilities
L = zcb_price
print(L.head())

# Hedging with Zero-Coupon Bonds
print("Hedging with Zero-Coupon Bonds")

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

# Hedging by Holding Cash
print("Hedging by Holding Cash")

# Calculate the future asset value when holding cash, accounting for compounding interest
asset_value_in_cash = asset_0 * (1 + rates/periods_per_year).cumprod()
print(asset_value_in_cash.head())

### Comparing the Two Investment Strategies
print("Comparing the Two Investment Strategies")

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


# Coupon-Bearing Bonds
print("Coupon-Bearing Bonds")

# Cash Flow from a Bond
print("Cash Flow from a Bond")

# Bond parameters
principal        = 100 
maturity         = 3
ytm              = 0.05
coupon_rate      = 0.03 
coupons_per_year = 2

# Calculating bond cash flows
cf = pok.bond_cash_flows(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year)
print(cf)

# Bond Price Calculation
print("Bond Price Calculation")

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


# Yield to Maturity and Bond Price Relationship
print("Yield to Maturity and Bond Price Relationship")

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


# Variations in Bond Price
print("Variations in Bond Price")

# Observing Price Changes with Interest Rate Fluctuations
print("Observing Price Changes with Interest Rate Fluctuations")

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

# Calculating Total Return of a Coupon-Bearing Bond
print("Calculating Total Return of a Coupon-Bearing Bond")

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


# Macaulay Duration
print("Macaulay Duration")

# Calculating Macaulay Duration for a Bond
print("Calculating Macaulay Duration for a Bond")