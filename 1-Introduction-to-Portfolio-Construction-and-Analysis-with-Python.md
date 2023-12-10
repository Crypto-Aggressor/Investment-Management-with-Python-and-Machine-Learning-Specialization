# Fundamentals and Strategies in Portfolio Management: A Python Approach

**Core Topics**:

1. Understanding Returns and Assessing Risks with Value at Risk
2. Essentials of Portfolio Optimization
3. Strategies Beyond Diversification
4. Fundamentals of Asset-Liability Management

## 1. Understanding Returns and Assessing Risks with Value at Risk

### Analyzing Returns

**Percentage Returns Explained**: Percentage return measures the financial gain or loss between two time points, $t$ and $t+1$. It's calculated as:

$$
P_{t+1} = P_{t} + R_{t,t+1}P_{t} = P_{t}(1+R_{t,t+1})
\qquad\Longrightarrow\qquad
R_{t,t+1} := \frac{P_{t+1} - P_t}{P_{t}} = \frac{P_{t+1}}{P_t} - 1.
$$

where $R_{t,t+1}$ is the return. For example, if a stock price rises from $100 to $104, the return is $R_{t,t+1}=104/100-1=0.04\$=4\%$

### Understanding Compound Returns

**Compound Returns over Multiple Periods**: The total return over several periods is not merely the sum of individual returns. Consider two consecutive time periods, with prices $P_0$ and $P_2$:

$$
P_1 = P_0 + R_{0,1}P_0
\qquad\text{and}\qquad
P_2 = P_1 + R_{1,2}P_1.
$$

So, the total return over two periods,

$$
R_{0,2} = \frac{P_2}{P_0} - 1
= 1 + R_{0,1}+R_{1,2}+R_{1,2}R_{0,1} - 1
= (1 + R_{0,1})(1 + R_{1,2}) - 1.
$$

In a timeframe $t$ to $t+k$, with $k > 1$, it generalizes to:

$$
R_{t,t+k} = (1+R)^{k} - 1.
\qquad\Longrightarrow\qquad
\prod_{i=0}^{k-1} (1+R_{t+i,t+i+1}) - 1
$$

#### Practical Examples

1. **Two-Day Stock Performance:** A stock increasing by 10% on day one and decreasing by 3% on day two has a compound return $R_{0,2} = (1 + 0.10)(1 - 0.03) - 1 = 6.7\%$, not simply $10\% - 3\%$.

2. **Annualized Return from Quarterly Returns:** A stock with consistent 1% quarterly returns yields an annualized return $R_{0,12} = (1 + 0.01)^4 - 1 = 4.06\%$.

### Monthly and Annual Returns

- **Monthly Returns:** Given monthly returns, the compound total return after two months $R_{total}$ is calculated. To find the equivalent monthly return $R_{pm}$, we solve $R_{total} = (1 + R_{pm})^2 - 1$, giving $R_{pm} = \sqrt{1 + R_{total}} - 1$.

- **Annualized Returns:** The annualized return $R_{py}$ is derived from monthly returns using $R_{py} = (1 + R_{pm})^{12} - 1$. For a series of returns, the formula becomes $R_{py} = (1 + R_{total})^{12/n} - 1$, where $n$ is the number of months.

#### Generalizing Annualized Returns

For different time intervals (daily, weekly, monthly), the annualized return formula adjusts the power in the equation, with \( P_y \) representing the number of periods per year:

$$
R_{py} = (1 + R_{total})^{P_{y}/N_{\text{rets}}} - 1,
\quad \text{where} \quad
P_{y} =
\begin{cases}
&252 & \text{if daily}, \\
&52 & \text{if weekly}, \\
&12 & \text{if monthly}.
\end{cases}
$$

### Assessing Volatility and Risk

Volatility, a risk measure, is the standard deviation of asset returns:

$$
\sigma := \sqrt{  \frac{1}{N-1} \sum_{t} (R_t - \mu)^2  },
$$

where $\mu$ is the mean return. For monthly returns, annualized volatility is $\sigma_{ann} = \sigma_{m} \sqrt{12}$.

When dealing with **monthly return data**, the calculation of volatility usually focuses on the monthly scale, termed as **monthly volatility**. However, to understand the asset's risk profile over a longer period, such as a year, this monthly volatility needs to be scaled up. This process is necessary because volatility metrics derived from different time intervals are not directly comparable.

The conversion to **annualized volatility** $\sigma_{ann}$ involves a simple mathematical adjustment:

$$
\sigma_{ann} = \sigma_{p} \sqrt{p},
$$

where $\sigma_{p}$ is the volatility calculated over the shorter time period $p$, and $\sigma_{ann}$ is the annualized volatility. The variable $p$ represents the number of periods in a year.

For different time frames, the calculation adjusts as follows:

- Monthly Volatility $\sigma_{m}$: To annualize, use $\sigma_{ann} = \sigma_{m} \sqrt{12}$
- Weekly Volatility $\sigma_{w}$: Annualize by calculating $\sigma_{ann} = \sigma_{w} \sqrt{12}$
- Daily Volatility $\sigma_{d}$: Convert to annual volatility with $\sigma_{ann} = \sigma_{d} \sqrt{12}$

This method standardizes volatility to a yearly scale, allowing for a consistent and comparable measure of risk across different time frames.

#### Zero Volatility Concept

**Question cosidering this scenario**: Asset A experiences a monthly decrease of 1% over a period of 12 months, while Asset B sees a consistent monthly increase of 1% during the same timeframe.

**Which asset exhibits greater volatility?**

Interestingly, the answer is that neither of them display any volatility (volatility is non-existent in both cases), as there are no real fluctuations; Asset A consistently decreases, whereas Asset B consistently increases.

```python
# Creating two hypothetical assets
a = [100]
b = [100]
for i in range(12):
    a.append(a[-1] * 0.99)  # Asset A loses 1% each month
    b.append(b[-1] * 1.01)  # Asset B gains 1% each month

# Convert to DataFrame and calculate returns
asset_df = pd.DataFrame({"Asset A": a, "Asset B": b})
asset_df["Return A"] = asset_df["Asset A"].pct_change()
asset_df["Return B"] = asset_df["Asset B"].pct_change()

print("Total Returns:", (1 + asset_df[["Return A", "Return B"]].iloc[1:]).prod() - 1)
print("Volatility:", asset_df[["Return A", "Return B"]].iloc[1:].std())
```

#### Python Example: Analyzing Stock Data

```python
# In this analysis, we first generate synthetic stock prices for two stocks with distinct volatilities. 
# We then calculate their monthly returns and visualize the data. 
# The total compound returns, mean returns, and volatility are computed to understand the risk profile of each stock. 
# Finally, we calculate the Return on Risk (ROR) for each stock, which provides insights into the risk-adjusted performance of the investments. 
# This analysis suggests that Stock A offers a better return per unit of risk compared to Stock B, even though their total returns are similar.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example of generating stock prices
np.random.seed(51)
stocks = pd.DataFrame({"Stock A": np.random.normal(10, 1, size=10), "Stock B": np.random.normal(10, 5, size=10)})
stocks.index.name = "Months"
stocks = round(stocks, 2)

# Calculating returns
stocks["Stock A Rets"] = stocks["Stock A"] / stocks["Stock A"].shift(1) - 1
stocks["Stock B Rets"] = stocks["Stock B"] / stocks["Stock B"].shift(1) - 1
stocks = round(stocks, 2)

# Visualizing stock prices and returns
f, ax = plt.subplots(1, 2, figsize=(20, 4))
ax[0].plot(stocks[["Stock A", "Stock B"]])
ax[0].set_title('Stock Prices')
ax[0].set_xlabel("Months")
ax[0].set_ylabel("Price (USD)")
ax[0].legend(["Stock A", "Stock B"])
ax[0].grid()
(stocks[["Stock A Rets", "Stock B Rets"]].drop(index=0) * 100).plot.bar(ax=ax[1])
ax[1].set_title('Stock Returns')
ax[1].set_xlabel("Months")
ax[1].set_ylabel("Returns (%)")
ax[1].legend(["Stock A", "Stock B"])
ax[1].grid()
plt.show()

# Calculating total compound returns
total_ret = (1 + stocks[["Stock A Rets", "Stock B Rets"]]).prod() - 1
print("Total Returns (%):", total_ret * 100)

# Computing means and volatility
means = stocks[["Stock A Rets", "Stock B Rets"]].mean()
volatility = stocks[["Stock A Rets", "Stock B Rets"]].std()
print("Mean Returns:", means)
print("Volatility:", volatility)

# Annualizing volatility
ann_volatility = volatility * np.sqrt(12)
print("Annualized Volatility:", ann_volatility)
```

### Evaluating Return on Risk

Return on Risk (ROR) measures the reward per unit of risk, calculated as:

$$
\text{ROR} := \frac{\text{RETURN}}{\text{RISK}} = \frac{R}{\sigma},
$$

where $R$ is the total compound return. This metric helps compare investments with different risk profiles.

```python
# This code segment computes the ROR for each stock, helping investors understand which stock offers better returns for the risk taken.

# Calculating Return on Risk
ROR = total_ret / volatility
print("Return on Risk:", ROR)
```

### Sharpe Ratio: Assessing Risk-Adjusted Returns

The Sharpe Ratio provides a more nuanced view of an investment's performance by considering the risk-free rate. This ratio adjusts the return on risk by accounting for the returns of a risk-free asset, like a US Treasury Bill. It's defined as the excess return per unit of risk:

$$
\lambda := \frac{E_R}{\sigma}
\quad\text{where}\quad
E_R := R - R_F,
$$

Here, $E_R$ is the excess return, calculated by subtracting the risk-free rate $R_F$ from the return $R$.

```python
# This calculation demonstrates how the Sharpe Ratio can provide additional insights into the risk-adjusted performance of an investment.

# Assuming a 3% risk-free rate
risk_free_rate = 0.03 
excess_return  = total_ret - risk_free_rate
sharpe_ratio   = excess_return / volatility
print("Sharpe Ratio:", sharpe_ratio)
```

### Illustrating Financial Concepts Using a Real-World Dataset

The performance of Small Cap and Large Cap US stocks using a dataset from Kaggle will be used to illustrate Financial Concepts. This dataset includes monthly returns from July 1926 to December 2018.

#### Data Analysis

**Dataset Source**: **`PortfolioOptimizationKit.py`** to load data from a CSV file will be used. This dataset categorizes US stocks into Small Caps (bottom 10% by market capitalization) and Large Caps (top 10%).

**Data Visualization**: The code visualizes monthly returns for both categories. This step is crucial for a quick assessment of performance over time.

```python
import pandas as pd
import matplotlib.pyplot as plt
import PortfolioOptimizationKit as pok

# Load the dataset
file_to_load = pok.path_to_data_folder() + "Portfolios_Formed_on_ME_monthly_EW.csv"
df = pd.read_csv(file_to_load, index_col=0, parse_dates=True, na_values=-99.99)

# Focus on Small Cap and Large Cap stocks
small_large_caps = df[["Lo 10", "Hi 10"]] / 100  # Dividing by 100 to convert to actual returns

# Plotting the returns
small_large_caps.plot(grid=True, figsize=(10,4))
plt.title("Monthly Returns of Small Cap and Large Cap Stocks")
plt.xlabel("Year")
plt.ylabel("Returns")
plt.show()
```

#### Calculating Volatility

**Monthly Volatility**: Using Python's standard deviation function to calculate the monthly volatility, a measure of how much stock returns vary from their average value.

**Annualizing Volatility**: Then scale up the monthly volatility to an annual figure. This conversion provides a broader view of the stocks' risk over a longer period.

```python
# Calculating monthly volatility
monthly_volatility = small_large_caps.std()

# Annualizing the volatility
annualized_volatility = monthly_volatility * (12 ** 0.5)

print("Monthly Volatility:", monthly_volatility)
print("Annualized Volatility:", annualized_volatility)
```

#### Returns Analysis

**Monthly & Annual Returns**: Python code is used to calculate both monthly and annual returns. This step is key to understanding the long-term growth potential of the stocks.

```python
# Number of months in the dataset
n_months = small_large_caps.shape[0]

# Total compound return
total_return = (1 + small_large_caps).prod() - 1

# Monthly and annualized returns
return_per_month = (1 + total_return) ** (1 / n_months) - 1
annualized_return = (1 + return_per_month) ** 12 - 1

print("Return per Month:", return_per_month)
print("Annualized Return:", annualized_return)
```

**Risk-Adjusted Returns**: Return on Risk and Sharpe Ratio are computed. These metrics help us understand which stocks offer better returns for their level of risk.

```python
# Assuming a risk-free rate
risk_free_rate = 0.03

# Return on Risk and Sharpe Ratio
return_on_risk = annualized_return / annualized_volatility
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

print("Return on Risk:", return_on_risk)
print("Sharpe Ratio:", sharpe_ratio)
```

### Drawdown

**``Drawdown``** is a crucial metric in portfolio management, indicating the maximum loss from a peak to a trough of a portfolio, before a new peak is achieved. It's a measure of the most significant drop in asset value and is often used to assess the risk of a particular investment or portfolio.

To calculate the drawdown, these steps are needed:

Calculate the **`Wealth Index`**: This represents the value of a portfolio as it evolves over time, taking into account the compounding of returns.

Determine the **`Previous Peaks`**: Identify the highest value of the portfolio before each time point.

Calculate the **`Drawdown`**: It is the difference between the current wealth index and the previous peak, represented as a percentage of the previous peak.

The formula can be expressed as:

$$
\text{Drawdown at time } t = \frac{P_t - \max_{t' \in [0, t]} P_{t'}}{\max_{t' \in [0, t]} P_{t'}} \
$$

where $P_t$ is the portfolio value at time $t$, and $\max_{t' \in [0, t]} P_{t'}$ is the maximum portfolio value up to time $t$.

In extension, the **`Maximum Drawdown`** is a metric that quantifies the most substantial loss experienced from a peak to a trough of a portfolio before the emergence of a new peak. This metric is crucial for measuring the most severe decrease in value. The Maximum Drawdown formula is described as:

$$
\text{Max Drawdown} = \max_{t \in [0, T]} \left( \max_{t' \in [0, t]} P_{t'} - P_t \right) \
$$

where $T$ is the entire time period under consideration, $P_t$ is the portfolio value at time $t$, and $\max_{t' \in [0, t]} P_{t'}$ is the maximum portfolio value up to time $t$.

#### Practical example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
file_to_load = "Portfolios_Formed_on_ME_monthly_EW.csv"
rets = pd.read_csv(file_to_load, index_col=0, parse_dates=True, na_values=-99.99)

# Focusing on Small Caps and Large Caps
rets = rets[["Lo 10", "Hi 10"]] / 100
rets.columns = ["Small Caps", "Large Caps"]
rets.index = pd.to_datetime(rets.index, format="%Y%m")

# Step 1: Calculate the Wealth Index
wealth_index = 100 * (1 + rets).cumprod()

# Step 2: Compute Previous Peaks
previous_peaks = wealth_index.cummax()

# Step 3: Calculate Drawdown
drawdown = (wealth_index - previous_peaks) / previous_peaks

# Visualizing Wealth Index and Drawdown
fig, ax = plt.subplots(3, 2, figsize=(20, 12))
wealth_index.plot(ax=ax[0, :], title="Wealth Index")
(drawdown * 100).plot(ax=ax[1, :], title="Drawdown (%)")
ax[2, 0].set_ylabel("%")
ax[2, 1].set_ylabel("%")
plt.show()

# Analyzing Specific Drawdowns
# '29 Crisis
print("1929 Crisis:")
print("Max Drawdown: ", drawdown.min().round(2) * 100)
print("Date of Max Drawdown:", drawdown.idxmin())

# Dot Com Crisis
print("Dot Com Crisis:")
print("Max Drawdown (1990-2005): ", drawdown["1990":"2005"].min().round(2) * 100)
print("Date of Max Drawdown:", drawdown["1990":"2005"].idxmin())

# Lehman Brothers Crisis
print("Lehman Brothers Crisis:")
print("Max Drawdown (2005 onwards): ", drawdown["2005":].min().round(2) * 100)
print("Date of Max Drawdown:", drawdown["2005":].idxmin())
```

#### Insights from Historical Crises

- 1929 Crisis:

    The Great Depression profoundly affected both Small Caps and Large Caps, leading to severe wealth erosion.
    This case study serves as a reminder of the potential extreme risks in the stock market.

- Dot Com Crisis (1990-2005):

    Characterized by the bursting of the dot-com bubble, this period saw significant losses, particularly in technology-heavy Large Caps.
    Reflects the volatility and potential downside of tech-driven market booms.

- Lehman Brothers Crisis (2005 Onwards):

    Triggered by the collapse of Lehman Brothers, this crisis led to a global financial meltdown.
    Highlights the interconnectedness of modern financial markets and the rapidity with which shocks can propagate.

### Gaussian Density & Distribution

A Gaussian random variable $X$, characterized by a mean $\mu$ and $\sigma^2$ (notated as $X\sim N(\mu,\sigma^2$)), is characterized by the following:

- A **Density function**:

$$
f(x) := \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(\frac{-(x-\mu)^2}{2\sigma^2}\right),
$$

- And a Cumulative distribution function (CDF):

$$
F_X(x) := \mathbb{P}(X\leq x) = \int_{-\infty}^x f(t)dt = \Phi(x).
$$

Here, $\Phi(x)$ gives the probability that $X$ is less than or equal to $x$.

Where $\mu=0$, and $\sigma^2=1$, $X$ is considered as **standard Gaussian random variable**. This standardization simplifies many statistical calculations.

- **Symmetry Property**:

For a standard normal random variable $X\sim N(0,1)$, it holds that:

$$
\Phi(x) = 1-\Phi(-x).
$$

This property reflects the symmetry of the standard normal distribution.

- **Distribution of Negative Random Variable**:

The exponential function $\exp(-t^2/2)$ is symmetric (i.e., an even function). Therefore, for a standard normal random variable $X\sim N(0,1)$, the distribution of $-X$ is also $N(0,1)$. Mathematically:

$$
\mathbb{P}(-X\leq x) 
= \mathbb{P}(X\geq -x)
= 1-\mathbb{P}(X\leq -x)
= 1-\Phi(-x)
= \Phi(x)
= \mathbb{P}(X\leq x),
$$

confirming that the cumulative distribution function of $-X$ is identical tot hat of $X$:

$$
F_{-X}(x) = F_X(x).
$$

### Quantiles

Imagine a random variable $X$, and let's consider a number $\alpha$ that lies between 0 and 1 (i.e., $\alpha\in(0,1)$). The quantile of order $\alpha$, denoted as $\phi_\alpha\in\mathbb{R}$, is a value such that the probability of $X$ being less than or equal to $\phi_\alpha$ equals $\alpha$. Mathematically, it's defined as:

$$
\mathbb{P}(X\leq \phi_\alpha) = \alpha.
$$

- **Quantiles in a Standard Normal Distribution**:

For a standard normal distribution, where $X\sim N(0,1)$, the concept of quantiles becomes particularly interesting. If $\phi_\alpha$ is an $\alpha$-quantile of a standard normal distribution, it satisfies:

$$
\Phi(-\phi_\alpha)
= \mathbb{P}(X\leq - \phi_\alpha)
= \mathbb{P}(-X\leq - \phi_\alpha)
= \mathbb{P}(X\geq \phi_\alpha)
= 1 - \mathbb{P}(X\leq \phi_\alpha)
= 1 - \Phi(\phi_\alpha)
= 1 - \alpha
= \mathbb{P}(X\leq \phi_{1-\alpha})
= \Phi(\phi_{1-\alpha}),
$$

This leads to a useful identity:

$$
-\phi_\alpha = \phi_{1-\alpha}.
$$

In essence, this identity illustrates the symmetry of the standard normal distribution.

- **Symmetry in Probability**:

In a standard normal distribution, the probability of the variable lying within certain symmetric bounds is given by:

$$
\mathbb{P}(|X|\leq \phi_{1-\alpha/2})
= \mathbb{P}(-\phi_{1-\alpha/2} \leq X \leq \phi_{1-\alpha/2})
= \Phi(\phi_{1-\alpha/2}) - \Phi(\underbrace{ -\phi_{1-\alpha/2} }_{= \phi_{\alpha/2}})
= (1-\alpha/2) - (\alpha/2)
= 1-\alpha.
$$

This formula is useful for understanding the distribution of values around the mean in a normal distribution.

#### Practical Application: Finding a Specific Quantile

For example, to determine the 0.9-quantile of a standard normal distribution (i.e., the value below which 90% of the data lies), let's look for $\phi_{0.9}$ such that:

$$
\phi_{0.9} = \Phi^{-1}(0.9).
$$

Using the **`norm.ppf()`** function from **`scipy.stats`**, which provides **`quantiles`** for the **`Gaussian distribution`**, we can calculate this value directly.

```python
import scipy.stats

# Calculating the 0.9-quantile of a standard normal distribution
phi_0_9 = scipy.stats.norm.ppf(0.9, 0, 1)
print(f"The 0.9-quantile (phi_0.9) of a standard normal distribution: {phi_0_9}")

# Double-checking by calculating the probability up to the quantile
probability_check = scipy.stats.norm.cdf(phi_0_9, 0, 1)
print(f"Probability check for phi_0.9: {probability_check}")
```

### Exploring Skewness and Kurtosis in Financial Data

#### Skewness: Assessing Asymmetry in Distributions

Skewness quantifies how asymmetrical a distribution is regarding its mean. It can be:

- **Positive**: Indicating a tail on the right side.

- **Negative**: Showing a tail on the left side.

Formally, skewness is defined using the third centered moment:

$$
S(X) := \frac{\mathbb{E}(X - \mathbb{E}(X)^3)}{\sigma^3},
$$

where $\sigma$ is the standard deviation of $X$.

#### Kurtosis: Analyzing Tails and Outliers

Kurtosis measures the "tailedness" or concentration of outliers in a distribution. It's calculated as:

$$
K(X) := \frac{\mathbb{E}(X - \mathbb{E}(X)^4)}{\sigma^4},
$$

where $\sigma$ is the standard deviation of $X$.

A normal distribution has a kurtosis of 3. Thus, **`"excess kurtosis"`** is often computed as $K(X)−3$, offering a comparison to a normal distribution.

#### Practical Analysis

Here's a practical demonstration using Python to understand skewness and kurtosis in financial datasets:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PortfolioOptimizationKit as pok

# Generate a normally distributed dataset
normal_data = pd.DataFrame({"A": np.random.normal(0, 2, size=800)})

# Retrieve returns from a financial dataset known to deviate from normality
market_returns = pok.get_ffme_returns()["Hi 10"]

# Plot histograms to visualize distributions
fig, axes = plt.subplots(1, 2, figsize=(18, 3))

axes[0].hist(normal_data.values, bins=60, density=True)
axes[0].set_title(f'Normal Distribution - Mean: {normal_data.mean().values.round(3)}, Std: {normal_data.std().values.round(3)}')
axes[0].grid()

axes[1].hist(market_returns.values, bins=60, density=True)
axes[1].set_title(f'Market Returns - Mean: {round(market_returns.mean(), 3)}, Std: {round(market_returns.std(), 3)}')
axes[1].grid()

plt.show()

# Calculating skewness and kurtosis
skewness_normal = ((normal_data - normal_data.mean())**3 / normal_data.std(ddof=0)**3).mean()
kurtosis_normal = ((normal_data - normal_data.mean())**4 / normal_data.std(ddof=0)**4).mean()

skewness_market = ((market_returns - market_returns.mean())**3 / market_returns.std(ddof=0)**3).mean()
kurtosis_market = ((market_returns - market_returns.mean())**4 / market_returns.std(ddof=0)**4).mean()

# Displaying the skewness and kurtosis
print(f"Skewness - Normal: {skewness_normal.values}, Market: {round(skewness_market, 3)}")
print(f"Kurtosis - Normal: {kurtosis_normal.values}, Market: {round(kurtosis_market, 3)}")

# Visual analysis of skewness and kurtosis
fig, axes = plt.subplots(2, 2, figsize=(18, 8))

axes[0, 0].plot(((normal_data - normal_data.mean())**3 / normal_data.std(ddof=0)**3).values)
axes[0, 0].axhline(y=skewness_normal[0], linestyle=":", color="red")
axes[0, 0].set_title('Normal Distribution - Skewness')

axes[1, 0].plot(((normal_data - normal_data.mean())**4 / normal_data.std(ddof=0)**4).values)
axes[1, 0].axhline(y=kurtosis_normal[0], linestyle=":", color="red")
axes[1, 0].set_title('Normal Distribution - Kurtosis')

axes[0, 1].plot(((market_returns - market_returns.mean())**3 / market_returns.std(ddof=0)**3).values)
axes[0, 1].axhline(y=skewness_market, linestyle=":", color="red")
axes[0, 1].set_title('Market Returns - Skewness')

axes[1, 1].plot(((market_returns - market_returns.mean())**4 / market_returns.std(ddof=0)**4).values)
axes[1, 1].axhline(y=kurtosis_market, linestyle=":", color="red")
axes[1, 1].set_title('Market Returns - Kurtosis')

plt
```

In this analysis, the normal distribution is expected to show skewness near zero and kurtosis close to three. In contrast, the market returns, deviating from normality, may exhibit different skewness and higher kurtosis values.

#### Advanced Analysis with Hedge Fund Indices incorporation

We extend our analysis of skewness and kurtosis to a dataset involving hedge fund indices. This dataset provides a different perspective, often diverging from the standard characteristics of normal distributions.

```python
import PortfolioOptimizationKit as pok
import scipy.stats

# Load the hedge fund index data
hfi = pok.get_hfi_returns()
print(hfi.head(3))

# Initialize a DataFrame to store skewness and kurtosis values
hfi_skew_kurt = pd.DataFrame(columns=["Skewness", "Kurtosis"])

# Calculate skewness and kurtosis for each column in the hedge fund index data
hfi_skew_kurt["Skewness"] = hfi.aggregate(pok.skewness)
hfi_skew_kurt["Kurtosis"] = hfi.aggregate(pok.kurtosis)

# Display the calculated skewness and kurtosis
print(hfi_skew_kurt)
```

When trying identifying Gaussian Distributions, **`CTA Global`**, shows skewness near zero and kurtosis close to three, indicating a possible normal distribution.

- **Using Jarque-Bera Test for Normality**

The Jarque-Bera test, a statistical test for normality, helps confirm if an index follows a Gaussian distribution.

```python
# Jarque-Bera test on CTA Global
jb_result_cta_global = scipy.stats.jarque_bera(hfi["CTA Global"])
print("CTA Global:", jb_result_cta_global)

# Check normality using custom function in erk toolkit
is_normal_cta_global = pok.is_normal(hfi["CTA Global"])
print("Is CTA Global Normal?", is_normal_cta_global)

# Jarque-Bera test on Convertible Arbitrage
jb_result_conv_arb = scipy.stats.jarque_bera(hfi["Convertible Arbitrage"])
print("Convertible Arbitrage:", jb_result_conv_arb)

# Check normality for Convertible Arbitrage
is_normal_conv_arb = pok.is_normal(hfi["Convertible Arbitrage"])
print("Is Convertible Arbitrage Normal?", is_normal_conv_arb)
```

- **Normality Across Indices**

Finally, we examine the normality of all indices in the hedge fund dataset.

```python
# Aggregate normality test across all indices
normality_test_results = hfi.aggregate(pok.is_normal)
print(normality_test_results)
```

This comprehensive analysis reveals that only the CTA Global index passes the normality test, suggesting it's the most normally distributed among the hedge fund indices.
