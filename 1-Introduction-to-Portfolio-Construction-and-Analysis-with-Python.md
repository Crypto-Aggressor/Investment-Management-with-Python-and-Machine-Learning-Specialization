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