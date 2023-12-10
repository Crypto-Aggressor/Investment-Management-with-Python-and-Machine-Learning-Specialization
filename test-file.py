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