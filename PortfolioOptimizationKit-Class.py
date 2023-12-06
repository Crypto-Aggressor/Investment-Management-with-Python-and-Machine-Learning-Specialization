import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from numpy.linalg import inv


# ---------------------------------------------------------------------------------
# Load and format data files
# ---------------------------------------------------------------------------------
class DataLoader:
    """
    DataLoader: A class for loading and formatting financial data files.

    This class provides methods to access and preprocess data from various financial sources. 
    It is designed to handle the import and initial processing of data such as stock prices, 
    bond yields, market indices, etc., typically from CSV files or other common data formats.

    Methods:
        path_to_data_folder(): Returns the path to the data folder.
        get_ffme_returns(): Loads the French-Fama dataset for US stock returns.
        get_hfi_returns(): Retrieves the EDHEC Hedge Fund Index returns.
        get_brka_rets(monthly): Loads Berkshire Hathaway's stock returns.
        get_fff_returns(): Accesses Fama-French Research Factors data.
        get_ind_file(filetype, nind, ew): Loads Kenneth French Industry Portfolios.
        get_ind_market_caps(nind, weights): Returns industry market caps or cap weights.
        get_total_market_index_returns(nind): Computes returns of a cap-weighted total market index.
        get_total_market_index(nind, capital): Returns the cap-weighted total market index.
    
    Usage:
        To load specific financial data, create an instance of the class and call its methods:
        loader = DataLoader()
        ffme_returns = loader.get_ffme_returns()
        hfi_returns = loader.get_hfi_returns()

    Note:
        The methods assume the presence of specific data files and formats. Paths and file names
        may need to be adjusted based on your data storage structure.
    """
    
    @staticmethod
    def path_to_data_folder():
        """
        Static method to provide the path to the data folder.
        
        Returns:
            str: A string representing the path to the data folder.
        """
        return "/Users/leonardorocchi/Documents/Coding/Python/finance-courses/data/"

    @staticmethod
    def get_ffme_returns():
        """
        Static method to load the French-Fama dataset for the returns of the bottom 
        and top deciles (Low 10 and Hi 10) of US stocks.

        Returns:
            pandas.DataFrame: A DataFrame containing the Low 10 and Hi 10 returns, indexed by month.
        """
        filepath = DataLoader.path_to_data_folder() + "Portfolios_Formed_on_ME_monthly_EW.csv"
        rets = pd.read_csv(filepath, index_col=0, parse_dates=True, na_values=-99.99)
        rets = rets[["Lo 10", "Hi 10"]] / 100
        rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
        return rets

    @staticmethod
    def get_hfi_returns():
        """
        Static method to load the EDHEC Hedge Fund Index returns.

        Returns:
            pandas.DataFrame: A DataFrame containing hedge fund index returns, indexed by date.
        """
        filepath = DataLoader.path_to_data_folder() + "edhec-hedgefundindices.csv"
        hfi = pd.read_csv(filepath, index_col=0, parse_dates=True, na_values=-99.99) / 100.0
        return hfi

    @staticmethod
    def get_brka_rets(monthly=False):
        '''
        Load and format Berkshire Hathaway's returns from 1990-01 to 2018-12.
        Default data are daily returns. 
        If monthly=True, then monthly data are returned. Here, the method used 
        the .resample method which allows to run an aggregation function on each  
        group of returns of the daily time series.
        '''
        filepath = path_to_data_folder() + "brka_d_ret.csv"
        rets = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if monthly:
            rets = rets.resample("M").apply( compound ).to_period("M")
        return rets
    
    @staticmethod
    def get_fff_returns():
        '''
        Load the Fama-French Research Factors Monthly Dataset.
        Factors returned are those of the Fama-French model:
        - Excess return of the market, i.e., Market minus Risk-Free Rate,
        - Small (size) Minus Big (size) SMB,
        - High (B/P ratio) Minus Low (B/P ratio) HML, 
        - and the Risk Free Rate 
        '''
        filepath = path_to_data_folder() + "F-F_Research_Data_Factors_m.csv"
        fff = pd.read_csv(filepath, index_col=0, parse_dates=True, na_values=-99.99) / 100
        fff.index = pd.to_datetime(fff.index, format="%Y%m").to_period("M")
        return fff 

    @staticmethod
    def get_ind_file(filetype="rets", nind=30, ew=False):
        '''
        Load and format the Kenneth French Industry Portfolios files.
        - filetype: can be "rets", "nfirms", "size"
        - nind: can be 30 or 49
        - ew: if True, equally weighted portfolio dataset are loaded.
        Also, it has a role only when filetype="rets".
        '''
        if nind!=30 and nind!=49:
            raise ValueError("Expected either 30 or 49 number of industries")
        if filetype == "rets":
            portfolio_w = "ew" if ew==True else "vw" 
            name = "{}_rets" .format( portfolio_w )
            divisor = 100.0
        elif filetype == "nfirms":
            name = "nfirms"
            divisor = 1
        elif filetype == "size":
            name = "size"
            divisor = 1
        else:
            raise ValueError("filetype must be one of: rets, nfirms, size")
        filepath = path_to_data_folder() + "ind{}_m_{}.csv" .format(nind, name)
        ind = pd.read_csv(filepath, index_col=0, parse_dates=True) / divisor
        ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
        ind.columns = ind.columns.str.strip()
        return ind

    @staticmethod
    def get_ind_market_caps(nind=30, weights=False):
        '''
        Load the industry portfolio dataset and returns single industries market caps.
        If weights=True, it returns single industries market cap-weights as a percentage of
        of the total market cap.
        '''
        ind_nfirms = get_ind_file(filetype="nfirms", nind=nind)
        ind_size   = get_ind_file(filetype="size", nind=nind)
        # compute the market capitalization of each industry sector
        ind_caps   = ind_nfirms * ind_size
        if weights:
            # compute the total market capitalization
            total_cap = ind_caps.sum(axis=1)
            # compute single market capitalizations as a percentage of the total market cap
            ind_cap_weight = ind_caps.divide(total_cap, axis=0)
            return ind_cap_weight
        else:
            return ind_caps 
    
    @staticmethod
    def get_total_market_index_returns(nind=30):
        '''
        Computes the returns of a cap-weighted total market index from Kenneth French Industry portfolios
        '''  
        # load the right returns 
        ind_rets = get_ind_file(filetype="rets", nind=nind) 
        # load the cap-weights of each industry 
        ind_cap_weight = get_ind_market_caps(nind=nind, weights=True)
        # total market returns         
        total_market_return = (ind_cap_weight * ind_rets).sum(axis=1)
        return total_market_return

    def get_total_market_index(nind=30, capital=1000):
        '''
        Return the cap-weighted total market index from Kenneth French Industry portfolios
        ''' 
        total_market_return = get_total_market_index_returns(nind=nind)
        total_market_index  = capital * (1 + total_market_return).cumprod()
        return total_market_index

# ---------------------------------------------------------------------------------
# Return Analysis and general statistics
# ---------------------------------------------------------------------------------
class ReturnAnalysis:
    """
    ReturnAnalysis: A class dedicated to the analysis of financial returns.

    This class includes methods for calculating various metrics and performing 
    analyses on return data of financial assets. The methods take financial 
    return data, typically in the form of pandas DataFrames or Series, and 
    compute useful statistics and metrics for portfolio analysis.

    Methods:
        terminal_wealth(s): Calculates the terminal wealth of a return series.
        compound(s): Computes the compounded return of a series.
        compound_returns(s, start): Compounds returns from an initial value.
        compute_returns(s): Calculates percentage change returns of a series.
        compute_logreturns(s): Calculates log returns of a series.
        drawdown(rets): Computes the drawdowns of a return series.
        skewness(s): Calculates the skewness of a return series.
        kurtosis(s): Calculates the kurtosis of a return series.
        exkurtosis(s): Calculates the excess kurtosis of a return series.
        is_normal(s, level): Tests if a series is normally distributed.
        semivolatility(s): Computes the semivolatility of a return series.
        var_historic(s, level): Calculates historical Value at Risk.
        var_gaussian(s, level, cf): Calculates Gaussian Value at Risk.
        cvar_historic(s, level): Computes Conditional VaR using historical method.
        annualize_rets(s, periods_per_year): Annualizes a series of returns.
        annualize_vol(s, periods_per_year, ddof): Annualizes the volatility of a series.

    Usage:
        To analyze return data, create an instance of the class and call its methods:
        analysis = ReturnAnalysis()
        drawdowns = analysis.drawdown(rets)
        ann_returns = analysis.annualize_rets(rets, 12)

    Note:
        The methods assume the input data is properly formatted as pandas DataFrame or Series.
    """
    
    @staticmethod
    def terminal_wealth(s):
        '''
        Computes the terminal wealth of a sequence of return, which is, in other words, 
        the final compounded return. 
        The input s is expected to be either a pd.DataFrame or a pd.Series
        '''
        if not isinstance(s, (pd.DataFrame, pd.Series)):
            raise ValueError("Expected either a pd.DataFrame or pd.Series")
        return (1 + s).prod()
    
    @staticmethod
    def compound(s):
        '''
        Single compound rule for a pd.Dataframe or pd.Series of returns. 
        The method returns a single number - using prod(). 
        See also the TERMINAL_WEALTH method.
        '''
        if not isinstance(s, (pd.DataFrame, pd.Series)):
            raise ValueError("Expected either a pd.DataFrame or pd.Series")
        return (1 + s).prod() - 1
        # Note that this is equivalent to (but slower than)
        # return np.expm1( np.logp1(s).sum() )
    
    @staticmethod
    def compound_returns(s, start=100):
        '''
        Compound a pd.Dataframe or pd.Series of returns from an initial default value equal to 100.
        In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
        The method returns a pd.Dataframe or pd.Series - using cumprod(). 
        See also the COMPOUND method.
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate(ReturnAnalysis.compound_returns, start=start)
        elif isinstance(s, pd.Series):
            return start * (1 + s).cumprod()
        else:
            raise TypeError("Expected pd.DataFrame or pd.Series")
    @staticmethod
    def compute_returns(s):
        '''
        Computes the returns (percentage change) of a Dataframe of Series. 
        In the former case, it computes the returns for every column (Series) by using pd.aggregate
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate( compute_returns )
        elif isinstance(s, pd.Series):
            return s / s.shift(1) - 1
        else:
            raise TypeError("Expected pd.DataFrame or pd.Series")
    
    @staticmethod
    def compute_logreturns(s):
        '''
        Computes the log-returns of a Dataframe of Series. 
        In the former case, it computes the returns for every column (Series) by using pd.aggregate
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate( ReturnAnalysis.compute_logreturns )
        elif isinstance(s, pd.Series):
            return np.log( s / s.shift(1) )
        else:
            raise TypeError("Expected pd.DataFrame or pd.Series")
    
    @staticmethod 
    def drawdown(rets: pd.Series, start=1000):
        '''
        Compute the drawdowns of an input pd.Series of returns. 
        The method returns a dataframe containing: 
        1. the associated wealth index (for an hypothetical starting investment of $1000) 
        2. all previous peaks 
        3. the drawdowns
        '''
        wealth_index   = ReturnAnalysis.compound_returns(rets, start=start)
        previous_peaks = wealth_index.cummax()
        drawdowns      = (wealth_index - previous_peaks ) / previous_peaks
        df = pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns} )
        return df

    @staticmethod
    def skewness(s):
        '''
        Computes the Skewness of the input Series or Dataframe.
        There is also the function scipy.stats.skew().
        '''
        return ( ((s - s.mean()) / s.std(ddof=0))**3 ).mean()

    @staticmethod
    def kurtosis(s):
        '''
        Computes the Kurtosis of the input Series or Dataframe.
        There is also the function scipy.stats.kurtosis() which, however, 
        computes the "Excess Kurtosis", i.e., Kurtosis minus 3
        '''
        return ( ((s - s.mean()) / s.std(ddof=0))**4 ).mean()

    @staticmethod
    def exkurtosis(s):
        '''
        Returns the Excess Kurtosis, i.e., Kurtosis minus 3
        '''
        return ReturnAnalysis.kurtosis(s) - 3

    @staticmethod
    def is_normal(s, level=0.01):
        '''
        Jarque-Bera test to see if a series (of returns) is normally distributed.
        Returns True or False according to whether the p-value is larger 
        than the default level=0.01.
        '''
        statistic, pvalue = scipy.stats.jarque_bera( s )
        return pvalue > level

    @staticmethod
    def semivolatility(s):
        '''
        Returns the semivolatility of a series, i.e., the volatility of
        negative returns
        '''
        return s[s<0].std(ddof=0) 

    @staticmethod
    def var_historic(s, level=0.05):
        '''
        Returns the (1-level)% VaR using historical method. 
        By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
        The method takes in input either a DataFrame or a Series and, in the former 
        case, it computes the VaR for every column (Series) by using pd.aggregate
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate( ReturnAnalysis.var_historic, level=level )
        elif isinstance(s, pd.Series):
            return - np.percentile(s, level*100)
        else:
            raise TypeError("Expected pd.DataFrame or pd.Series")
    
    @staticmethod    
    def var_gaussian(s, level=0.05, cf=False):
        '''
        Returns the (1-level)% VaR using the parametric Gaussian method. 
        By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
        The variable "cf" stands for Cornish-Fisher. If True, the method computes the 
        modified VaR using the Cornish-Fisher expansion of quantiles.
        The method takes in input either a DataFrame or a Series and, in the former 
        case, it computes the VaR for every column (Series).
        '''
        # alpha-quantile of Gaussian distribution 
        za = scipy.stats.norm.ppf(level,0,1) 
        if cf:
            S = ReturnAnalysis.skewness(s)
            K = ReturnAnalysis.kurtosis(s)
            za = za + (za**2 - 1)*S/6 + (za**3 - 3*za)*(K-3)/24 - (2*za**3 - 5*za)*(S**2)/36    
        return -( s.mean() + za * s.std(ddof=0) )

    @staticmethod
    def cvar_historic(s, level=0.05):
        '''
        Computes the (1-level)% Conditional VaR (based on historical method).
        By default it computes the 95% CVaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
        The method takes in input either a DataFrame or a Series and, in the former 
        case, it computes the VaR for every column (Series).
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate( ReturnAnalysis.cvar_historic, level=level )
        elif isinstance(s, pd.Series):
            # find the returns which are less than (the historic) VaR
            mask = s < -ReturnAnalysis.var_historic(s, level=level)
            # and of them, take the mean 
            return -s[mask].mean()
        else:
            raise TypeError("Expected pd.DataFrame or pd.Series")

    @staticmethod
    def annualize_rets(s, periods_per_year):
        '''
        Computes the return per year, or, annualized return.
        The variable periods_per_year can be, e.g., 12, 52, 252, in 
        case of monthly, weekly, and daily data.
        The method takes in input either a DataFrame or a Series and, in the former 
        case, it computes the annualized return for every column (Series) by using pd.aggregate
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate( ReturnAnalysis.annualize_rets, periods_per_year=periods_per_year )
        elif isinstance(s, pd.Series):
            growth = (1 + s).prod()
            n_period_growth = s.shape[0]
            return growth**(periods_per_year/n_period_growth) - 1

    @staticmethod
    def annualize_vol(s, periods_per_year, ddof=1):
        '''
        Computes the volatility per year, or, annualized volatility.
        The variable periods_per_year can be, e.g., 12, 52, 252, in 
        case of monthly, weekly, and daily data.
        The method takes in input either a DataFrame, a Series, a list or a single number. 
        In the former case, it computes the annualized volatility of every column 
        (Series) by using pd.aggregate. In the latter case, s is a volatility 
        computed beforehand, hence only annulization is done
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate(ReturnAnalysis.annualize_vol, periods_per_year=periods_per_year )
        elif isinstance(s, pd.Series):
            return s.std(ddof=ddof) * (periods_per_year)**(0.5)
        elif isinstance(s, list):
            return np.std(s, ddof=ddof) * (periods_per_year)**(0.5)
        elif isinstance(s, (int,float)):
            return s * (periods_per_year)**(0.5)

    @staticmethod
    def sharpe_ratio(s, risk_free_rate, periods_per_year, v=None):
        '''
        Computes the annualized sharpe ratio. 
        The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
        The variable risk_free_rate is the annual one.
        The method takes in input either a DataFrame, a Series or a single number. 
        In the former case, it computes the annualized sharpe ratio of every column (Series) by using pd.aggregate. 
        In the latter case, s is the (allready annualized) return and v is the (already annualized) volatility 
        computed beforehand, for example, in case of a portfolio.
        '''
        if isinstance(s, pd.DataFrame):
            return s.aggregate( ReturnAnalysis.sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, v=None)
        elif isinstance(s, pd.Series):
            # convert the annual risk free rate to the period assuming that:
            # RFR_year = (1+RFR_period)^{periods_per_year} - 1. Hence:
            rf_to_period = (1 + risk_free_rate)**(1/periods_per_year) - 1        
            excess_return = s - rf_to_period
            # now, annualize the excess return
            ann_ex_rets = ReturnAnalysis.annualize_rets(excess_return, periods_per_year)
            # compute annualized volatility
            ann_vol = ReturnAnalysis.annualize_vol(s, periods_per_year)
            return ann_ex_rets / ann_vol
        elif isinstance(s, (int,float)) and v is not None:
            # Portfolio case: s is supposed to be the single (already annnualized) 
            # return of the portfolio and v to be the single (already annualized) volatility. 
            return (s - risk_free_rate) / v


# ---------------------------------------------------------------------------------
# Modern Portfolio Theory 
# ---------------------------------------------------------------------------------
class PortfolioTheoryTools:
    """
    PortfolioTheoryTools class contains methods for various portfolio theory calculations 
    and optimizations, including portfolio returns and volatility, efficient frontier, 
    summary statistics, and optimization techniques under the Modern Portfolio Theory framework.
    """
        
    def portfolio_return(weights, vec_returns):
        '''
        Computes the return of a portfolio. 
        It takes in input a row vector of weights (list of np.array) 
        and a column vector (or pd.Series) of returns
        '''
        return np.dot(weights, vec_returns)
        
    def portfolio_volatility(weights, cov_rets):
        '''
        Computes the volatility of a portfolio. 
        It takes in input a vector of weights (np.array or pd.Series) 
        and the covariance matrix of the portfolio asset returns
        '''
        return ( np.dot(weights.T, np.dot(cov_rets, weights)) )**(0.5) 

    def efficient_frontier(n_portfolios, rets, covmat, periods_per_year, risk_free_rate=0.0, 
                        iplot=False, hsr=False, cml=False, mvp=False, ewp=False):
        '''
        Returns (and plots) the efficient frontiers for a portfolio of rets.shape[1] assets. 
        The method returns a dataframe containing the volatilities, returns, sharpe ratios and weights 
        of the portfolios as well as a plot of the efficient frontier in case iplot=True. 
        Other inputs are:
            hsr: if true the method plots the highest return portfolio,
            cml: if True the method plots the capital market line;
            mvp: if True the method plots the minimum volatility portfolio;
            ewp: if True the method plots the equally weigthed portfolio. 
        The variable periods_per_year can be, e.g., 12, 52, 252, in case of monthly, weekly, and daily data.
        '''   
        
        def append_row_df(df,vol,ret,spr,weights):
            temp_df = list(df.values)
            temp_df.append( [vol, ret, spr,] + [w for w in weights] )
            return pd.DataFrame(temp_df)
            
        ann_rets = ReturnAnalysis.annualize_rets(rets, periods_per_year)
        
        # generates optimal weights of porfolios lying of the efficient frontiers
        weights = ReturnAnalysis.optimal_weights(n_portfolios, ann_rets, covmat, periods_per_year) 
        # in alternative, if only the portfolio consists of only two assets, the weights can be: 
        #weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_portfolios)]

        # portfolio returns
        portfolio_ret = [ReturnAnalysis.portfolio_return(w, ann_rets) for w in weights]
        
        # portfolio volatility
        vols          = [ReturnAnalysis.portfolio_volatility(w, covmat) for w in weights] 
        portfolio_vol = [ReturnAnalysis.annualize_vol(v, periods_per_year) for v in vols]
        
        # portfolio sharpe ratio
        portfolio_spr = [ReturnAnalysis.sharpe_ratio(r, risk_free_rate, periods_per_year, v=v) for r,v in zip(portfolio_ret,portfolio_vol)]
        
        df = pd.DataFrame({"volatility": portfolio_vol,
                        "return": portfolio_ret,
                        "sharpe ratio": portfolio_spr})
        df = pd.concat([df, pd.DataFrame(weights)],axis=1)
        
        if iplot:
            ax = df.plot.line(x="volatility", y="return", style="--", color="coral", grid=True, label="Efficient frontier", figsize=(8,4))
            if hsr or cml:
                w   = ReturnAnalysis.maximize_shape_ratio(ann_rets, covmat, risk_free_rate, periods_per_year)
                ret = ReturnAnalysis.portfolio_return(w, ann_rets)
                vol = ReturnAnalysis.annualize_vol( ReturnAnalysis.portfolio_volatility(w,covmat), periods_per_year)
                spr = ReturnAnalysis.sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
                df  = append_row_df(df,vol,ret,spr,w)
                if cml:
                    # Draw the CML: the endpoints of the CML are [0,risk_free_rate] and [port_vol,port_ret]
                    ax.plot([0, vol], [risk_free_rate, ret], color="g", linestyle="-.", label="CML")
                    ax.set_xlim(left=0)
                    ax.legend()
                if hsr:
                    # Plot the highest sharpe ratio portfolio
                    ax.scatter([vol], [ret], marker="o", color="g", label="MSR portfolio")
                    ax.legend()
            if mvp:
                # Plot the global minimum portfolio:
                w   = ReturnAnalysis.minimize_volatility(ann_rets, covmat)
                ret = ReturnAnalysis.portfolio_return(w, ann_rets)
                vol = ReturnAnalysis.annualize_vol( ReturnAnalysis.portfolio_volatility(w,covmat), periods_per_year)
                spr = ReturnAnalysis.sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
                df  = append_row_df(df,vol,ret,spr,w)
                ax.scatter([vol], [ret], color="midnightblue", marker="o", label="GMV portfolio")
                ax.legend()  
            if ewp:
                # Plot the equally weighted portfolio:
                w   = np.repeat(1/ann_rets.shape[0], ann_rets.shape[0])
                ret = ReturnAnalysis.portfolio_return(w, ann_rets)
                vol = ReturnAnalysis.annualize_vol( ReturnAnalysis.portfolio_volatility(w,covmat), periods_per_year)
                spr = ReturnAnalysis.sharpe_ratio(ret, risk_free_rate, periods_per_year, v=vol)
                df  = append_row_df(df,vol,ret,spr,w)
                ax.scatter([vol], [ret], color="goldenrod", marker="o", label="EW portfolio")
                ax.legend()
            return df, ax
        else: 
            return df
        
    def summary_stats(s, risk_free_rate=0.03, periods_per_year=12, var_level=0.05):
        '''
        Returns a dataframe containing annualized returns, annualized volatility, sharpe ratio, 
        skewness, kurtosis, historic VaR, Cornish-Fisher VaR, and Max Drawdown
        '''
        if isinstance(s, pd.Series):
            stats = {
                "Ann. return"  : ReturnAnalysis.annualize_rets(s, periods_per_year=periods_per_year),
                "Ann. vol"     : ReturnAnalysis.annualize_vol(s, periods_per_year=periods_per_year),
                "Sharpe ratio" : ReturnAnalysis.sharpe_ratio(s, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
                "Skewness"     : ReturnAnalysis.skewness(s),
                "Kurtosis"     : ReturnAnalysis.kurtosis(s),
                "Historic CVar": ReturnAnalysis.cvar_historic(s, level=var_level),
                "C-F Var"      : ReturnAnalysis.var_gaussian(s, level=var_level, cf=True),
                "Max drawdown" : ReturnAnalysis.drawdown(s)["Drawdown"].min()
            }
            return pd.DataFrame(stats, index=["0"])
        
        elif isinstance(s, pd.DataFrame):        
            stats = {
                "Ann. return"  : s.aggregate( ReturnAnalysis.annualize_rets, periods_per_year=periods_per_year ),
                "Ann. vol"     : s.aggregate( ReturnAnalysis.annualize_vol,  periods_per_year=periods_per_year ),
                "Sharpe ratio" : s.aggregate( ReturnAnalysis.sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year ),
                "Skewness"     : s.aggregate( ReturnAnalysis.skewness ),
                "Kurtosis"     : s.aggregate( ReturnAnalysis.kurtosis ),
                "Historic CVar": s.aggregate( ReturnAnalysis.cvar_historic, level=var_level ),
                "C-F Var"      : s.aggregate( ReturnAnalysis.var_gaussian, level=var_level, cf=True ),
                "Max Drawdown" : s.aggregate( lambda r: ReturnAnalysis.drawdown(r)["Drawdown"].min() )
            } 
            return pd.DataFrame(stats)
        
    def summary_stats_terminal(rets, floor=0.8, periods_per_year=2, name="Stats", target=np.inf):
        '''
        Return a dataframe of statistics for a given input pd.DataFrame of asset returns. 
        Statistics computed are:
        - the mean annualized return
        - the mean terminal wealth (compounded return)
        - the mean terminal wealth volatility
        - the probability that an input floor is breached by terminal wealths
        - the expected shortfall of those terminal wealths breaching the input floor 
        '''    
        # terminal wealths over scenarios, i.e., compounded returns
        terminal_wlt = ReturnAnalysis.terminal_wealth(rets)
        
        # boolean vector of terminal wealths going below the floor 
        floor_breach = terminal_wlt < floor

        stats = pd.DataFrame.from_dict({
            "Mean ann. ret.":  ReturnAnalysis.annualize_rets(rets, periods_per_year=periods_per_year).mean(),              # mean annualized returns over scenarios
            "Mean wealth":     terminal_wlt.mean(),                                                         # terminal wealths mean 
            "Mean wealth std": terminal_wlt.std(),                                                          # terminal wealths volatility
            "Prob breach":     floor_breach.mean() if floor_breach.sum() > 0 else 0,                        # probability of breaching the floor
            "Exp shortfall":   (floor - terminal_wlt[floor_breach]).mean() if floor_breach.sum() > 0 else 0 # expected shortfall if floor is reached  
        }, orient="index", columns=[name])
        return stats
        
    def optimal_weights(n_points, rets, covmatrix, periods_per_year):
        '''
        Returns a set of n_points optimal weights corresponding to portfolios (of the efficient frontier) 
        with minimum volatility constructed by fixing n_points target returns. 
        The weights are obtained by solving the minimization problem for the volatility. 
        '''
        target_rets = np.linspace(rets.min(), rets.max(), n_points)    
        weights = [ReturnAnalysis.minimize_volatility(rets, covmatrix, target) for target in target_rets]
        return weights

    def minimize_volatility(rets, covmatrix, target_return=None):
        '''
        Returns the optimal weights of the minimum volatility portfolio on the effient frontier. 
        If target_return is not None, then the weights correspond to the minimum volatility portfolio 
        having a fixed target return. 
        The method uses the scipy minimize optimizer which solves the minimization problem 
        for the volatility of the portfolio
        '''
        n_assets = rets.shape[0]    
        # initial guess weights
        init_guess = np.repeat(1/n_assets, n_assets)
        weights_constraint = {
            "type": "eq",
            "fun": lambda w: 1.0 - np.sum(w)  
        }
        if target_return is not None:
            return_constraint = {
                "type": "eq",
                "args": (rets,),
                "fun": lambda w, r: target_return - ReturnAnalysis.portfolio_return(w, r)
            }
            constr = (return_constraint, weights_constraint)
        else:
            constr = weights_constraint
            
        result = minimize(ReturnAnalysis.portfolio_volatility, 
                        init_guess,
                        args = (covmatrix,),
                        method = "SLSQP",
                        options = {"disp": False},
                        constraints = constr,
                        bounds = ((0.0,1.0),)*n_assets ) # bounds of each individual weight, i.e., w between 0 and 1
        return result.x

    def minimize_volatility_2(rets, covmatrix, target_return=None, weights_norm_const=True, weights_bound_const=True):
        '''
        Returns the optimal weights of the minimum volatility portfolio.
        If target_return is not None, then the weights correspond to the minimum volatility portfolio 
        having a fixed target return (such portfolio will be on the efficient frontier).
        The variables weights_norm_const and weights_bound_const impose two more conditions, the firt one on 
        weight that sum to 1, and the latter on the weights which have to be between zero and 1
        The method uses the scipy minimize optimizer which solves the minimization problem 
        for the volatility of the portfolio
        '''
        n_assets = rets.shape[0]    
        
        # initial guess weights
        init_guess = np.repeat(1/n_assets, n_assets)
        
        if weights_bound_const:
            # bounds of the weights (between 0 and 1)
            bounds = ((0.0,1.0),)*n_assets
        else:
            bounds = None
        
        constraints = []
        if weights_norm_const:
            weights_constraint = {
                "type": "eq",
                "fun": lambda w: 1.0 - np.sum(w)  
            }
            constraints.append( weights_constraint )    
        if target_return is not None:
            return_constraint = {
                "type": "eq",
                "args": (rets,),
                "fun": lambda w, r: target_return - portfolio_return(w, r)
            }
            constraints.append( return_constraint )
        
        result = minimize(ReturnAnalysis.portfolio_volatility, 
                        init_guess,
                        args = (covmatrix,),
                        method = "SLSQP",
                        options = {"disp": False},
                        constraints = tuple(constraints),
                        bounds = bounds)
        return result.x

    def maximize_shape_ratio(rets, covmatrix, risk_free_rate, periods_per_year, target_volatility=None):
        '''
        Returns the optimal weights of the highest sharpe ratio portfolio on the effient frontier. 
        If target_volatility is not None, then the weights correspond to the highest sharpe ratio portfolio 
        having a fixed target volatility. 
        The method uses the scipy minimize optimizer which solves the maximization of the sharpe ratio which 
        is equivalent to minimize the negative sharpe ratio.
        '''
        n_assets   = rets.shape[0] 
        init_guess = np.repeat(1/n_assets, n_assets)
        weights_constraint = {
            "type": "eq",
            "fun": lambda w: 1.0 - np.sum(w)  
        }
        if target_volatility is not None:
            volatility_constraint = {
                "type": "eq",
                "args": (covmatrix, periods_per_year),
                "fun": lambda w, cov, p: target_volatility - ReturnAnalysis.annualize_vol(ReturnAnalysis.portfolio_volatility(w, cov), p)
            }
            constr = (volatility_constraint, weights_constraint)
        else:
            constr = weights_constraint
            
        def neg_portfolio_sharpe_ratio(weights, rets, covmatrix, risk_free_rate, periods_per_year):
            '''
            Computes the negative annualized sharpe ratio for minimization problem of optimal portfolios.
            The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
            The variable risk_free_rate is the annual one.
            '''
            # annualized portfolio returns
            portfolio_ret = ReturnAnalysis.portfolio_return(weights, rets)        
            # annualized portfolio volatility
            portfolio_vol = ReturnAnalysis.annualize_vol(ReturnAnalysis.portfolio_volatility(weights, covmatrix), periods_per_year)
            return - ReturnAnalysis.sharpe_ratio(portfolio_ret, risk_free_rate, periods_per_year, v=portfolio_vol)    
            #i.e., simply returns  -(portfolio_ret - risk_free_rate)/portfolio_vol
            
        result = minimize(neg_portfolio_sharpe_ratio,
                        init_guess,
                        args = (rets, covmatrix, risk_free_rate, periods_per_year),
                        method = "SLSQP",
                        options = {"disp": False},
                        constraints = constr,
                        bounds = ((0.0,1.0),)*n_assets)
        return result.x

    def weigths_max_sharpe_ratio(covmat, mu_exc, scale=True):
        '''
        Optimal (Tangent/Max Sharpe Ratio) portfolio weights using the Markowitz Optimization Procedure:
        - mu_exc is the vector of Excess expected Returns (has to be a column vector as a pd.Series)
        - covmat is the covariance N x N matrix as a pd.DataFrame
        Look at pag. 188 eq. (5.2.28) of "The econometrics of financial markets", by Campbell, Lo, Mackinlay.
        '''
        w = UtilityMethods.inverse_df(covmat).dot(mu_exc)
        if scale:
            # normalize weigths
            w = w/sum(w) 
        return w

# ---------------------------------------------------------------------------------
# CPPI backtest strategy
# ---------------------------------------------------------------------------------
class CPPIBacktest:
    """
    Class for running a CPPI (Constant Proportion Portfolio Insurance) backtest strategy.
    This strategy provides a way to participate in the upside of a risky asset while offering 
    a protection mechanism against downside risk.

    Attributes:
        risky_rets (pd.Series or pd.DataFrame): Returns of the risky asset.
        safe_rets (pd.Series or pd.DataFrame, optional): Returns of the safe asset. If None, artificial safe returns are created.
        start_value (float, optional): Initial investment value. Default is 1000.
        floor (float, optional): The minimum acceptable value of the investment, as a fraction of the initial investment. Default is 0.8.
        m (float, optional): The multiplier to determine the risky asset allocation. Default is 3.
        drawdown (float, optional): The maximum acceptable drawdown. Default is None, which means no drawdown constraint.
        risk_free_rate (float, optional): The risk-free rate used for safe asset returns. Default is 0.03.
        periods_per_year (int, optional): The number of periods per year. Default is 12.
    """

    def __init__(self, risky_rets, safe_rets=None, start_value=1000, floor=0.8, m=3, drawdown=None, risk_free_rate=0.03, periods_per_year=12):
        self.risky_rets = risky_rets
        self.safe_rets = safe_rets
        self.start_value = start_value
        self.floor = floor
        self.m = m
        self.drawdown = drawdown
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run_backtest(self):
        """
        Executes the CPPI backtest strategy based on the initialized parameters. 
        This method computes the account value history, cushion history, risky weight history, 
        and the corresponding returns throughout the investment period.

        Returns:
            dict: A dictionary containing various metrics from the backtest, including risky wealth, 
                  CPPI wealth, CPPI returns, cushion history, risky allocation history, and safe returns. 
                  If drawdown is specified, additional metrics like floor value and peak history are included.
        """

        # Compute the risky wealth (100% investment in the risky asset)
        risky_wealth = self.start_value * (1 + self.risky_rets).cumprod()

        # Initialize CPPI parameters
        account_value = self.start_value
        floor_value = self.floor * account_value

        # Create DataFrames for historical data
        account_history = pd.DataFrame().reindex_like(self.risky_rets)
        cushion_history = pd.DataFrame().reindex_like(self.risky_rets)
        risky_w_history = pd.DataFrame().reindex_like(self.risky_rets)

        # Additional dataframes for drawdown
        if self.drawdown is not None:
            peak_history = pd.DataFrame().reindex_like(self.risky_rets)
            floor_history = pd.DataFrame().reindex_like(self.risky_rets)
            peak = self.start_value
            self.m = 1 / self.drawdown

        # Loop over dates
        for step in range(len(self.risky_rets.index)):
            if self.drawdown is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak * (1 - self.drawdown)
                floor_history.iloc[step] = floor_value

            cushion = (account_value - floor_value) / account_value
            risky_w = self.m * cushion
            risky_w = np.minimum(risky_w, 1).clip(0)

            safe_w = 1 - risky_w

            # Calculate allocations
            risky_allocation = risky_w * account_value
            safe_allocation = safe_w * account_value

            # Update account value
            account_value = risky_allocation * (1 + self.risky_rets.iloc[step]) + safe_allocation * (1 + self.safe_rets.iloc[step])

            # Save historical data
            account_history.iloc[step] = account_value
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w

        # Calculate CPPI returns
        cppi_rets = (account_history / account_history.shift(1) - 1).dropna()

        # Results dictionary
        backtest_result = {
            "Risky wealth": risky_wealth,
            "CPPI wealth": account_history,
            "CPPI returns": cppi_rets,
            "Cushions": cushion_history,
            "Risky allocation": risky_w_history,
            "Safe returns": self.safe_rets
        }

        if self.drawdown is not None:
            backtest_result.update({
                "Floor value": floor_history,
                "Peaks": peak_history,
                "Multiplier": self.m
            })

        return backtest_result


# ---------------------------------------------------------------------------------
# Random walks
# ---------------------------------------------------------------------------------
class RandomWalks:
    """
    Class for simulating random walks using Geometric Brownian Motion (GBM) and visualizing 
    the outcomes. This class includes methods for simulating stock prices and returns using GBM, 
    as well as methods for plotting the simulated paths.

    Methods include simulations based on percentage returns and log-prices, and visualization tools 
    for GBM and CPPI strategies.
    """

    @staticmethod
    def simulate_gbm_from_returns(n_years=10, n_scenarios=20, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0):
        """
        Simulates evolution of an initial stock price using Geometric Brownian Motion based on percentage returns.
        
        Args:
            n_years (int): Number of years to simulate.
            n_scenarios (int): Number of scenarios to simulate.
            mu (float): Expected return.
            sigma (float): Standard deviation of returns.
            periods_per_year (int): Number of periods per year.
            start (float): Starting price of the stock.

        Returns:
            tuple: DataFrames containing simulated prices and returns.
        """
        dt = 1 / periods_per_year
        n_steps = int(n_years * periods_per_year)
        
        rets = pd.DataFrame(np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios)))
        prices = ReturnAnalysis.compound_returns(rets, start=start)
        prices = UtilityMethods.insert_first_row_df(prices, start)
        
        return prices, rets

    @staticmethod
    def simulate_gbm_from_prices(n_years=10, n_scenarios=20, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0):
        """
        Simulates evolution of an initial stock price using Geometric Brownian Motion based on log-prices.

        Args:
            n_years, n_scenarios, mu, sigma, periods_per_year, start: Same as in simulate_gbm_from_returns.

        Returns:
            tuple: DataFrames containing simulated prices and returns.
        """
        dt = 1 / periods_per_year
        n_steps = int(n_years * periods_per_year)
        
        prices_dt = np.exp(np.random.normal(loc=(mu - 0.5*sigma**2)*dt, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios)))
        prices = start * pd.DataFrame(prices_dt).cumprod()
        prices = UtilityMethods.insert_first_row_df(prices, start)
        
        rets = ReturnAnalysis.compute_logreturns(prices).dropna()
        
        return prices, rets

    @staticmethod
    def show_gbm(n_years=10, n_scenarios=10, mu=0.05, sigma=0.15, periods_per_year=12, start=100):
        """
        Plots the evolution of stock prices generated by GBM.

        Args:
            n_years, n_scenarios, mu, sigma, periods_per_year, start: Same as in simulate_gbm_from_returns.

        Returns:
            matplotlib.axes.Axes: Axes object of the plot.
        """
        prices, _ = RandomWalks.simulate_gbm_from_returns(n_years, n_scenarios, mu, sigma, periods_per_year, start)
        ax = prices.plot(figsize=(12, 5), grid=True, legend=False, color="sandybrown", alpha=0.7, linewidth=2)
        ax.axhline(y=start, ls=":", color="black")
        xlab = "Months" if periods_per_year == 12 else "Weeks" if periods_per_year == 52 else "Days"
        ax.set_xlabel(xlab)
        ax.set_ylabel("Price")
        ax.set_title("Prices generated by GBM")
        return ax

    @staticmethod
    def show_cppi(n_years=10, n_scenarios=50, m=3, floor=0, mu=0.04, sigma=0.15, risk_free_rate=0.03, periods_per_year=12, start=100, ymax=100):
        """
        Simulates CPPI strategy using GBM-generated returns and plots the resulting CPPI wealths and a histogram.

        Args:
            n_years, n_scenarios, mu, sigma, risk_free_rate, periods_per_year, start, ymax: As in other methods.
            m (float): CPPI multiplier.
            floor (float): CPPI floor as a fraction of start value.

        Returns:
            tuple: matplotlib.axes.Axes objects for the two plots.
        """
        _, risky_rets = RandomWalks.simulate_gbm_from_returns(n_years, n_scenarios, mu, sigma, periods_per_year, start)
        cppiw = CPPIBacktest.cppi(risky_rets, start_value=start, floor=floor, m=m, drawdown=None, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)["CPPI wealth"]
        cppiw = UtilityMethods.insert_first_row_df(cppiw, {i: start for i in range(cppiw.shape[1])})

        fig, (wealth_ax, hist_ax) = plt.subplots(figsize=(20, 7), nrows=1, ncols=2, sharey=True, gridspec_kw={"width_ratios": [3, 2]})
        plt.subplots_adjust(wspace=0.005)
        simclr, floorclr, startclr = "sandybrown", "red", "black"
        ymax = (cppiw.values.max() - start) / 100 * ymax + start

        cppiw.plot(ax=wealth_ax, grid=True, legend=False, color=simclr, alpha=0.5, linewidth=2)
        wealth_ax.axhline(y=start, ls=":", color=startclr)
        wealth_ax.axhline(y=start*floor, ls=":", color=floorclr, linewidth=2)
        xlab = "Months" if periods_per_year == 12 else "Weeks" if periods_per_year == 52 else "Days"
        wealth_ax.set_xlabel(xlab)
        wealth_ax.set_ylim(top=ymax)
        wealth_ax.set_title("CPPI wealths due to brownian motion generated returns", fontsize=14)

        violations_per_scenarios = (cppiw < start*floor).sum()
        total_violations = violations_per_scenarios.sum()
        terminal_wealth = cppiw.iloc[-1]
        tw_mean, tw_median = terminal_wealth.mean(), terminal_wealth.median()
        failure_mask = np.less(terminal_wealth, start*floor)
        n_failures, p_fail = failure_mask.sum(), n_failures / n_scenarios
        e_shorfall = np.dot(terminal_wealth - start*floor, failure_mask) / n_failures if n_failures > 0.0 else 0.0

        terminal_wealth.hist(ax=hist_ax, bins=50, ec="white", fc=simclr, orientation="horizontal")
        hist_ax.axhline(y=start, ls=":", color=startclr)
        hist_ax.axhline(y=start*floor, ls=":", color=floorclr, linewidth=2)
        hist_ax.axhline(y=tw_mean, ls=":", color=simclr)
        hist_ax.axhline(y=tw_median, ls=":", color=simclr)
        hist_ax.annotate("Mean: ${:.2f}".format(tw_mean), xy=(0.5, 0.9), xycoords="axes fraction", fontsize=15)
        hist_ax.annotate("Median: ${:.2f}".format(tw_median), xy=(0.5, 0.85), xycoords="axes fraction", fontsize=15)
        if floor > 0.0:
            hist_ax.annotate("Violations (overall): {}".format(total_violations), xy=(0.5, 0.75), xycoords="axes fraction", fontsize=15)
            hist_ax.annotate("Violations (end period): {} ({:.1f}%)".format(n_failures, p_fail*100), xy=(0.5, 0.7), xycoords="axes fraction", fontsize=15)
            hist_ax.annotate("E(shortfall) (end period): ${:.2f}".format(e_shorfall), xy=(0.5, 0.65), xycoords="axes fraction", fontsize=15)
        hist_ax.set_title("Histogram of the CPPI wealth at the end of the period", fontsize=14)

        return wealth_ax, hist_ax


# ---------------------------------------------------------------------------------
# Securities 
# ---------------------------------------------------------------------------------
class Securities:
    """
    Class providing methods for calculations and simulations related to securities. 
    This includes discount bond pricing, present value calculations, funding ratio computations, 
    interest rate simulations using the CIR model, and bond-related calculations.
    """

    @staticmethod
    def discount(t, r):
        """
        Computes the price of a pure discount bond that pays 1 at time t.

        Args:
            t (pd.Series or list): Times at which the bond pays out.
            r (float or list): Interest rate or a list of interest rates.

        Returns:
            pd.DataFrame: Dataframe representing the discount bond prices.
        """
        if not isinstance(t, pd.Series):
            t = pd.Series(t)

        if not isinstance(r, list):
            r = [r]

        ds = pd.DataFrame([1 / (1 + rate) ** t for rate in r]).T
        ds.index = t
        return ds

    @staticmethod
    def present_value(L, r):
        """
        Computes the present value of a DataFrame of liabilities at a given interest rate.

        Args:
            L (pd.DataFrame): Dataframe of liabilities.
            r (float): Interest rate.

        Returns:
            float: The present value of the liabilities.
        """
        if not isinstance(L, pd.DataFrame):
            raise TypeError("Expected pd.DataFrame")

        dates = pd.Series(L.index)
        ds = Securities.discount(dates, r)  # Present values of future cashflows
        return (ds * L).sum()

    @staticmethod
    def funding_ratio(asset_value, liabilities, r):
        """
        Computes the funding ratio between the value of holding assets and the present value of the liabilities.

        Args:
            asset_value (float): The total value of the assets.
            liabilities (pd.DataFrame): Dataframe of liabilities.
            r (float or list): Interest rate or a list of interest rates.

        Returns:
            float: The funding ratio.
        """
        return asset_value / Securities.present_value(liabilities, r)

    @staticmethod
    def compounding_rate(r, periods_per_year=None):
        """
        Converts a nominal rate to a continuously compounded rate.

        Args:
            r (float): Nominal rate.
            periods_per_year (int, optional): Number of compounding periods per year. Defaults to None.

        Returns:
            float: Continuously compounded rate if periods_per_year is None, otherwise discrete compounded rate.
        """
        if periods_per_year is None:
            return np.exp(r) - 1
        else:
            return (1 + r / periods_per_year) ** periods_per_year - 1

    @staticmethod
    def compounding_rate_inv(R, periods_per_year=None):
        """
        Converts a compounded rate back to a nominal rate.

        Args:
            R (float): Compounded rate.
            periods_per_year (int, optional): Number of compounding periods per year. Defaults to None.

        Returns:
            float: Nominal rate.
        """
        if periods_per_year is None:
            return np.log(1 + R)
        else:
            return periods_per_year * ((1 + R) ** (1 / periods_per_year) - 1)

    @staticmethod
    def simulate_cir(n_years=10, n_scenarios=10, a=0.05, b=0.03, sigma=0.05, periods_per_year=12, r0=None):
        """
        Simulates the evolution of interest rates using the Cox-Ingersoll-Ross (CIR) model.

        Args:
            n_years (int, optional): Number of years for the simulation. Defaults to 10.
            n_scenarios (int, optional): Number of scenarios to simulate. Defaults to 10.
            a (float, optional): Speed of mean reversion. Defaults to 0.05.
            b (float, optional): Long-term mean interest rate. Defaults to 0.03.
            sigma (float, optional): Volatility of interest rate. Defaults to 0.05.
            periods_per_year (int, optional): Number of periods per year. Defaults to 12.
            r0 (float, optional): Initial interest rate. Defaults to None, which means using b as initial rate.

        Returns:
            tuple of pd.DataFrame: Dataframes containing simulated interest rates and zero-coupon bond prices.
        """
        if r0 is None:
            r0 = b

        dt = 1 / periods_per_year
        n_steps = int(n_years * periods_per_year) + 1
        r0 = Securities.compounding_rate_inv(r0)

        shock = np.random.normal(loc=0, scale=(dt)**0.5, size=(n_steps, n_scenarios))
        rates = np.zeros_like(shock)
        rates[0] = r0
        zcb_prices = np.zeros_like(shock)
        h = np.sqrt(a**2 + 2*sigma**2)
        zcb_prices[0] = Securities._zcb_price(n_years, r0, h, a, b, sigma)

        for step in range(1, n_steps):
            r_t = rates[step - 1]
            rates[step] = r_t + a * (b - r_t) + sigma * np.sqrt(r_t) * shock[step]
            zcb_prices[step] = Securities._zcb_price(n_years - dt * step, r_t, h, a, b, sigma)

        rates = pd.DataFrame(Securities.compounding_rate(rates))
        zcb_prices = pd.DataFrame(zcb_prices)

        return rates, zcb_prices

    @staticmethod
    def _zcb_price(ttm, r, h, a, b, sigma):
        """
        Helper method to compute the price of a zero-coupon bond using the CIR model.

        Args:
            ttm (float): Time to maturity.
            r (float): Interest rate.
            h, a, b, sigma: CIR model parameters.

        Returns:
            float: Price of the zero-coupon bond.
        """
        A = ((2 * h * np.exp(0.5 * (a + h) * ttm)) / (2 * h + (a + h) * (np.exp(h * ttm) - 1))) ** (2 * a * b / (sigma ** 2))
        B = (2 * (np.exp(h * ttm) - 1)) / (2 * h + (a + h) * (np.exp(h * ttm) - 1))
        return A * np.exp(-B * r)

    @staticmethod
    def bond_cash_flows(principal=100, maturity=10, coupon_rate=0.03, coupons_per_year=2):
        """
        Generates cash flows for a regular bond.

        Args:
            principal (float, optional): Principal amount of the bond. Defaults to 100.
            maturity (float, optional): Maturity of the bond in years. Defaults to 10.
            coupon_rate (float, optional): Coupon rate of the bond. Defaults to 0.03.
            coupons_per_year (int, optional): Number of coupons paid per year. Defaults to 2.

        Returns:
            pd.Series: Series of cash flows for the bond.
        """
        n_coupons = round(maturity * coupons_per_year)
        coupon_amount = (coupon_rate / coupons_per_year) * principal
        cash_flows = pd.DataFrame(coupon_amount, index=np.arange(1, n_coupons + 1), columns=[0])
        cash_flows.iloc[-1] = cash_flows.iloc[-1] + principal
        return cash_flows

    @staticmethod
    def bond_price(principal=100, maturity=10, coupon_rate=0.02, coupons_per_year=2, ytm=0.03, cf=None):
        """
        Returns the price of a regular coupon-bearing bond.

        Args:
            principal (float, optional): Principal amount of the bond. Defaults to 100.
            maturity (float, optional): Maturity of the bond in years. Defaults to 10.
            coupon_rate (float, optional): Coupon rate. Defaults to 0.02.
            coupons_per_year (int, optional): Number of coupons per year. Defaults to 2.
            ytm (float or pd.DataFrame, optional): Yield to maturity. Defaults to 0.03.
            cf (pd.DataFrame, optional): Cash flows. Defaults to None.

        Returns:
            float or pd.DataFrame: Price of the bond.
        """
        # single bond price 
        def single_price_bond(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year, ytm=ytm, cf=cf):
            if cf is None:            
                # compute the bond cash flow on the fly
                cf = Securities.bond_cash_flows(maturity=maturity, principal=principal, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year)             
            bond_price = Securities.present_value(cf, ytm/coupons_per_year)[0]
            return bond_price
        
        if isinstance(ytm,pd.Series):
            raise TypeError("Expected pd.DataFrame or a single value for ytm")

        if isinstance(ytm,pd.DataFrame):
            # ytm is a dataframe of rates for different scenarios 
            n_scenarios = ytm.shape[1]
            bond_price  = pd.DataFrame()
            # we have a for over each scenarios of rates (ytms)
            for i in range(n_scenarios):
                # for each scenario, a list comprehension computes bond prices according to ytms up to time maturity minus 1
                prices = [single_price_bond(principal=principal, maturity=maturity - t/coupons_per_year, coupon_rate=coupon_rate,
                                            coupons_per_year=coupons_per_year, ytm=y, cf=cf) for t, y in zip(ytm.index[:-1], ytm.iloc[:-1,i]) ] 
                bond_price = pd.concat([bond_price, pd.DataFrame(prices)], axis=1)
            # rename columns with scenarios
            bond_price.columns = ytm.columns
            # concatenate one last row with bond prices at maturity for each scenario
            bond_price = pd.concat([ bond_price, 
                                    pd.DataFrame( [[principal+principal*coupon_rate/coupons_per_year] * n_scenarios], index=[ytm.index[-1]]) ], 
                                    axis=0)
            return bond_price 
        else:
            # base case: ytm is a value and a single bond price is computed 
            return single_price_bond(principal=principal, maturity=maturity, coupon_rate=coupon_rate, 
                                    coupons_per_year=coupons_per_year, ytm=ytm, cf=cf)        

    def bond_returns(principal, bond_prices, coupon_rate, coupons_per_year, periods_per_year, maturity=None):
        '''
        Computes the total return of a coupon-paying bond. 
        The bond_prices can be a pd.DataFrame of bond prices for different ytms and scenarios 
        as well as a single bond price for a fixed ytm. 
        In the first case, remind to annualize the computed returns.
        In the latter case, the maturity of the bond has to passed since cash-flows needs to be recovered. 
        Moreover, the computed return does not have to be annualized.
        '''
        if isinstance(bond_prices, pd.DataFrame):
            coupons = pd.DataFrame(data=0, index=bond_prices.index, columns=bond_prices.columns)
            last_date = bond_prices.index.max()
            pay_date = np.linspace(periods_per_year/coupons_per_year, last_date, int(coupons_per_year*last_date/periods_per_year), dtype=int  )
            coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
            tot_return = (bond_prices + coupons)/bond_prices.shift(1) - 1 
            return tot_return.dropna()
        else:
            cf = Securities.bond_cash_flows(principal=principal, maturity=maturity, coupon_rate=coupon_rate, coupons_per_year=coupons_per_year) 
            tot_return = ( cf.sum() / bond_prices )**(1/maturity) - 1
            return tot_return[0]

    def mac_duration(cash_flows, discount_rate):
        '''
        Computed the Macaulay duration of an asset involving regular cash flows a given discount rate
        Note that if the cash_flows dates are normalized, then the discount_rate is simply the YTM. 
        Otherwise, it has to be the YTM divided by the coupons per years.
        '''
        if not isinstance(cash_flows,pd.DataFrame):
            raise ValueError("Expected a pd.DataFrame of cash_flows")

        dates = cash_flows.index

        # present value of single cash flows (discounted cash flows)
        discount_cf = Securities.discount.discount( dates, discount_rate ) * cash_flows
        
        # weights: the present value of the entire payment, i.e., discount_cf.sum() is equal to the principal 
        weights = discount_cf / discount_cf.sum()
        
        # sum of weights * dates
        return ( weights * pd.DataFrame(dates,index=weights.index) ).sum()[0]


# ---------------------------------------------------------------------------------
# Liability driven strategies 
# ---------------------------------------------------------------------------------
class LiabilityDrivenStrategies:
    """
    Class for implementing Liability-Driven Investment (LDI) strategies. 
    Includes various allocation strategies to balance between PSP (Pension Surplus Portfolios) 
    and LHP (Liability Hedging Portfolios), considering factors like floors, drawdowns, and glide paths.
    """

    @staticmethod
    def ldi_mixer(psp_rets, lhp_rets, allocator, **kwargs):
        """
        Combines PSP and LHP asset returns based on a specified allocation strategy.

        Args:
            psp_rets (pd.DataFrame): Returns of PSP assets.
            lhp_rets (pd.DataFrame): Returns of LHP assets.
            allocator (function): Allocator function to determine weights.
            **kwargs: Additional arguments for the allocator function.

        Returns:
            pd.DataFrame: Weighted average of returns based on the allocation strategy.
        """
        # ... [Rest of the code]

    @staticmethod
    def ldi_fixed_allocator(psp_rets, lhp_rets, w1, **kwargs):
        """
        Allocator for a fixed-mix strategy between PSP and LHP asset returns.

        Args:
            psp_rets (pd.DataFrame): Returns of PSP assets.
            lhp_rets (pd.DataFrame): Returns of LHP assets.
            w1 (float): Fixed weight in PSP assets.

        Returns:
            pd.DataFrame: DataFrame consisting of the fixed weight w1.
        """
        # ... [Rest of the code]

    @staticmethod
    def ldi_glidepath_allocator(psp_rets, lhp_rets, start=1, end=0):
        """
        Allocator for a glide path fixed-mix strategy between PSP and LHP asset returns.

        Args:
            psp_rets (pd.DataFrame): Returns of PSP assets.
            lhp_rets (pd.DataFrame): Returns of LHP assets.
            start (float): Starting weight in PSP assets.
            end (float): Ending weight in PSP assets.

        Returns:
            pd.DataFrame: DataFrame of linearly spaced weights from start to end.
        """
        # ... [Rest of the code]

    @staticmethod
    def ldi_floor_allocator(psp_rets, lhp_rets, zcb_price, floor, m=3):
        """
        Allocator using a CPPI-style dynamic risk budgeting algorithm with a floor constraint.

        Args:
            psp_rets (pd.DataFrame): Returns of PSP assets.
            lhp_rets (pd.DataFrame): Returns of LHP assets.
            zcb_price (pd.DataFrame): Prices of Zero-Coupon Bonds.
            floor (float): Floor value as a fraction of the initial asset value.
            m (float): Multiplier for cushion in PSP.

        Returns:
            pd.DataFrame: DataFrame containing weights in PSP.
        """
        # ... [Rest of the code]

    @staticmethod
    def ldi_drawdown_allocator(psp_rets, lhp_rets, maxdd=0.2):
        """
        Allocator using a CPPI-style dynamic risk budgeting algorithm with a drawdown constraint.

        Args:
            psp_rets (pd.DataFrame): Returns of PSP assets.
            lhp_rets (pd.DataFrame): Returns of LHP assets.
            maxdd (float): Maximum drawdown constraint as a fraction.

        Returns:
            pd.DataFrame: DataFrame containing weights in PSP.
        """
        if not psp_rets.shape == lhp_rets.shape:
            raise ValueError("PSP and LHP returns must have the same shape")
            
        # define the multipler as the inverse of the maximum drawdown
        m = 1 / maxdd
        dates, n_scenarios = psp_rets.shape
        account_value  = np.repeat(1,n_scenarios)
        floor_value    = np.repeat(1,n_scenarios)
        peak_value     = np.repeat(1,n_scenarios)
        weight_history = pd.DataFrame(index=psp_rets.index, columns=psp_rets.columns)
        
        for date in range(dates):
            floor_value = (1 - maxdd)*peak_value
            cushion = (account_value - floor_value) / account_value
            # weights in the PSP and LHP 
            psp_w = (m * cushion).clip(0,1)
            lhp_w = 1 - psp_w
            # update
            account_value = psp_w*account_value*(1 + psp_rets.iloc[date]) + lhp_w*account_value*(1 + lhp_rets.iloc[date])
            peak_value = np.maximum(peak_value, account_value)
            weight_history.iloc[date] = psp_w
        return weight_history 


# ---------------------------------------------------------------------------------
# Factor and Style analysis 
# ---------------------------------------------------------------------------------
class FactorStyleAnalysis:
    """
    Class for conducting factor and style analysis in finance. 
    Includes methods for linear regression, CAPM beta calculations, 
    tracking error calculation, and Sharpe style analysis optimization.
    """

    @staticmethod
    def linear_regression(dep_var, exp_vars, alpha=True):
        """
        Performs a linear regression to decompose the dependent variable into the explanatory variables.

        Args:
            dep_var (pd.Series or np.array): Dependent variable.
            exp_vars (pd.DataFrame or np.array): Explanatory variables.
            alpha (bool): If True, includes an intercept in the regression.

        Returns:
            RegressionResults: Results object from statsmodels OLS regression.
        """
        if alpha:
            # the OLS methods assume a bias equal to 0, hence a specific variable for the bias has to be given
            if isinstance(exp_vars,pd.DataFrame):
                exp_vars = exp_vars.copy()
                exp_vars["Alpha"] = 1
            else:
                exp_vars = np.concatenate( (exp_vars, np.ones((exp_vars.shape[0],1))), axis=1 )
        return sm.OLS(dep_var, exp_vars).fit()

    @staticmethod
    def capm_betas(ri, rm):
        """
        Computes CAPM betas for each asset given market returns.

        Args:
            ri (pd.DataFrame): Returns of individual assets.
            rm (pd.Series or pd.DataFrame): Market returns.

        Returns:
            pd.Series: CAPM beta values for each asset.
        """
        market_var = ( rm.std()**2 )[0]
        betas = []
        for name in ri.columns:
            cov_im = pd.concat( [ri[name],rm], axis=1).cov().iloc[0,1]
            betas.append( cov_im / market_var )
        return pd.Series(betas, index=ri.columns)

    @staticmethod
    def tracking_error(r_a, r_b):
        """
        Calculates the tracking error between two return series.

        Args:
            r_a (pd.Series or pd.DataFrame): First return series.
            r_b (pd.Series or pd.DataFrame): Second return series.

        Returns:
            float: Tracking error between the two return series.
        """
        return ( ((r_a - r_b)**2).sum() )**(0.5)

    @staticmethod
    def style_analysis_tracking_error(weights, ref_r, bb_r):
        """
        Objective function for Sharpe style analysis.

        Args:
            weights (np.array): Portfolio weights.
            ref_r (pd.Series): Reference returns.
            bb_r (pd.DataFrame): Building block returns.

        Returns:
            float: Tracking error for the style analysis optimization.
        """
        return FactorStyleAnalysis.tracking_error(ref_r, (weights*bb_r).sum(axis=1))

    @staticmethod
    def style_analysis(dep_var, exp_vars):
        """
        Conducts Sharpe style analysis optimization.

        Args:
            dep_var (pd.Series or pd.DataFrame): Dependent variable (asset or portfolio returns).
            exp_vars (pd.DataFrame): Explanatory variables (potential style or factor returns).

        Returns:
            pd.Series: Optimal weights that minimize the tracking error.
        """
        # dep_var is expected to be a pd.Series
        if isinstance(dep_var,pd.DataFrame):
            dep_var = dep_var[dep_var.columns[0]]
        
        n = exp_vars.shape[1]
        init_guess = np.repeat(1/n, n)
        weights_const = {
            'type': 'eq',
            'fun': lambda weights: 1 - np.sum(weights)
        }
        solution = minimize(FactorStyleAnalysis.style_analysis_tracking_error, 
                            init_guess,
                            method='SLSQP',
                            options={'disp': False},
                            args=(dep_var, exp_vars),
                            constraints=(weights_const,),
                            bounds=((0.0, 1.0),)*n)
        weights = pd.Series(solution.x, index=exp_vars.columns)
        return weights


# ---------------------------------------------------------------------------------
# Covariance matrix estimators
# ---------------------------------------------------------------------------------
class CovarianceEstimators:
    """
    Class for various covariance matrix estimators including sample covariance, 
    constant correlation model, and shrinkage estimator.
    """

    @staticmethod
    def sample_cov(r, **kwargs):
        """
        Computes the sample covariance matrix of a series of returns.

        Args:
            r (pd.DataFrame): DataFrame containing return series.

        Returns:
            pd.DataFrame: Sample covariance matrix.
        """
        if not isinstance(r, pd.DataFrame):
            raise ValueError("Expected r to be a pd.DataFrame of returns series")
        return r.cov()

    @staticmethod
    def cc_cov(r, **kwargs):
        """
        Estimates a covariance matrix using the Elton/Gruber Constant Correlation model.

        Args:
            r (pd.DataFrame): DataFrame containing return series.

        Returns:
            pd.DataFrame: Constant correlation covariance matrix.
        """
        # correlation coefficents  
        rhos = r.corr()
        n = rhos.shape[0]
        # compute the mean correlation: since the matrix rhos is a symmetric with diagonals all 1, 
        # the mean correlation can be computed by:
        mean_rho = (rhos.values.sum() - n) / (n**2-n) 
        # create the constant correlation matrix containing 1 on the diagonal and the mean correlation outside
        ccor = np.full_like(rhos, mean_rho)
        np.fill_diagonal(ccor, 1.)
        # create the new covariance matrix by multiplying mean_rho*std_i*std_i 
        # the product of the stds is done via np.outer
        ccov = ccor * np.outer(r.std(), r.std())
        return pd.DataFrame(ccov, index=r.columns, columns=r.columns)

    @staticmethod
    def shrinkage_cov(r, delta=0.5, **kwargs):
        """
        Computes a shrinkage estimator that combines the sample covariance matrix 
        with the constant correlation covariance matrix.

        Args:
            r (pd.DataFrame): DataFrame containing return series.
            delta (float): Shrinkage parameter, a value between 0 and 1.

        Returns:
            pd.DataFrame: Shrinkage covariance matrix.
        """
        samp_cov  = CovarianceEstimators.sample_cov(r, **kwargs)
        const_cov = CovarianceEstimators.cc_cov(r, **kwargs)
        return delta*const_cov + (1-delta)*samp_cov


# ---------------------------------------------------------------------------------
# Back-test weigthing schemes
# ---------------------------------------------------------------------------------
class BacktestWeightingSchemes:
    """
    A class for backtesting various portfolio weighting schemes.

    This class includes methods to compute different portfolio weights such as equally-weighted,
    cap-weighted, risk parity, minimum volatility, and maximum Sharpe ratio portfolios.
    It also provides a method for backtesting a given weighting scheme over a rolling window.
    """

    @staticmethod
    def weight_ew(r, cap_ws=None, max_cw_mult=None, microcap_thr=None, **kwargs):
        """
        Returns the weights of an Equally-Weighted (EW) portfolio based on asset returns.
        Optionally applies a cap-weight tether and removes microcaps.

        Args:
            r: DataFrame of asset returns.
            cap_ws: Optional DataFrame of market capitalizations to modify the EW scheme.
            max_cw_mult: Optional cap on the weight relative to the cap-weight.
            microcap_thr: Optional threshold to exclude microcap stocks.

        Returns:
            Series of portfolio weights.
        """
        ew = pd.Series(1/len(r.columns), index=r.columns)
        if cap_ws is not None:
            cw = cap_ws.loc[r.index[0]]  # starting cap weight
            if microcap_thr is not None and microcap_thr > 0.0:
                ew[cw < microcap_thr] = 0
                ew = ew / ew.sum()
            if max_cw_mult is not None and max_cw_mult > 0:
                ew = np.minimum(ew, cw*max_cw_mult)
                ew = ew / ew.sum()
        return ew

    @staticmethod
    def weight_cw(r, cap_ws, **kwargs):
        """
        Returns the weights of a Cap-Weighted (CW) portfolio based on the time series of cap weights.

        Args:
            r: DataFrame of asset returns.
            cap_ws: DataFrame of market capitalizations.

        Returns:
            Series of portfolio weights.
        """
        return cap_ws.loc[r.index[0]]

    @staticmethod
    def weight_rp(r, cov_estimator=CovarianceEstimators.sample_cov, **kwargs):
        """
        Produces the weights of a risk parity portfolio given a covariance matrix of the returns.

        Args:
            r: DataFrame of asset returns.
            cov_estimator: Function to estimate the covariance matrix.

        Returns:
            Series of portfolio weights.
        """
        est_cov = cov_estimator(r)
        return ReturnAnalysis.risk_parity_weigths(est_cov)

    @staticmethod
    def backtest_weight_scheme(r, window=36, weight_scheme=weight_ew, **kwargs):
        """
        Backtests a given weighting scheme.

        Args:
            r: DataFrame of asset returns to build the portfolio.
            window: The rolling window used for the backtest.
            weight_scheme: Function representing the weighting scheme to use.

        Returns:
            Series of portfolio returns.
        """
        n_periods = r.shape[0]
        windows = [(start, start + window) for start in range(0, n_periods - window)]
        weights = [weight_scheme(r.iloc[win[0]:win[1]]) for win in windows]
        weights = pd.DataFrame(weights, index=r.iloc[window:].index, columns=r.columns)
        returns = (weights * r).sum(axis=1, min_count=1)
        return returns

    @staticmethod
    def annualize_vol_ewa(r, decay=0.95, periods_per_year=12):
        """
        Computes the annualized exponentially weighted average volatility of a series of returns.

        Args:
            r: DataFrame or Series of asset returns.
            decay: Decay factor for the exponential weighting.
            periods_per_year: Number of periods in a year.

        Returns:
            Annualized volatility.
        """
        N = r.shape[0]
        times = np.arange(1, N + 1)
        sq_errs = pd.DataFrame((r - r.mean())**2)
        weights = [decay**(N-t) for t in times] / np.sum(decay**(N-times))
        weights = pd.DataFrame(weights, index=r.index)
        vol_ewa = (weights * sq_errs).sum()**0.5
        ann_vol_ewa = vol_ewa[0] * np.sqrt(periods_per_year)
        return ann_vol_ewa

    @staticmethod
    def weight_minvar(r, cov_estimator=CovarianceEstimators.sample_cov, periods_per_year=12, **kwargs):
        """
        Produces the weights of the Minimum Volatility Portfolio given a covariance matrix of the returns.

        Args:
            r: DataFrame of asset returns.
            cov_estimator: Function to estimate the covariance matrix.
            periods_per_year: Number of periods in a year for annualization.

        Returns:
            Series of portfolio weights.
        """
        est_cov = cov_estimator(r)
        ann_ret = ReturnAnalysis.annualize_rets(r, periods_per_year)
        return ReturnAnalysis.minimize_volatility(ann_ret, est_cov)

    @staticmethod
    def weight_maxsharpe(r, cov_estimator=ReturnAnalysis.sample_cov, periods_per_year=12, risk_free_rate=0.03):
        """
        Produces the weights of the Maximum Sharpe Ratio Portfolio given a covariance matrix of the returns.

        Args:
            r: DataFrame of asset returns.
            cov_estimator: Function to estimate the covariance matrix.
            periods_per_year: Number of periods in a year for annualization.
            risk_free_rate: Risk-free rate for the Sharpe Ratio calculation.

        Returns:
            Series of portfolio weights.
        """
        est_cov = cov_estimator(r)
        ann_ret = ReturnAnalysis.annualize_rets(r, periods_per_year)
        return ReturnAnalysis.maximize_shape_ratio(ann_ret, est_cov, risk_free_rate, periods_per_year)
# ---------------------------------------------------------------------------------
# Black-Litterman model
# ---------------------------------------------------------------------------------
class BlackLittermanModel:
    """
    Implements the Black-Litterman model for combining investor views with the market equilibrium.

    This class includes methods for computing implied returns from market weights, constructing
    the Omega matrix in the absence of explicit view uncertainty, and the full Black-Litterman
    model for posterior expected returns and covariances.
    """

    @staticmethod
    def implied_returns(covmat, weights, delta=2.5):
        """
        Computes the implied expected returns using the Black-Litterman model.

        Args:
            covmat: N x N covariance matrix as pd.DataFrame.
            weights: N x 1 portfolio weights as pd.Series.
            delta: Risk aversion coefficient.

        Returns:
            Implied returns as pd.Series.
        """
        imp_rets = delta * covmat.dot(weights).squeeze() # to get a series from a 1-column dataframe
        imp_rets.name = 'Implied Returns'
        return imp_rets

    @staticmethod
    def omega_uncertain_prior(covmat, tau, P):
        """
        Computes the He-Litterman Omega matrix in cases where investor view uncertainty is not explicit.

        Args:
            covmat: N x N covariance matrix as pd.DataFrame.
            tau: Scalar denoting the uncertainty of the CAPM prior.
            P: K x N projection matrix of asset weights for views as pd.DataFrame.

        Returns:
            K x K Omega matrix as pd.DataFrame, representing prior uncertainties.
        """
        he_lit_omega = P.dot(tau * covmat).dot(P.T)
        return pd.DataFrame(np.diag(np.diag(he_lit_omega.values)), index=P.index, columns=P.index)

    @staticmethod
    def black_litterman(w_prior, Sigma_prior, P, Q, Omega=None, delta=2.5, tau=0.02):
        """
        Computes the Black-Litterman posterior expected returns and covariances.

        Args:
            w_prior: N x 1 pd.Series of prior weights.
            Sigma_prior: N x N covariance matrix as pd.DataFrame.
            P: K x N projection matrix of asset weights for views, a pd.DataFrame.
            Q: K x 1 pd.Series of views.
            Omega: K x K matrix as pd.DataFrame representing the uncertainty of views (optional).
            delta: Risk aversion coefficient.
            tau: Uncertainty of the CAPM prior.

        Returns:
            Tuple of posterior expected returns (mu_bl) and covariance matrix (sigma_bl).
        """
        if Omega is None:
            Omega = BlackLittermanModel.omega_uncertain_prior(Sigma_prior, tau, P)
        
        N = w_prior.shape[0]
        K = Q.shape[0]
        
        Pi = BlackLittermanModel.implied_returns(Sigma_prior, w_prior, delta)
        
        invmat = inv(P.dot(tau * Sigma_prior).dot(P.T) + Omega)
        mu_bl = Pi + (tau * Sigma_prior).dot(P.T).dot(invmat.dot(Q - P.dot(Pi).values))
        sigma_bl = Sigma_prior + (tau * Sigma_prior) - (tau * Sigma_prior).dot(P.T).dot(invmat).dot(P).dot(tau * Sigma_prior)
        
        return (mu_bl, sigma_bl)
    

# ---------------------------------------------------------------------------------
# Risk contributions analysis 
# ---------------------------------------------------------------------------------
class RiskContributionAnalysis:
    """
    Class for analyzing and optimizing risk contributions in a portfolio.

    Includes methods for computing the Effective Number of Constituents (ENC),
    Effective Number of Correlated Bets (ENCB), individual asset risk contributions,
    and optimization functions for risk parity.
    """

    @staticmethod
    def enc(weights):
        """
        Computes the Effective Number of Constituents (ENC) of a portfolio.

        Args:
            weights: Vector of portfolio weights as pd.Series or np.array.

        Returns:
            ENC value as a float.
        """
        return (weights**2).sum()**(-1)

    @staticmethod
    def encb(risk_contrib):
        """
        Computes the Effective Number of Correlated Bets (ENCB) of a portfolio.

        Args:
            risk_contrib: Vector of portfolio risk contributions as pd.Series or np.array.

        Returns:
            ENCB value as a float.
        """
        return (risk_contrib**2).sum()**(-1)

    @staticmethod
    def portfolio_risk_contributions(weights, matcov):
        """
        Computes the risk contributions of each asset in a portfolio.

        Args:
            weights: Portfolio weights as pd.Series or np.array.
            matcov: Covariance matrix of asset returns as pd.DataFrame.

        Returns:
            Vector of risk contributions as pd.Series.
        """
        portfolio_var = RiskContributionAnalysis.portfolio_volatility(weights, matcov)**2
        marginal_contrib = matcov @ weights
        risk_contrib = np.multiply(marginal_contrib, weights.T) / portfolio_var
        return risk_contrib

    @staticmethod
    def msd_risk_contrib(weights, target_risk, mat_cov):
        """
        Mean Squared Difference between target and actual risk contributions.

        Args:
            weights: Portfolio weights as np.array.
            target_risk: Target risk contributions as np.array.
            mat_cov: Covariance matrix as pd.DataFrame.

        Returns:
            Mean squared difference value as float.
        """
        w_risk_contribs = RiskContributionAnalysis.portfolio_risk_contributions(weights, mat_cov)
        msd = (w_risk_contribs - target_risk)**2 
        return msd.sum()

    @staticmethod
    def portfolio_risk_contrib_optimizer(target_risk_contrib, mat_cov):
        """
        Optimizes portfolio weights to match target risk contributions.

        Args:
            target_risk_contrib: Target risk contributions as np.array or pd.Series.
            mat_cov: Covariance matrix as pd.DataFrame.

        Returns:
            Optimized portfolio weights as np.array.
        """
        n = mat_cov.shape[0]
        init_guess = np.repeat(1/n, n)
        weights_sum_to_one = {'type': 'eq', 'fun': lambda weights: 1 - np.sum(weights)}
        
        weights = minimize(RiskContributionAnalysis.msd_risk_contrib, 
                           init_guess,
                           args=(target_risk_contrib, mat_cov), 
                           method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_one,),
                           bounds=((0.0, 1.0),)*n )
        return weights.x

    @staticmethod
    def risk_parity_weights(mat_cov):
        """
        Computes the risk parity portfolio weights.

        Args:
            mat_cov: Covariance matrix as pd.DataFrame.

        Returns:
            Risk parity portfolio weights as pd.Series.
        """
        n = mat_cov.shape[0]
        weights = RiskContributionAnalysis.portfolio_risk_contrib_optimizer(target_risk_contrib=np.repeat(1/n,n), mat_cov=mat_cov)
        return pd.Series(weights, index=mat_cov.index)


# ---------------------------------------------------------------------------------
# Auxiliary methods 
# ---------------------------------------------------------------------------------
class UtilityMethods:
    """
    Class containing auxiliary methods for various mathematical and dataframe manipulations. 
    These methods are designed to assist in financial modeling and data processing tasks.
    """

    @staticmethod
    def as_colvec(x):
        """
        Converts an input array to a column vector.

        Args:
            x (np.array or np.matrix): Input array or matrix.

        Returns:
            np.matrix: A column vector representation of the input.
        """
        if x.ndim == 2:
            return x
        else:
            return np.expand_dims(x, axis=1)

    @staticmethod
    def inverse_df(d):
        """
        Calculates the inverse of a pandas DataFrame.

        Args:
            d (pd.DataFrame): The DataFrame to be inverted.

        Returns:
            pd.DataFrame: The inverse of the input DataFrame.
        """
        return pd.DataFrame(np.linalg.inv(d.values), index=d.columns, columns=d.index)

    @staticmethod
    def insert_first_row_df(df, row):
        """
        Inserts a row at the beginning of a DataFrame and shifts existing rows down.

        Args:
            df (pd.DataFrame): The DataFrame to be modified.
            row (single element or dict): The row to be inserted. Single element for 1-column DataFrame or dictionary for multi-column DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the row inserted at the beginning.
        """
        df.loc[-1] = row
        df.index = df.index + 1
        return df.sort_index()
