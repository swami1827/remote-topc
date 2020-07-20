import pandas as pd
def drawdown(return_series:pd.Series):
    """
    Takes in a time series of asset returns
    computes and returns a dataframe consisting of:
    wealth index
    previous peaks
    percentage drawdowns
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdown= (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "drawdowns": drawdown
    })

def get_ffme_returns():
    """
    Load the Fama French data set for the returns of top and botttom deciles by market cap
    """
    me_m=pd.read_csv("info/Imp_data.csv",
           header=0,index_col=0,parse_dates=True,na_values=-99.99
           )  #monthly returns based on market equity.
    rets= me_m[['Lo 10','Hi 10']]
    rets.columns=['SmallCap','LargeCap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index,format="%Y%m")
    return rets
def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted monthly returns.
    """
    ind=pd.read_csv("/home/mukund/Downloads/JupyterWorkflow-master/ind30_m_vw_rets.csv",header=0,index_col=0,parse_dates=True)/100
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    """
    ind=pd.read_csv("/home/mukund/Downloads/JupyterWorkflow-master/ind30_m_size.csv",header=0,index_col=0,parse_dates=True)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    """
    ind=pd.read_csv("/home/mukund/Downloads/JupyterWorkflow-master/ind30_m_nfirms.csv",header=0,index_col=0,parse_dates=True)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind    

    
def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    
    """
    hfi=pd.read_csv("/home/mukund/Downloads/JupyterWorkflow-master/edhec-hedgefundindices.csv",
                   header=0,index_col=0,parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def semideviation(r):
    """
    Computes semideviation/aka negative semideviation of r.
    r must be a series or DataFrame
    """
    is_negative= r<0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative to skipy.stats.skew()(direct method)
    Computes the skewness of the given DataFrame or supplied series.
    Returns float or series.
    """
    demeaned_r= r-r.mean()
    # use the popular standard deviation, so setting ddof=0(degree of freedom)
    sigma_r=r.std(ddof=0)
    exp= (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to skipy.stats.kurtosis()(direct method)
    Computes the kurtosis of the given DataFrame or supplied series.
    Returns float or series.
    """
    demeaned_r= r-r.mean()
    # use the popular standard deviation, so setting ddof=0(degree of freedom)
    sigma_r=r.std(ddof=0)
    exp= (demeaned_r**4).mean()
    return exp/sigma_r**4

import scipy.stats
def is_normal(r,level=0.01):
    """
    Aplies Jarque bera test to verify if a series is normal or not. 
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise.
    
    """
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value>level

import numpy as np
def var_historic(r, level=5):
    """
    VaR historic
    """
    if isinstance(r, pd.DataFrame):
       return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
       return -np.percentile(r,level) #since we want positive VaR
    else:
        raise TypeError("Expected r to be series or DataFrame")
        

from scipy.stats import norm
def var_gaussian(r, level=5,modified=False):
    """
    Returns the parametric gaussian VaR of a series or DataFrame
    If modified is "True", modified VaR is returned, using the Cornish-Fisher modification.
    """
    #Compute z-score assuming it is gaussian.
    z=norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis.
        s=skewness(r)
        k=kurtosis(r)
        z=(z+
              (z**2-1)*s/6+
              (z**3-3*z)*(k-3)/24-
               (2*z**3-5*z)*(s**2)/36
          )
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r,level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r,pd.Series):
        is_beyond= r<= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic,level=level)
    else:
        raise TypeError("Expected r to be a series or DataFrame")

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year.
    """
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r,periods_per_year):
    """
    Annualizes the volatility of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r,riskfree_rate,periods_per_year):
    """
    Calculates annualized sharpe ratio for a set of returns
    """
    #Convert annual risk free rate to per period
    rf_per_period=(1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret= annualize_rets(excess_ret,periods_per_year)
    ann_vol=annualize_vol(r,periods_per_year)
    return ann_ex_ret/ann_vol   

def portfolio_return(weights,returns):
    """
    Weights-> Returns
    """
    return weights.T @ returns #@symbolizes matrix multiplication. 

def portfolio_vol(weights,covmat):
    """
    Weights->volatility
    """
    return (weights.T @ covmat @ weights)**0.5   

def plot_ef2(n_points,er,cov):
    """
    Plots the 2 asset coefficient frontier
    """
    if er.shape[0] !=2 or er.shape[0] !=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights= [np.array([w, 1-w]) for w in np.linspace(0,1,n_points)]
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov)for w in weights]
    ef= pd.DataFrame({"Returns":rets,"Volatility":vols})
    return ef.plot.line(x="Volatility",y="Returns",style=".-")

import numpy as np
from scipy.optimize import minimize
def minimize_vol(target_return,er,cov):
    """
    will use the target_return to get the weights.
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    return_is_target={
        'type':'eq',
        'args':(er,),
        'fun': lambda weights,er: target_return - portfolio_return(weights,er)}
    weights_sum_to_1= {
         'type':'eq',
         'fun': lambda weights: np.sum(weights)-1
         }
    results = minimize(portfolio_vol,init_guess,
                       args=(cov,), method="SLSQP",
                       options={'disp':False},
                       constraints=(return_is_target,weights_sum_to_1),
                       bounds=bounds   
                       )
    return results.x  

def gmv(cov):
    """
    Returns the Global Minimum Variance Portfolio given the covariance matrix.
    """
    n=cov.shape[0]
    return msr(0,np.repeat(1,n),cov)

def optimal_weights(n_points,er,cov):
    """
    -> List of weights to run the optimizer to minimize the volatility
    """
    target_rs = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights
    
def plot_ef(n_points,er,cov,show_cml=False,style=".-",riskfree_rate=0,show_ew=False,show_gmv=False):

    """
    Plots the multi-asset coefficient frontier.
    """
    
    weights= optimal_weights(n_points,er,cov)
    rets= [portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov)for w in weights]
    ef= pd.DataFrame({"Returns":rets,"Volatility":vols})
    ax= ef.plot.line(x="Volatility",y="Returns",style=".-")
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew,er)
        vol_ew=portfolio_vol(w_ew,cov)
        #display EW
        ax.plot([vol_ew],[r_ew],color="goldenrod",marker="o",markersize=8)

    if show_gmv:
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv,er)
        vol_gmv=portfolio_vol(w_gmv,cov)
        #Display GMV
        ax.plot([vol_gmv],[r_gmv],color="midnightblue",marker="o",markersize=12)

    if show_cml:
      ax.set_xlim(left=0)
      riskfree_rate=0.1
      w_msr= msr(riskfree_rate,er,cov)
      r_msr= portfolio_return(w_msr,er)
      vol_msr= portfolio_vol(w_msr,cov)
      # Drawing cml-Capital market line.
      cml_x=(0,vol_msr)
      cml_y=(riskfree_rate,r_msr)
      ax.plot(cml_x,cml_y,color="Purple",marker="o",linestyle="dashed",markersize=12,linewidth=2)

    return ax

def msr(riskfree_rate,er,cov):
    """
    computing max sharpe ratio. 
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1= {
         'type':'eq',
         'fun': lambda weights: np.sum(weights)-1
         }
    def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
        """
        Returns negative of sharpe ratio given weights
        """
        r=portfolio_return(weights,er)
        vol=portfolio_vol(weights,cov)
        return -(r - riskfree_rate)/vol

    results = minimize(neg_sharpe_ratio,init_guess,
                       args=(riskfree_rate,er,cov,), method="SLSQP",
                       options={'disp':False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds   
                       )
    return results.x  


def get_total_market_index_returns():
    ind_size= get_ind_size()
    ind_nfirms= get_ind_nfirms()
    ind_returns= get_ind_returns()
    ind_mktcap=ind_nfirms*ind_size
    total_mktcap=ind_mktcap.sum(axis="columns")
    ind_capweight=ind_mktcap.divide(total_mktcap,axis="rows")
    total_market_return=(ind_capweight*ind_returns).sum(axis="columns")
    return total_market_return

def run_cppi(risky_r,safe_r=None,m=3,start=1000,floor=0.8,riskfree_rate=0.03,drawdown=None):
    """
    Run a backtest of the CPPI strategy,given a set of returns for the risky asset.
    Returns a dictionary containing: Asset Value History, Risk Budget History, risky weight history.
    """
    #Set the CPPI parameters.
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=start*floor
    peak=start
    if isinstance(risky_r,pd.Series):
        risky_r=pd.DataFrame(risky_r,columns=["R"])
        #Here, even if we are given a series, to make it work, we initialize risky returns as data frame.
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=riskfree_rate/12# fast way to set all values to a number.

    #Set up some data frames to save all intermediate values.
    account_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    risky_w_history=pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None: #doing this to make floor value dynamic.
            peak=np.maximum(peak,account_value)
            floor_value=peak*(1-drawdown)
        cushion=(account_value-floor_value)/account_value
        risky_w =m*cushion
        risky_w = np.minimum(risky_w,1)
        risky_w = np.maximum(risky_w,0)
        safe_w=1- risky_w
        risky_alloc=account_value*risky_w
        safe_alloc=account_value*safe_w
        #Update the account value for this time step.
        account_value=risky_alloc*(1+risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
        # Save the values for later reference.
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value

    risky_wealth=start*(1+risky_r).cumprod()        
    
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "multiplier m":m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }


    return backtest_result

def summary_stats(r,riskfree_rate=0.03):
    """
    Returns a DataFrame that contains aggregated summary stats for the returns in the columns of r.
    """
    ann_r=r.aggregate(annualize_rets,periods_per_year=12)
    ann_vol=r.aggregate(annualize_vol,periods_per_year=12)
    ann_sr=r.aggregate(sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_year=12)
    dd=r.aggregate(lambda r:drawdown(r).drawdowns.min())
    skew=r.aggregate(skewness)
    kurt=r.aggregate(kurtosis)
    cf_var5= r.aggregate(var_gaussian, modified=True)
    hist_cvar5= r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR(5%)": cf_var5,
        "Historic CVaR(5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown":dd
    })    

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

    