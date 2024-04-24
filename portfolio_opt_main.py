import yfinance as yf
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from variable import *


def get_historical_data(symbol, start_date, end_date,exchange):
    if exchange=="NS":
        symbol = symbol+f".{exchange}"
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)


def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

def OptimizePortfolio(adj_CPdf):
    tickers = adj_CPdf.columns.tolist()
    risk_free_rate = 0.02
    log_returns = np.log(adj_CPdf/adj_CPdf.shift(1))
    log_returns = log_returns.dropna()
    cov_matrix = log_returns.cov() * 252
    risk_free_rate = .02
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(tickers))]
    initial_weights = np.array([1/len(tickers)]*len(tickers))
    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
    optimal_weights = optimized_results.x
    weights_json = {}
    print("Optimal Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        weights_json[ticker] = round(weight,10)
        print(f"{ticker}: {weight:.4f}")

    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
    print(f"Expected Annual Return: {optimal_portfolio_return*100:.4f}")
    print(f"Expected Volatility: {optimal_portfolio_volatility*100:.4f}")
    print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

    final_output_format = {
        "portfolio_weights":weights_json,
        "optimal_expected_return":optimal_portfolio_return,
        "optimal_expected_volatility":optimal_portfolio_volatility,
        "sharpe_ratio":optimal_sharpe_ratio
    }
    return final_output_format

if __name__ =="__main__":
    end_date = datetime.today()
    start_date = end_date-timedelta(days=365*period)
    if exchange=="NS":
        df_master = pd.read_csv("/Users/nayanchoudhary/Documents/Portfolio Optimization/StocksTraded.csv")
        ticker_col = "Symbol "
        company_col = None
    else:
        df_master = pd.read_excel("/Users/nayanchoudhary/Documents/Portfolio Optimization/Publicly_Traded_Companies_US_NYSE_NASADAQ_AMEX.xlsx")
        ticker_col = "symbol"
        company_col = "name"
    if choose_random:
        df_master = df_master.sample(no_tickers)
        tickers = df_master[ticker_col].tolist()
        
    adj_CPdf = pd.DataFrame()
    for ticker in tickers:
        ticker_data = get_historical_data(ticker,start_date,end_date, exchange)
        adj_CPdf[ticker] = ticker_data['Adj Close']
    adj_CPdf.dropna(axis=1, how='all', inplace=True)
    final_output_format = OptimizePortfolio(adj_CPdf)
    
