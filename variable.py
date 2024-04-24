import numpy as np
exchange = "NS" # define thte exchange, NS -> nse india, NASDAQ, NYSE, AMEX
tickers = ["RELIANCE","ADANIPORTS","TATASTEEL","YESBANK"] # add your tickers here available options in Publicly_Traded_Companies_US_NYSE_NASADAQ_AMEX.xlsx and StocksTraded.csv
if len(tickers)==0:
    choose_random = 1
    no_tickers = 10
else:
    choose_random = 0
    no_tickers = len(tickers)
period =3