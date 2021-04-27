import pandas as pd

# This module contains functions to read in the .csv files containing the relevant data

def load_votingdata(fname):
    votes = pd.read_csv(fname)
    votes['Meeting Date'] = pd.to_datetime(votes['Meeting Date'], dayfirst=True)
    return votes

def load_engagements(fname):
    engagements = pd.read_csv(fname)
    return engagements

def load_sp500constituents(fname):
    data = pd.read_csv(fname)
    data.index = data.co_tic
    data = data.drop(columns=['gvkey', 'gvkeyx', 'conm', 'tic', 'co_conm', 'co_tic'])
    # values missing from S&P constituent dataset downloaded from WRDS
    missing = ['ACE', 'ADT', 'ALTR', 'CBG', 'CCE', 'COH', 'CSC', 'CVC', 'DTV', 'DV', 'ESV', 'GAS', 'GMCR', 'HCN',
     'HNZ', 'HRS', 'IGT', 'KORS', 'LIFE', 'LSI', 'LUK', 'MHFI', 'NWS', 'PCLN', 'PLL', 'PX', 'SE', 'SLM', 'SUN', 'TEG',
     'TMK', 'TSO', 'UA', 'WIN', 'WYN', 'YHOO']
    data['from'] = pd.to_datetime(data['from'], dayfirst=True)
    data['thru'] = pd.to_datetime(data['thru'], dayfirst=True)
    data['nan'] = data['thru'].isna()
    return data

def load_holdings(fname):
    holdings = pd.read_csv(fname)
    # Get all the dates for which there are holdings data
    dates = holdings.fDate.drop_duplicates().to_list()
    # Make a dictionary to store the value of all holdings at each date
    dic = {i: holdings.loc[holdings.fDate == i, 'VALUE (x$1000)'].sum() for i in dates}
    # Make a column equal to the whole portfolio value on that date
    holdings['portfolio_value'] = holdings.apply(lambda row: dic[row['fDate']], axis=1)
    # Make a portfolio weight column; both are in x1000 units
    holdings['portfolio_weight'] = holdings['VALUE (x$1000)'] / holdings['portfolio_value']
    return holdings

def load_sharepricedata(fname):
    price_data = pd.read_csv(fname)
    return price_data

def load_statestreet_price_data(fname):
    ssga_price = pd.read_csv(fname)
#     returns state street stock price data
    return ssga_price

def load_crsp_data(fname):
    crsp = pd.read_csv(fname)
    crsp.date = pd.to_datetime(crsp.date, dayfirst=True)
    return crsp

def load_fundamental_data(path_to_main_data, path_to_missing_data):
    f = pd.read_csv(path_to_main_data)
    f_miss = pd.read_csv(path_to_missing_data)
    # Make a list of all the companies already in the 'f' dataset
    already_in = f.tic.drop_duplicates().to_list()
    # Keep the last row for all duplicates
    f = f.drop_duplicates(subset=['tic', 'fyear'], keep='last')
    # Make columns in both datasets match; i.e. make them both uppercase
    f.columns = f.columns.str.upper()
    f_miss.columns = f_miss.columns.str.upper()
    # Get the relevant columns from the missing dataset
    columns = f.columns.to_list()
    f_miss = f_miss[columns][:]
    # Only take the data from the missing dataset that isn't already in the main data
    f_miss = f_miss.loc[f_miss.TIC.isin(already_in) == False][:]
    # Join the two dataframes together
    df = pd.concat([f, f_miss])
    return df