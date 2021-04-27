import pandas as pd

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