from typing import List
import pandas as pd
import datetime
from file_loaders import load_sp500constituents

# This module contains functions to increment the base dataframe

def get_columns(engagements: pd.DataFrame) -> dict:
    '''
    Returns a dictionary of dictionaries.
    {2012:{'Multiple Engagements': ['AAPL', 'MSFT]}, etc.}
    '''
#   Set up dictionary
    needed = ['Multiple Engagements', 'Governance', 'Proxy Contest/M&A', 'Pay', 'ES', 'Letter', 'Comprehensive Engagement']
    cols = {}
    for i in range(2012, 2019):
        cols[i] = {}
        for col in needed:
            ticks = engagements.loc[(engagements.Year == i) & (engagements.Market == 'USA') & (engagements[col] == 1), 'Ticker'].to_list()
            cols[i][col] = ticks
    return cols

def get_tickers(engagements: pd.DataFrame) -> list:
    tickers = engagements['Ticker'].unique().tolist()
    return tickers[1:]


def add_engagement_details(row, details_dict: dict, col: str):
    '''
    To be applied to each row of the dataframe. 
    Adds data relating to the type of engagement conducted by State Street.
    For example, whether they conducted 'Multiple Engagements', a 'Comprehensive engagement' etc.
    '''
    ticker = row['Ticker']
    year = row['Meeting Date'].to_pydatetime().year
    return 1 if ticker in details_dict[year][col] else 0 


def add_engagements(row, enagements_dict: dict, lag: int):
    '''
    To be applied to each row of the dataframe.
    Checks to see if a firm was engaged by State Street within a given period.
    '''
    ticker = row['Ticker']
    year = row['Meeting Date'].to_pydatetime().year
    if ticker in enagements_dict.keys():
        return 1 if (year - lag in enagements_dict[ticker]) else 0
    else:
        return 0

def add_engagement_data(votes: pd.DataFrame, engagements: pd.DataFrame) -> pd.DataFrame:
    '''
    This function adds 3 columns to the votes data - 'engaged_0year', 'engaged_1year' and 'engaged_2year'.
    These columns indicate whether State Street Global Advisors engaged the firm within the said time period.
    '''
    engagements_dict = {}
    engagement_columns = get_columns(engagements)
    tickers = get_tickers(engagements)
    needed = ['Multiple Engagements', 'Governance', 'Proxy Contest/M&A', 'Pay', 'ES', 'Letter',
              'Comprehensive Engagement']
    for firm in tickers:
        years = engagements.loc[engagements['Ticker'] == firm, 'Year'].to_list()
        engagements_dict[firm] = years
    for col in needed:
        votes[col] = votes.apply(add_engagement_details, args=(engagement_columns, col), axis=1)
    votes['engaged_0year'] = votes.apply(add_engagements, args=(engagements_dict, 0), axis=1)
    votes['engaged_1year'] = votes.apply(add_engagements, args=(engagements_dict, 1), axis=1)
    votes['engaged_2year'] = votes.apply(add_engagements, args=(engagements_dict, 2), axis=1)
    return votes

def add_timespent_data(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    '''
    Adds the length of time the constituent firm has been in the S&P 500 Index at the time of meeting date
    '''
    dates = load_sp500constituents(fname)
    first_date = datetime.date(1993, 1, 22)
    times = []
    for ind, row in df.iterrows():
        ticker = row['Ticker']
        meeting_date = row['Meeting Date'].date()
        x = dates.loc[[ticker]]
        time = 0
        for i, r in x.iterrows():
            from_d = r['from'].date()
            thru_d = r['thru'].date()
            d = max(from_d,first_date)
            # for rows where thru is nan (and hence the company is still in the S&P 500 today)
            if r['nan']:
                time_spent = (meeting_date - d).days
            # for rows where from and thru are both defined
            else:
                time_spent = (thru_d - d).days
            time += time_spent
        times.append(time)
    df['time_insp500'] = times
    return df

def add_holdings_data(df: pd.DataFrame) -> pd.DataFrame: 
    '''
    Adds the number of shares, type of shares, and market value of the shares owned by State Street of the constituent
    firm at the closest quarter to the meeting date
    '''
    tol = pd.Timedelta('120 days')
    hh = get_holdings()
    hh.rDate = pd.to_datetime(hh.rDate)
    hh = hh[['TITLE OF CLASS', 'CUSIP', 'VALUE (x$1000)', 'SHRS OR PRN AMT', 'SOLE', 'SHARED', 'NONE', 'rDate', 'portfolio_weight']]
    combined = pd.merge_asof(left=df, right=hh, left_on='Meeting Date', right_on='rDate', left_by='Security ID', right_by='CUSIP',
                  tolerance=tol, direction='nearest')

    return combined