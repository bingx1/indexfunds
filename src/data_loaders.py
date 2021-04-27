from json import load
from typing import List
import pandas as pd
import datetime
from file_loaders import load_governance_data, load_sp500constituents
from file_loaders import load_holdings
from file_loaders import load_sharepricedata
from file_loaders import load_statestreet_price_data
from file_loaders import load_crsp_data
from file_loaders import load_fundamental_data
from file_loaders import load_governance_data
from file_loaders import load_iss_directors_data
import difflib
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

def add_holdings_data(df: pd.DataFrame, fpath: str) -> pd.DataFrame: 
    '''
    Adds the number of shares, type of shares, and market value of the shares owned by State Street of the constituent
    firm at the closest quarter to the meeting date
    '''
    tol = pd.Timedelta('120 days')
    holdings_data = load_holdings(fpath)
    holdings_data.rDate = pd.to_datetime(holdings_data.rDate)
    holdings_data = holdings_data[['TITLE OF CLASS', 'CUSIP', 'VALUE (x$1000)', 'SHRS OR PRN AMT', 'SOLE', 'SHARED', 'NONE', 'rDate', 'portfolio_weight']]
    combined = pd.merge_asof(left=df, right=holdings_data, left_on='Meeting Date', right_on='rDate', left_by='Security ID', right_by='CUSIP',
                  tolerance=tol, direction='nearest')

    return combined

def add_price_data(df: pd.DataFrame, fpath: str) -> pd.DataFrame:
    '''
    Adds a column containing the market capitalisation of the target firm on the meeting date.
    '''
    p = load_sharepricedata(fpath)
    pp = p[['date', 'TICKER', 'MKT_CAP']][:]
    pp.date = pd.to_datetime(pp.date, dayfirst=True)
    pp.sort_values(by='date', inplace=True)
    combined = pd.merge_asof(left=df, right=pp, left_on='Meeting Date', right_on='date', left_by='Ticker',
                             right_by='TICKER',
                             tolerance=pd.Timedelta('50 days'), direction='backward')
    return combined


def add_ssga_price_data(df: pd.DataFrame, fpath: str) -> pd.DataFrame:
    '''
    Adds a column containing the market capitalisation of State Street on the meeting date.
    '''
    p = load_statestreet_price_data(fpath)
    p = p[['date', 'market_cap']][:]
    p.date = pd.to_datetime(p.date, dayfirst=True)
    c2 = pd.merge_asof(left=df, right=p, left_on='Meeting Date', right_on='date',
                       tolerance=pd.Timedelta('3 days'), direction='backward')
    return c2

def add_annual_return_data(df: pd.DataFrame, fpath: str):
    # Set up dictionary to not do the same ones twice:
    crsp = load_crsp_data(fpath)
    # Dictionary for firm returns
    dict = {i:{} for i in df.Ticker.drop_duplicates()}
    # Dictionary for sp500 returns
    dict2 = {i:{} for i in df.Ticker.drop_duplicates()}
    returns_col = []
    spreturns_col = []
    for ind, row in df.iterrows():
        ticker = row['Ticker']
        date = row['meeting_date']
        # First check if it's already been calculated:
        if date in dict[ticker]:
            annual_return = dict[ticker][date]
            sp_ret = dict2[ticker][date]
        else:
        # Index into the CRSP dataframe, grab price data for that stock
            returns = crsp.loc[crsp.TICKER == ticker]
        # Find the closest date + closest date 1y ago to the meeting date for where there is stock return data in the CRSP file
            close_date = nearest(returns['date'], pd.Timestamp(date))
            far_date = nearest(returns['date'], (pd.Timestamp(date) - pd.Timedelta('365 days')))
        # Get the returns between these two dates and add 1 to them
            rets = returns.loc[(returns.date > far_date) & (returns.date <= close_date)].vwretd + 1
            sprets = returns.loc[(returns.date > far_date) & (returns.date <= close_date)].sprtrn + 1
        # Take the cumproduct to convert 12 monthly returns to an annual return, take the last value in the cumprod
            sp_ret = sprets.cumprod().values[-1]
            annual_return = rets.cumprod().values[-1]
        # Since these values weren't in the dictionary, add them to the storage dictionaries
            dict[ticker][date] = annual_return
            dict2[ticker][date] = sp_ret
#         Add the annual return to the firm-returns list
        returns_col.append(annual_return)
    #     Add the S&P return to the sp500-returns list
        spreturns_col.append(sp_ret)
    #     Use the returns list to make 1y return columns
    df['1y_return'] = returns_col
    df['1y_return'] = df['1y_return'] - 1
    df['1y_spreturn'] = spreturns_col
    df['1y_spreturn'] = df['1y_spreturn'] - 1
    # Get excess return as the difference between firm 1 y return and sp 500 1 y return
    df['excess_return'] = df['1y_return'] - df['1y_spreturn']
    return df

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def add_fundamental_data(df: pd.DataFrame, path_to_main_compustat_data: str, path_to_missing_compustat_data: str) -> pd.DataFrame:
    '''
    Adds the target firms fundamental data (as of the meeting date) to the dataframe.
    '''
        # Columns we need to use to calculate the things we need
    columns = ['NI', 'AT', 'LT', 'TXDITC', 'PSTK', 'FYEAR', 'TIC']
    fundamentals = load_fundamental_data(path_to_main_compustat_data, path_to_missing_compustat_data)
    # Only keep the columns we need
    fundamentals = fundamentals[columns]
    # Replace all NaN values with 0
    fundamentals[columns] = fundamentals[columns].fillna(0)
    # Merge with the dataframe on year and Ticker
    mgd = pd.merge(left=df, right=fundamentals, left_on=['year', 'Ticker'], right_on=['FYEAR', 'TIC'], how='left')
    # mgd[columns] = mgd[columns].fillna(0)
    # Make the columns we need; Return on Assets, Book value of Equity, Market leverage, Book/Market  ratio
    mgd['ROA'] = mgd.NI / mgd.AT
    mgd['book_equity'] = mgd.AT - mgd.LT + mgd.TXDITC - mgd.PSTK
    mgd['market_leverage'] = mgd.LT / (mgd['firm_marketcap(x1000)'] / 1000)
    mgd['bm_equityratio'] = mgd.book_equity / (mgd['firm_marketcap(x1000)'] / 1000)
    return mgd

def add_governance_data(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    '''
    Adds firm-specific governance data to the dataframe. 
    This is related to the characteristics of the firms' charter
    '''
    df['proposal_year'] = df.meeting_date.dt.year
    df['year'] = df.meeting_date.dt.year
    gov = load_governance_data(fname)
    gov['majority_vote'] = 1 * (gov['MAJOR_VOTE_COMM'].str.contains('majority vote standard'))
    gov_to_merge = gov[
        ['CBOARD', 'year', 'TICKER', 'DUALCLASS', 'FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE',
         'majority_vote']]
    gov_to_merge = gov_to_merge.fillna(0)
    gov_to_merge = gov_to_merge.replace(to_replace='YES', value=1)
    merged = pd.merge(left=df,right=gov_to_merge,left_on=['proposal_year','Ticker'],right_on=['year','TICKER'],how='left')
    merged['Proposal'] = merged.Proposal.str.strip()
    merged['iss_for_mgt'] = 1 * (merged.iss_recommendation == merged['Mgt Rec'])
    return merged

def merge_iss_director_data_with_votes(directors: pd.DataFrame, path_to_iss_directors_data: str) -> pd.DataFrame:
    iss_director_data = load_iss_directors_data(path_to_iss_directors_data)
    # Match director names in Voting data to ISS data
    matches = []
    for ind,row in directors.iterrows():
        # Lookup director and match in iss data
        ticker = row['Ticker']
        year = row['year_x']
        name = row['name']
        d = iss_director_data.loc[(iss_director_data.ticker == ticker)& (iss_director_data.year == year)].FULLNAME.to_list()
        match = difflib.get_close_matches(name,d,1)
        # See if matching failed
        if len(match) == 0:
        # Expand the match range
            d = iss_director_data.loc[(iss_director_data.ticker == ticker) & ((iss_director_data.year <= year+2) | (iss_director_data.year >= year-2) )].FULLNAME.to_list()
            match = difflib.get_close_matches(name, d, 1)
        # Check if matching failed again, and add match to the list
        if len(match) == 0:
            matches.append('')
        else:
            matches.append(match[0])
    directors['matched_name'] = matches
    # Convert date to datetime
    iss_director_data.MeetingDate = pd.to_datetime(iss_director_data.MeetingDate, format='%Y%m%d')
    # Sort by date before performing merge
    iss_director_data.sort_values(by='MeetingDate', inplace=True)
    # Merge on meeting date
    merged = pd.merge_asof(left=directors, right=iss_director_data, left_on='meeting_date', right_on='MeetingDate',
                         left_by=['matched_name', 'Ticker'], right_by=['FULLNAME', 'ticker'], direction='nearest')
    merged['independent'] = 1 * (merged.classification == 'I')
    merged['incumbent'] = 1 * (merged.classification == 'E')
    merged['ceo'] = 1 * (merged.Employment_CEO == 'Yes')
    merged['outside_seats'] = merged.Outside_Public_Boards
    merged['attendedless75'] = 1 * (merged.Attend_LESS75_PCT == 'Yes')
    merged['tenure'] = merged.year - merged.DirSince
    merged['above65'] = 1 * (merged.Age > 65)
    merged['female'] = 1 * (merged.female == 'Yes')
    # # Get rid of NaNs
    merged = merged.drop(merged.loc[merged.company_id.isna()].index.to_list())[:]
    return merged
    