import pandas as pd
import datetime
import loaders

PATH_TO_VOTE_DATA = "../data/votes.csv"
PATH_TO_ENGAGEMENT_DATA = "../data/stewardship/engagements.csv"
PATH_TO_SPCONSTITUENT_DATA = "../data/sp500historical_constituents.csv"

def get_columns(engagements):
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

def get_tickers(engagements):
    tickers = engagements['Ticker'].unique().tolist()
    return tickers[1:]


def add_engagement_details(row, details_dict, col):
    '''
    To be applied to each row of the dataframe. 
    Adds data relating to the type of engagement conducted by State Street.
    For example, whether they conducted 'Multiple Engagements', a 'Comprehensive engagement' etc.
    '''
    ticker = row['Ticker']
    year = row['Meeting Date'].to_pydatetime().year
    return 1 if ticker in details_dict[year][col] else 0 


def add_engagements(row, enagements_dict, lag):
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

def add_engagement_data(votes, engagements):
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

def add_timespent_data(df):
    '''
    Adds the length of time the constituent firm has been in the S&P 500 Index at the time of meeting date
    '''
    dates = loaders.load_sp500constituents(PATH_TO_SPCONSTITUENT_DATA)
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
    
def make_dataframe():
    '''
    Entrypoint to the application.
    Builds a dataframe with data from all relevant sources and joins into 1. 
    '''
    votes = loaders.load_votingdata(PATH_TO_VOTE_DATA)
    engagements = loaders.load_engagements(PATH_TO_ENGAGEMENT_DATA)
    votes = add_engagement_data(votes, engagements)
    votes = clean_df(votes)
    # Add time spent in S&P 500 column
    df = add_timespent_data(votes)
    # Make sure all votes cast are For/Withold/Against
    df = remove_na(df)
    # Add 13F State Street holdings data
    df = add_holdings(df)
    # Add price
    df = add_constituentprice(df)
    df = add_ssgaprice(df)
    df = add_followedmgt(df)
    df['followed_ISS'] = df.apply(checker, axis=1)
    return df

if __name__ == "__main__":
    df = make_dataframe()