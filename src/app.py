import pandas as pd
import file_loaders
import data_loaders

PATH_TO_VOTE_DATA = "../data/votes.csv"
PATH_TO_ENGAGEMENT_DATA = "../data/stewardship/engagements.csv"
PATH_TO_SPCONSTITUENT_DATA = "../data/sp500historical_constituents.csv"
PATH_TO_HOLDINGS_DATA = "../data/ss13fholdings.csv"
PATH_TO_PRICE_DATA = "../data/sp500constituent_returns.csv"
PATH_TO_STATESTREET_PRICE_DATA = "../data/ss_price_data.csv"

def clean_df(df):
    '''
    Simple function to tidy up the dataframe and remove votes with insufficient data.
    ie. Drops all datapoints without ISS recommendations.
    '''
    df = df.rename(columns={'ISS Recommendation': 'iss_recommendation'})
    obs = len(df)
    meetings = len(df.drop_duplicates(subset=['Ticker', 'Meeting Date']))
    print('{} proposals across {} meetings'.format(obs, meetings))
    df = df[df.iss_recommendation.notnull()][:]
    obs = len(df)
    meetings = len(df.drop_duplicates(subset=['Ticker', 'Meeting Date']))
    print('{} proposals across {} meetings after dropping proposals with no ISS recommendation'.format(obs, meetings))
    df['Vanguard'] = df['Vanguard'].str.title()
    df = df.loc[df.iss_recommendation != 'Refer']
    df = df.replace(['Did Not Vote','Do Not Vote','Yes','Abstain','None'], ['Withhold','Withhold','For','Withhold','Withhold'])
    return df


def add_followed_management(df: pd.DataFrame):
    '''
    Adds a indicator column equal to 1 if State Street followed management else 0. 
    '''
    df['followed_management'] = df.apply(followed_mgt, axis=1)

def followed_mgt(row):
    # When the vote cast for State Street and Management are equal; return 1
    if row['State Street'] == row['Mgt Rec']:
        return 1
    # If its a proxy contest and they did not follow the dissidents recommendation; then count as 1 because they followed management by disagreeing
    elif (row['Diss Rec'] == 'For') and (row['State Street'] != 'For'):
        return 1
    else:
        return 0

def add_followed_ISS(df: pd.DataFrame):
    '''
    Adds a indicator column equal to 1 if State Street followed ISS else 0. 
    '''
    df['followed_ISS'] = df.apply(followed_ISS, axis=1)
    

def followed_ISS(row):
    return 1 if row['State Street'] == row['iss_recommendation'] else 0

def make_dataframe():
    '''
    Entrypoint to the application.
    Builds a dataframe with data from all relevant sources and joins into 1. 
    '''
    votes = file_loaders.load_votingdata(PATH_TO_VOTE_DATA)
    engagements = file_loaders.load_engagements(PATH_TO_ENGAGEMENT_DATA)
    votes = data_loaders.add_engagement_data(votes, engagements)
    votes = clean_df(votes)
    df = data_loaders.add_timespent_data(votes, PATH_TO_SPCONSTITUENT_DATA)
    df = data_loaders.add_holdings_data(df, PATH_TO_HOLDINGS_DATA)
    df = data_loaders.add_price_data(df, PATH_TO_PRICE_DATA)
    df = data_loaders.add_ssga_price_data(df, PATH_TO_STATESTREET_PRICE_DATA)
    add_followed_management(df)
    add_followed_ISS(df)
    return df

if __name__ == "__main__":
    df = make_dataframe()
    print(df.head())