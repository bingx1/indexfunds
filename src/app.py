import pandas as pd
import file_loaders
import data_loaders

PATH_TO_VOTE_DATA = "../data/votes.csv"
PATH_TO_ENGAGEMENT_DATA = "../data/stewardship/engagements.csv"
PATH_TO_SPCONSTITUENT_DATA = "../data/sp500historical_constituents.csv"

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
    # Add 13F State Street holdings data
    # df = add_holdings(df)
    # # Add price
    # df = add_constituentprice(df)
    # df = add_ssgaprice(df)
    # df = add_followedmgt(df)
    # df['followed_ISS'] = df.apply(checker, axis=1)
    return df

if __name__ == "__main__":
    df = make_dataframe()
