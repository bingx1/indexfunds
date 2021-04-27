import pandas as pd
from pandas.core.frame import DataFrame
import file_loaders
import data_loaders
import datetime
import numpy as np

PATH_TO_VOTE_DATA = "../data/votes.csv"
PATH_TO_ENGAGEMENT_DATA = "../data/stewardship/engagements.csv"
PATH_TO_SPCONSTITUENT_DATA = "../data/sp500historical_constituents.csv"
PATH_TO_HOLDINGS_DATA = "../data/ss13fholdings.csv"
PATH_TO_PRICE_DATA = "../data/sp500constituent_returns.csv"
PATH_TO_STATESTREET_PRICE_DATA = "../data/ss_price_data.csv"
PATH_TO_RETURNS_DATA = "../data/crsp_data.csv"
PATH_TO_FUNDAMENTALS_DATA = "../data/fundamentals_data.csv"
PATH_TO_EXTRA_FUNDAMENTALS_DATA = "../data/missing_fundamentals_data.csv"
PATH_TO_GOVERNANCE_DATA = "../data/iss_governance_data.csv"
PATH_TO_DIRECTORS_DATA = "../data/iss_director_data.csv"


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
    df = df.replace(['Did Not Vote', 'Do Not Vote', 'Yes', 'Abstain', 'None'], [
                    'Withhold', 'Withhold', 'For', 'Withhold', 'Withhold'])
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


def add_followed_iss(df: pd.DataFrame):
    '''
    Adds a indicator column equal to 1 if State Street followed ISS else 0. 
    '''
    df['followed_ISS'] = df.apply(followed_iss, axis=1)


def followed_iss(row):
    return 1 if row['State Street'] == row['iss_recommendation'] else 0


def tidyup_df(df: pd.DataFrame):
    '''
    Removes duplicate columns from dataframe and useless columns, make the column descriptions more accurate
    Divide the market cap columns by 1000 to stay consistent with the VALUE column
    '''
    df = df.drop(labels=['SHRS OR PRN AMT', 'TITLE OF CLASS', 'CUSIP', 'SOLE',
                         'SHARED', 'NONE', 'rDate', 'date_x', 'TICKER', 'date_y'], axis=1)
    df['MKT_CAP'] = df['MKT_CAP']/1000
    df['market_cap'] = df['market_cap']/1000
    df = df.rename({'MKT_CAP': 'firm_marketcap(x1000)', 'market_cap': 'STT_marketcap(x1000)', 'VALUE (x$1000)': 'mktvalue_holdings(x1000)',
                    'Meeting Type (from BlackRock data)': 'meeting_type', 'Record Date (from BlackRock Data)': 'record_date',
                    'Security ID': 'CUSIP', 'Meeting Date': 'meeting_date'}, axis=1)
    df['ownership_stake'] = df['mktvalue_holdings(x1000)'] / \
        df['firm_marketcap(x1000)']
    df.loc[df['Mgt Rec'].isna(), 'Mgt Rec'] = 'Withhold'
    return df


def restrict_df_by_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    '''
    Removes all datapoints/votes before the specified date.
    '''
    return df.loc[df['meeting_date'] > date][:]


def make_year_column(df: pd.DataFrame):
    df['year'] = df.meeting_date.dt.year


def add_sponsor_cols(df: pd.DataFrame):
    '''
    Adds binary indicator columns, 'Director' and 'Shareholder' denoting who sponsored the vote.
    '''
    df['Director'] = 1 * df.description.str.contains('Elect Director')
    df['Shareholder'] = 1 * (df['Sponsor'] == 'Shareholder')


def drop_bad_cols(df: pd.DataFrame):
    '''
    Remove low-quality and repeated datapoints. 
    '''
    df = df.drop(df.loc[df.Ticker == 'BXLT'].index.to_list())
    # Drop ones where vote outcome is not applicable (i.e. duplicate proposals)
    return df.loc[df.vote_outcome != 'Not Applicable']


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
    add_followed_iss(df)
    df = tidyup_df(df)
    df = restrict_df_by_date(df, pd.Timestamp(datetime.date(2014, 1, 1)))
    df = data_loaders.add_annual_return_data(df, PATH_TO_RETURNS_DATA)
    make_year_column(df)
    df = data_loaders.add_fundamental_data(
        df, PATH_TO_FUNDAMENTALS_DATA, PATH_TO_EXTRA_FUNDAMENTALS_DATA)
    df = data_loaders.add_governance_data(df, PATH_TO_GOVERNANCE_DATA)
    df['log_assets'] = np.log(df.AT)
    add_sponsor_cols(df)
    df = drop_bad_cols(df)
    return df


def clean_director_data(directors):
    # Change proposals 'In the Elect ... as Director' format to 'Elect Director ....'
    directors.loc[directors.Proposal.str.contains('as Director'), 'Proposal'] = directors.Proposal.apply(
        lambda x: 'Elect Director ' + ' '.join(x.split(' ')[1:-2]))
    # Change proposals 'In the Elect ... as a Director' format to 'Elect Director ....'
    directors.loc[directors.Proposal.str.contains('as a Director'), 'Proposal'] = directors.Proposal.apply(
        lambda x: 'Elect Director ' + ' '.join(x.split(' ')[1:-3]))
    # Add names
    directors['name'] = directors['Proposal'].apply(
        lambda x: x.split('Elect Director ')[-1])
    # Make upper case and remove commas from names
    directors.name = directors.name.str.upper()
    directors.name = directors.name.str.replace(',', '')


def separate_director_votes(df):
    '''
    Most votes are related to the election of directors.
    This function splits the votes into two parts: votes pertaining to director elections, and others
    '''
    directors = df.loc[df.description.str.contains('Elect Director')][:]
    clean_director_data(directors)
    directors = data_loaders.merge_iss_director_data_with_votes(
        directors, PATH_TO_DIRECTORS_DATA)
    print(directors.head())
    other_proposals = df.loc[df.description.str.contains(
        'Elect Director') == False][:]
    print(other_proposals.head())
    return directors, other_proposals
