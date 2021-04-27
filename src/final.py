import PyPDF2 as pdf
import pandas as pd
import tabula
import numpy as np
import re
import datetime
from datetime import datetime as dt
import glob
import os
import json
from statsmodels import api as sm
import statsmodels
import difflib

# pdfobj = open(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\annual-stewardship-report-2017.pdf", 'rb')
# pdfreader = pdf.PdfFileReader(pdfobj)

def gen_privateengagements():
    columns = ['Company Name', 'Market', 'Multiple Engagements', 'Governance', 'Proxy Contest/M&A', 'Pay', 'ES']
    cols_2018 = ['Company Name', 'Market', 'Letter', 'Comprehensive Engagement', 'Multiple Engagements', 'Governance',
                 'ES']
    df_2018 = tabula.read_pdf(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Annual Stewardship Reports\SSGA 2018 Stewardship.pdf",
        pages='106-149')
    df_2018 = df_2018.drop(0)
    df_2018.columns = cols_2018

    df_2017 = tabula.read_pdf(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Annual Stewardship Reports\SSGA 2017 Stewardship.pdf", pages='79-98')
    df_2017 = df_2017.drop(0)
    df_2016 = tabula.read_pdf(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Annual Stewardship Reports\SSGA 2016 Stewardship.pdf", pages='41-53', pandas_options={'header': None})
    df_2015 = tabula.read_pdf(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Annual Stewardship Reports\SSGA 2015 Stewardship.pdf", pages='32-42')
    df_2014 = tabula.read_pdf(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Annual Stewardship Reports\SSGA 2014 Stewardship.pdf", pages='28-37', pandas_options={'header': None})
    for count, df in enumerate([df_2014, df_2015, df_2016, df_2017, df_2018], 2014):
        df.columns = columns
        df[columns[2:]] = np.where(pd.isna(df[columns[2:]]), 0, 1)
        df['Year'] = count

    df_2018.loc[(df_2018['Letter'] == 0) & (df_2018['Comprehensive Engagement'] == 0) & (
                df_2018['Multiple Engagements'] == 0) & (df_2018['Governance'] == 0) & (df_2018['ES'] == 0)]
    bad_rows = df_2018.loc[df_2018['Company Name'] == 'Name'].index.to_list()
    more_bad_rows = [x - 1 for x in bad_rows]
    bad_rows2 = df_2018.loc[(df_2018['Company Name'] == 'Name Market')].index.to_list()
    more_badrows2 = [ x - 1 for x in bad_rows2]
    bad_rows += more_bad_rows
    bad_rows += bad_rows2
    bad_rows += more_badrows2
    for i in bad_rows:
        df_2018.drop(i, inplace=True)
    markets = df_2018['Market'].unique().tolist()

    df = df_2014.append(df_2015)
    df = df.append(df_2016)
    df = df.append(df_2017)

    return df

RE_D = re.compile('\d')


def f3(string):
    return bool(RE_D.search(string))


def f4(string):
    return '#' in string


def split_data(file):
    data = file.read()
    data = data.split("=================== State Street Equity 500 Index Portfolio ====================")
    data = data[1]
    if "========= State Street International Developed Equity Index Portfolio ==========" in data:
        data = data.split('========= State Street International Developed Equity Index Portfolio ==========')
    else:
        data = data.split('===================== State Street Money Market Portfolio ======================')
    data = data[0]
    data = data.split('--------------------------------------------------------------------------------')
    return data


def split_vanguard(file):
    vg_data = file.read()
    vg_data = vg_data.split("VANGUARD 500 INDEX FUND")
    vg_data = vg_data[1]
    vg_data = vg_data.split('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    del vg_data[0]
    return vg_data


def split_blackrock(file):
    blk = file.read()
    blk = blk.split("=========================== iShares Core S&P 500 ETF ===========================")
    blk = blk[1]
    blk = blk.split("--------------------------------------------------------------------------------")
    return blk


def parse_vanguard(text):
    data = []
    x = text.replace('\n', '  ')
    x = re.split(r'\s{2,}', x)
    company_name = x[2]
    ticker = x[4]
    security_id = x[6]
    meeting_date = dt.strftime(dt.strptime(x[8], '%m/%d/%Y'), '%d-%m-%Y')
    general_info = [company_name, ticker, security_id, meeting_date]
    y = x[x.index(next(i for i in x if f4(i))):]
    n = 0  # counter within each data entry
    for i in y:
        if f4(i):  # first check if it contains a hash or not
            if n > 0:  # add the last entry to the data and start a new data entry, reset within data entry count to 0
                entry = general_info + entry
                entry = [item.strip() for item in entry]
                data.append(entry)
    #  # establish the empty list
            entry = []
            n = 0
        # take what we need
            i = i.split(':')
            proposal = i[-1]
            proposal_no = i[0].split('#')[-1]
            entry.append(proposal_no)
            entry.append(proposal)
        elif n > 4:
            entry[1] += ' '
            entry[1] += i
        else:
            entry.append(i)
        n += 1
    # add the last data entry to 'data'
    entry = general_info + entry
    data.append(entry)
    return data


def parse_votes(text):
    data = []
    x = text.replace('\n', '  ')
    x = re.split(r'\s{2,}', x)
    company_name = x[1]
    ticker = x[3]
    security_id = x[5]
    meeting_date = dt.strftime(dt.strptime(x[6][14:], '%b %d, %Y'), '%d-%m-%Y')
    meeting_type = x[7][14:]
    try:
        record_date = dt.strftime(dt.strptime(x[9], '%b %d, %Y'), '%d-%m-%Y')
    except:
        record_date = None
    general_info = [company_name, ticker, security_id, meeting_date, meeting_type, record_date]
    print (general_info)
# 3rd, 4th and 5th objects are mgt_rec,, vote_cast and sponsor accordingly
    y = x[10:]
    # make sure we only take the bit that we want - in case of weird formatting
    y = y[y.index(next(i for i in y if f3(i))):]
    n = 0  # counter within each data entry
    for i in y:
        if f3(i) and sum([iterable.isdigit() for iterable in i]) < 4 and len(i) < 5:  # first check if it is a number
            if n > 0:  # add the last entry to the data and start a new data entry, reset within data entry count to 0
                entry = general_info + entry
                entry = [item.strip() for item in entry]
                data.append(entry)
    #  # establish the empty list
            entry = []
            n = 0
        if n > 4:
            entry[1] += ' '
            entry[1] += i
        else:
            entry.append(i)
        n += 1
    # add the last data entry to 'data'
    entry = general_info + entry
    data.append(entry)
    return data


def gen_statestreet():
    output = []
    cols = ['Company Name', 'Ticker', 'Security ID', 'Meeting Date', 'Meeting Type', 'Record Date', 'Item number',
            'Proposal', 'Mgt Rec', 'Vote Cast', 'Sponsor']
    for filename in glob.glob(os.path.join(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\RAW NPX Form data\State Street", '*.txt')):
        votes = open(filename, "r")
        split = split_data(votes)
        for firm in split:
            parsed = parse_votes(firm)
            output += parsed
    df = pd.DataFrame(output, columns=cols)
    data_2013 = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\RAW NPX Form data\State Street\2013 NPX State Street Equity 500.csv")
    data_2014 = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\RAW NPX Form data\State Street\2014 NPX State Street Equity 500.csv")
    data_2013.columns = cols
    data_2014.columns = cols
    df = df.append(data_2013, ignore_index=True)
    df = df.append(data_2014, ignore_index=True)
    # df.to_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Cleaned NPX data\Formatted v4 trial.csv")
    return df

def gen_blackrock():
    output = []
    cols = ['Company Name', 'Ticker', 'Security ID', 'Meeting Date', 'Meeting Type', 'Record Date', 'Item number',
            'Proposal', 'Mgt Rec', 'Vote Cast', 'Sponsor']
    for filename in glob.glob(os.path.join(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\RAW NPX Form data\BlackRock", '*.txt')):
        votes = open(filename, "r")
        split = split_blackrock(votes)
        for firm in split:
            parsed = parse_votes(firm)
            output += parsed
    df = pd.DataFrame(output, columns=cols)
    return df


def gen_vanguard():
    output = []
    cols = ['Company Name', 'Ticker', 'Security ID', 'Meeting Date', 'Item number',
            'Proposal', 'Sponsor', 'Voted?', 'Vote Cast', 'Mgt Rec']
    for filename in glob.glob(os.path.join(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\RAW NPX Form data\Vanguard", '*.txt')):
        votes = open(filename, "r")
        split = split_vanguard(votes)
        for firm in split:
            parsed = parse_vanguard(firm)
            output += parsed
    df = pd.DataFrame(output, columns = cols)
    return df


def clean_npx(df):
    '''

    :param df: output from the gen_ functions (dataframe output from the NP-X parsing functions)
    :return: cleaned dataframe
    '''
    # Empty string
    x = df.loc[df['Sponsor'] == '', 'Proposal']
    Sponsor = df.loc[df['Sponsor'] == '', 'Vote Cast']
    vote_cast = df.loc[df['Sponsor'] == '', 'Mgt Rec']
    x = x.str.split()
    y = list(x)
    mgt_rec = [i[-1] for i in y]
    proposal = [' '.join(i[:-1]) for i in y]
    df.loc[df['Sponsor'] == '', 'Vote Cast'] = vote_cast
    df.loc[df['Sponsor'] == '', 'Mgt Rec'] = mgt_rec
    df.loc[df['Sponsor'] == '', 'Proposal'] = proposal
    df.loc[df['Sponsor'] == '', 'Sponsor'] = Sponsor

    # None
    x = df.loc[df['Sponsor'].isna(), 'Proposal']
    Sponsor = df.loc[df['Sponsor'].isna(), 'Vote Cast']
    vote_cast = df.loc[df['Sponsor'].isna(), 'Mgt Rec']
    x = x.str.split()
    y = list(x)
    mgt_rec = [i[-1] for i in y]
    proposal = [' '.join(i[:-1]) for i in y]
    df.loc[df['Sponsor'].isna(), 'Vote Cast'] = vote_cast
    df.loc[df['Sponsor'].isna(), 'Mgt Rec'] = mgt_rec
    df.loc[df['Sponsor'].isna(), 'Proposal'] = proposal
    df.loc[df['Sponsor'].isna(), 'Sponsor'] = Sponsor

    vote_cast2 = df.loc[(df['Sponsor'] != 'Management') & (df['Sponsor'] != 'Shareholder'), 'Sponsor']
    mgt_rec2 = df.loc[(df['Sponsor'] != 'Management') & (df['Sponsor'] != 'Shareholder'), 'Vote Cast']
    z = df.loc[(df['Sponsor'] != 'Management') & (df['Sponsor'] != 'Shareholder'), 'Proposal']
    z = list(z)
    sponsor = [i[-1] for i in z]
    proposal2 = [' '.join(i[:-1]) for i in z]

    df.loc[(df['Sponsor'] != 'Management') & (df['Sponsor'] != 'Shareholder'), 'Mgt Rec'] = mgt_rec2
    df.loc[(df['Sponsor'] != 'Management') & (df['Sponsor'] != 'Shareholder'), 'Vote Cast'] = vote_cast2
    df.loc[(df['Sponsor'] != 'Management') & (df['Sponsor'] != 'Shareholder'), 'Proposal'] = proposal2
    df.loc[(df['Sponsor'] != 'Management') & (df['Sponsor'] != 'Shareholder'), 'Sponsor'] = sponsor
    return df


def label_engagements(row, dict, lag):
    ticker = row['Ticker']
    year = row['Meeting Date'].to_pydatetime().year
    if ticker in dict.keys():
        if year - lag in dict[ticker]:
            return 1
        else:
            return 0
    else:
        return 0


def get_engagements():
    engagements = pd.read_csv(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Annual Stewardship Reports\Annual Engagements.csv")
    return engagements


def get_votes():
    v = pd.read_csv(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Cleaned NPX data\Combined (dropped multiple columns).csv", converters={'Security ID': lambda x: str(x)})
    iss = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\ISS Vote data\ISS Votes.csv",
                      converters={'CUSIP': lambda x: str(x)}, encoding='ISO-8859-1')
    # Fix the datapoints where State Street's vote is equal to 'ShareÂ Holder'
    # First save the indexes of the rows that need fixing; i.e. the ones where State Street's vote == 'ShareÂ Holder'
    indicies =  v.loc[(v['State Street'].str.contains('Share')) & (v['State Street'].isna() == False)].index.to_list()
    #     # As State Street's vote cast has been recorded under 'Mgt Rec'; make State Street vote cast equal to Mgt Rec
    v.loc[indicies, 'State Street'] = v['Mgt Rec']
    #
    v.loc[indicies, 'Proposal'] = v.loc[indicies].apply(lambda row: ' '.join([''.join(i.split()) for i in re.split(r' {2,}', row['Proposal'])]) + row.Sponsor, axis=1)
    #       As all the rows that need fixing are Shareholder sponsored; set the sponsor == Shareholder
    v.loc[indicies, 'Sponsor'] = 'Shareholder'
    #      Get the management recommendations; outcomes and descriptions from ISS data
    # First convert dates in both dataframes to pandas datettimes so we can match
    v['Meeting Date'] = pd.to_datetime(v['Meeting Date'], dayfirst=True)
    iss.MeetingDate = pd.to_datetime(iss.MeetingDate, dayfirst=True)
    # Go through rows and fix
    for i in indicies:
        ticker = v.iloc[i].Ticker
        date = v.iloc[i]['Meeting Date']
        number = str(v.iloc[i]['Item number'])
        v.loc[i,'Mgt Rec'] = iss.loc[(iss.ticker == ticker) & (iss['MeetingDate'] == date) & (iss['BallotItemNumber'] == number), 'MGMTrec'].item()
        v.loc[i, 'vote_outcome'] = iss.loc[(iss.ticker == ticker) & (iss['MeetingDate'] == date) & (iss['BallotItemNumber'] == number), 'voteResult'].item()
        v.loc[i, 'description'] = iss.loc[(iss.ticker == ticker) & (iss['MeetingDate'] == date) & (iss['BallotItemNumber'] == number), 'AgendaGeneralDesc'].item()

    # FIX Broken CUSIPS by taking them from ISS data - Broken as in the ones that 'E+'; i.e. scientific notation saved as text
    broken = v.loc[v['Security ID'].str.contains('\+')].Ticker.drop_duplicates().to_list()
    # Save the relevant tickers and CUSIP from the ISS data, as a dataframe with just those columns
    save = iss.loc[iss.ticker.isin(broken)].drop_duplicates(subset='ticker')[['ticker', 'CUSIP']][:]
    # Save the broken CUSIPS and ticker
    old_cusip = v.loc[v['Security ID'].str.contains('\+'), ['Security ID', 'Ticker']].drop_duplicates()[:]
    # Merge with save to have a dataframe with old security ID, Ticker and fixed CUSIP
    data = pd.merge(left=old_cusip, right=save, left_on='Ticker', right_on='ticker')
    # Save a dict with the broken security ID as the keys and the fixed CUSIP as the value
    cusips_dict = dict(zip(data['Security ID'], data.CUSIP))
    # Go through and fix one by one
    for b_cusip in cusips_dict:
        v.loc[v['Security ID'] == b_cusip, 'Security ID'] = cusips_dict[b_cusip]
    # Add leading zeroes to CUSIPS less than 9 characters long - NOT using this yet; might break some of the merges?
    # v['CUSIP'] = v['Security ID'].apply('{0:0>9}'.format)
    return v

def get_votes2():
#     New get votes to retrieve the already fixed votes data; no need to open the one with stuff that needs fixing every time and fix it when we run
    votes = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Cleaned NPX data\Combined votes with fixed CUSIPS and fixed rows (the ones where ShareÂ Holder).csv", converters={'Security ID': lambda x: str(x)})
    votes['Meeting Date'] = pd.to_datetime(votes['Meeting Date'], dayfirst=True)
    return votes

def get_votes3():
#     Additional fix of proxy contetsts
    votes = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Cleaned NPX data\Voting data (with proxy contests fixed).csv")
    votes['Meeting Date'] = pd.to_datetime(votes['Meeting Date'], dayfirst=True)
    return votes


def get_wrds():
    wrds = pd.read_csv(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Recent data\WRDS 13F Insto ownership (done with CUSIPS).csv")
    return wrds

def link_wrds_votes(v):
    wrds = get_wrds()
    # Get unique companies from both datasets
    wrdscomps = wrds.stkname.drop_duplicates().to_list()
    comps = v['Company NAME'].drop_duplicates().to_list()
    # Make both lists of companies uppercase
    comps = [str.upper(i) for i in comps]
    wrdscomps = [str.upper(i) for i in wrdscomps]
    # Setup results dictionary
    dic = {}
    # Use difflib to do the heavy lifting
    for i in wrdscomps:
        dic[i] = difflib.get_close_matches(i, comps, n=1)
    # Do the remainder manually
    dic['ADT SECURITY CORP'] = ["The ADT Corporation"]
    dic['CLEVELAND-CLIFFS INC'] = ["Cliffs Natural Resources Inc."]
    dic['ROBERT HALF'] = ["ROBERT HALF INTERNATIONAL INC."]
    dic['SMUCKER J M CO'] = ["The J. M. Smucker Company"]
    dic['ULTA SALON COSMETICS AND FRA'] = ["ULTA BEAUTY, INC."]
    dic['WEC ACQUISITION CORP'] = ["WEC ENERGY GROUP, INC."]
    dic['XYLEM INC (DUPLICATE)'] = ["Xylem Inc."]
    dic.pop("MERRILL LYNCH CAPITAL TRUST")
    dic.pop("WASHINGTON MUTUAL CAPITAL TR")

    return dic


def checker(row):
    # return row['State Street'] == row['iss_recommendation']
    ''' function to see if State Street has followed ISS for a given proposal '''
    # When the vote cast for State Street and ISS are equal; return 1
    if row['State Street'] == row['iss_recommendation']:
        return 1
    else:
        return 0

def label_details(row, dict, col):
#     check to see if the np.sum of the column is >= 1
    ticker = row['Ticker']
    year = row['Meeting Date'].to_pydatetime().year
    if ticker in dict[year][col]:
        return 1
    else:
        return 0


def make_cols_dict(engagements):
#     set up dictionary
    needed = ['Multiple Engagements', 'Governance', 'Proxy Contest/M&A', 'Pay', 'ES', 'Letter', 'Comprehensive Engagement']
    dict = {}
    for i in range(2012, 2019):
        dict[i] = {}
        for col in needed:
            ticks = engagements.loc[(engagements.Year == i) & (engagements.Market == 'USA') & (engagements[col] == 1), 'Ticker'].to_list()
            dict[i][col] = ticks
    return dict

def gen_final_df():
    votes = get_votes3()
    # votes['Meeting Date'] = pd.to_datetime(votes['Meeting Date'], dayfirst=True); No longer needed because do it in get_votes()
    engagements = get_engagements()
    tickers = engagements['Ticker'].unique().tolist()
    tickers = tickers[1:]
    engagements_dict = {}
    needed_cols = make_cols_dict(engagements)
    needed = ['Multiple Engagements', 'Governance', 'Proxy Contest/M&A', 'Pay', 'ES', 'Letter',
              'Comprehensive Engagement']
    for firm in tickers:
        years = engagements.loc[engagements['Ticker'] == firm, 'Year'].to_list()
        engagements_dict[firm] = years
    for col in needed:
        votes[col] = votes.apply(label_details, args=(needed_cols, col), axis=1)
    votes['engaged_0year'] = votes.apply(label_engagements, args=(engagements_dict, 0), axis=1)
    votes['engaged_1year'] = votes.apply(label_engagements, args=(engagements_dict, 1), axis=1)
    votes['engaged_2year'] = votes.apply(label_engagements, args=(engagements_dict, 2), axis=1)
    votes = clean_df(votes)
    # Add time spent in S&P 500 column
    df = add_timespent(votes)
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


def clean_df(df):
    df = df.rename(columns={'ISS Recommendation': 'iss_recommendation'})
    obs = len(df)
    meetings = len(df.drop_duplicates(subset=['Ticker', 'Meeting Date']))
    print('{} proposals across {} meetings'.format(obs, meetings))
    df = df[df.iss_recommendation.notnull()][:]
    obs = len(df)
    meetings = len(df.drop_duplicates(subset=['Ticker', 'Meeting Date']))
    print('{} proposals across {} meetings after dropping proposals with no ISS recommendation'.format(obs, meetings))
    df['Vanguard'] = df['Vanguard'].str.title()
    return df


def generate_yearly_stats(df):
#     generates annual stats for engagement
# # # Reporting period = July 1st to 30th June
    for year in range(2014, 2019):
        sample = df.loc[(df2['Meeting Date'] >= datetime.date(year - 1, 7, 1)) & (
                df['Meeting Date'] <= datetime.date(year, 6, 30))]
        engaged = sample.loc[(sample['Engaged in the current year'] == 1)]
        engagements = len(engaged.drop_duplicates(subset=['Ticker', 'Meeting Date']))
        followed = sample['followed_ISS'].sum()
        iss = engaged['followed_ISS']
        deviated = sample.loc[sample['Engaged in the current year'] == 0, 'followed_ISS']
        print('{} engagements in {}'.format(engagements, year))
        print('{}% - {} of {} proposals in {} voted with ISS recommendation'.format(round(100 * followed / len(sample), 4),
                                                                                    followed, len(sample), year))
        print('For companies engaged with - Percentage voted with ISS recommendation: {}%'.format(
            round(100 * iss.sum() / len(iss), 4)))
        print('For companies not engaged with - Percentage voted with ISS recommendation: {}%'.format(round(
            100 * deviated.sum() / len(deviated), 4)))
    return


def get_sp500constituents():
    data = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\List of S&P 500 Companies\S&P 500 Historical Constituents.csv")
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


def gen_sp500data():
    # use dataset from github
    with open(
            r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\List of S&P 500 Companies\sp500_constituents.json") as f:
        data = json.load(f)
    start = {}
    end = {}
    # first observed
    for date in data:
        for ticker in data[date]:
            if ticker in start.keys():
                pass
            else:
                start[ticker] = date
#     last observed
    for date2 in reversed(list(data.keys())):
        for ticker in data[date2]:
            if ticker in end.keys():
                pass
            else:
                end[ticker] = date2
    start = pd.DataFrame.from_dict(start, orient='index', columns = ['start_date'])
    end = pd.DataFrame.from_dict(end, orient='index', columns = ['end_date'])
    df = start.join(end, how='left')
    # add more detail from other dataset
    more_det = pd.read_csv(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\List of S&P 500 Companies\changes from 08 to 06.csv",
        index_col='Ticker')
    det2000_2006 = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\List of S&P 500 Companies\additions 2000 - 06.csv", index_col='Ticker')
    df = df.join(more_det, how='left')
    df.loc[df.DateAfterChange.isna() == False, 'start_date'] = df.DateAfterChange
    df = df.drop('DateAfterChange',axis=1)
    df = df.join(det2000_2006, how='left')
    df = df.groupby(df.index).first()
    df.loc[df.Date.isna() == False, 'start_date'] = df.Date
    df = df.drop('Date', axis=1)
    return df


def dynamics_analysis(df):
    # first clean it up by only keeping
    df = df.loc[(df.Vanguard.isna() == False) & (df['State Street'].isna() == False) & (df.BlackRock.isna() == False)]
    agreed = df.loc[((df['State Street'] == df['BlackRock']) & (df['Vanguard'] == df['State Street']))]
    disagreed = df.loc[((df['State Street'] == df['BlackRock']) & (df['Vanguard'] == df['State Street'])) == False, 'Proposal']
    disagreed_on = df.loc[((df['State Street'] == df['BlackRock']) & (
                df['Vanguard'] == df['State Street'])) == False, 'Proposal'].to_list()
    director_elections = ['Elect Director' in i for i in disagreed_on]
    de_disagreed = sum(director_elections)
    remove = []
    leftover = disagreed[disagreed.str.contains('Elect Director') == False]
    dic = {}
    for words in ['Pay Frequency','Management Nominee','as Director','A Director','Declassify','Named Executive',
                  'Political', 'Independent Board','Lobbying','Emissions', 'Adjourn Meeting']:
        x = leftover[leftover.str.contains(words)].index.to_list()
        remove += x
        dic[words] = len(x)
    dic['director'] = de_disagreed + dic['as Director'] + dic['A Director'] + dic['Management Nominee']
    return dic


def get_holdings():
    holdings = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\State Street Holdings\13F Holdings with fixed CUSIPS.csv")
    # Get all the dates for which there are holdings data
    dates = holdings.fDate.drop_duplicates().to_list()
    # Make a dictionary to store the value of all holdings at each date
    dic = {i: holdings.loc[holdings.fDate == i, 'VALUE (x$1000)'].sum() for i in dates}
    # Make a column equal to the whole portfolio value on that date
    holdings['portfolio_value'] = holdings.apply(lambda row: dic[row['fDate']], axis=1)
    # Make a portfolio weight column; both are in x1000 units
    holdings['portfolio_weight'] = holdings['VALUE (x$1000)'] / holdings['portfolio_value']
    return holdings

def get_ssgaprice():
    ssga_price = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\List of S&P 500 Companies\State Street historical daily price data.csv")
#     returns state street stock price data
    return ssga_price


def get_pricedata():
    '''
    :return: stock price data for all stocks in the votes data in the data timeframe
    '''
    # price_data = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\List of S&P 500 Companies\S&P 500 Constituent returns.csv")
    price_data = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\List of S&P 500 Companies\S&P 500 Constituent returns 3 changed UAA UA LEN STZ TAP.csv")
    return price_data


def get_closestdate(date):
    # reports occur on 31/03, 30/06, 30/09, 31/12
    '''

    :param date: meeting date (datetime object)
    :return: closest date on which holdings is reported
    '''
    year = date.year
    report_dates = [(3,31),(6,30),(9,30),(12,31)]
    dates = [ datetime.date(year,i[0],i[1]) for i in report_dates]
    return min(dates, key=lambda s: np.abs((date - s).days))


def add_holdings(df):
    '''
    Adds the number of shares, type of shares, and market value of the shares owned by State Street of the constituent
    firm at the closest quarter to the meeting date
    :param df: votes df
    :return: votes df with column representing holdings
    '''
    tol = pd.Timedelta('120 days')
    hh = get_holdings()
    hh.rDate = pd.to_datetime(hh.rDate)
    hh = hh[['TITLE OF CLASS', 'CUSIP', 'VALUE (x$1000)', 'SHRS OR PRN AMT', 'SOLE', 'SHARED', 'NONE', 'rDate', 'portfolio_weight']]
    combined = pd.merge_asof(left=df, right=hh, left_on='Meeting Date', right_on='rDate', left_by='Security ID', right_by='CUSIP',
                  tolerance=tol, direction='nearest')

    return combined


def add_timespent(df):
    '''
    Adds the length of time the constituent firm has been in the S&P 500 Index at the time of meeting date
    :return: votes dataframe with timespent in the S&P500 column added
    '''
    dates = get_sp500constituents()
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


def clean_holdings(h):
#    WRDS holdinss data is missing a lot of tickers - DEPRECIATED NOW - USING 13f data as opposed to WRDS
    int = h.drop_duplicates(subset=['cusip', 'ticker'])
    # int[['cusip', 'ticker']]
    # int.sort_values(by='cusip', inplace=True)
    # int.loc[int.ticker.isna(), ['cusip', 'ticker']]
    test = int[:]
    test = test[['cusip', 'ticker']]
    test.sort_values(by=['cusip', 'ticker'], inplace=True)
    missing = test.loc[test.ticker.isna(), 'cusip']
    lookup = test.dropna()
    lookedup = pd.merge(lookup, missing)
    test = h.merge(lookedup, on='cusip', how='left')
    test.loc[(test.ticker_x.isna()) & (test.ticker_y.isna() == False), 'ticker_x'] = test.ticker_y
    test = test.drop('ticker_y', axis=1)
    test.rename({'ticker_x': 'ticker'}, inplace=True, axis=1)
    test.drop_duplicates(subset=['rdate', 'ticker','value'], keep='first', inplace=True)
    return test


def add_constituentprice(df):
    '''
    Adds the market capitalisation of the firm at the meeting date to the dataframe
    :param df:
    :return:
    '''
    p = get_pricedata()
    pp = p[['date', 'TICKER', 'MKT_CAP']][:]
    pp.date = pd.to_datetime(pp.date, dayfirst=True)
    pp.sort_values(by='date', inplace=True)
    combined = pd.merge_asof(left=df, right=pp, left_on='Meeting Date', right_on='date', left_by='Ticker',
                             right_by='TICKER',
                             tolerance=pd.Timedelta('50 days'), direction='backward')

    return combined


def add_ssgaprice(df):
    '''
    Adds the market capitalisation of State Stree at the meeting date to the dataframe
    :param df:
    :return:
    '''
    p = get_ssgaprice()
    p = p[['date', 'market_cap']][:]
    p.date = pd.to_datetime(p.date, dayfirst=True)
    c2 = pd.merge_asof(left=df, right=p, left_on='Meeting Date', right_on='date',
                       tolerance=pd.Timedelta('3 days'), direction='backward')
    return c2

def last_clean(df):
    '''
    Removes duplicate columns from dataframe and useless columns, make the column descriptions more accurate
    Divide the market cap columns by 1000 to stay consistent with the VALUE column
    :param df:
    :return: Cleaned df
    '''
    df = df.drop(labels=['SHRS OR PRN AMT','TITLE OF CLASS', 'CUSIP', 'SOLE', 'SHARED', 'NONE', 'rDate', 'date_x', 'TICKER', 'date_y'], axis=1)
    df['MKT_CAP'] = df['MKT_CAP']/1000
    df['market_cap'] = df['market_cap']/1000
    df = df.rename({'MKT_CAP': 'firm_marketcap(x1000)', 'market_cap': 'STT_marketcap(x1000)', 'VALUE (x$1000)': 'mktvalue_holdings(x1000)',
               'Meeting Type (from BlackRock data)': 'meeting_type', 'Record Date (from BlackRock Data)': 'record_date',
               'Security ID': 'CUSIP', 'Meeting Date': 'meeting_date'}, axis=1)
    # df['mktvalue_holdings(x1000)'] = df['mktvalue_holdings(x1000)'] / 1000
    df['ownership_stake'] = df['mktvalue_holdings(x1000)'] / df['firm_marketcap(x1000)']
    return df


def apply_board(row):
    ''' function to see if State Street has followed management for a given proposal '''
    # When the vote cast for State Street and Management are equal; return 1
    if row['State Street'] == row['Mgt Rec']:
        return 1
    # If its a proxy contest and they did not follow the dissidents recommendation; then count as 1 because they
    # followed management by disagreeing
    elif (row['Diss Rec'] == 'For') and (row['State Street'] != 'For'):
        return 1
    else:
        return 0


def get_compileddata():
    data = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Compiled data\Final data v7.csv",
                       parse_dates=['meeting_date'], dayfirst=True)
    return data


def regress(df, dep, indep, constant = True):
    '''

    :param indep: y variable (string)
    :param dep:  list of strings - x variables
    :return: Statasmodels regression results class - use .summary() and .get_margeff() to get what u need
    '''
    # y = data.followed_ISS
    y = df[dep]
    x = df[indep]
    if constant:
        x = sm.add_constant(x)
    model = statsmodels.discrete.discrete_model.Probit(y, x)
    out = model.fit()
    # out.summary()
    # margeffs = out.get_margeff(at='median', method='dydx')
    # margeffs.summary()
    # margeffs2 = out.get_margeff(at='mean', method='dydx')

    return out

def add_followedmgt(df):
    df['followed_management'] = df.apply(apply_board, axis=1)
    return df


def get_gov():
    df = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\ISS Vote data\ISS Gov Data with redundant columns removed .csv")
    return df


def get_crsp():
    crsp = pd.read_csv(r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Recent data\CRSP Monthly Stock data (combined OG and missing).csv")
    crsp.date = pd.to_datetime(crsp.date, dayfirst=True)
    return crsp

def add_annual_return(df):
    # Set up dictionary to not do the same ones twice:
    crsp = get_crsp()
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


def remove_na(df):
    df = df.loc[df.iss_recommendation != 'Refer']
    df = df.replace(['Did Not Vote','Do Not Vote','Yes','Abstain','None'], ['Withhold','Withhold','For','Withhold','Withhold'])
    return df

def clean_na(df):
    df = df.loc[(df.BlackRock.isna() | df.Vanguard.isna()) == False]
    df = df.loc[df['State Street'].isna() == False]
    return df


def get_director(df):
    director = director = df.loc[(df.Proposal.str.contains('Elect Director')) | (df.Proposal.str.contains('as Director')) | (
        df.Proposal.str.contains('A Director')) | (df.Proposal.str.contains('Management Nominee'))]
    return director

def get_topshareholderproposals(test):
    test['desc'] = test.description
    # test.loc[test.Proposal.str.contains('Politic') | test.Proposal.str.contains(
    #     'Lobbying'), 'desc'] = 'Report on Lobbying Payments/Political Contributions'
    test.loc[test.Proposal.str.contains('Independent Board'), 'desc'] = test.Proposal
    test.loc[test.Proposal.str.contains('Written Consent'), 'desc'] = 'Provide Right to Act by Written Consent'
    test.loc[test.Proposal.str.contains('Proxy Access'), 'desc'] = 'Proxy Access'
    test.loc[test.Proposal.str.contains('Special Meeting'), 'desc'] = 'Amend Articles - Call Special Meeting'
    test.loc[test.Proposal.str.contains('Vesting'), 'desc'] = 'Vesting of Equity Awards'
    test.loc[test.Proposal.str.contains('Retention'), 'desc'] = 'Stock Retention'
    test.loc[test.Proposal.str.contains('Recapitalization'), 'desc'] = 'Approve Recapitalization Plan for all Stock to Have One-vote per Share '
    test.loc[test.Proposal.str.contains('Majority Vote'), 'desc'] = 'Require a Majority Vote for the Election of Directors'
    test.loc[(test.Proposal.str.contains('Sustain') & test.Proposal.str.contains(' on')), 'desc'] = 'Report on Sustainability'
    top10 = pd.Series(test.loc[test.Sponsor == 'Shareholder', 'desc'].value_counts()).iloc[:12].index.to_list()
    # top10.remove('Elect Directors (Opposition Slate)')
    top10.append('Report on Climate Change')
    dic = {}
    for i in top10:
        dic[i] = {}
        dic[i]['obs'] = test.loc[test.Sponsor == 'Shareholder','desc'].value_counts().loc[i]
        dic[i]['iss_for_mgt'] = round(test.loc[(test.desc == i) & (test.Sponsor == 'Shareholder'), 'iss_for_mgt'].sum()/dic[i]['obs'], 3)
        dic[i]['ss_for_mgt'] = round(test.loc[(test.desc == i) & (test.Sponsor == 'Shareholder'), 'followed_management'].sum()/dic[i]['obs'], 3)
        engaged = test.loc[(test.desc == i) & (test.Sponsor == 'Shareholder') & (test.engaged_0year == 1)]
        dic[i]['ss_for_mgt_engaged'] = round(engaged.followed_management.sum() / len(engaged), 3)
        not_engaged = test.loc[(test.desc == i) & (test.Sponsor == 'Shareholder') & (test.engaged_0year == 0)]
        dic[i]['ss_for_mgt_notengaged'] = round(not_engaged.followed_management.sum() / len(not_engaged),3)
    shareholderproposals  = pd.DataFrame(dic).T
    return shareholderproposals


def summary_to_df(model):
    '''

    :param model: statsmodels probit output
    :return:
    '''
    margeffs = model.get_margeff(at='median',method='dydx')
    params = model.params
    param_zvalues = model.tvalues
    params.name = 'Parameters'
    param_zvalues.name = 'Parameter z-values'
    marg_effects = np.array([np.nan] + margeffs.margeff.tolist())
    marg_effects_zvalues = np.array([np.nan] + margeffs.tvalues.tolist())
    marg_effects_pvalues = np.array([np.nan] + margeffs.pvalues.tolist())
    df = pd.merge(params, param_zvalues, left_index=True,right_index=True)
    df['Marginal Effects'] = marg_effects
    df['Marginal Effects z-values'] = marg_effects_zvalues
    df['Marginal Effects p-values'] = marg_effects_pvalues
    df['Significance'] = np.nan
    df.loc[df['Marginal Effects p-values'] < 0.1, 'Significance'] = '*'
    df.loc[df['Marginal Effects p-values'] < 0.05, 'Significance'] = '**'
    df.loc[df['Marginal Effects p-values'] < 0.01, 'Significance'] = '***'
    df['Pseudo-rsquared'] = model.prsquared
    df['Number of observations'] = model.nobs
    return df


def get_fundamentals():
    f = pd.read_csv(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Recent data\Compustat firm fundamentals data v2 (extra vars) 2.csv")
    f_miss = pd.read_csv(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Recent data\COMPUSTAT Fundamentals missing data.csv")
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



# FIX VOTES
# v['length'] = v.apply(lambda row: len(row['Security ID']), axis=1)
# v['CUSIP'] = v.apply( lambda row: (9-row['length'])*'0' + str(row['Security ID']), axis=1)


# engages by market by year
# for year in range(2014, 2019):
#     dict2[year] = {}
#     markets = e.loc[e.Year == year, 'Market'].drop_duplicates().to_list()
#     for var in markets:
#         if year == 2018:
#             dict2[year][var] = len(e.loc[(e['Year'] == year) & (e['Comprehensive Engagement'] == 1) & (e.Market == var)])
#         else:
#             dict2[year][var] = len(e.loc[(e['Year'] == year) & (e.Market == var)])


# engages by category by year
# dict = {}
# for year in range(2014, 2019):
#     dict[year] = {}
#     for var in ['Governance', 'Proxy Contest/M&A', 'Pay', 'ES']:
#         if year == 2018:
#             dict[year][var] = e.loc[(e['Year'] == year) & (e['Comprehensive Engagement'] == 1), var].sum()
#         else:
#             dict[year][var] = e.loc[(e['Year'] == year) & (e['Letter'] != 1), var].sum()


def add_gov(data):
    '''
    Adds governance data to the dataframe
    :param df: generates governance dataframe from cleaned votes dataframe
    :return:  gov dataframe
    '''
    data['proposal_year'] = data.meeting_date.dt.year
    data['year'] = data.meeting_date.dt.year
    gov = get_gov()
    gov['majority_vote'] = 1 * (gov['MAJOR_VOTE_COMM'].str.contains('majority vote standard'))
    gov_to_merge = gov[
        ['CBOARD', 'year', 'TICKER', 'DUALCLASS', 'FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE',
         'majority_vote']]
    gov_to_merge = gov_to_merge.fillna(0)
    gov_to_merge = gov_to_merge.replace(to_replace='YES', value=1)
    test = pd.merge(left=data,right=gov_to_merge,left_on=['proposal_year','Ticker'],right_on=['year','TICKER'],how='left')
    test['Proposal'] = test.Proposal.str.strip()
    test['iss_for_mgt'] = 1 * (test.iss_recommendation == test['Mgt Rec'])

    return test


def add_fundamentals(df):
    '''
    ADDS DATA REGARDING FIRM FUNDAMENTALS TO THE DATAFRAME
    :param df:
    :return:
    '''
    # Columns we need to use to calculate the things we need
    columns = ['NI', 'AT', 'LT', 'TXDITC', 'PSTK', 'FYEAR', 'TIC']
    fundamentals = get_fundamentals()
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

def get_df():
    ''' output the final df for data manipulation'''
    all_votes = gen_final_df()
    all_votes = last_clean(all_votes)
    votes_20142018 = all_votes.loc[all_votes['meeting_date'] > pd.Timestamp(datetime.date(2014, 1, 1))][:]
    votes_20142018 = add_annual_return(votes_20142018)
    votes_20142018['year'] = votes_20142018.meeting_date.dt.year
    final_votes = add_fundamentals(votes_20142018)
    final_votes2 = add_gov(final_votes)
    final_votes2['log_assets'] = np.log(final_votes2.AT)
    final_votes2['Director'] = 1 * final_votes2.description.str.contains('Elect Director')
    final_votes2['Shareholder'] = 1 * (final_votes2['Sponsor'] == 'Shareholder')
    # Drop Baxalta from both
    final_votes2 = final_votes2.drop(final_votes2.loc[final_votes2.Ticker == 'BXLT'].index.to_list())
    # Drop ones where vote outcome is not applicable (i.e. duplicate proposals)
    final_votes2 = final_votes2.loc[final_votes2.vote_outcome != 'Not Applicable']
    # SET mGT REC ON PROXY CONTESTS TO WITHHOLD
    final_votes2.loc[final_votes2['Mgt Rec'].isna(), 'Mgt Rec'] = 'Withhold'
    return final_votes2


if __name__ == "__main__":
    final_votes2 = get_df()

    # Get directors and others
    directors = final_votes2.loc[final_votes2.description.str.contains('Elect Director')][:]
    other_proposals = final_votes2.loc[final_votes2.description.str.contains('Elect Director') == False][:]
    # Change proposals 'In the Elect ... as Director' format to 'Elect Director ....'
    directors.loc[directors.Proposal.str.contains('as Director'), 'Proposal'] = directors.Proposal.apply(
        lambda x: 'Elect Director ' + ' '.join(x.split(' ')[1:-2]))
    # Change proposals 'In the Elect ... as a Director' format to 'Elect Director ....'
    directors.loc[directors.Proposal.str.contains('as a Director'), 'Proposal'] = directors.Proposal.apply(
        lambda x: 'Elect Director ' + ' '.join(x.split(' ')[1:-3]))
    # Add names
    directors['name'] = directors['Proposal'].apply(lambda x: x.split('Elect Director ')[-1])
    # Make upper case and remove commas from names
    directors.name = directors.name.str.upper()
    directors.name = directors.name.str.replace(',', '')
    # # Remove duplicate white spaces
    # directors.name= director_data.FULLNAME.apply(lambda s: re.sub("\s\s+", " ", s))
    # Read-in ISS director data
    director_data = pd.read_csv(
        r"C:\Users\Bing\OneDrive - The University of Melbourne\Thesis\Recent data\ISS Director Data.csv",
        encoding='ISO-8859-1')
    # Remove commas from ISS data
    director_data.FULLNAME = director_data.FULLNAME.str.replace(',', '')
    # Remove duplicate white spaces from ISS data
    director_data.FULLNAME = director_data.FULLNAME.apply(lambda s: re.sub("\s\s+", " ", s))
    # Keep only the columns we need
    keep = ['company_id', 'ticker', 'year', 'MeetingDate', 'FULLNAME', 'classification', 'Employment_CEO',
            'Outside_Public_Boards', 'Attend_LESS75_PCT', 'DirSince', 'Age', 'female']
    director_data = director_data[keep][:]
    # Match director names in Voting data to ISS data
    matches = []
    for ind,row in directors.iterrows():
        # Lookup director and match in iss data
        ticker = row['Ticker']
        year = row['year_x']
        name = row['name']
        d = director_data.loc[(director_data.ticker == ticker)& (director_data.year == year)].FULLNAME.to_list()
        match = difflib.get_close_matches(name,d,1)
        # See if matching failed
        if len(match) == 0:
        # Expand the match range
            d = director_data.loc[(director_data.ticker == ticker) & ((director_data.year <= year+2) | (director_data.year >= year-2) )].FULLNAME.to_list()
            match = difflib.get_close_matches(name, d, 1)
        # Check if matching failed again, and add match to the list
        if len(match) == 0:
            matches.append('')
        else:
            matches.append(match[0])
    directors['matched_name'] = matches
    # Manually look at thesse:
    mia = directors.loc[directors.matched_name == '']
    # Convert date to datetime
    director_data.MeetingDate = pd.to_datetime(director_data.MeetingDate, format='%Y%m%d')
    # Sort by date before performing merge
    director_data.sort_values(by='MeetingDate', inplace=True)
    # Merge on meeting date
    test = pd.merge_asof(left=directors, right=director_data, left_on='meeting_date', right_on='MeetingDate',
                         left_by=['matched_name', 'Ticker'], right_by=['FULLNAME', 'ticker'], direction='nearest')
    # # regressors = ['independent', 'incumbent', 'ceo', 'outside_seats', 'attendedless75', 'tenure', 'above65', 'female']
    test['independent'] = 1 * (test.classification == 'I')
    test['incumbent'] = 1 * (test.classification == 'E')
    test['ceo'] = 1 * (test.Employment_CEO == 'Yes')
    test['outside_seats'] = test.Outside_Public_Boards
    test['attendedless75'] = 1 * (test.Attend_LESS75_PCT == 'Yes')
    test['tenure'] = test.year - test.DirSince
    test['above65'] = 1 * (test.Age > 65)
    test['female'] = 1 * (test.female == 'Yes')
    # # Get rid of NaNs
    directors_merged = test.drop(test.loc[test.company_id.isna()].index.to_list())[:]
    # Drop the shareholder sponsored director elections
    # directors2 = directors_merged.loc[directors_merged.Sponsor != 'Shareholder']
    directors2 = directors_merged[:]
    # Set up contentious dataframes and subsample
    # Directors
    contentious_directors = directors2.loc[directors2.iss_for_mgt == 0]
    consensus_directors = directors2.loc[directors2.iss_for_mgt == 1]
    # Other proposals
    contentious_other = other_proposals.loc[other_proposals.iss_for_mgt == 0]
    consensus_other = other_proposals.loc[other_proposals.iss_for_mgt == 1]
    director_x = ['engaged_0year', 'Multiple Engagements',
         'portfolio_weight', 'ownership_stake',
         'excess_return',
         'CBOARD', 'DUALCLASS','FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE', 'majority_vote',
          'independent', 'incumbent', 'ceo', 'outside_seats', 'attendedless75', 'tenure', 'above65', 'female']

    other_x = ['engaged_0year', 'Multiple Engagements',
         'portfolio_weight', 'ownership_stake',
         'excess_return',
         'CBOARD', 'DUALCLASS','FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE', 'majority_vote',
          'Shareholder']
    y = 'followed_management'
    content_d = summary_to_df(regress(contentious_directors, y, director_x))
    consens_d = summary_to_df(regress(consensus_directors, y, director_x))
    content_o = summary_to_df(regress(contentious_other, y, other_x))
    consens_o = summary_to_df(regress(consensus_other, y, other_x))
    # THIS FILE HAS the way FOLLOWING ISS AND MANAGEMENT IS DEFINED DIFFERENTLY
    content_d_iss = summary_to_df(regress(contentious_directors, 'followed_ISS', director_x))
    consens_d_iss = summary_to_df(regress(consensus_directors, 'followed_ISS', director_x))
    content_o_iss = summary_to_df(regress(contentious_other, 'followed_ISS', other_x))
    consens_o_iss = summary_to_df(regress(consensus_other, 'followed_ISS', other_x))

    final_votes3 = clean_na(final_votes2)

    # Percentage of votes cast with management/ISS per year for each firm
    res ={}
    for year in range(2014, 2019):
        res[year] = {}
        for fund in ['State Street', 'BlackRock', 'Vanguard','iss_recommendation']:
            if fund == 'BlackRock' and year == 2017:
                dat = final_votes3.loc[
                    (final_votes3.year_x == year) & (final_votes3.Proposal.str.contains('Pay Frequency') == False)]
            else:
                dat = final_votes3.loc[final_votes2.year_x == year]
            total = len(dat)
            res[year][fund] = round((dat[fund] == dat['Mgt Rec']).sum() / total, 4)
    res = pd.DataFrame(res)

    res0 = {}
    for fund in ['State Street', 'BlackRock', 'Vanguard']:
        res0[fund] = {}
        consensus = final_votes3.loc[(final_votes3.Sponsor == 'Shareholder') &(final_votes3.iss_for_mgt == 1) & ( (final_votes3.year_x == 2017) & (final_votes3.Proposal.str.contains('Pay Frequency') == False))]
        contentious = final_votes3.loc[(final_votes3.Sponsor == 'Shareholder')& (final_votes3.iss_for_mgt == 0) & ( (final_votes3.year_x == 2017) & (final_votes3.Proposal.str.contains('Pay Frequency') == False))]
        cons_t = len(consensus)
        cont_t = len(contentious)
        res0[fund]['consensus'] = round((consensus[fund] == consensus['iss_recommendation']).sum() / cons_t, 4)
        res0[fund]['contentious'] = round((contentious[fund] == contentious['iss_recommendation']).sum() / cont_t, 4)