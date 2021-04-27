import pandas as pd
import numpy as np
import re
import datetime
from datetime import datetime as dt
import glob
import os
import json


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