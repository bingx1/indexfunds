import json
import pandas as pd

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


def apply_timespent(row, dict, index):
    df = pd.DataFrame.from_dict(dict)
    df.index = index
    ticker = row['Ticker']
    first_date = datetime.date(1993, 1, 22)
    meeting_date = row['Meeting Date'].date()
    x = df.loc[[ticker]]
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
    return time


# REGRESSIONS

    y = 'engaged_0year'
    # x = ['CBOARD', 'DUALCLASS', 'FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE',
    #      'time_insp500', 'mktvalue_holdings(x1000)', 'firm_marketcap(x1000)']
    x = ['CBOARD', 'DUALCLASS', 'FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE', 'portfolio_weight', 'ownership_stake']
    y = 'followed_ISS'
    # # y = 'followed_management'
    x = ['engaged_0year', 'portfolio_weight', 'ownership_stake', 'Multiple Engagements', 'CBOARD', 'DUALCLASS', 'FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE',
         'majority_vote',]
    # model = regress(data, 'followed_ISS', x)
    # model.summary()