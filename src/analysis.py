import pandas as pd
import numpy as np
from statsmodels import api as sm
import statsmodels
import datetime
from data import separate_director_votes, make_dataframe

DIRECTOR_INDEPENDENT_VARS = ['engaged_0year', 'Multiple Engagements',
         'portfolio_weight', 'ownership_stake',
         'excess_return',
         'CBOARD', 'DUALCLASS','FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE', 'majority_vote',
          'independent', 'incumbent', 'ceo', 'outside_seats', 'attendedless75', 'tenure', 'above65', 'female']

OTHER_PROPOSAL_INDEPENDENT_VARS = ['engaged_0year', 'Multiple Engagements',
         'portfolio_weight', 'ownership_stake',
         'excess_return',
         'CBOARD', 'DUALCLASS','FAIRPRICE', 'GPARACHUTE', 'LSPMT', 'PPILL', 'UNEQVOTE', 'majority_vote', 'Shareholder']


def generate_yearly_stats(df):
    '''
    Generates State Street's annual statistics for private engagement.
    Reporting period = July 1st to 30th June 
    '''
    for year in range(2014, 2019):
        sample = df.loc[(df['Meeting Date'] >= datetime.date(year - 1, 7, 1)) & (
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


def dynamics_analysis(df):
    '''
    Analyse which proposals the big three index funds disagreed on. 
    Returns a dictionary containing the topic and the number of times the funds' disagreed on said topic.
    '''
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


def get_topshareholderproposals(df):
    df['desc'] = df.description
    # df.loc[df.Proposal.str.contains('Politic') | df.Proposal.str.contains(
    #     'Lobbying'), 'desc'] = 'Report on Lobbying Payments/Political Contributions'
    df.loc[df.Proposal.str.contains('Independent Board'), 'desc'] = df.Proposal
    df.loc[df.Proposal.str.contains('Written Consent'), 'desc'] = 'Provide Right to Act by Written Consent'
    df.loc[df.Proposal.str.contains('Proxy Access'), 'desc'] = 'Proxy Access'
    df.loc[df.Proposal.str.contains('Special Meeting'), 'desc'] = 'Amend Articles - Call Special Meeting'
    df.loc[df.Proposal.str.contains('Vesting'), 'desc'] = 'Vesting of Equity Awards'
    df.loc[df.Proposal.str.contains('Retention'), 'desc'] = 'Stock Retention'
    df.loc[df.Proposal.str.contains('Recapitalization'), 'desc'] = 'Approve Recapitalization Plan for all Stock to Have One-vote per Share '
    df.loc[df.Proposal.str.contains('Majority Vote'), 'desc'] = 'Require a Majority Vote for the Election of Directors'
    df.loc[(df.Proposal.str.contains('Sustain') & df.Proposal.str.contains(' on')), 'desc'] = 'Report on Sustainability'
    top10 = pd.Series(df.loc[df.Sponsor == 'Shareholder', 'desc'].value_counts()).iloc[:12].index.to_list()
    # top10.remove('Elect Directors (Opposition Slate)')
    top10.append('Report on Climate Change')
    dic = {}
    for i in top10:
        dic[i] = {}
        dic[i]['obs'] = df.loc[df.Sponsor == 'Shareholder','desc'].value_counts().loc[i]
        dic[i]['iss_for_mgt'] = round(df.loc[(df.desc == i) & (df.Sponsor == 'Shareholder'), 'iss_for_mgt'].sum()/dic[i]['obs'], 3)
        dic[i]['ss_for_mgt'] = round(df.loc[(df.desc == i) & (df.Sponsor == 'Shareholder'), 'followed_management'].sum()/dic[i]['obs'], 3)
        engaged = df.loc[(df.desc == i) & (df.Sponsor == 'Shareholder') & (df.engaged_0year == 1)]
        dic[i]['ss_for_mgt_engaged'] = round(engaged.followed_management.sum() / len(engaged), 3)
        not_engaged = df.loc[(df.desc == i) & (df.Sponsor == 'Shareholder') & (df.engaged_0year == 0)]
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

def analyse_votes_by_fund(df: pd.DataFrame):
        # Percentage of votes cast with management/ISS per year for each firm
    res ={}
    for year in range(2014, 2019):
        res[year] = {}
        for fund in ['State Street', 'BlackRock', 'Vanguard','iss_recommendation']:
            if fund == 'BlackRock' and year == 2017:
                dat = df.loc[
                    (df.year_x == year) & (df.Proposal.str.contains('Pay Frequency') == False)]
            else:
                dat = df.loc[df.year_x == year]
            total = len(dat)
            res[year][fund] = round((dat[fund] == dat['Mgt Rec']).sum() / total, 4)
    res = pd.DataFrame(res)

    res0 = {}
    for fund in ['State Street', 'BlackRock', 'Vanguard']:
        res0[fund] = {}
        consensus = df.loc[(df.Sponsor == 'Shareholder') &(df.iss_for_mgt == 1) & ( (df.year_x == 2017) & (df.Proposal.str.contains('Pay Frequency') == False))]
        contentious = df.loc[(df.Sponsor == 'Shareholder')& (df.iss_for_mgt == 0) & ( (df.year_x == 2017) & (df.Proposal.str.contains('Pay Frequency') == False))]
        cons_t = len(consensus)
        cont_t = len(contentious)
        res0[fund]['consensus'] = round((consensus[fund] == consensus['iss_recommendation']).sum() / cons_t, 4)
        res0[fund]['contentious'] = round((contentious[fund] == contentious['iss_recommendation']).sum() / cont_t, 4)
    return res, res0

def split_contentious_directors(directors: pd.DataFrame):
    contentious_directors = directors.loc[directors.iss_for_mgt == 0]
    consensus_directors = directors.loc[directors.iss_for_mgt == 1]
    return contentious_directors, consensus_directors

def split_contentious_other_proposals(other_proposals: pd.DataFrame):
    contentious_other = other_proposals.loc[other_proposals.iss_for_mgt == 0]
    consensus_other = other_proposals.loc[other_proposals.iss_for_mgt == 1]
    return contentious_other, consensus_other

if __name__ == "__main__":
    df = make_dataframe()
    directors, others = separate_director_votes(df)

    #---------SUMMARY STATS------------
    # Top shareholder proposals
    top_shareholder_proposals = get_topshareholderproposals(df)
    print(top_shareholder_proposals)
    # Fund voting habits
    with_management, with_iss = analyse_votes_by_fund(df)

    #---------State Street ENGAGEMENT EFFECT ---------- 
    # Split into contentious and consensus
    contentious_directors, consensus_directors = split_contentious_directors(directors)
    contentious_other, consensus_other = split_contentious_other_proposals(others)

    # Investigate State Street following management recommendations:
    DEP_VAR = 'followed_management'
    contentious_directors_management = summary_to_df(regress(contentious_directors, DEP_VAR, DIRECTOR_INDEPENDENT_VARS))
    consensus_directors_management = summary_to_df(regress(consensus_directors, DEP_VAR, DIRECTOR_INDEPENDENT_VARS))
    contentious_other_proposals_management = summary_to_df(regress(contentious_other, DEP_VAR, OTHER_PROPOSAL_INDEPENDENT_VARS))
    consensus_other_proposals_management = summary_to_df(regress(consensus_other, DEP_VAR, OTHER_PROPOSAL_INDEPENDENT_VARS))

    # Investigate State Street following ISS recommendations:
    DEP_VAR = 'followed_ISS'
    contentious_directors_iss = summary_to_df(regress(contentious_directors, DEP_VAR, DIRECTOR_INDEPENDENT_VARS))
    consensus_directors_iss = summary_to_df(regress(consensus_directors, DEP_VAR, DIRECTOR_INDEPENDENT_VARS))
    contentious_other_proposals_iss = summary_to_df(regress(contentious_other, DEP_VAR, OTHER_PROPOSAL_INDEPENDENT_VARS))
    consensus_other_proposals_iss = summary_to_df(regress(consensus_other, DEP_VAR, OTHER_PROPOSAL_INDEPENDENT_VARS))

