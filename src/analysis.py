import pandas as pd
import numpy as np
from statsmodels import api as sm
import statsmodels

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