import pandas as pd
from datetime import datetime


def get_help_data(path):
    """
    Function that loads help data used for the estimation and prediction of the sentiment
    Args:
        path: root directory of the files as a string

    Returns:
        company_mapping: pandas.DataFrame object that contains the information of the companies (id, name, ticker, ...)
        emojis_positive: regular expression as string that identifies positive emoticons
        emojis_negative: regular expression as string that identifies negative emoticons
        closing_info: pandas.DataFrame with the information for each day indicating whether the stock market was
                      open or closed

    """
    # Information about problems in our data set:
    data_issue = pd.read_csv(
        path + '/data_issue_info.tsv',
        delimiter='\t')
    # Determine which stocks have to be excluded due to data issues (select their IDs)
    to_be_excluded = data_issue.loc[data_issue['exclude'] == 1, 'rpid'].values
    # Load mapping file and lower case tickers and company names
    company_mapping = pd.read_csv(
        path + "/SP500_Company_Mapping.tsv",
        delimiter="\t")
    company_mapping['taq_ticker'] = company_mapping['taq_ticker'].map(lambda ticker: ticker.lower())
    company_mapping['original_name'] = company_mapping['original_name'].map(lambda name: name.lower())
    company_mapping['cleaned_name'] = company_mapping['cleaned_name'].map(lambda name: name.lower())
    # Remove companies with data issues
    to_remove = company_mapping['rpid'].map(lambda x: x in to_be_excluded)
    company_mapping = company_mapping.loc[~to_remove, ]

    # Load emoticon data
    emojis = pd.read_csv(path + '/emojis.csv', delimiter=';', index_col='unicode')
    # Load tagged (positive vs. negative) emoticons
    emojis_tags = pd.read_csv(path + '/emojis_tags.csv', delimiter=';', index_col='unicode')
    # Define regular expression for positive and negative emoticons
    emojis_positive = '|'.join('(' + pd.concat([emojis_tags.loc[emojis_tags['tag'] == 'positive'], emojis],
                                               join='inner', axis=1)['ftu8'] + ')')
    emojis_negative = '|'.join('(' + pd.concat([emojis_tags.loc[emojis_tags['tag'] == 'negative'], emojis],
                                               join='inner', axis=1)['ftu8'] + ')')

    # Load data with information about closing times/days of the NYSE:
    holidays = pd.read_csv(path + '/NYSE_closing_days.tsv', delimiter='\t')
    holidays.columns = ['date', 'time', 'holiday']
    holidays = holidays.drop('time', axis=1)
    holidays['holiday'] = holidays['holiday'].map(lambda x: x == 1)
    holidays['date'] = holidays['date'].map(lambda x: pd.Timestamp(x))
    # Construct pandas.DataFrame with the information for each day indicating whether the stock market was
    # open or closed
    closing_info = pd.DataFrame(
        {'date': pd.date_range(start=datetime(2010, 1, 1), end=datetime(2019, 1, 1))})
    closing_info['weekend'] = closing_info['date'].map(lambda x: x.weekday() in [5, 6])
    closing_info = closing_info.merge(holidays, how='left', on='date')
    closing_info['holiday'] = closing_info['holiday'].fillna(False)
    closing_info['closed'] = closing_info.apply(lambda x: x['weekend'] or x['holiday'], axis=1)
    closing_info['date'] = closing_info['date'].dt.date
    closing_info = closing_info.drop(['weekend', 'holiday'], axis=1)

    return company_mapping, emojis_positive, emojis_negative, closing_info
