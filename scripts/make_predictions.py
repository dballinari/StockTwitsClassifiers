from __future__ import print_function
# noinspection PyUnresolvedReferences
import script_helper
from joblib import load
from src.global_variables import MODEL_DATA_PATH, STOCKTWITS_DATA_PATH, RAW_DATA_PATH, \
    TWITTER_DATA_PATH, PROCESSED_DATA_PATH
from src.data_preparation import TextHandler
from src.load_help_data import get_help_data
from nltk.stem import WordNetLemmatizer
from src.prediction_functions import predict_in_chunks, bull
import pandas as pd
from datetime import timedelta
import numpy as np

"""
Define general settings and load global data
"""
# Define vectorization method:
vectorization_method = 'tfidf'
# Define aggregation scheme
aggregation_tweets = 'c2c'
if aggregation_tweets == 'c2c':
    hours_shift = 8
    minutes_shift = 0
elif aggregation_tweets == 'o2o':
    hours_shift = -9
    minutes_shift = -30
else:
    hours_shift = 0
    minutes_shift = 0
# Define filtering approach
has_cashtag = False
unique_cashtag = False
unique_cashtag = unique_cashtag and has_cashtag

# Load help data
company_mapping, emojis_positive, emojis_negative, closing_info = get_help_data(path=RAW_DATA_PATH)
# Initialize the lemmatizer:
lemmer = WordNetLemmatizer()
# Load models
logistic = load(f"{MODEL_DATA_PATH}/logistic_{vectorization_method}.joblib")
naive_bayes = load(f"{MODEL_DATA_PATH}/naive_bayes_{vectorization_method}.joblib")

"""
Compute and aggregate sentiment of StockTwits messages
"""
# Initialize data handler for StockTwits:
th_stocktwits = TextHandler(data_path=STOCKTWITS_DATA_PATH, mapping=company_mapping,
                            tz_in='UTC', tz_out='America/New_York',
                            outcome_name='StockTwits_sentiment', datetime_name='tweet_datetime', text_name='text')

# Load vectorizer
th_stocktwits.vectorizer = load(f"{MODEL_DATA_PATH}/vectorizer_{vectorization_method}.joblib")

# Predict and aggregate the daily sentiment of StockTwits
sentiment_stocktwits = pd.DataFrame()
for data_i in th_stocktwits.text_generator(emojis_positive, emojis_negative, estimation=False,
                                           has_cashtag=has_cashtag, unique_cashtag=unique_cashtag,
                                           lemmer=lemmer, hour_shift=hours_shift, minute_shift=minutes_shift):
    # Vectorize data:
    x_i = th_stocktwits.vectorizer.transform(data_i['text'])
    # Make predictions:
    data_i['logistic'] = (logistic.predict(x_i) - 0.5) * 2
    data_i['naive_bayes'] = (predict_in_chunks(naive_bayes, x_i, 10000) - 0.5) * 2
    # For the aggregation, we shift the date of messages posted during holidays or weekends to the next trading day:
    data_i = data_i.merge(closing_info, how='left', left_on='date_out', right_on='date')[
        ['rpid', 'date', 'closed', 'logistic', 'naive_bayes']]
    while any(data_i.closed):
        data_i['date'] = data_i.apply(lambda x: x['date'] + timedelta(days=1) if x['closed'] else x['date'],
                                      axis=1)
        data_i = data_i.drop('closed', axis=1).merge(closing_info, how='left', on='date')
    # Aggregate sentiments on a daily basis:
    sentiment_i = data_i.drop(['closed', 'rpid'], axis=1).groupby('date').aggregate(
        {'logistic': [bull, np.mean], 'naive_bayes': [bull, np.mean]})
    # Transform multi-index column names to single level:
    sentiment_i.columns = ['_'.join(col).strip() for col in sentiment_i.columns.values]
    # Date (which acts as an index) to a column:
    sentiment_i.reset_index(level=0, inplace=True)
    # Add information about RavenPack ID:
    sentiment_i['rpid'] = data_i['rpid'][0]
    # Append data:
    sentiment_stocktwits = sentiment_stocktwits.append(sentiment_i, ignore_index=True)

# Save predicted and aggregated daily sentiments obtained from Twitter
path_stocktwits_sentiment = f"{PROCESSED_DATA_PATH}/StockTwits_Logistic_NB_daily_" \
                         f"{aggregation_tweets}{'_cashtag_only' if has_cashtag else ''}" \
                         f"{'_unique' if unique_cashtag else ''}.csv"
sentiment_stocktwits.to_csv(path_stocktwits_sentiment)
del sentiment_stocktwits

"""
Compute and aggregate sentiment of Twitter messages
"""
# Initialize data handler for StockTwits:
th_twitter = TextHandler(data_path=TWITTER_DATA_PATH, mapping=company_mapping,
                         tz_in='Europe/Zurich', tz_out='America/New_York',
                         outcome_name=None, datetime_name='datetime', text_name='text')

# Load vectorizer
th_twitter.vectorizer = load(f"{MODEL_DATA_PATH}/vectorizer_{vectorization_method}.joblib")

# Predict and aggregate the daily sentiment of Twitter
sentiment_twitter = pd.DataFrame()
for data_i in th_twitter.text_generator(emojis_positive, emojis_negative, estimation=False,
                                        has_cashtag=has_cashtag, unique_cashtag=unique_cashtag,
                                        lemmer=lemmer, hour_shift=hours_shift, minute_shift=minutes_shift):
    # Vectorize data:
    x_i = th_twitter.vectorizer.transform(data_i['text'])
    # Make predictions:
    data_i['logistic'] = (logistic.predict(x_i) - 0.5) * 2
    data_i['naive_bayes'] = (predict_in_chunks(naive_bayes, x_i, 10000) - 0.5) * 2
    # For the aggregation, we shift the date of messages posted during holidays or weekends to the next trading day:
    data_i = data_i.merge(closing_info, how='left', left_on='date_out', right_on='date')[
        ['rpid', 'date', 'closed', 'logistic', 'naive_bayes']]
    while any(data_i.closed):
        data_i['date'] = data_i.apply(lambda x: x['date'] + timedelta(days=1) if x['closed'] else x['date'],
                                      axis=1)
        data_i = data_i.drop('closed', axis=1).merge(closing_info, how='left', on='date')
    # Aggregate sentiments on a daily basis:
    sentiment_i = data_i.drop(['closed', 'rpid'], axis=1).groupby('date').aggregate(
        {'logistic': [bull, np.mean], 'naive_bayes': [bull, np.mean]})
    # Transform multi-index column names to single level:
    sentiment_i.columns = ['_'.join(col).strip() for col in sentiment_i.columns.values]
    # Date (which acts as an index) to a column:
    sentiment_i.reset_index(level=0, inplace=True)
    # Add information about RavenPack ID:
    sentiment_i['rpid'] = data_i['rpid'][0]
    # Append data:
    sentiment_twitter = sentiment_twitter.append(sentiment_i, ignore_index=True)

# Save predicted and aggregated daily sentiments obtained from Twitter
path_twitter_sentiment = f"{PROCESSED_DATA_PATH}/Twitter_Logistic_NB_daily_" \
                         f"{aggregation_tweets}{'_cashtag_only' if has_cashtag else ''}" \
                         f"{'_unique' if unique_cashtag else ''}.csv"
sentiment_twitter.to_csv(path_twitter_sentiment)
del sentiment_twitter
