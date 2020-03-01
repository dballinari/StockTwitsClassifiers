import pandas as pd
import re
from text_unidecode import unidecode
import string
import pytz
from datetime import datetime, timedelta
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from joblib import dump


class TextHandler:
    def __init__(self, data_path, mapping, tz_in, tz_out, outcome_name, datetime_name, text_name,
                 vectorization_method='count', ngram_range=(1, 1), stop_words=None, min_df=1):
        """
        Class that handles the textual data from Twitter and StockTwits. The class is designed to load the raw textual
        data, clean it, vectorize it and split the observations into train, validation and test set
        Args:
            data_path: path as string where the data tweet-data is located
            mapping: pandas.DataFrame containing information about the companies' name and ticker over time
            tz_in: timezone as string of original data
            tz_out: timezone as string in which the observation have to analysed
            outcome_name: name of the outcome column as a string
            datetime_name: name of the date-time column as a string
            text_name: name of the text column as a string
            vectorization_method: vectorization method as string; possible values are 'count' and 'tfidf'
            ngram_range: tuple indicating the range of n-grams to consider for the bag-of-words approach
            stop_words: list of stop words
            min_df: minimum document frequency for a token to be considered as a feature
        """
        self.data_path = data_path
        self.mapping = mapping
        self.tz_in = pytz.timezone(tz_in)
        self.tz_out = pytz.timezone(tz_out)
        self.outcome_name = outcome_name
        self.datetime_name = datetime_name
        self.text_name = text_name
        self.nagram_range = ngram_range
        self.vectorization_method = vectorization_method
        self.stop_words = stop_words
        self.min_df = min_df
        self.data = None
        self.vectorizer = None

    def text_generator(self, emojis_positive='', emojis_negative='', estimation=True,
                       has_cashtag=False, unique_cashtag=False,
                       lemmer=None, stemmer=None,
                       hour_shift=0, minute_shift=0):
        """
        Generator that loads and cleans the data company-by-company generating an iterable of cleaned data sets
        Args:
            emojis_positive: regular expression as a string of positive emoticons
            emojis_negative: regular expression as a string of negative emoticons
            estimation: boolean indicating if the data is used for estimating a model
            unique_cashtag: boolean indicating if only text with a unique cashtag should be kept in the data set
            has_cashtag: boolean indicating if only text with the company's cashtag should be kept in the data set
            lemmer: lemmer-object from the class nltk.stem.WordNetLemmatizer
            stemmer: stemmer-object from the class nltk.stem.PorterStemmer
            hour_shift: integer indicating by how many hours the date-times should be shifted
            minute_shift: integer indicating by how many minutes the date-times should be shifted

        Returns: yields a cleaned data set of text messages company-by-company

        """

        for rpid_i in self.mapping['rpid'].unique():
            # Load data for the company with ID 'rpid_i':
            data_i = pd.read_csv(
                f"{self.data_path}/{rpid_i}_tweets.tsv",
                encoding="ANSI", quotechar='"', delimiter="\t", engine='python')
            # Select relevant columns and change the column names:
            if estimation:
                data_i = data_i[[self.outcome_name, self.datetime_name, self.text_name]]
                data_i.columns = ['outcome', 'datetime_in', 'text']
                data_i = data_i.loc[data_i['outcome'] != 'None', :]
            else:
                data_i = data_i[[self.datetime_name, self.text_name]]
                data_i.columns = ['datetime_in', 'text']
            # Define regular expression for the company's cashtag:
            cashtag_regex_i = '|'.join(
                r'([$]{1}\b' + self.mapping.loc[self.mapping['rpid'] == rpid_i, 'taq_ticker'] + r'\b)')
            ticker_regex_i = '|'.join(
                r'(\b' + self.mapping.loc[self.mapping['rpid'] == rpid_i, 'taq_ticker'] + r'\b)')
            # Define regular expression for the company's name:
            name_regex_i = '|'.join(
                r'(\b' + self.mapping.loc[self.mapping['rpid'] == rpid_i, 'original_name'] + r'\b)')
            nameclean_regex_i = '|'.join(
                r'(\b' + self.mapping.loc[self.mapping['rpid'] == rpid_i, 'cleaned_name'] + r'\b)')
            # Clean text data:
            data_i['text'] = data_i['text'].map(lambda x: clean_text(x,
                                                                     cashtag_regex_i,
                                                                     ticker_regex_i,
                                                                     name_regex_i,
                                                                     nameclean_regex_i,
                                                                     emojis_positive,
                                                                     emojis_negative,
                                                                     lemmer,
                                                                     stemmer))

            if has_cashtag or unique_cashtag:
                # Count number of company cashtags:
                data_i['num_companycashtag'] = data_i['text'].map(lambda x: len(re.findall(r'\bcompanycashtag\b', x)))
                # Count number of other cashtags:
                data_i['num_cashtag'] = data_i['text'].map(lambda x: len(re.findall(r'\bcashtag\b', x)))
                # If wanted, remove tweets that do not mention the company's cashtag:
                if has_cashtag:
                    data_i = data_i.loc[data_i['num_companycashtag'] > 0]
                # If wanted, remove tweets that mention other cashtags:
                if unique_cashtag:
                    data_i = data_i.loc[data_i['num_cashtag'] == 0]
                data_i.drop(['num_companycashtag', 'num_cashtag'], axis=1, inplace=True)

            # Add RavenPack ID:
            data_i['rpid'] = rpid_i
            # Adjust date-time information:
            data_i['datetime_in'] = pd.to_datetime(data_i['datetime_in']). \
                dt.tz_localize(None). \
                dt.tz_localize(tz=self.tz_in, ambiguous=True)
            data_i['datetime_out'] = data_i['datetime_in'].map(lambda x: x.astimezone(self.tz_out))
            data_i['datetime_out'] = data_i['datetime_out'].map(
                lambda x: x + timedelta(hours=hour_shift, minutes=minute_shift))
            data_i['date_out'] = data_i['datetime_out'].dt.date

            yield data_i

    def load_text(self, emojis_positive='', emojis_negative='', estimation=True,
                  has_cashtag=False, unique_cashtag=False,
                  lemmer=None, stemmer=None,
                  hour_shift=0, minute_shift=0):
        """
        Method that loads and cleans the data
        Args:
            emojis_positive: regular expression as a string of positive emoticons
            emojis_negative: regular expression as a string of negative emoticons
            estimation: boolean indicating if the data is used for estimating a model
            unique_cashtag: boolean indicating if only text with a unique cashtag should be kept in the data set
            has_cashtag: boolean indicating if only text with the company's cashtag should be kept in the data set
            lemmer: lemmer-object from the class nltk.stem.WordNetLemmatizer
            stemmer: stemmer-object from the class nltk.stem.PorterStemmer
            hour_shift: integer indicating by how many hours the date-times should be shifted
            minute_shift: integer indicating by how many minutes the date-times should be shifted

        """
        data_clean = pd.DataFrame()
        for data_i in self.text_generator(emojis_positive, emojis_negative, estimation,
                                          has_cashtag, unique_cashtag, lemmer, stemmer, hour_shift, minute_shift):
            data_clean = data_clean.append(data_i, ignore_index=True)
        self.data = data_clean

    def get_train_validation_test(self, start_train_validation, end_train_validation,
                                  balanced=True, val_size=0.3, seed=None):
        """
        Method that vectorizes the text data and splits the data set into train, validation and test sets
        Args:
            start_train_validation: start date of the train and validation time period as a string
            end_train_validation: end date of the train and validation time period as a string
            balanced: boolean indicating if the train and validation data has to be balanced
            val_size: float between 0 and 1 indicating the share of data from the train/validation data used for
                      validation
            seed: integer fixing the seed

        Returns: feature matrices and outcome vectors for train, validation and test data

        """
        if self.data is None:
            print('Data has not been loaded!')
            pass
        if seed is not None and type(seed) is int:
            random.seed(seed)
        # Parse date-strings to datetime objects
        start_train_validation = datetime.strptime(start_train_validation, '%Y-%m-%d')
        end_train_validation = datetime.strptime(end_train_validation, '%Y-%m-%d')
        # Define the range of days which are considered for the training and validation set:
        training_date_range = pd.date_range(start=start_train_validation, end=end_train_validation)
        # Define which observations are in the window for training and validation
        self.data['for_train_val'] = self.data['date_out'].map(lambda x: x in training_date_range)
        # If we want, we can create a balanced train and validation set
        if balanced:
            # Check how many bearish and bullish observations are in the sample
            in_range_bearish = self.data.apply(lambda x: x['outcome'] == 'Bearish' and x['for_train_val'], axis=1)
            in_range_bullish = self.data.apply(lambda x: x['outcome'] == 'Bullish' and x['for_train_val'], axis=1)
            num_bearish = np.sum(in_range_bearish)
            num_bullish = np.sum(in_range_bullish)
            # For the class with the larger number of observations in the train/validation set, get their indices
            if num_bullish > num_bearish:
                ind_train_val_max = [i for i in range(len(in_range_bullish)) if in_range_bullish[i]]
            else:
                ind_train_val_max = [i for i in range(len(in_range_bearish)) if in_range_bearish[i]]
            # Randomly remove a number of indices form the train/test set
            ind_to_test = random.sample(ind_train_val_max, np.abs(num_bullish - num_bearish))
            self.data.loc[ind_to_test, 'for_train_val'] = False

        ind_train_val = [i for i in range(len(self.data['for_train_val'])) if self.data['for_train_val'][i]]
        ind_train, ind_val = train_test_split(ind_train_val, test_size=val_size)
        ind_test = [i for i in range(len(self.data['for_train_val'])) if not self.data['for_train_val'][i]]
        # Initialize vectorizer
        self.initialize_vectorizer()
        # Create bag-of-words text for train, validation and test data:
        x_train = self.vectorizer.fit_transform(self.data.loc[ind_train, 'text'])
        x_validation = self.vectorizer.transform(self.data.loc[ind_val, 'text'])
        x_test = self.vectorizer.transform(self.data.loc[ind_test, 'text'])
        # Process the labels
        lb = preprocessing.LabelBinarizer()
        y_train = lb.fit_transform(self.data.loc[ind_train, 'outcome']).flatten()
        y_validation = lb.fit_transform(self.data.loc[ind_val, 'outcome']).flatten()
        y_test = lb.transform(self.data.loc[ind_test, 'outcome']).flatten()

        return x_train, x_validation, x_test, y_train, y_validation, y_test

    def initialize_vectorizer(self):
        """
        Method that initializes the vectorizer (either count of words or tf-idf)

        """
        if self.vectorization_method == 'count':
            self.vectorizer = CountVectorizer(stop_words=self.stop_words, ngram_range=self.nagram_range,
                                              min_df=self.min_df)
        elif self.vectorization_method == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.nagram_range,
                                              min_df=self.min_df)
        else:
            print('Invalid vectorization method!')

    def save_vectorizer(self, path):
        """
        Method that saves the fitted vectorizer
        Args:
            path: path as a string where the vectorizer will be stored

        """
        dump(self.vectorizer, f"{path}/vectorizer_{self.vectorization_method}.joblib")


def clean_text(text, regex_cashtag, regex_ticker, regex_name, regex_clean_name,
               regex_pos_emoji, regex_neg_emoji, lemmer=None, stemmer=None):
    """
    Function that cleans the text of a tweet for before using on it a bag-of-words approach
    Args:
        text: the text of the tweet as a string
        regex_cashtag: the regular expression for the cashtag of the current company
        regex_ticker: regular expression of the company's ticker
        regex_name: regular expression of the company's name
        regex_clean_name: regular expression of the cleaned company name (e.g. without punctuation)
        regex_pos_emoji: regular expression of positive emojis
        regex_neg_emoji: regular expression of negative emojis
        lemmer: lemmer-object
        stemmer: stemmer-object

    Returns: cleaned text

    """
    # Transform text to unicode:
    text = unidecode(text)
    # Handle emoticons
    text = tag_emoticons(text, regex_pos_emoji, regex_neg_emoji)
    # Remove HTML tags:
    text = remove_html_tags(text)
    # Change encoding to remove non-english words
    text = text.encode("ascii", errors="ignore").decode()
    # Lower case all letters
    text = text.lower()
    # Remove "'s"
    text = re.sub(r"'s(?=\s)", ' ', text)
    # Replace usernames with "usernametag"
    text = tag_usernames(text)
    # Replace Twitter picuters with picturetag
    text = tag_twitter_pictures(text)
    # Replace URLs with "urltag"
    text = tag_url(text)
    # Tag business quarters
    text = tag_quarters(text)
    # Tag numbers
    text = tag_numbers(text)
    # Tag references to companies:
    text = tag_company(text, regex_cashtag, regex_ticker, regex_name, regex_clean_name)
    # Characters that appear more two or more times are shortened (e.g. loooool -> lool):
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    # Remove remaining punctuation
    text = re.sub('['+string.punctuation+']', ' ', text)
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    # Lemmatize or stemmatize text:
    text = lemmatize_stemmatize(text, lemmer, stemmer)
    return text


def tag_emoticons(text, positive_emoticons, negative_emoticons):
    """
    Function that handles emoticons in a text. Positive and negative emoticons are replaced by respective tags, other
    emoticons are instead removed from the text
    Args:
        text: text of to be modified as a string
        positive_emoticons: regular expression for positive emoticons as a string
        negative_emoticons: regular expression for negative emoticons as a string

    Returns: text with tagged emoticons

    """
    # Replace positive emojis:
    text = re.sub(positive_emoticons, ' emojipostag ', text)
    text = re.sub('(:[)])|(;[)])|(:-[)])|(=[)])|(:D)', ' emojipostag ', text)
    # Replace negative emojis:
    text = re.sub(negative_emoticons, ' emojinegtag ', text)
    text = re.sub('(:[(])|(:-[(])|(=[(])', ' emojinegtag ', text)
    # Remove other emojis:
    text = re.sub('[<][a-z0-9]+[>]', ' ', text)
    return text


def remove_html_tags(text):
    """
    Function that remove html tags
    Args:
        text: text from which html-tags have to be removed as a string

    Returns: text without html-tags as a string

    """
    clean_html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(clean_html, '', text)
    return text


def tag_usernames(text):
    """
    Function that replaces usernames by a tag
    Args:
        text: text as a string where the usernames have to be tagged

    Returns: text as a string with tags in place of usernames

    """
    text = re.sub(r'[@]\w+(?=\s|$)', ' usernametag ', text)
    return text


def tag_twitter_pictures(text):
    """
    Function that replaces picture-links of Twitter by a tag
    Args:
        text: text as a string where the picture-links have to be tagged

    Returns: text as a string with tags in place of picutre-links

    """
    text = re.sub(r'pic.twitter.com/[0-9a-zA-Z]*(?=\s|$)', ' picturetag ', text)
    return text


def tag_url(text):
    """
    Function that replaces URLs by a tag
    Args:
        text: text as a string where the URLs have to be tagged

    Returns: text as a string with tags in place of URLs

    """
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' urltag ', text)
    return text


def tag_quarters(text):
    """
    Function that tags references to the fiscal quarters in a text
    Args:
        text: text as a string

    Returns: text as string with tagged fiscal quarters

    """
    # Replace Q1 with first quarter tag:
    text = re.sub('q1', ' firstquartertag ', text)
    # Replace Q2 with first quarter tag:
    text = re.sub('q2', ' secondquartertag ', text)
    # Replace Q3 with first quarter tag:
    text = re.sub('q3', ' thirdquartertag ', text)
    # Replace Q4 with first quarter tag:
    text = re.sub('q4', ' fourthquartertag ', text)
    return text


def tag_numbers(text):
    """
    Function that replaces numbers by a tag
    Args:
        text: text as string for which we want to tag numbers

    Returns: text as string with tags instead of number information

    """
    # Replace percent numbers with tag:
    text = re.sub(r'([+-]*\d+[.,:]\d+[%])|([+-]*\d+[%])', ' numbertag ', text)
    # Replace numbers with tag:
    text = re.sub(r'([+-]*\d+[.,:]\d+)|([+-]*\d+)', ' numbertag ', text)
    return text


def tag_company(text, regex_cashtag, regex_ticker, regex_name, regex_clean_name):
    """
    Function that tags all references to companies in a text
    Args:
        text: text as a string
        regex_cashtag: regular expression as string that identifies a company's cashtag
        regex_ticker: regular expression as string that identifies a company's ticker
        regex_name: regular expression as string that identifies a company's name
        regex_clean_name: regular expression as string that identifies a company's cleaned name

    Returns: text as string with tagged company references

    """
    # Replace company cashtag
    text = re.sub(regex_cashtag, ' companycashtag ', text)
    # Replace company ticker
    text = re.sub(regex_ticker, ' companytickertag ', text)
    # Replace all other cashtags with a tag
    text = re.sub(r'[$]\b[a-zA-z]+\b', ' cashtag ', text)
    # Replace company name with tag:
    text = re.sub(regex_name, ' companynametag ', text)
    text = re.sub(regex_clean_name, ' companynametag ', text)
    return text


def lemmatize_stemmatize(text, lemmer=None, stemmer=None):
    """
    Function that lemmatizes and/or stemmatizes a text
    Args:
        text: text as string
        lemmer: a lemmer instance of the class nltk.stem.WordNetLemmatizer
        stemmer: a stemmer instance of the class nltk.stem,PorterStemmer

    Returns: lemmatized and/or stemmatized text as string

    """
    if stemmer is not None:
        text = ' '.join([stemmer.stem(word) for word in text.split(' ')])
    if lemmer is not None:
        text = ' '.join([lemmer.lemmatize(word) for word in text.split(' ')])
    return text
