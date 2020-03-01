from __future__ import print_function
# noinspection PyUnresolvedReferences
import script_helper
from src.data_preparation import TextHandler
from src.load_help_data import get_help_data
from src.global_variables import RAW_DATA_PATH, STOCKTWITS_DATA_PATH, MODEL_DATA_PATH, OUTPUT_DATA_PATH
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from joblib import dump
from src.prediction_functions import predict_in_chunks
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Define vectorization method:
vectorization_method = 'tfidf'
# Load help data:
company_mapping, emojis_positive, emojis_negative, _ = get_help_data(path=RAW_DATA_PATH)
# Initialize the lemmatizer:
lemmer = WordNetLemmatizer()
# Initialize data handler:
th_stocktwits = TextHandler(data_path=STOCKTWITS_DATA_PATH, mapping=company_mapping,
                            tz_in='UTC', tz_out='America/New_York',
                            outcome_name='StockTwits_sentiment', datetime_name='tweet_datetime', text_name='text',
                            vectorization_method=vectorization_method, ngram_range=(1, 2),
                            stop_words=['a', 'an', 'the'], min_df=0.001)
# Load data:
th_stocktwits.load_text(emojis_positive, emojis_negative, estimation=True, lemmer=lemmer)
# Create feature matrix and split the data set in train, validation and test set
x_train, x_validation, x_test, y_train, y_validation, y_test = \
    th_stocktwits.get_train_validation_test('2013-06-01', '2014-08-31', balanced=True, seed=11)
# Save the fitted vectorizer
th_stocktwits.save_vectorizer(MODEL_DATA_PATH)

# Logistic regression:
logistic_cv = LogisticRegressionCV(random_state=0, max_iter=2000, cv=5, Cs=20, scoring='f1').fit(x_train, y_train)
y_logistic_validation = logistic_cv.predict(x_validation)
y_logistic_test = logistic_cv.predict(x_test)
# Save estimated logistic model:
dump(logistic_cv, f"{MODEL_DATA_PATH}/logistic_{vectorization_method}.joblib")

# Naive Bayes:
naive_bayes = MultinomialNB().fit(x_train.toarray(), y_train)
y_naive_bayes_validation = predict_in_chunks(naive_bayes, x_validation, 10000)
y_naive_bayes_test = predict_in_chunks(naive_bayes, x_test, 10000)
# Save estimated naive bayes:
dump(naive_bayes, f"{MODEL_DATA_PATH}/naive_bayes_{vectorization_method}.joblib")

# Compute performance
data = {'Model': ['Logistic', 'Logistic', 'Naive-Bayes', 'Naive-Bayes'],
        'Data': ['Validation', 'Test', 'Validation', 'Test'],
        'Accuracy': [accuracy_score(y_validation, y_logistic_validation),
                     accuracy_score(y_test, y_logistic_test),
                     accuracy_score(y_validation, y_naive_bayes_validation),
                     accuracy_score(y_test, y_naive_bayes_test)],
        'F1': [f1_score(y_validation, y_logistic_validation),
               f1_score(y_test, y_logistic_test),
               f1_score(y_validation, y_naive_bayes_validation),
               f1_score(y_test, y_naive_bayes_test)]}
model_performance = pd.DataFrame(data)
# Export performance
model_performance.to_csv(f"{OUTPUT_DATA_PATH}/model_performance_{vectorization_method}.csv", index=False)
