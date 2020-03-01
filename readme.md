# Sentiment classification of StockTwits messages
## Introduction
This projects trains a Naive-Bayes model and a Logistic regression model for classifying the self-reported
sentiment (bearish or bullish) of StockTwits messages. 

## Set-up
The project is written using Python 3.7. The data processing and the modelling part are done using 
the `sklearn` module. Other helpful modules used for this project are `pandas`, `numpy`, `joblib` 
and `pytz`. For a quick installation of these dependencies, navigate to the root of the project and run
`pip install -r requirements.txt`.

## Project structure
The projects consist of mainly four folders:
* data: this folder is empty as the data is stored locally on the machine. The folder has the following 
structure:
    * 00_raw: the raw data and help-data
    * 01_processed: the output of the sentiment prediction, i.e. the daily sentiment
    * 02_models: the estimated models saved as `joblib` files
    * 03_model_output: csv-file with the model performance (accuracy, F1-score)
* notebook: jupyter-notebooks for the initial data analysis
* scripts: contains the scripts used to run the estimation and prediction
* src: contains functions and classes used used for preparing the data, estimating the models and
making the predictions.

## Estimation details
The models are estimated using StockTwits messages mentioning one of 360 US stocks of the S&P 500. 
For the estimation only messages from 2013-06-01 to 2014-08-31 are used; this matches the data 
estimation window used by [Renault (2017)](https://www.sciencedirect.com/science/article/abs/pii/S0378426617301589).
An oversampling technique is then used to create a balanced train data set. 
The models are estimated by running the script `model_estimation.py`. The textual data is vectorized 
by using a bag-of-words approach. More precisely, one- and two-grams are considered and word-document 
matrix is constructed using TF-IDF. The dictionary of words considered is defined using only training
data.

Predictions are made by running the file `model_predictions.py`. It can be specified if predictions
are made for all messages, only messages containing the companies cashtag, or only messages which
contain a unique cashtag. Moreover, it can be specified whether the sentiment is aggregated 
close-to-close (4pm to 4pm) or from open-to-open (9pm to 9pm). The prediction is made also for 
Twitter messages. The daily aggregation is done by a simple empirical average of intra-day sentiment
scores and by the bullish-measure of [Antweiler and Frank (2004)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00662.x).






