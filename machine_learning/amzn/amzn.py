# distribution is the same as movieLens, Nflx, abort.
# https://www.kaggle.com/qwikfix/amazon-recommendation-dataset?select=database.sqlite
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer


def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'


def get_data():
    file = 'Reviews.csv'
    con = sqlite3.connect('database.sqlite')
    messages = pd.read_sql_query("""SELECT Score, Summary
                FROM Reviews
                WHERE Score != 3""", con)

    Score = messages['Score']
    Score = Score.map(partition)
    Summary = messages['Summary']
    X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=0.2, random_state=42)


def main():
    ratings = get_data()
    # count = np.unique(ratings[:, 2], return_counts=True)
    # print(count)


if __name__ == '__main__':
    main()
