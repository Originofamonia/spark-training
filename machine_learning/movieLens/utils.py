import itertools
import os
import pickle
import sys
from time import time
from os.path import isfile

import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel
from pyspark.sql import SparkSession
from scipy.sparse import coo_matrix


def parse_xoy(mat, n_users, n_items):
    """
    Parses a sparse matrix to x, o, y
    :return:
    """
    o = np.clip(mat, 0, 1)  # O
    # x = (mat >= 3) * np.ones((n_users, n_items))  # X, split [0, 1, 2] -> 0, [3, 4, 5] -> 1
    x = mat / 5
    # y = o - x
    y = (6 - mat) / 5  # Y
    return x, o, y


def parse_xoy_binary(mat, n_users, n_items):
    """
    Parses a sparse matrix to x, o, y
    :return:
    """
    o = np.clip(mat, 0, 1)  # O
    x = (mat >= 3) * np.ones((n_users, n_items))  # X, split [0, 1, 2] -> 0, [3, 4, 5] -> 1
    # x = mat / 5
    y = o - x
    # y = (6 - mat) / 5  # Y
    return x, o, y


def generate_xoy(coo_mat, rating_shape):
    """
    convert coordinate matrix [i, j, value] to sparse matrix (2d)
    :return: sparse matrix (2d)
    """
    mat = coo_matrix((coo_mat[:, 2], (coo_mat[:, 0], coo_mat[:, 1])), shape=rating_shape).toarray()
    x, o, y = parse_xoy(mat, mat.shape[0], mat.shape[1])
    x = x.astype(np.float32)
    o = o.astype(np.float32)
    y = y.astype(np.float32)
    return x, o, y


def generate_xoy_binary(coo_mat, rating_shape):
    """
    convert coordinate matrix [i, j, value] to sparse matrix (2d)
    :return: sparse matrix (2d)
    """
    mat = coo_matrix((coo_mat[:, 2], (coo_mat[:, 0], coo_mat[:, 1])), shape=rating_shape).toarray()
    x, o, y = parse_xoy_binary(mat, mat.shape[0], mat.shape[1])
    return x, o, y


def load_ratings(ratings_file):
    """
    Load ratings from file into ndarray
    return: ndarray, [i, j, rating, timestamp]
    """
    if not isfile(ratings_file):
        print("File %s does not exist." % ratings_file)
        sys.exit(1)
    f = open(ratings_file, 'r')
    ratings = np.loadtxt(ratings_file, dtype=int, delimiter='::')
    f.close()
    if not ratings.any():
        print("No ratings provided.")
        sys.exit(1)
    else:
        return ratings
