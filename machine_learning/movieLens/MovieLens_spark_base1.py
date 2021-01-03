# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. stuck at line 195: model = ALS.train(t_rdd, rank, numIter, lmbda)
"""
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
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import SparkSession
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
add_path(root_path)

from machine_learning.movieLens.MovieLens_spark_hcf import spark_matrix_completion
from machine_learning.movieLens.MovieLens_sklearn_hcf_nn import split_ratings_by_time
from machine_learning.movieLens.MovieLens_sklearn_hcf2vcat import diversity_excludes_train, diversity_rerank
from machine_learning.movieLens.utils import load_ratings, generate_xoy, generate_xoy_binary


def compute_s(x_train):
    s = np.dot(x_train.T, x_train)
    mask = s > 0
    s_norm = (s - np.min(s[mask])) / (np.max(s[mask]) - np.min(s[mask]))  # only normalize t1 > 0
    s_norm = s_norm * mask
    s_norm[s_norm < 1e-1] = 0

    return s_norm


def parse_s(t):
    """
    convert sparse matrix (2d) to list of tuple (i, j, value)
    :return: t_list_tuple
    """
    t_list_tuple = []
    for i in tqdm.tqdm(range(t.shape[0])):
        for j in range(t.shape[1]):
            if t[i][j] > 1e-2:
                t_list_tuple.append((i, j, t[i][j]))

    # sparse_t = csr_matrix(t, dtype=float).tocoo()  # used for spark
    # a = np.unique(sparse_t.data, return_counts=True)
    # mat = np.vstack((sparse_t.row, sparse_t.col, sparse_t.data)).T  # has data == 0
    # t_list_tuple = list(map(tuple, mat))
    return t_list_tuple


def normalize_validation(validation):
    validation[:, 2] = validation[:, 2] / 5
    return validation


def get_list_tuples():
    # load personal ratings
    pkl_file = 'base1.pkl'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)  # [i, j, rating, timestamp]
    training, test = split_ratings_by_time(ratings, 0.8)  # (i, j, value)
    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))
    if not os.path.isfile(pkl_file):
        s = compute_s(x_train)

        # train models and evaluate them on the validation set
        s_list_tuple = parse_s(s)
        with open(pkl_file, 'wb') as fh:
            pickle.dump(s_list_tuple, fh)
        exit()
    else:
        with open(pkl_file, "rb") as fh:
            s_list_tuple = pickle.load(fh)

    test = normalize_validation(test)

    test_list_tuple = list(map(tuple, test))  # i, j, value
    return s_list_tuple, test_list_tuple, o_train, x_train


def manual_inference(s_hat):
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)  # [i, j, rating, timestamp]
    training, test = split_ratings_by_time(ratings, 0.8)
    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))
    x_test, o_test, y_test = generate_xoy_binary(test, (6041, 3953))

    all_scores = np.dot(x_train, s_hat)  # [6041, 3953]
    # all_scores intersect with o_test
    all_scores_norm = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))
    y_scores = all_scores_norm[o_test > 0]
    y_true = x_test[o_test > 0]
    auc_score = roc_auc_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.show()
    return auc_score, all_scores_norm


def spark_inference(model, data):
    """
    :param model:
    :param data:
    :return:
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictions_and_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
        .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
        .values()

    metrics = BinaryClassificationMetrics(predictions_and_ratings)
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)

    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    return metrics.areaUnderROC


def main():
    train_list_tuple, test_list_tuple, o_train, x_train = get_list_tuples()
    # set up environment
    spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "7g") \
        .getOrCreate()
    sc = spark.sparkContext

    num_partitions = 2
    t_rdd = sc.parallelize(train_list_tuple).filter(lambda x: x[2] > 0).repartition(num_partitions)
    test_rdd = sc.parallelize(test_list_tuple) \
        .map(lambda x: (x[0], x[1], float(x[2])))\
        .repartition(num_partitions)
    ranks = [16]
    lambdas = [0.1]
    num_iters = [10]
    best_model = None
    best_validation_auc = float("-inf")
    best_rank = 0
    best_lambda = -1.0
    best_num_iter = -1
    start_time = time()
    for rank, lmbda, numIter in itertools.product(ranks, lambdas, num_iters):

        model = ALS.train(t_rdd, rank, numIter, lmbda, nonnegative=True, seed=999)
        s_hat = spark_matrix_completion(model, (3953, 3953), rank)  # s_hat: [3953, 3953]
        validation_auc, r_hat = manual_inference(s_hat)
        div_score = diversity_rerank(s_hat, r_hat, o_train, x_train)
        print("The current model was trained with rank = {} and lambda = {}, and numIter = {}, and its AUC on the "
              "validation set is {}.".format(rank, lmbda, numIter, validation_auc))
        if validation_auc > best_validation_auc:
            best_model = model
            best_validation_auc = validation_auc
            best_rank = rank
            best_lambda = lmbda
            best_num_iter = numIter

    test_auc = spark_inference(best_model, test_rdd)
    end_time = time() - start_time
    # evaluate the best model on the test set
    print("The best model was trained with rank = {} and lambda = {}, and numIter = {}, and its AUC on the test set is"
          " {}; runtime is {}".format(best_rank, best_lambda, best_num_iter, test_auc, end_time))

    # clean up
    sc.stop()


if __name__ == "__main__":
    main()
