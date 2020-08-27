# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)

"""
import sys
import itertools
from math import sqrt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
# from operator import add
# from os.path import join, isfile, dirname

from machine_learning.movieLens.MovieLens_spark_hcf import *
from machine_learning.movieLens.MovieLens_sklearn_hcf import mf_sklearn


def compute_s(x_train):
    s = np.dot(x_train.T, x_train)
    s_norm = (s - np.mean(s)) / np.std(s)
    s = sigmoid(s_norm)
    return s


def baseline_inference(s_hat, training, test):
    """
    sklearn version AUROC
    """
    x_train, o_train, y_train = generate_xoy_binary(training)
    x_test, o_test, y_test = generate_xoy_binary(test)

    all_scores = np.dot(x_train, s_hat)  # [6041, 3953]
    all_scores_norm = (all_scores - np.mean(all_scores)) / np.std(all_scores)
    all_scores = sigmoid(all_scores_norm)
    y_scores = all_scores[o_test > 0]  # exclude unobserved
    y_true = x_test[o_test > 0] 
    auc = roc_auc_score(y_true, y_scores)
    return auc


def pyspark_auc(model, data):
    """
    https://spark.apache.org/docs/2.4.0/mllib-evaluation-metrics.html
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
    # load personal ratings
    movie_lens_home_dir = '../../data/movielens/medium/'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, validation, test = split_ratings(ratings, 6, 8)

    x_train, o_train, y_train = generate_xoy(training)

    # train_mat = coo_matrix((training[:, 2], (training[:, 0], training[:, 1])), shape=(6041, 3953)).toarray()
    # test_mat = coo_matrix((test[:, 2], (test[:, 0], test[:, 1])), shape=(6041, 3953)).toarray()

    s = compute_s(x_train)

    ranks = [30, 40]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0

    best_num_iter = -1

    for rank, num_iter in itertools.product(ranks, num_iters):
        s_hat = mf_sklearn(s, n_components=rank, n_iter=num_iter)  # [0, 23447]
        valid_auc = baseline_inference(s_hat, training, validation)
        print("The current model was trained with rank = {}, and num_iter = {}, and its AUC on the "
              "validation set is {}.".format(rank, num_iter, valid_auc))
        if valid_auc > best_validation_auc:
            best_t = s_hat
            best_validation_auc = valid_auc
            best_rank = rank
            best_num_iter = num_iter

    test_auc = baseline_inference(best_t, training, test)
    print("The best model was trained with rank = {}, and num_iter = {}, and its AUC on the "
          "test set is {}.".format(best_rank, best_num_iter, test_auc))


if __name__ == "__main__":
    main()
