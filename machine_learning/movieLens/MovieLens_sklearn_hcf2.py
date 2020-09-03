# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. T = concat(X, Y), evaluate on left half of T* only
"""
import sys
import os
import itertools
# from math import sqrt
import numpy as np
# from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score, precision_recall_curve
# from operator import add
# from os.path import join, isfile, dirname


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
add_path(root_path)


from machine_learning.movieLens.MovieLens_spark_hcf import generate_xoy, generate_xoy_binary, split_ratings,\
    sigmoid, load_ratings


def mf_sklearn(t, n_components, n_iter):
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=n_iter)
    w = model.fit_transform(t)  # MF
    h = model.components_
    t_hat = np.dot(w, h)  # matrix completion [1783， 0]
    # a, b = np.max(t_hat), np.min(t_hat)
    return t_hat


def compute_t(x_train, y_train):
    t = np.concatenate((x_train, y_train), axis=0)

    mask = t > 0
    t_norm = (t - np.min(t[mask])) / (np.max(t[mask]) - np.min(t[mask]))  # only normalize t > 0
    t_norm = t_norm * mask
    t_norm[t_norm < 2e-1] = 0

    return t_norm


def hcf_inference(t_hat, training, test, rating_shape, pr_curve_filename):
    """
    sklearn version AUROC
    """
    t_hat = t_hat[:int(t_hat.shape[0] / 2), :]
    x_test, o_test, y_test = generate_xoy_binary(test, rating_shape)

    # all_scores intersect with o_test
    all_scores_norm = (t_hat - np.min(t_hat)) / (np.max(t_hat) - np.min(t_hat))

    y_scores = all_scores_norm[o_test > 0]
    y_true = x_test[o_test > 0]
    auc_score = roc_auc_score(y_true, y_scores)
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score


def main():
    # load personal ratings
    pr_curve_filename = 'movieLen_base2.npy'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, validation, test = split_ratings(ratings, 6, 8)
    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))
    t = compute_t(x_train, y_train)

    ranks = [8, 12]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0

    best_num_iter = -1

    for rank, num_iter in itertools.product(ranks, num_iters):
        t_hat = mf_sklearn(t, n_components=rank, n_iter=200)

        valid_auc = hcf_inference(t_hat, training, validation, (6041, 3953), pr_curve_filename)
        print("The current model was trained with rank = {}, and num_iter = {}, and its AUC on the "
              "validation set is {}.".format(rank, num_iter, valid_auc))
        if valid_auc > best_validation_auc:
            best_t = t_hat
            best_validation_auc = valid_auc
            best_rank = rank
            best_num_iter = num_iter

    test_auc = hcf_inference(best_t, training, test, (6041, 3953), pr_curve_filename)
    print("The best model was trained with rank = {}, and num_iter = {}, and its AUC on the "
          "test set is {}.".format(best_rank, best_num_iter, test_auc))


if __name__ == "__main__":
    main()
