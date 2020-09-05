# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. T = concat(X, Y*Y.T*X)
    3. R* = X* + 2 * (Y*Y.T*X)
"""
import sys
import os
import itertools
# from math import sqrt
import numpy as np
# from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score, precision_recall_curve


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
add_path(root_path)

from machine_learning.movieLens.MovieLens_sklearn_hcf_nn import split_ratings
from machine_learning.movieLens.MovieLens_spark_hcf import generate_xoy, generate_xoy_binary,\
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
    mask1 = x_train > 0
    t1_norm = (x_train - np.min(x_train[mask1])) / (np.max(x_train[mask1]) - np.min(x_train[mask1]))
    t1_norm = t1_norm * mask1

    t2 = np.dot(y_train, np.dot(y_train.T, x_train))
    mask2 = t2 > 0
    t2_norm = (t2 - np.min(t2[mask2])) / (np.max(t2[mask2]) - np.min(t2[mask2]))  # only normalize t > 0
    t2_norm = t2_norm * mask2
    t2_norm[t2_norm < 2e-1] = 0

    t = np.concatenate((t1_norm, t2_norm), axis=0)

    return t  # [12082, 3953]


def hcf_inference(t_hat, training, test, rating_shape, pr_curve_filename):
    """
    sklearn version AUROC
    """
    t1_hat = t_hat[:int(t_hat.shape[0] / 2), :]
    t2_hat = t_hat[int(t_hat.shape[0] / 2):, :]
    mask1 = t1_hat > 0
    t1_hat_norm = (t1_hat - np.min(t1_hat[mask1])) / (np.max(t1_hat[mask1]) - np.min(t1_hat[mask1]))
    t1_hat_norm *= mask1

    mask2 = t2_hat > 0
    t2_hat_norm = (t2_hat - np.min(t2_hat[mask2])) / (np.max(t2_hat[mask2]) - np.min(t2_hat[mask2]))
    t2_hat_norm *= mask2

    r_hat = t1_hat_norm + 0.5 * t2_hat_norm
    x_test, o_test, y_test = generate_xoy_binary(test, rating_shape)

    # all_scores intersect with o_test
    all_scores_norm = (r_hat - np.min(r_hat)) / (np.max(r_hat) - np.min(r_hat))

    y_scores = all_scores_norm[o_test > 0]
    y_true = x_test[o_test > 0]
    auc_score = roc_auc_score(y_true, y_scores)
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score


def main():
    # load personal ratings
    pr_curve_filename = 'movieLen_hcf22.npy'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, test = split_ratings(ratings, 8)
    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))
    t = compute_t(x_train, y_train)

    ranks = [16, 25]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0

    best_num_iter = -1

    for rank, num_iter in itertools.product(ranks, num_iters):
        t_hat = mf_sklearn(t, n_components=rank, n_iter=num_iter)

        valid_auc = hcf_inference(t_hat, training, test, (6041, 3953), pr_curve_filename)
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
