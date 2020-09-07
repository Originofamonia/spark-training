# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. stuck at line 195: model = ALS.train(t_rdd, rank, numIter, lmbda)
"""
import sys
import os
import itertools
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
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

from machine_learning.movieLens.MovieLens_sklearn_hcf_nn import split_ratings_by_time
from machine_learning.movieLens.MovieLens_spark_hcf import generate_xoy, generate_xoy_binary,\
    compute_t, load_ratings
from machine_learning.movieLens.MovieLens_sklearn_hcf2vcat import diversity, mf_sklearn, diversity_excludes_train


def hcf_inference(t_hat, training, test, rating_shape, pr_curve_filename):
    """
    sklearn version AUROC
    """
    t1_hat = t_hat[:int(t_hat.shape[0] / 2)]
    t2_hat = t_hat[int(t_hat.shape[0] / 2):]
    mask1 = t1_hat > 0
    t1_hat_norm = (t1_hat - np.min(t1_hat[mask1])) / (np.max(t1_hat[mask1]) - np.min(t1_hat[mask1]))
    t1_hat_norm *= mask1

    mask2 = t2_hat > 0
    t2_hat_norm = (t2_hat - np.min(t2_hat[mask2])) / (np.max(t2_hat[mask2]) - np.min(t2_hat[mask2]))
    t2_hat_norm *= mask2
    t_hat = np.concatenate((t1_hat_norm, t2_hat_norm), axis=0)
    x_train, o_train, y_train = generate_xoy(training, rating_shape)
    x_test, o_test, y_test = generate_xoy_binary(test, rating_shape)

    u = np.concatenate((x_train, 0.5 * y_train), axis=1)
    all_scores = np.dot(u, t_hat)  # [6041, 3953]
    # all_scores intersect with o_test
    mask = all_scores > 0
    all_scores_norm = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))
    all_scores_norm *= mask
    y_scores = all_scores_norm[o_test > 0]
    y_true = x_test[o_test > 0]
    auc_score = roc_auc_score(y_true, y_scores)
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score, all_scores_norm


def main():
    # load personal ratings
    movie_lens_home_dir = '../../data/movielens/medium/'
    pr_curve_filename = 'movieLen_base2.npy'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, test = split_ratings_by_time(ratings, 0.8)

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
        valid_auc, r_hat = hcf_inference(t_hat, training, test, (6041, 3953), pr_curve_filename)
        t1_hat = t_hat[:int(t_hat.shape[0] / 2), :]  # x.T * x
        sim_matrix = diversity_excludes_train(t1_hat, r_hat, o_train, x_train)
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
