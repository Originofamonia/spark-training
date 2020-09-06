# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)

"""
import sys
import itertools
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score
from itertools import combinations
import matplotlib.pyplot as plt


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
lib_dir = os.path.join(root_path, 'lib')
add_path(root_path)


from machine_learning.movieLens.MovieLens_spark_hcf import generate_xoy, generate_xoy_binary,\
    sigmoid, load_ratings
from machine_learning.movieLens.MovieLens_sklearn_hcf import mf_sklearn, split_ratings
from machine_learning.movieLens.MovieLens_sklearn_hcf2vcat import diversity


def compute_s(x_train):
    s = np.dot(x_train.T, x_train)
    s_norm = (s - np.min(s)) / (np.max(s) - np.min(s))
    return s_norm


# def diversity(s_hat, r_hat):
#     # s_hat is sim_matrix
#     s_hat = (s_hat - np.min(s_hat)) / (np.max(s_hat) - np.min(s_hat))
#     div_matrix = 1 - s_hat
#
#     k = 10
#     diversity_list = []
#     for i, row in enumerate(r_hat):
#         topk_indices = row.argsort()[-k:][::-1]
#         comb = np.array(list(combinations(topk_indices, 2)))
#         topk_diversity = div_matrix[comb[:, 0], comb[:, 1]]
#         diversity_list.append(np.sum(topk_diversity) / comb.shape[0])
#
#     diversity_score = sum(diversity_list) / r_hat.shape[0]
#
#     plt.hist(sorted(diversity_list, reverse=True), bins=50, color=np.random.rand(1, 3))
#     plt.grid()
#     plt.show()
#     return diversity_score


def baseline_inference(s_hat, training, test, rating_shape, pr_curve_filename):
    """
    sklearn version AUROC
    """
    x_train, o_train, y_train = generate_xoy_binary(training, rating_shape)
    x_test, o_test, y_test = generate_xoy_binary(test, rating_shape)

    all_scores = np.dot(x_train, s_hat)  # [6041, 3953]
    all_scores_norm = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))

    y_scores = all_scores_norm[o_test > 0]  # exclude unobserved
    y_true = x_test[o_test > 0] 
    auc_score = roc_auc_score(y_true, y_scores)
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score, all_scores_norm


def main():
    # load personal ratings
    pr_curve_filename = 'movieLens_base1.npy'
    movie_lens_home_dir = '../../data/movielens/medium/'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, test = split_ratings(ratings, 8)

    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))

    s = compute_s(x_train)

    ranks = [16, 25]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0

    best_num_iter = -1

    for rank, num_iter in itertools.product(ranks, num_iters):
        s_hat = mf_sklearn(s, n_components=rank, n_iter=num_iter)  # [0, 23447]
        valid_auc, all_scores_norm = baseline_inference(s_hat, training, test, (6041, 3953), pr_curve_filename)
        diversity_score = diversity(s_hat, all_scores_norm)
        print("The current model was trained with rank = {}, and num_iter = {}, and its AUC on the "
              "validation set is {}.".format(rank, num_iter, valid_auc))
        if valid_auc > best_validation_auc:
            best_t = s_hat
            best_validation_auc = valid_auc
            best_rank = rank
            best_num_iter = num_iter

    test_auc = baseline_inference(best_t, training, test, (6041, 3953), pr_curve_filename)
    print("The best model was trained with rank = {}, and num_iter = {}, and its AUC on the "
          "test set is {}.".format(best_rank, best_num_iter, test_auc))


if __name__ == "__main__":
    main()
