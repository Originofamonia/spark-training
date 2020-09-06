# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
    1. use sklearn's MF (done)
    2. this baseline is MF(x_train)
"""
import sys
import itertools
from itertools import combinations
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
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


def normalize_s(x_train):
    s_norm = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    return s_norm


# def diversity(r_hat):
#     sim_matrix = np.dot(r_hat.T, r_hat)
#     sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
#     div_matrix = 1 - sim_matrix
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


def baseline2_inference(s_hat, test, rating_sahpe, pr_curve_filename):
    """
    sklearn version AUROC
    """
    fpr = dict()
    tpr = dict()
    # x_train, o_train, y_train = generate_xoy_binary(training, rating_sahpe)
    x_test, o_test, y_test = generate_xoy_binary(test, rating_sahpe)

    y_scores = s_hat[o_test > 0]  # exclude unobserved
    y_true = x_test[o_test > 0]  # [0, 1]
    a = len(np.unique(y_true))
    auc_score = roc_auc_score(y_true, y_scores)
    # for i in range(len(np.unique(y_true))):
    #     pass
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score


def draw_pr():
    prth_baseline = 'prth.npy'
    prth_baseline2 = 'prth_baseline2.npy'
    prth_hcf = 'prth_hcf.npy'
    pr_b1, re_b1, th_b1 = np.load(prth_baseline, allow_pickle=True)
    pr_b2, re_b2, th_b2 = np.load(prth_baseline2, allow_pickle=True)
    pr_h, re_h, th_h = np.load(prth_hcf, allow_pickle=True)

    base1 = plt.plot(re_b1, pr_b1, c='r', label='base1')
    base2 = plt.plot(re_b2, pr_b2, c='b', label='base2')
    hcf = plt.plot(re_h, pr_h, c='y', label='hcf')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend()
    plt.show()


def main():
    # load personal ratings
    pr_curve_filename = 'movieLens_base2.npy'
    movie_lens_home_dir = '../../data/movielens/medium/'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, test = split_ratings(ratings, 8)

    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))

    s = normalize_s(x_train)

    ranks = [16, 25]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0

    best_num_iter = -1

    for rank, num_iter in itertools.product(ranks, num_iters):
        s_hat = mf_sklearn(s, n_components=rank, n_iter=num_iter)  # [0, 23447]
        diversity_score = diversity(np.dot(s_hat.T, s_hat), s_hat)
        valid_auc = baseline2_inference(s_hat, test, (6041, 3953), pr_curve_filename)
        print("The current model was trained with rank = {}, and num_iter = {}, and its AUC on the "
              "validation set is {}.".format(rank, num_iter, valid_auc))
        if valid_auc > best_validation_auc:
            best_t = s_hat
            best_validation_auc = valid_auc
            best_rank = rank
            best_num_iter = num_iter

    test_auc = baseline2_inference(best_t, test, (6041, 3953), pr_curve_filename)
    print("The best model was trained with rank = {}, and num_iter = {}, and its AUC on the "
          "test set is {}.".format(best_rank, best_num_iter, test_auc))


if __name__ == "__main__":
    main()
    # draw_pr()
