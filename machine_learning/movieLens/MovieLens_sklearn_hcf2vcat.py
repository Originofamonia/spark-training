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
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score, precision_recall_curve
from random import random


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
add_path(root_path)

from machine_learning.movieLens.MovieLens_sklearn_hcf_nn import split_ratings_by_time
from machine_learning.movieLens.utils import generate_xoy, generate_xoy_binary, load_ratings


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
    # t2_norm[t2_norm < 1e-1] = 0

    t = np.concatenate((t1_norm, t2_norm), axis=0)

    return t  # [12082, 3953]


def diversity(sim_matrix, r_hat):
    # mask = sim_matrix > 0
    sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
    # sim_matrix *= mask
    div_matrix = 1 - sim_matrix

    k = 10
    diversity_list = []
    for i, row in enumerate(r_hat):
        topk_indices = row.argsort()[-k:][::-1]
        comb = np.array(list(combinations(topk_indices, 2)))
        topk_diversity = div_matrix[comb[:, 0], comb[:, 1]]
        diversity_list.append(np.sum(topk_diversity) / comb.shape[0])

    diversity_score = sum(diversity_list) / r_hat.shape[0]

    plt.hist(sorted(diversity_list, reverse=True), bins=50, color=np.random.rand(1, 3))
    plt.grid()
    plt.show()
    return diversity_score


def diversity_excludes_train(sim_matrix, r_hat, o_train, x_train):
    mask = sim_matrix > 0
    sim_matrix = (sim_matrix - np.min(sim_matrix[mask])) / (np.max(sim_matrix[mask]) - np.min(sim_matrix[mask]))
    sim_matrix *= mask
    div_matrix = 1 - sim_matrix
    r_hat = r_hat * (o_train < 1)  # f1: must have

    k = 50
    all_users_div = []
    for i, row in enumerate(r_hat):
        # if np.max(x_train[i]) < 0.5:  # f2: user must have positive rating in x_train (optional)
        #     continue

        topk_indices = row.argsort()[-k:][::-1]
        topk_diversity = get_user_div_list(div_matrix, topk_indices)
        all_users_div.append(np.mean(topk_diversity))

    avg_div = sum(all_users_div) / len(all_users_div)
    diversity_median = sorted(all_users_div)[int(len(all_users_div) / 2)]
    diversity_quarter = sorted(all_users_div)[int(len(all_users_div) / 4)]

    np.save('hcf1_div.npy', all_users_div)
    plt.hist(sorted(all_users_div, reverse=True), bins=50, color=np.random.rand(1, 3))
    plt.grid()
    plt.xlabel('Diversity')
    plt.show()
    return avg_div


def get_user_div_list(div_matrix, topk_indices):
    comb = np.array(list(combinations(topk_indices, 2)))
    topk_diversity = div_matrix[comb[:, 0], comb[:, 1]]
    return topk_diversity


def diversity_rerank(sim_matrix, r_hat, o_train, x_train):
    # rerank by largest diversity among topk R*
    mask = sim_matrix > 0
    sim_matrix = (sim_matrix - np.min(sim_matrix[mask])) / (np.max(sim_matrix[mask]) - np.min(sim_matrix[mask]))
    sim_matrix *= mask
    div_matrix = 1 - sim_matrix
    r_hat = r_hat * (o_train < 1)  # f1: must have

    k = 50
    topk2 = 9
    all_users_div = []
    for i, row in enumerate(r_hat):  # for each user
        # if np.max(x_train[i]) < 0.5:  # f2: user must have positive rating in x_train (optional)
        #     continue

        div_idx_list = []
        topk_indices = row.argsort()[-k:][::-1]
        div_idx_list.append(topk_indices[0])
        topk_indices = np.delete(topk_indices, 0)  # delete by index
        for i in range(topk2):
            candidate_idx = select_largest_div(div_idx_list, div_matrix, topk_indices)
            div_idx_list.append(topk_indices[candidate_idx])
            topk_indices = np.delete(topk_indices, candidate_idx)

        user_div_list = get_user_div_list(div_matrix, div_idx_list)
        all_users_div.append(np.mean(user_div_list))

    avg_div = sum(all_users_div) / len(all_users_div)
    diversity_median = sorted(all_users_div)[int(len(all_users_div) / 2)]
    diversity_quarter = sorted(all_users_div)[int(len(all_users_div) / 4)]

    np.save('base1_rerank.npy', all_users_div)
    plt.hist(sorted(all_users_div, reverse=True), bins=50, color=np.random.rand(1, 3))
    plt.grid()
    plt.xlabel('Diversity')
    plt.show()
    return avg_div


def select_largest_div(div_list, div_matrix, topk_indices):
    largest_avg_div = float('-inf')
    candidate_idx = -1
    for i, idx in enumerate(topk_indices):  # for topk item indices
        candidate_div_list = div_matrix[div_list, idx]
        avg_div = np.mean(candidate_div_list)
        if avg_div > largest_avg_div:
            candidate_idx = i
            largest_avg_div = avg_div

    return candidate_idx


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
    return auc_score, t1_hat_norm, r_hat


def main():
    # load personal ratings
    pr_curve_filename = 'movieLen_hcf22.npy'
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

        valid_auc, t1_hat_norm, r_hat = hcf_inference(t_hat, training, test, (6041, 3953), pr_curve_filename)
        diversity_score = diversity_excludes_train(np.dot(t1_hat_norm.T, t1_hat_norm), r_hat, o_train, x_train)
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
