# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. use NN to train rating = g(u+, w, u-)
"""
import sys
import os
import itertools
import torch
import numpy as np
# from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score, precision_recall_curve
from machine_learning.movieLens.hcf_nn import Hcf
# from os.path import join, isfile, dirname


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
add_path(root_path)


from machine_learning.movieLens.MovieLens_spark_hcf import generate_xoy, generate_xoy_binary, split_ratings,\
    compute_t, sigmoid, load_ratings


def mf_sklearn(t, n_components, n_iter):
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=n_iter)
    w = model.fit_transform(t)  # MF
    h = model.components_
    t_hat = np.dot(w, h)  # matrix completion [1783， 0]
    # a, b = np.max(t_hat), np.min(t_hat)
    return t_hat


def get_u_w(x, y, t, n_components, n_iter):
    split = int(len(t) / 2)

    t1 = t[: split]  # x.T * x
    t2 = t[split:]  # y.T * x
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=n_iter)
    p = model.fit_transform(t1)
    q = model.components_
    r = model.fit_transform(t2)
    s = model.components_

    pos_u = np.dot(x, p)
    neg_u = np.dot(y, r)
    w = np.dot(s, q.T)
    return w, pos_u, neg_u


def hcf_inference(t_hat, training, test, rating_shape, pr_curve_filename):
    """
    sklearn version AUROC
    """
    x_train, o_train, y_train = generate_xoy(training, rating_shape)
    x_test, o_test, y_test = generate_xoy_binary(test, rating_shape)
    # a = np.unique(x_test)
    # b = np.count_nonzero(x_test)
    u = np.concatenate((x_train, 0.2 * y_train), axis=1)
    all_scores = np.dot(u, t_hat)  # [6041, 3953]
    # all_scores intersect with o_test
    all_scores_norm = (all_scores - np.mean(all_scores)) / np.std(all_scores)
    all_scores = sigmoid(all_scores_norm)
    y_scores = all_scores[o_test > 0]
    y_true = x_test[o_test > 0]
    auc_score = roc_auc_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 20
    num_epochs = 10
    lr = 1e-3

    movie_lens_home_dir = '../../data/movielens/medium/'
    pr_curve_filename = 'movieLen_base2.npy'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, validation, test = split_ratings(ratings, 6, 8)

    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))
    # x_train, o_train, y_train = generate_xoy_binary(training)

    t = compute_t(x_train, y_train)

    ranks = [16, 25]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0

    best_num_iter = -1
    net = Hcf().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    for rank, num_iter in itertools.product(ranks, num_iters):
        w, pos_u, neg_u = get_u_w(x_train, y_train, t, rank, num_iter)

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
