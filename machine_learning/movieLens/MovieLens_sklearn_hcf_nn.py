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
from tqdm import tqdm
import random
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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


from machine_learning.movieLens.utils import generate_xoy, generate_xoy_binary, load_ratings


def split_ratings(ratings, b1):
    """
    :param ratings: matrix, row is (i, j, value, timestamp)
    :param b1: boundary1: between training and validation
    :return: training, test: [i, j, rating]
    """
    training = np.array([row for row in ratings if row[3] % 10 < b1])  # [0, 5]
    test = np.array([row for row in ratings if b1 <= row[3] % 10])  # [8, 9]
    training = np.delete(training, 3, 1)
    test = np.delete(test, 3, 1)

    return training, test


def split_ratings_by_time(ratings, b1):
    """
    :param ratings: matrix, row is (i, j, value, timestamp)
    :param b1: boundary1: between training and validation
    :return: training, test: [i, j, rating]
    """
    time_order = ratings[:, 3].argsort()
    new_ratings = ratings[time_order]
    training = new_ratings[:int(new_ratings.shape[0] * b1)]
    test = new_ratings[int(new_ratings.shape[0] * b1):]
    training = np.delete(training, 3, 1)
    test = np.delete(test, 3, 1)

    return training, test


def mf_sklearn(t, n_components, n_iter):
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=n_iter)
    w = model.fit_transform(t)  # MF
    h = model.components_
    t_hat = np.dot(w, h)  # matrix completion [1783， 0]
    # a, b = np.max(t_hat), np.min(t_hat)
    return t_hat


def dense_to_sparse(o):
    """
    convert dense matrix (2d) to sparse list of tuple (i, j, value)
    :return:
    """
    o_list_tuple = []
    for i in tqdm(range(o.shape[0])):
        for j in range(o.shape[1]):
            if o[i][j] > 1e-1:
                o_list_tuple.append((i, j, o[i][j]))

    return o_list_tuple


def get_u_v_label(x, o, y, t, n_components, n_iter):
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
    u = np.concatenate((pos_u, neg_u), axis=1)  # [6041, 32]
    v = np.concatenate((q, s), axis=0).T  # [3953, 32]
    o_list = dense_to_sparse(o)

    np.save('uv_16.npy', (u, v, x, o_list, y))
    return u, v


def build_dataset(uv_file):
    u, v, x, o_list, y = np.load(uv_file, allow_pickle=True)
    samples = o_list[np.random.choice(len(o_list), size=5, replace=False)]  # sample from o_list
    u_sample = u[samples[:, 0]]
    v_sample = v[:, samples[:, 1]]
    x_sample = x[samples[:, 0], samples[:, 1]]
    y_sample = y[samples[:, 0], samples[:, 1]]
    print(y_sample)


def hcf_nn_inference(net, uv_file, device):
    net.eval()
    u, v, x, _, y = np.load(uv_file, allow_pickle=True)
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    training, test = split_ratings(ratings, 8)
    x_test, o_test, y_test = generate_xoy_binary(test, (6041, 3953))
    # o_list = dense_to_sparse(o_test)
    # np.save('test_o_list.npy', o_list)
    o_list = np.load('test_o_list.npy')
    y_true = x_test[o_test > 0]
    y_hat = np.zeros(y_true.shape)
    for i, row in enumerate(o_list):
        i_index = int(row[0])
        j_index = int(row[1])
        u_vec = torch.from_numpy(u[i_index]).unsqueeze(0).to(device)
        v_vec = torch.from_numpy(v[j_index]).unsqueeze(0).to(device)

        rating_hat = net(u_vec, v_vec)
        y_hat[i] = rating_hat.squeeze(0).cpu().detach().numpy()[0]

    auc_score = roc_auc_score(y_true, y_hat)
    precision, recall, thresholds = precision_recall_curve(y_true, y_hat)
    return auc_score


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
    all_scores_norm = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))

    y_scores = all_scores_norm[o_test > 0]
    y_true = x_test[o_test > 0]
    auc_score = roc_auc_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score


def main():
    rank = 25
    num_iter = 2000
    # path = '../../data/movielens/medium/ratings.dat'
    # ratings = load_ratings(path)
    # training, test = split_ratings(ratings, 8)
    # x_train, o_train, y_train = generate_xoy(training, (6041, 3953))
    # t = compute_t(x_train, y_train)
    # u, v = get_u_v_label(x_train, o_train, y_train, t, rank, num_iter)
    uv_file = 'uv_25.npy'
    u, v, x, o_list, y = np.load(uv_file, allow_pickle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 50
    lr = 1e-3

    net = Hcf(in_feature=rank*2).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    for i in range(num_iter):
        # sample_index = o_list[np.random.choice(len(o_list), size=batch_size, replace=False)]
        sample_index = random.sample(o_list, batch_size)
        i_index = [x[0] for x in sample_index]
        j_index = [x[1] for x in sample_index]
        u_sample = torch.from_numpy(u[i_index]).to(device)
        v_sample = torch.from_numpy(v[j_index]).to(device)
        x_sample = x[i_index, j_index]
        y_sample = y[i_index, j_index]
        label = torch.from_numpy(np.array((x_sample, y_sample)).T).to(device)
        optimizer.zero_grad()

        rating_hat = net(u_sample, v_sample)
        loss = F.mse_loss(rating_hat, label)
        loss.backward()
        optimizer.step()

    auc_score = hcf_nn_inference(net, uv_file, device)
    print('auc score: {}'.format(auc_score))


if __name__ == "__main__":
    main()
    # uv_file = 'uv_16.npy'
    # build_dataset(uv_file)
