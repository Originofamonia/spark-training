# Download the data from: https://www.kaggle.com/laowingkin/netflix-movie-recommendation
import pandas as pd
import numpy as np
import itertools
import sys
import os
from scipy.sparse import coo_matrix, csr_matrix
# import seaborn as sns


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
add_path(root_path)

from machine_learning.movieLens.MovieLens_spark_hcf import generate_xoy, parse_xoy, parse_xoy_binary, compute_t
from machine_learning.movieLens.MovieLens_sklearn_hcf import mf_sklearn, hcf_inference
from machine_learning.movieLens.MovieLens_sklearn_baseline2 import baseline2_inference, normalize_s
from machine_learning.netflix.netflix_sklearn_hcf import split_nflx_ratings, gen_nflx_xoy


def main():
    rating_filename = "nflx_rating.npy"
    # ratings = get_nflx_rating()  # only need run once
    # np.save(rating_filename, ratings)

    ratings = np.load(rating_filename)
    training, validation, test = split_nflx_ratings(ratings, 0.6, 0.8)
    x_train, o_train, y_train = gen_nflx_xoy(training, ratings.shape)
    # x_train, o_train, y_train = gen_nflx_xoy_binary(training, ratings.shape)

    s = normalize_s(x_train)

    ranks = [30, 40]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0
    best_num_iter = 0

    for rank, num_iter in itertools.product(ranks, num_iters):
        s_hat = mf_sklearn(s, n_components=rank, n_iter=num_iter)
        valid_auc = baseline2_inference(s_hat, validation, ratings.shape)
        print("The current model was trained with rank = {}, and num_iter = {}, and its AUC on the "
              "validation set is {}.".format(rank, num_iter, valid_auc))
        if valid_auc > best_validation_auc:
            best_t = s_hat
            best_validation_auc = valid_auc
            best_rank = rank
            best_num_iter = num_iter

    test_auc = baseline2_inference(best_t, test, ratings.shape)
    print("The best model was trained with rank = {}, and num_iter = {}, and its AUC on the "
          "test set is {}.".format(best_rank, best_num_iter, test_auc))


if __name__ == '__main__':
    main()
