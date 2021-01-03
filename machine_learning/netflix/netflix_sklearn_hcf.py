# Download the data from: https://www.kaggle.com/laowingkin/netflix-movie-recommendation
import pandas as pd
import numpy as np
import itertools
import sys
import os
from scipy.sparse import coo_matrix, csr_matrix


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
add_path(root_path)

from machine_learning.movieLens.MovieLens_spark_hcf import generate_xoy, parse_xoy, parse_xoy_binary, compute_t
from machine_learning.movieLens.MovieLens_sklearn_hcf import mf_sklearn, hcf_inference


def get_nflx_rating():
    # Skip date
    df1 = pd.read_csv('../nflx_data/combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df2 = pd.read_csv('../nflx_data/combined_data_2.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df3 = pd.read_csv('../nflx_data/combined_data_3.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df4 = pd.read_csv('../nflx_data/combined_data_4.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

    print('Dataset 1 shape: {}'.format(df1.shape))

    # load less data for speed
    df = df1
    df = df.append(df2)  # Uncomment these to use the whole dataset
    df = df.append(df3)
    df = df.append(df4)
    df.index = np.arange(0, len(df))
    print('Full dataset shape: {}'.format(df.shape))

    # Data cleaning
    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        # numpy approach
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Account for last record and corresponding length
    # numpy approach
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    print('Movie numpy: {}'.format(movie_np))
    print('Length: {}'.format(len(movie_np)))

    # remove those Movie ID rows
    df = df[pd.notnull(df['Rating'])]

    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)

    # Data slicing
    f = ['count', 'mean']
    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    print('Movie minimum times of review: {}'.format(movie_benchmark))

    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.7), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    print('Customer minimum times of review: {}'.format(cust_benchmark))

    print('Original Shape: {}'.format(df.shape))
    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]
    print('After Trim Shape: {}'.format(df.shape))

    # df_p is the rating matrix
    df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')
    print(df_p.shape)
    df_p = df_p.fillna(0)
    rating = df_p.to_numpy()

    return rating


def sparse_to_coo(t):
    """
    convert sparse matrix (2d) to list of tuple (i, j, value)
    :return: [i, j, rating]
    """
    sparse_t = csr_matrix(t, dtype=float).tocoo()
    coo = np.vstack((sparse_t.row, sparse_t.col, sparse_t.data)).T
    np.random.shuffle(coo)
    return coo


def split_nflx_ratings(ratings, b1):
    """
    :param ratings: sparse matrix
    :return: training, validation, test: [i, j, rating]
    """
    coo = sparse_to_coo(ratings)
    full_len = len(coo)
    training = coo[:int(full_len * b1)]
    test = coo[int(full_len * b1):]

    return training, test


def gen_nflx_xoy(coo_mat, rating_shape):
    """
    convert coordinate matrix [i, j, value] to sparse matrix (2d)
    :return: sparse matrix (2d)
    """
    mat = coo_matrix((coo_mat[:, 2], (coo_mat[:, 0], coo_mat[:, 1])), shape=rating_shape).toarray()
    x, o, y = parse_xoy(mat, mat.shape[0], mat.shape[1])
    return x, o, y


def gen_nflx_xoy_binary(coo_mat, rating_shape):
    """
    convert coordinate matrix [i, j, value] to sparse matrix (2d)
    :return: sparse matrix (2d)
    """
    mat = coo_matrix((coo_mat[:, 2], (coo_mat[:, 0], coo_mat[:, 1])), shape=rating_shape).toarray()
    x, o, y = parse_xoy_binary(mat, mat.shape[0], mat.shape[1])
    return x, o, y


def main():
    pr_curve_filename = 'nflx_hcf.npy'
    rating_filename = "nflx_rating.npy"
    # ratings = get_nflx_rating()  # only need run once
    # np.save(rating_filename, ratings)

    ratings = np.load(rating_filename)
    training, test = split_nflx_ratings(ratings, 0.8)
    x_train, o_train, y_train = gen_nflx_xoy(training, ratings.shape)
    # x_train, o_train, y_train = gen_nflx_xoy_binary(training, ratings.shape)

    t = compute_t(x_train, y_train)

    ranks = [30, 40]
    num_iters = [50, 80]
    best_t = None
    best_validation_auc = float("-inf")
    best_rank = 0
    best_num_iter = 0

    for rank, num_iter in itertools.product(ranks, num_iters):
        t_hat = mf_sklearn(t, n_components=rank, n_iter=num_iter)  # [0, 23447]
        valid_auc = hcf_inference(t_hat, training, test, ratings.shape, pr_curve_filename)
        print("The current model was trained with rank = {}, and num_iter = {}, and its AUC on the "
              "validation set is {}.".format(rank, num_iter, valid_auc))
        if valid_auc > best_validation_auc:
            best_t = t_hat
            best_validation_auc = valid_auc
            best_rank = rank
            best_num_iter = num_iter

    test_auc = hcf_inference(best_t, training, test, ratings.shape, pr_curve_filename)
    print("The best model was trained with rank = {}, and num_iter = {}, and its AUC on the "
          "test set is {}.".format(best_rank, best_num_iter, test_auc))


if __name__ == '__main__':
    main()
