# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. stuck at line 195: model = ALS.train(t_rdd, rank, numIter, lmbda)
"""
import itertools
import os
import pickle
import sys
from time import time
from os.path import isfile

import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import SparkSession
from scipy.sparse import coo_matrix


def parse_rating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return int(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))


def parse_movie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]


def parse_xoy(mat, n_users, n_items):
    """
    Parses a sparse matrix to x, o, y
    :return:
    """
    o = np.clip(mat, 0, 1)  # O
    # x = (mat >= 3) * np.ones((n_users, n_items))  # X, split [0, 1, 2] -> 0, [3, 4, 5] -> 1
    x = mat / 5
    # y = o - x
    y = (6 - mat) / 5  # Y
    return x, o, y


def parse_xoy_binary(mat, n_users, n_items):
    """
    Parses a sparse matrix to x, o, y
    :return:
    """
    o = np.clip(mat, 0, 1)  # O
    x = (mat >= 3) * np.ones((n_users, n_items))  # X, split [0, 1, 2] -> 0, [3, 4, 5] -> 1
    # x = mat / 5
    y = o - x
    # y = (6 - mat) / 5  # Y
    return x, o, y


def parse_o(line):
    """
    Parse a rating matrix to o
    :param line:
    :return:
    """
    if float(line[1][2]) > 0:  # observed
        return int(line[1][0]), int(line[1][1]), 1
    else:
        return int(line[1][0]), int(line[1][1]), 0


def load_ratings(ratings_file):
    """
    Load ratings from file into ndarray
    return: ndarray, [i, j, rating, timestamp]
    """
    if not isfile(ratings_file):
        print("File %s does not exist." % ratings_file)
        sys.exit(1)
    f = open(ratings_file, 'r')
    ratings = np.loadtxt(ratings_file, dtype=int, delimiter='::')
    f.close()
    if not ratings.any():
        print("No ratings provided.")
        sys.exit(1)
    else:
        return ratings


def split_ratings(ratings, b1, b2):
    """
    split ratings into train (60%), validation (20%), and test (20%) based on the
    last digit of the timestamp, add my_ratings to train, and cache them
    training, validation, test are all RDDs of (userId, movieId, rating)
    split ratings into training, validation, test
    :param ratings: matrix, row is (i, j, value, timestamp)
    :param b1: boundary1: between training and validation
    :param b2: boundary2: between validation and test
    :return: training, validation, test: [i, j, rating]
    """
    training = np.array([row for row in ratings if row[3] % 10 < b1])  # [0, 5]
    validation = np.array([row for row in ratings if b1 <= row[3] % 10 < b2])  # [6, 7]
    test = np.array([row for row in ratings if b2 <= row[3] % 10])  # [8, 9]
    training = np.delete(training, 3, 1)
    validation = np.delete(validation, 3, 1)
    test = np.delete(test, 3, 1)

    return training, validation, test


def compute_s(x_train):
    s = np.dot(x_train.T, x_train)
    mask = s > 0
    s_norm = (s - np.min(s[mask])) / (np.max(s[mask]) - np.min(s[mask]))  # only normalize t1 > 0
    s_norm = s_norm * mask
    s_norm[s_norm < 1e-1] = 0

    return s_norm


def parse_s(t):
    """
    convert sparse matrix (2d) to list of tuple (i, j, value)
    :return:
    """
    t_list_tuple = []
    for i in tqdm.tqdm(range(t.shape[0])):
        for j in range(t.shape[1]):
            if t[i][j] > 1e-6:
                t_list_tuple.append((i, j, t[i][j]))

    # sparse_t = csr_matrix(t, dtype=float).tocoo()  # used for spark
    # a = np.unique(sparse_t.data, return_counts=True)
    # mat = np.vstack((sparse_t.row, sparse_t.col, sparse_t.data)).T  # has data == 0
    # t_list_tuple = list(map(tuple, mat))
    return t_list_tuple


def normalize_validation(validation):
    validation[:, 2] = validation[:, 2] / 5
    return validation


def generate_xoy(coo_mat, rating_shape):
    """
    convert coordinate matrix [i, j, value] to sparse matrix (2d)
    :return: sparse matrix (2d)
    """
    mat = coo_matrix((coo_mat[:, 2], (coo_mat[:, 0], coo_mat[:, 1])), shape=rating_shape).toarray()
    x, o, y = parse_xoy(mat, mat.shape[0], mat.shape[1])
    return x, o, y


def generate_xoy_binary(coo_mat, rating_shape):
    """
    convert coordinate matrix [i, j, value] to sparse matrix (2d)
    :return: sparse matrix (2d)
    """
    mat = coo_matrix((coo_mat[:, 2], (coo_mat[:, 0], coo_mat[:, 1])), shape=rating_shape).toarray()
    x, o, y = parse_xoy_binary(mat, mat.shape[0], mat.shape[1])
    return x, o, y


def get_list_tuples():
    # load personal ratings
    movie_lens_home_dir = '../../data/movielens/medium/'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)  # [i, j, rating, timestamp]
    training, validation, test = split_ratings(ratings, 6, 8)  # (i, j, value)
    if not os.path.isfile('s.pkl'):
        x_train, o_train, y_train = generate_xoy(training, (6041, 3953))

        s = compute_s(x_train)

        # train models and evaluate them on the validation set
        s_list_tuple = parse_s(s)
        with open('s.pkl', 'wb') as fh:
            pickle.dump(s_list_tuple, fh)
        exit()
    else:
        with open("s.pkl", "rb") as fh:
            s_list_tuple = pickle.load(fh)

    validation = normalize_validation(validation)
    test = normalize_validation(test)

    validation_list_tuple = list(map(tuple, validation))  # i, j, value
    test_list_tuple = list(map(tuple, test))
    return s_list_tuple, validation_list_tuple, test_list_tuple


def manual_inference(s_hat):
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)  # [i, j, rating, timestamp]
    training, validation, test = split_ratings(ratings, 6, 8)
    x_train, o_train, y_train = generate_xoy(training, (6041, 3953))
    x_test, o_test, y_test = generate_xoy_binary(test, (6041, 3953))

    all_scores = np.dot(x_train, s_hat)  # [6041, 3953]
    # all_scores intersect with o_test
    all_scores_norm = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))
    y_scores = all_scores_norm[o_test > 0]
    y_true = x_test[o_test > 0]
    auc_score = roc_auc_score(y_true, y_scores)
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # np.save(pr_curve_filename, (precision, recall, thresholds))
    return auc_score


def spark_matrix_completion(model, t_shape, rank):
    """
    complete the usr, item matrices with 0s
    :param model:
    :param t_shape:
    :return: completed usr, item matrices
    """
    user_matrix = model.userFeatures().collect()
    item_matrix = model.productFeatures().collect()
    user_complete_matrix = np.zeros((t_shape[0], rank))
    item_complete_matrix = np.zeros((t_shape[1], rank))
    for row in user_matrix:
        user_complete_matrix[row[0]] = np.array(row[1])

    for row in item_matrix:
        item_complete_matrix[row[0]] = np.array(row[1])

    t_hat = np.dot(user_complete_matrix, item_complete_matrix.T)
    return t_hat


def spark_inference(model, data):
    """
    :param model:
    :param data:
    :return:
    """

    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictions_and_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
        .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
        .values()

    metrics = BinaryClassificationMetrics(predictions_and_ratings)
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)

    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    return metrics.areaUnderROC


def main():
    t_list_tuple, validation_list_tuple, test_list_tuple = get_list_tuples()
    # set up environment
    spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "5g") \
        .getOrCreate()
    sc = spark.sparkContext

    num_partitions = 2
    t_rdd = sc.parallelize(t_list_tuple).filter(lambda x: x[2] > 0).repartition(num_partitions)
    validation_rdd = sc.parallelize(validation_list_tuple)\
        .map(lambda x: (x[0], x[1], float(x[2])))\
        .repartition(num_partitions)
    test_rdd = sc.parallelize(test_list_tuple) \
        .map(lambda x: (x[0], x[1], float(x[2])))\
        .repartition(num_partitions)
    ranks = [8, 12]
    lambdas = [0.1, 0.01]
    num_iters = [10, 20]
    best_model = None
    best_validation_auc = float("-inf")
    best_rank = 0
    best_lambda = -1.0
    best_num_iter = -1
    start_time = time()
    for rank, lmbda, numIter in itertools.product(ranks, lambdas, num_iters):

        model = ALS.train(t_rdd, rank, numIter, lmbda, nonnegative=True, seed=444)
        t_hat = spark_matrix_completion(model, (3953, 3953), rank)
        validation_auc = manual_inference(t_hat)
        # validation_auc = spark_inference(model, validation_rdd)
        print("The current model was trained with rank = {} and lambda = {}, and numIter = {}, and its AUC on the "
              "validation set is {}.".format(rank, lmbda, numIter, validation_auc))
        if validation_auc > best_validation_auc:
            best_model = model
            best_validation_auc = validation_auc
            best_rank = rank
            best_lambda = lmbda
            best_num_iter = numIter

    test_auc = spark_inference(best_model, test_rdd)
    end_time = time() - start_time
    # evaluate the best model on the test set
    print("The best model was trained with rank = {} and lambda = {}, and numIter = {}, and its AUC on the test set is"
          " {}; runtime is {}".format(best_rank, best_lambda, best_num_iter, test_auc, end_time))

    # clean up
    sc.stop()


if __name__ == "__main__":
    main()
