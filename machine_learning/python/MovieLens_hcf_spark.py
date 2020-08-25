# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. stuck at line 195: model = ALS.train(t_rdd, rank, numIter, lmbda)
"""
import sys
import itertools
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import BinaryClassificationMetrics


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
    Parses a training matrix to x, o, y
    :return:
    """
    o = np.clip(mat, 0, 1)  # O
    x = (mat >= 3) * np.ones((n_users, n_items))  # X, split [0, 1, 2] -> 0, [3, 4, 5] -> 1
    y = o - x  # Y
    # a = np.unique(o)
    # b = np.unique(x)
    # c = np.unique(y)
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


def parse_t(t):
    """
    convert sparse matrix (2d) to list of tuple (i, j, value)
    :return:
    """
    sparse_t = csr_matrix(t, dtype=float).tocoo()  # used for spark
    mat = np.vstack((sparse_t.row, sparse_t.col, sparse_t.data)).T
    t_list_tuple = list(map(tuple, mat))
    return t_list_tuple


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def compute_t(x_train, y_train):
    t1 = np.dot(x_train.T, x_train)
    t1_norm = (t1 - np.mean(t1)) / np.std(t1)
    t2 = np.dot(y_train.T, x_train)
    t2_norm = (t2 - np.mean(t2)) / np.std(t2)
    t_norm = np.concatenate((t1_norm, t2_norm), axis=0)  # [7906, 3953]
    t = sigmoid(t_norm)
    # a = np.unique(t)
    return t


def generate_xoy(coo_mat):
    """
    convert coordinate matrix [i, j, value] to sparse matrix (2d)
    :return: sparse matrix (2d)
    """
    mat = coo_matrix((coo_mat[:, 2], (coo_mat[:, 0], coo_mat[:, 1])), shape=(6041, 3953)).toarray()
    x, o, y = parse_xoy(mat, mat.shape[0], mat.shape[1])
    return x, o, y


def hcf_inference(t, x, y):
    """

    :param t: t is after matrix completion
    :param x: x_train
    :param y: y_train
    :return: AUROC
    """
    pass


def spark_inference(model, data):
    """
    :param model:
    :param data:
    :param n:
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
    # set up environment
    conf = SparkConf().setAppName("MovieLensALS") \
                      .set("spark.executor.memory", "10g")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)  # solve the ParallelRDD issue

    # load personal ratings
    movie_lens_home_dir = '../../data/movielens/medium/'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)  # [i, j, rating, timestamp]

    training, validation, test = split_ratings(ratings, 6, 8)

    x_train, o_train, y_train = generate_xoy(training)
    x_valid, o_valid, y_valid = generate_xoy(validation)
    x_test, o_test, y_test = generate_xoy(test)

    t = compute_t(x_train, y_train)

    # train models and evaluate them on the validation set
    num_partitions = 1000
    t_list_tuple = parse_t(t)
    validation_list_tuple = list(map(tuple, validation))
    test_list_tuple = list(map(tuple, test))
    # not sure whether should use filter?
    t_rdd = sc.parallelize(t_list_tuple, num_partitions).filter(lambda x: x[2] > 0)
    validation_rdd = sc.parallelize(validation_list_tuple, num_partitions)
    test_rdd = sc.parallelize(test_list_tuple, num_partitions)
    ranks = [8, 12]
    lambdas = [0.1, 10.0]
    num_iters = [10, 20]
    best_model = None
    best_validation_auc = float("-inf")
    best_rank = 0
    best_lambda = -1.0
    best_num_iter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, num_iters):
        model = ALS.train(t_rdd, rank, numIter, lmbda)
        validation_auc = spark_inference(model, validation_rdd)
        print("The best model was trained with rank = {} and lambda = {}, and numIter = {}, and its AUC on the "
              "validation set is {}.".format(best_rank, best_lambda, best_num_iter, validation_auc))
        if validation_auc > best_validation_auc:
            best_model = model
            best_validation_auc = validation_auc
            best_rank = rank
            best_lambda = lmbda
            best_num_iter = numIter

    test_auc = spark_inference(best_model, test_rdd)

    # evaluate the best model on the test set
    print("The best model was trained with rank = {} and lambda = {}, and numIter = {}, and its AUC on the test set is"
          " {}.".format(best_rank, best_lambda, best_num_iter, test_auc))

    # clean up
    sc.stop()


if __name__ == "__main__":
    main()
