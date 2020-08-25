# http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
# https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
"""
use numpy to process matrices
    1. use sklearn's MF (done)
    2. stuck at line 195: model = ALS.train(t_rdd, rank, numIter, lmbda)
"""
import sys
import itertools
from math import sqrt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry, BlockMatrix
from pyspark.mllib.evaluation import MulticlassMetrics as metric


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


def parse_xoy(train_mat, n_users, n_items):
    """
    Parses a training matrix to x, o, y
    :param n_items:
    :param n_users:
    :param train_mat:
    :return:
    """
    o = np.clip(train_mat, 0, 1)  # O
    x = (train_mat >= 3) * np.ones((n_users, n_items))  # X, split [0, 1, 2] -> 0, [3, 4, 5] -> 1
    y = o - x  # Y
    a = np.unique(o)
    b = np.unique(x)
    c = np.unique(y)
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
    split ratings into training, validation, test
    :param ratings: matrix, row is (i, j, value, timestamp)
    :param b1: boundary1: between training and validation
    :param b2: boundary2: between validation and test
    :return: training, validation, test
    """
    training = np.array([row for row in ratings if row[3] % 10 < b1])  # [0, 5]
    validation = np.array([row for row in ratings if b1 <= row[3] % 10 < b2])  # [6, 7]
    test = np.array([row for row in ratings if b2 < row[3] % 10])  # [8, 9]
    training = np.delete(training, 3, 1)
    validation = np.delete(validation, 3, 1)
    test = np.delete(test, 3, 1)

    return training, validation, test


def mf_sklearn(t, x_test, y_test, n_components, n_iter):
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=n_iter)
    w = model.fit_transform(t)  # MF
    h = model.components_
    t_hat = np.dot(w, h)  # matrix completion [1783， 0]
    a, b = np.max(t_hat), np.min(t_hat)
    u = np.concatenate((x_test, y_test), axis=1)
    scores = np.dot(u, t_hat)
    return scores


def parse_t(t):
    sparse_t = csr_matrix(t, dtype=int).tocoo()  # used for spark
    mat = np.vstack((sparse_t.row, sparse_t.col, sparse_t.data)).T
    return mat


def compute_rmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictions_and_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
        .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
        .values()
    return sqrt(predictions_and_ratings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


def compute_auc(model, data, n):
    """
    https://spark.apache.org/docs/2.4.0/mllib-evaluation-metrics.html
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
    # load personal ratings
    movie_lens_home_dir = '../../data/movielens/medium/'
    path = '../../data/movielens/medium/ratings.dat'
    ratings = load_ratings(path)
    # split ratings into train (60%), validation (20%), and test (20%) based on the
    # last digit of the timestamp, add my_ratings to train, and cache them
    # training, validation, test are all RDDs of (userId, movieId, rating)
    training, validation, test = split_ratings(ratings, 6, 8)

    num_training = training.shape
    num_validation = validation.shape
    num_test = test.shape
    print("Training: {}, validation: {}, test: {}".format(num_training, num_validation, num_test))

    train_mat = coo_matrix((training[:, 2], (training[:, 0], training[:, 1])), shape=(6041, 3953)).toarray()
    test_mat = coo_matrix((test[:, 2], (test[:, 0], test[:, 1])), shape=(6041, 3953)).toarray()
    x_train, o_train, y_train = parse_xoy(train_mat, 6041, 3953)
    x_test, o_test, y_test = parse_xoy(test_mat, 6041, 3953)
    a, b = np.max(x_train), np.min(x_train)
    c, d = np.max(o_train), np.min(o_train)
    e = np.unique(x_test)
    f = np.unique(o_test)
    g = np.unique(y_test)
    t1 = np.dot(x_train.T, x_train)  # [0, 1948]
    t2 = np.dot(y_train.T, x_train)  # [0, 156]
    # a, b = np.max(t1), np.min(t1)
    # c, d = np.max(t2), np.min(t2)
    t = np.concatenate((t1, t2), axis=0)  # [7906, 3953]

    ranks = [40, 80]
    # lambdas = [0.1, 10.0]
    num_iters = [200, 300]
    best_model = None
    best_validation_auc = float("-inf")
    best_rank = 0
    best_lambda = -1.0
    best_num_iter = -1

    for rank, numIter in itertools.product(ranks, num_iters):
        scores = mf_sklearn(t, x_test, y_test, n_components=rank, n_iter=numIter)  # [0, 23447]
        a, b = np.max(scores), np.min(scores)
        auc = roc_auc_score(x_test, scores)
        print('auc: {}'.format(auc))
        # print("RMSE (validation) = %f for the model trained with " % validation_rmse + \
        #       "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter))
        # if validation_rmse < best_validation_rmse:
        #     best_model = model
        #     best_validation_rmse = validation_rmse
        #     best_rank = rank
        #     best_lambda = lmbda
        #     best_num_iter = numIter

    test_rmse = compute_rmse(best_model, test, num_test)

    # evaluate the best model on the test set
    print("The best model was trained with rank = %d and lambda = %.1f, " % (best_rank, best_lambda) \
          + "and numIter = %d, and its RMSE on the test set is %f." % (best_num_iter, test_rmse))

    # compare the best model with a naive baseline that always returns the mean rating
    mean_rating = training.union(validation).map(lambda x: x[2]).mean()
    baseline_rmse = sqrt(test.map(lambda x: (mean_rating - x[2]) ** 2).reduce(add) / num_test)
    improvement = (baseline_rmse - test_rmse) / baseline_rmse * 100
    print("The best model improves the baseline by %.2f" % improvement + "%.")

    # make personalized recommendations

    my_rated_movie_ids = set([x[1] for x in my_ratings])
    candidates = sc.parallelize([m for m in movies if m not in my_rated_movie_ids])
    predictions = best_model.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    print("Movies recommended for you:")
    for i in range(len(recommendations)):
        print("%2d: %s" % (i + 1, movies[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop()


if __name__ == "__main__":
    main()
