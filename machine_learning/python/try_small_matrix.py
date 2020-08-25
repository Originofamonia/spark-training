import os
import sys
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-1])
lib_dir = os.path.join(root_path, 'lib')
add_path(root_path)
add_path(abs_current_path)

# from machine_learning.python.MovieLensALS import compute_rmse



def small_matrix_sklearn():
    mat = np.array([(1, 0, 0, 0),
                    (0, 0, 1, 0),
                    (0, 1, 0, 1),
                    (0, 0, 0, 0)])


def try_small_matrix_spark():
    rank = 2
    num_iter = 22
    lmbda = 0.1
    ratings = [(0, 0, 1),
               (1, 2, 1),
               (2, 3, 1),
               (3, 1, 1)]

    validation = [(0, 3, 1),
                  (2, 1, 1)]
    num_validation = 2

    conf = SparkConf().setAppName("MovieLensALS") \
        .set("spark.executor.memory", "1g")
    sc = SparkContext(conf=conf)
    # spark = SparkSession(sc)  # solve the ParallelRDD issue
    ratings_rdd = sc.parallelize(ratings)
    validation_rdd = sc.parallelize(validation)
    model = ALS.train(ratings_rdd, rank, num_iter, lmbda)
    validation_rmse = compute_rmse(model, validation_rdd, num_validation)
    print(validation_rmse)


if __name__ == '__main__':
    try_small_matrix_spark()
