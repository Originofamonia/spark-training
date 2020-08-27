# HCF dev

Create a python virtual environment (pyspark). Required packages: pyspark, scipy, numpy, itertools, 


movieLens: in `machine_learning/movieLens/`

sklearn hcf is in `MovieLens_sklearn_hcf.py`

sklearn baseline (x.T * x) is in `MovieLens_sklearn_baseline.py` 

sklearn baseline2 (MF(x_train)) is in `MovieLens_sklearn_baseline2.py`

Change [0, 1] rating threshold in `MovieLens_spark_hcf.py`, function `parse_xoy_label(mat, n_users, n_items)`, line 55: `mat >= 3`

Change training data between continuous rating and binary rating [0, 1]: in `MovieLens_sklearn_hcf.py`, `MovieLens_sklearn_baseline.py`, `MovieLens_sklearn_baseline2.py`'s main function: change between `x_train, o_train, y_train = generate_xoy(training)` and `x_train, o_train, y_train = generate_xoy_binary(training)`

Change hcf's beta * Y: in `MovieLens_sklearn_hcf.py` function `hcf_inference(t_hat, training, test)` line 39: `u = np.concatenate((x_train, 0.2 * y_train), axis=1)`. 0.2 is beta.

## progress
MovieLens is done.

Netflix is ongoing.