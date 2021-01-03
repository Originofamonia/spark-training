# HCF dev
Create a python virtual environment (pyspark). Required packages: pyspark, scipy, numpy, itertools, sklearn
## MovieLens
Code in `machine_learning/movieLens/`

Data in `data/movielens/medium/ratings.dat`

Read data steps: 1st: `load_ratings` in `utils.py`; 2nd: `split_ratings_by_time` in `MovieLens_sklean_hcf_nn.py`; 3rd: `generate_xoy` in `utils.py`.  

`MovieLens_sklearn_hcf.py`： sklearn version HCF. `compute_t` is in `MovieLens_spark_hcf.py`; `mf_sklearn` is in `MovieLens_sklearn_hcf2vcat.py`; `hcf_inference` in this file.

`MovieLens_sklearn_hcf2.py`：T = concat(X, Y), evaluate on left half of T* only.

`MovieLens_sklearn_hcf2vcat.py`: T = concat(X, Y\*Y.T\*X); R* = X* + 2 * (Y\*Y.T*X)

`MovieLens_sklearn_hcf_nn.py`: DL version of HCF. 

`MovieLens_sklearn_baseline.py`: sklearn baseline MF(x.T * x)  

`MovieLens_sklearn_baseline2.py`: sklearn baseline2 MF(x)

`MovieLens_spark_base1.py`: spark ALS.train(x.T * x); `compute_s`: x.T * x, filter out `s_norm[s_norm < 1e-1] = 0`.

`MovieLens_spark_base2.py`: spark ALS.train(x);

`MovieLens_spark_hcf.py`: spark ALS.train(T); T = cat(x.T * x, y.T * x)

`MovieLens_spark_hcf2.py`: spark ALS.train(T); T = concat(X, Y), evaluate on left half of T* only

`MovieLens_spark_hcf2hcat.py`: spark ALS.train(T); T = hcat(X, Y * Y.T * X)

`MovieLensALS.py`: original ALS example.

`MovieLensALS_baseline.py` and `MovieLensALS_hcf.py`: unsuccessful attempts.
### Change hyperparameters
Change [0, 1] rating threshold in `MovieLens_spark_hcf.py`, function `parse_xoy_label(mat, n_users, n_items)`, line 55: `mat >= 3`

Change training data between continuous rating and binary rating [0, 1]: in `MovieLens_sklearn_hcf.py`, `MovieLens_sklearn_baseline.py`, `MovieLens_sklearn_baseline2.py`'s main function: change between `x_train, o_train, y_train = generate_xoy(training)` and `x_train, o_train, y_train = generate_xoy_binary(training)`

Change hcf's beta * Y: in `MovieLens_sklearn_hcf.py` function `hcf_inference(t_hat, training, test)` line 39: `u = np.concatenate((x_train, 0.2 * y_train), axis=1)`. 0.2 is beta.
## Netflix
Netflix code is in `machine_learning/netflix`

Download the dataset from `https://www.kaggle.com/laowingkin/netflix-movie-recommendation`, and put into `nflx_data` folder next to `netflix`.

sklearn hcf is in `netflix_sklearn_hcf.py`

sklearn baseline (x.T * x) is in `netflix_sklearn_baseline.py` 

sklearn baseline2 (MF(x_train)) is in `netflix_sklearn_baseline2.py`

### Change hyperparameters
[0, 1] rating threshold: same as above

Change training data between continuous rating and binary rating [0, 1]: in `main()`, switch between `gen_nflx_xoy` and `gen_nflx_xoy_binary`.

HCF beta * Y: same as above


### hcf_nn
hcf_nn is in `machine_learning/movieLens/MovieLens_sklearn_hcf_nn.py` and `hcf_nn.py`.


## Progress
MovieLens is done.

Netflix is done. (OOM)