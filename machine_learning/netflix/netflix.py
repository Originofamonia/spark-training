# https://www.kaggle.com/laowingkin/netflix-movie-recommendation
import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

sns.set_style("darkgrid")


def recommend(movie_title, min_count, df_title, df_p, df_movie_summary):
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(df_title.index[df_title['Name'] == movie_title][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns=['PearsonR'])
    corr_target.dropna(inplace=True)
    corr_target = corr_target.sort_values('PearsonR', ascending=False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]
    print(corr_target[corr_target['count'] > min_count][:10].to_string(index=False))


def main():
    # Skip date
    df1 = pd.read_csv('../nflx_data/combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df2 = pd.read_csv('../nflx_data/combined_data_2.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df3 = pd.read_csv('../nflx_data/combined_data_3.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df4 = pd.read_csv('../nflx_data/combined_data_4.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

    df1['Rating'] = df1['Rating'].astype(float)

    print('Dataset 1 shape: {}'.format(df1.shape))
    print('-Dataset examples-')
    print(df1.iloc[::5000000, :])

    # load less data for speed

    df = df1
    # df = df.append(df2)
    # df = df.append(df3)
    # df = df.append(df4)
    df.index = np.arange(0, len(df))
    print('Full dataset shape: {}'.format(df.shape))
    print('-Dataset examples-')
    print(df.iloc[::5000000, :])

    p = df.groupby('Rating')['Rating'].agg(['count'])

    # get movie count
    movie_count = df.isnull().sum()[1]

    # get customer count
    cust_count = df['Cust_Id'].nunique() - movie_count

    # get rating count
    rating_count = df['Cust_Id'].count() - movie_count

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
    print('-Dataset examples-')
    print(df.iloc[::5000000, :])

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
    print('-Data Examples-')
    print(df.iloc[::5000000, :])

    # df_p is the rating matrix (143458, 1350)
    df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')

    print(df_p.shape)
    df_p = df_p.fillna(0)
    df_p_nd = df_p.to_numpy()
    a = np.unique(df_p_nd, return_counts=True)
    print(a)
    # until above is useful

    df_title = pd.read_csv('../nflx_data/movie_titles.csv', encoding="ISO-8859-1", header=None,
                           names=['Movie_Id', 'Year', 'Name'])
    df_title.set_index('Movie_Id', inplace=True)
    print(df_title.head(10))

    reader = Reader()

    # get just top 100K rows for faster run time
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)
    # data.split(n_folds=3)

    svd = SVD()
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3)

    df_785314 = df[(df['Cust_Id'] == 785314) & (df['Rating'] == 5)]
    df_785314 = df_785314.set_index('Movie_Id')
    df_785314 = df_785314.join(df_title)['Name']
    print(df_785314)

    user_785314 = df_title.copy()
    user_785314 = user_785314.reset_index()
    user_785314 = user_785314[~user_785314['Movie_Id'].isin(drop_movie_list)]

    # getting full dataset
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)

    trainset = data.build_full_trainset()
    # Run 5-fold cross-validation and print results
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    user_785314['Estimate_Score'] = user_785314['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)

    user_785314 = user_785314.drop('Movie_Id', axis=1)

    user_785314 = user_785314.sort_values('Estimate_Score', ascending=False)
    print(user_785314.head(10))

    recommend("What the #$*! Do We Know!?", 0, df_title, df_p, df_movie_summary)


if __name__ == '__main__':
    main()
