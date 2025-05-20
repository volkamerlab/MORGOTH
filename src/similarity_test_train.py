from typing import Union
import fireducks.pandas as pd
import copy
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine, euclidean
from multiprocessing import Pool
from functools import partial
import pandas

def r_hat(x: np.array, y: np.array) -> float:
    '''
        calculates r hat, which is needed to calculate rank magnitude

        @param x: a numpy array
        @param y: a numpy array

        @return: r hat as a float
    '''
    sorted_y = y.argsort()
    y = y[sorted_y]
    x = x[sorted_y]

    order = x.argsort()
    ranks = order.argsort()
    for i in np.arange(len(ranks)):
        ranks[i] += 1
    max_rank = 0
    min_rank = 0
    n_y = len(y)
    for i in np.arange(1, n_y+1):
        y_i = y[i-1]
        min_rank += y_i * (n_y-i+1)
        max_rank += y_i*i
    sum = 0
    for i in np.arange(len(x)):
        sum += ranks[i]*y[i]
    sum *= 2
    sum -= (max_rank+min_rank)
    sum /= (max_rank-min_rank)
    return sum


def rank_magnitude(x: np.array, y: np.array) -> float:
    '''
        calculates the rank magnitude between x and y and transforms it into a distance 

        @param x: a numpy array
        @param y: a numpy array
    '''
    intermediate_1 = r_hat(x, y)
    intermediate_2 = r_hat(y, x)
    res = ((intermediate_1+intermediate_2)/2)
    return 1-res


def precompute_correlation_matrix(normalized_train_samples: pd.DataFrame, distance_measure: str) -> pd.DataFrame:
    '''
    @param normalized_train_samples: pandas DataFrame containing the normalized gene expression values for training samples
    @param distance_measure: a string indicating which distance measure should be used, available: 'cosine', 'pearson', 'spearman', 'euclidean', 'rank_magnitude'
    @return: a pandas DataFrame with the pairwise correlations/distances
    '''
    train_samples = normalized_train_samples.transpose()
    if distance_measure == 'pearson' or distance_measure == 'spearman':
        return train_samples.p_corr(method=distance_measure)
    if distance_measure == 'cosine':
        dist_fun = cosine
    elif distance_measure == 'euclidean':
        dist_fun = euclidean
    elif distance_measure == 'rank_magnitude':
        dist_fun = rank_magnitude

    return train_samples.p_corr(method=dist_fun)


def calculate_silhoutte_score_train_test(train_samples: pd.DataFrame, precomputed_similarities: pd.DataFrame, test_sample: np.array, distance_measure: str, precomputed: bool) -> float:
    '''
    @param train_samples: pandas DataFrame containing all training samples gene expression values
    @param test_sample: a numpy array with the respective gene expression values for one test sample
    @param distance_measure: a string indicating which distance measure should be used, available: 'cosine', 'pearson', 'spearman', 'euclidian', 'rank_magnitude'

    @return: the silhouette score assuming the train samples being a cluster and the test sample being the other one
        if the score is < 0 or close to 0, the clusters overlap which indicates that the test sample is similar to the training samples
    '''
    if distance_measure == 'pearson' or distance_measure == 'spearman':
        if not precomputed:
            pcc_similarities_train = train_samples.p_corr(
                method=distance_measure)
        else:
            pcc_similarities_train = precomputed_similarities

        train_sample_list = []
        for train_sample_id in range(len(train_samples.columns)):
            train_sample_list.append(copy.deepcopy(
                train_samples.iloc[:, train_sample_id].values))
        if distance_measure == 'pearson':
            with Pool() as pool:
                distance_p_val_list = pool.map(
                    partial(pearsonr, y=test_sample), train_sample_list)
        elif distance_measure == 'spearman':
            with Pool() as pool:
                distance_p_val_list = pool.map(
                    partial(spearmanr, b=test_sample), train_sample_list)
        distance_list = []
        for entry in distance_p_val_list:
            distance_list.append(entry[0])

    elif distance_measure == 'cosine' or distance_measure == 'euclidian' or distance_measure == 'rank_magnitude':
        if distance_measure == 'cosine':
            dist_fun = cosine
        elif distance_measure == 'euclidian':
            dist_fun = euclidean
        elif distance_measure == 'rank_magnitude':
            dist_fun = rank_magnitude

        if not precomputed:
            distances_train = train_samples.p_corr(method=dist_fun)
            # symmetric matrix with 1 at the diagonal => for distances we want 0 at the diagonal,
            # but we do not consider it anyways for the calculation of silhouette score
        else:
            distances_train = precomputed_similarities

        train_sample_list = []
        for train_sample_id in range(len(train_samples.columns)):
            train_sample_list.append(copy.deepcopy(
                train_samples.iloc[:, train_sample_id].values))
        if distance_measure == 'rank_magnitude':
            with Pool() as pool:
                distance_list = pool.map(
                    partial(dist_fun, y=test_sample), train_sample_list)
        else:
            with Pool() as pool:
                distance_list = pool.map(
                    partial(dist_fun, v=test_sample), train_sample_list)
        distance_matrix_sample_to_cluster = pd.DataFrame(
            data=[np.array(distance_list)], columns=train_samples.columns)
        return calculate_silhouette_score(distance_matrix_cluster=distances_train, distance_matrix_sample_to_cluster=distance_matrix_sample_to_cluster, distance=True)

    distance_matrix_sample_to_cluster = pd.DataFrame(
        data=[np.array(distance_list)], columns=train_samples.columns)
    return calculate_silhouette_score(distance_matrix_cluster=pcc_similarities_train, distance_matrix_sample_to_cluster=distance_matrix_sample_to_cluster, distance=False)


def calculate_silhouette_score(distance_matrix_cluster: pd.DataFrame, distance_matrix_sample_to_cluster: Union[pd.DataFrame, pd.Series], distance: bool) -> float:
    if type(distance_matrix_sample_to_cluster) == pd.Series:
        distance_matrix_sample_to_cluster = pd.Series(
            distance_matrix_sample_to_cluster).to_frame()

    if distance:
        distances = []
        array = np.tril(np.ones(distance_matrix_cluster.shape)).astype(bool)
        distances = distance_matrix_cluster.where(~array).values.flatten()
        distances = distances[~np.isnan(distances)]
        mean_intra_cluster_dist = np.mean(distances)
        mean_inter_cluster_dist = np.mean(
            distance_matrix_sample_to_cluster.values[0])

    else:
        similarities = []
        array = np.tril(np.ones(distance_matrix_cluster.shape)).astype(bool)
        similarities = distance_matrix_cluster.where(~array).values.flatten()
        similarities = similarities[~np.isnan(similarities)]
        for i in np.arange(len(similarities)):
            similarities[i] = 1-similarities[i]
        if type(distance_matrix_sample_to_cluster) is type(pandas.Series(dtype=float)):
            distance_matrix_sample_to_cluster = distance_matrix_sample_to_cluster.to_frame()
        for sample in distance_matrix_sample_to_cluster.columns:
            distance_matrix_sample_to_cluster[sample] = 1 - \
                distance_matrix_sample_to_cluster[sample]
        mean_inter_cluster_dist = np.mean(
            distance_matrix_sample_to_cluster.values[0])
        mean_intra_cluster_dist = np.mean(similarities)
    silhouette_score = (mean_intra_cluster_dist - mean_inter_cluster_dist) / \
        max((mean_inter_cluster_dist, mean_intra_cluster_dist))
    return silhouette_score


if __name__ == '__main__':
    main()
