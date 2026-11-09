from typing import Union
import sys
if sys.platform.startswith('linux'):
    import fireducks.pandas as pd
else:
    import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from scipy.spatial.distance import cosine, euclidean, cdist
import pandas

# measures for which the precomputed matrix contains similarities (correlations) instead of distances
SIMILARITY_MEASURES = {'pearson', 'spearman'}
DISTANCE_MEASURES = {'cosine', 'euclidean', 'euclidian', 'rank_magnitude'}


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

    ranks = x.argsort().argsort() + 1
    n_y = len(y)
    i = np.arange(1, n_y + 1)
    min_rank = np.sum(y * (n_y - i + 1))
    max_rank = np.sum(y * i)
    sum = 2 * np.sum(ranks * y)
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


######################## vectorized distance computations ########################

def _standardized_rows(A: np.ndarray) -> np.ndarray:
    centered = A - A.mean(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return centered / np.sqrt(np.einsum('ij,ij->i', centered, centered))[:, None]


def _r_hat_matrix(ranks_x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    '''
        r_hat(x=X[a], y=Y[b]) for all pairs (a, b)

        @param ranks_x: ranks (1..p) of the rows of X
        @param Y: matrix whose rows are the y vectors
    '''
    p = Y.shape[1]
    sorted_y = np.sort(Y, axis=1)
    totals = sorted_y.sum(axis=1)
    spread = sorted_y @ (2 * np.arange(1, p + 1) - p - 1).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (2 * (ranks_x @ Y.T) - (p + 1) * totals[None, :]) / spread[None, :]


def _ordinal_ranks(A: np.ndarray) -> np.ndarray:
    return (np.argsort(np.argsort(A, axis=1, kind='stable'), axis=1, kind='stable') + 1).astype(np.float64)


def pairwise_distances(A: np.ndarray, B: np.ndarray, distance_measure: str) -> np.ndarray:
    '''
        computes the similarity/distance between all rows of A and all rows of B (inputs must not contain NaN)

        @param A: numpy array (samples x features)
        @param B: numpy array (samples x features)
        @param distance_measure: 'pearson', 'spearman' (similarities), 'cosine', 'euclidean', 'rank_magnitude' (distances)
        @return: numpy array (len(A) x len(B))
    '''
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if distance_measure == 'pearson':
        return np.clip(_standardized_rows(A) @ _standardized_rows(B).T, -1, 1)
    if distance_measure == 'spearman':
        return pairwise_distances(rankdata(A, axis=1), rankdata(B, axis=1), 'pearson')
    if distance_measure == 'cosine':
        with np.errstate(divide='ignore', invalid='ignore'):
            norms_a = np.sqrt(np.einsum('ij,ij->i', A, A))
            norms_b = np.sqrt(np.einsum('ij,ij->i', B, B))
            return np.clip(1 - (A @ B.T) / norms_a[:, None] / norms_b[None, :], 0, 2)
    if distance_measure in ('euclidean', 'euclidian'):
        return cdist(A, B, 'euclidean')
    if distance_measure == 'rank_magnitude':
        # note: ties are ranked in order of appearance
        r_ab = _r_hat_matrix(_ordinal_ranks(A), B)
        r_ba = _r_hat_matrix(_ordinal_ranks(B), A).T
        return 1 - (r_ab + r_ba) / 2
    raise ValueError(f'Unsupported distance measure {distance_measure}')


def _scalar_distance_function(distance_measure: str):
    '''
        returns the function used to compute a single similarity/distance (used by pandas.DataFrame.corr)
    '''
    if distance_measure == 'pearson':
        return lambda x, y: pearsonr(x, y)[0]
    if distance_measure == 'spearman':
        return lambda x, y: spearmanr(x, y)[0]
    if distance_measure == 'cosine':
        return cosine
    if distance_measure in ('euclidean', 'euclidian'):
        return euclidean
    if distance_measure == 'rank_magnitude':
        return rank_magnitude
    raise ValueError(f'Unsupported distance measure {distance_measure}')


def distances_to_samples(train_samples: np.ndarray, test_samples: np.ndarray, distance_measure: str) -> np.ndarray:
    '''
        computes the similarity/distance of every test sample to every training sample.
        Missing values: only features that are finite for all training samples and the respective test sample are used.

        @param train_samples: numpy array (training samples x features)
        @param test_samples: numpy array (test samples x features)
        @return: numpy array (test samples x training samples)
    '''
    train_samples = np.asarray(train_samples, dtype=np.float64)
    test_samples = np.atleast_2d(np.asarray(test_samples, dtype=np.float64))
    usable = np.isfinite(train_samples).all(axis=0)
    test_finite = np.isfinite(test_samples)
    if usable.all() and test_finite.all():
        return pairwise_distances(test_samples, train_samples, distance_measure)
    result = np.empty((len(test_samples), len(train_samples)))
    for i, test_sample in enumerate(test_samples):
        features = usable & test_finite[i]
        result[i] = pairwise_distances(
            test_sample[None, features], train_samples[:, features], distance_measure)[0]
    return result


def precompute_correlation_matrix(normalized_train_samples: pd.DataFrame, distance_measure: str) -> pd.DataFrame:
    '''
    @param normalized_train_samples: pandas DataFrame containing the normalized gene expression values for training samples
    @param distance_measure: a string indicating which distance measure should be used, available: 'cosine', 'pearson', 'spearman', 'euclidean', 'rank_magnitude'
    @return: a pandas DataFrame with the pairwise correlations/distances
    '''
    matrix = distance_matrix(normalized_train_samples.to_numpy(
        dtype=np.float64), distance_measure)
    index = normalized_train_samples.index
    return pd.DataFrame(matrix, index=index, columns=index)


def distance_matrix(values: np.ndarray, distance_measure: str) -> np.ndarray:
    '''
        numpy version of precompute_correlation_matrix

        @param values: numpy array (samples x features)
        @return: numpy array (samples x samples) with the pairwise correlations/distances.
            As pandas.DataFrame.corr does, the diagonal is set to 1.
    '''
    values = np.asarray(values, dtype=np.float64)
    # features that are missing for all samples are ignored (as pandas does with pairwise complete observations)
    values = values[:, ~np.isnan(values).all(axis=0)]
    if np.isnan(values).any():
        # missing values in single samples: fall back to pandas (pairwise complete observations)
        train_samples = pandas.DataFrame(values.T)
        if distance_measure in SIMILARITY_MEASURES:
            return train_samples.corr(method=distance_measure).to_numpy()
        return train_samples.corr(method=_scalar_distance_function(distance_measure)).to_numpy()
    matrix = pairwise_distances(values, values, distance_measure)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    return matrix


def calculate_silhoutte_score_train_test(train_samples: pd.DataFrame, precomputed_similarities: pd.DataFrame, test_sample: np.array, distance_measure: str, precomputed: bool) -> float:
    '''
    @param train_samples: pandas DataFrame containing all training samples gene expression values (features x samples)
    @param test_sample: a numpy array with the respective gene expression values for one test sample
    @param distance_measure: a string indicating which distance measure should be used, available: 'cosine', 'pearson', 'spearman', 'euclidean', 'rank_magnitude'

    @return: the silhouette score assuming the train samples being a cluster and the test sample being the other one
        if the score is < 0 or close to 0, the clusters overlap which indicates that the test sample is similar to the training samples
    '''
    if precomputed:
        cluster = precomputed_similarities
    else:
        cluster = precompute_correlation_matrix(
            train_samples.transpose(), distance_measure)
    distance_list = distances_to_samples(
        train_samples.to_numpy(dtype=np.float64).T, test_sample, distance_measure)[0]
    distance_matrix_sample_to_cluster = pandas.DataFrame(
        data=[np.array(distance_list)])
    return calculate_silhouette_score(distance_matrix_cluster=cluster, distance_matrix_sample_to_cluster=distance_matrix_sample_to_cluster, distance=distance_measure not in SIMILARITY_MEASURES)


def calculate_silhouette_score(distance_matrix_cluster: pd.DataFrame, distance_matrix_sample_to_cluster: Union[pd.DataFrame, pd.Series], distance: bool) -> float:
    if isinstance(distance_matrix_sample_to_cluster, (pd.Series, pandas.Series)):
        distance_matrix_sample_to_cluster = distance_matrix_sample_to_cluster.to_frame()
    cluster = np.asarray(distance_matrix_cluster.values if hasattr(
        distance_matrix_cluster, 'values') else distance_matrix_cluster, dtype=np.float64)
    # strict upper triangle (row-major order, as in the original implementation)
    values = cluster[np.triu(np.ones(cluster.shape, dtype=bool), k=1)]
    values = values[~np.isnan(values)]
    sample_to_cluster = np.asarray(
        distance_matrix_sample_to_cluster.values, dtype=np.float64)[0]

    if distance:
        mean_intra_cluster_dist = np.mean(values)
        mean_inter_cluster_dist = np.mean(sample_to_cluster)
    else:
        mean_intra_cluster_dist = np.mean(1 - values)
        mean_inter_cluster_dist = np.mean(1 - sample_to_cluster)
    silhouette_score = (mean_intra_cluster_dist - mean_inter_cluster_dist) / \
        max((mean_inter_cluster_dist, mean_intra_cluster_dist))
    return silhouette_score


def multiset_intra_cluster_mean(cluster_matrix: np.ndarray, counts: np.ndarray, distance: bool) -> np.ndarray:
    '''
        mean distance between all pairs of members of clusters that may contain samples several times.
        Equivalent to building the (huge) matrix cluster_matrix[members, members] for every cluster and averaging
        its strict upper triangle (ignoring NaNs), but only needs O(#clusters * #samples) memory.

        @param cluster_matrix: symmetric numpy array (samples x samples) with the pairwise similarities/distances
        @param counts: numpy array (clusters x samples), how often each sample is a member of each cluster
        @param distance: True if cluster_matrix contains distances, False if it contains similarities (then 1 - similarity is used)
        @return: numpy array with the mean for every cluster
    '''
    missing = np.isnan(cluster_matrix)
    values = np.where(missing, 0.0, cluster_matrix)
    present = (~missing).astype(np.float64)
    total = np.einsum('ij,ij->i', counts @ values, counts) - counts @ np.diag(values)
    number = np.einsum('ij,ij->i', counts @ present,
                       counts) - counts @ np.diag(present)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = total / number
    if distance:
        return mean
    return 1 - mean


def silhouette_scores(intra: np.ndarray, inter: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return (intra - inter) / np.maximum(inter, intra)
