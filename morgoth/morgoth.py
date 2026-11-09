from collections import Counter
from collections.abc import Mapping
import operator
from typing import Union, List
import numpy as np
import math
import time
import copy
from morgoth.multivariate_dt import MultivariateDecisionTree, weighted_class_count
import sys
if sys.platform.startswith('linux'):
    import fireducks.pandas as pd
else:
    import pandas as pd
from multiprocessing import Pool, shared_memory
from functools import partial
from morgoth.similarity_test_train import calculate_silhoutte_score_train_test, precompute_correlation_matrix, calculate_silhouette_score, \
    distance_matrix, distances_to_samples, multiset_intra_cluster_mean, silhouette_scores, SIMILARITY_MEASURES
import re


######################## parallel tree growing ########################
# The training data is put into shared memory once and every worker process only receives the seeds of
# the trees it should grow. Previously, the whole MORGOTH object (incl. two copies of the training data)
# was pickled and sent to the workers for every single tree.

_WORKER_STATE = {}


def _attach_shared_memory(name: str) -> shared_memory.SharedMemory:
    try:
        # python >= 3.13: the creating process is responsible for cleaning up
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _init_tree_worker(shm_name: str, shape: tuple, dtype: str, XT: np.ndarray, settings: dict) -> None:
    if shm_name is not None:
        shm = _attach_shared_memory(shm_name)
        XT = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
        _WORKER_STATE['shm'] = shm
    _WORKER_STATE['XT'] = XT
    _WORKER_STATE['settings'] = settings


def _grow_tree_worker(random_object: np.random.RandomState) -> 'tuple[MultivariateDecisionTree, float]':
    tree, seconds = _grow_tree(
        _WORKER_STATE['XT'], random_object, _WORKER_STATE['settings'])
    # the feature names and the data frame are shared between all trees and are re-attached in the main process
    tree._detach_shared()
    return tree, seconds


def _grow_tree(XT: np.ndarray, random_object: np.random.RandomState, settings: dict) -> 'tuple[MultivariateDecisionTree, float]':
    '''
        grows a MultivariateDecisionTree with the given random object

        @param XT: transposed training matrix (features x samples)
        @param random_object: a np.random.RandomState instance that is used to randomly sample the bootstrap samples and also to
            sample the features that are considered per split
        @param settings: dict with the training data and the parameters of the tree
        @return: a newly created and fitted MultivariateDecisionTree
    '''
    start_time = time.perf_counter()
    number_of_samples = XT.shape[1]
    # draw bootstrap sample with replacement
    index_list_bootstrap = random_object.randint(
        0, number_of_samples, number_of_samples)
    new_tree = MultivariateDecisionTree(X_train=XT, feature_names=settings['feature_names'], _col_of=settings['col_of'], bootstrap_indices=index_list_bootstrap,
                                        y_train=settings['y_train'][index_list_bootstrap], class_names=settings['class_names'], criterion_class=settings['criterion_class'],
                                        criterion_reg=settings['criterion_reg'], min_number_of_samples_per_leaf=settings['min_number_of_samples_per_leaf'], max_depth=settings['max_depth'],
                                        number_of_features_per_split=settings['number_of_features_per_split'], random_object=random_object,
                                        impact_classification=settings['impact_classification'], output_format=settings['output_format'],
                                        sample_weights=settings['sample_weights'][index_list_bootstrap],
                                        sample_names_train=settings['sample_names_train'][index_list_bootstrap], distance_measure=settings['distance_measure'])
    new_tree.fit()
    end_time = time.perf_counter()
    return new_tree, end_time-start_time


class _TrainWeightTable(Mapping):
    '''
        read-only dict-like view {test sample: {training sample: weight}} on the weight matrix used for quantile regression
    '''

    def __init__(self, matrix: np.ndarray, test_names, train_names: np.ndarray, train_name_ids: np.ndarray):
        self._matrix = matrix
        self._row = {}
        for i, name in enumerate(test_names):
            self._row.setdefault(name, i)
        self._train_names = train_names
        self._train_name_ids = train_name_ids

    def __getitem__(self, key):
        row = self._matrix[self._row[key]]
        return {name: row[i] for name, i in zip(self._train_names, self._train_name_ids)}

    def __iter__(self):
        return iter(self._row)

    def __len__(self):
        return len(self._row)


class MORGOTH:

    def __init__(self, X_train, y_train: np.array, criterion_class: str, criterion_reg: str, sample_names_train: np.array,
                 min_number_of_samples_per_leaf: int, number_of_trees_in_forest: int, number_of_features_per_split: Union[float, str], class_names: list,
                 output_format: str, threshold: List[float], time_file: str, sample_weights_included: str, random_state: int, max_depth: int,
                 impact_classification: float, sample_info_file: str, analysis_name: str, leaf_assignment_file_train: str, feature_imp_output_file: str,
                 tree_weights: bool = False, silhouette_score_file: str = None, distance_measure: str = '', cluster_assignment_file: str = None, graph_path: str = None, draw_graph: bool = False,
                 silhouette_score_train_file: str = None):
        '''
            constructor of MORGOTH

            @param X_train: a pandas DataFrame with columns as features and rows as samples
            @param y_train: a numpy array with either single outputs or lists of outputs as specified by output_format
            @param criterion_class: str defining the splitting criterion that should be used to optimize the split for the discrete response,
                only needed if output_format is set to 'classification', or 'multioutput', available criteria: 'gini', 'hellinger', and 'entropy', default of decision tree = 'gini'
            @param criterion_reg: str defining the splitting criterion that should be used to optimize the split for the continuous response,
                only needed if output_format is set to 'regression', or 'multioutput', available criteria: 'mse', default of decision trees = 'mse'
            @param sample_names_train: a list of sample names corresponding to X_train and y_train
            @param min_number_of_samples_per_leaf: int specifying the minimal number of samples per leaf, default of decision trees = 1
            @param number_of_trees_in_forest: int specifying the number of trees that should be fitted
            @param number_of_features_per_split: float or string, defining the number of features that should be used per split.
                If a float is given, it should be in [0,1], then number_of_features_per_split*number_of_features are drawn randomly.
                If a string is given, we expect it to be in wither 'sqrt' or 'log2'.
                If 'sqrt', then number_of_features_per_split=sqrt(number_of_features).
                If 'log2', then number_of_features_per_split=log2(number_of_features).
                Default of decision trees = 'sqrt'
            @param class_names: a list containing the names of all available classes or None, only needed if output_format is set to
                'classification', or 'multioutput', default = None
            @param output_format: string defining the output format, i.e., if we have only classification, or regression data, respectively,
                or a multioutput given by a list of lists with length 2 where position 0 is the continuous response and pos 1 the class
                allowed values: 'regression', 'classification', or 'multioutput', default of decision trees = 'multioutput'
            @param threshold: list of floats indicating the threshold(s) that was/were used to obtain the class labels, needed to calculate sample weights
            @param time_file: str specifying the path to a file with the fitting time information
            @param sample_weights_included: str indicating, whether/which sample weights to use.
                If 'simple', we calculate the simple weights
                If 'linear', we calculat the linear weights
                otherwise, all samples are weighted equally with 1
            @param random_state: int specifying the seed to use when creating the random objects
            @param max_depth: int specifying the maximal depth of the tree, default of the decision trees = 20
            @param impact_classification: float in [0,1], defining how much impact the classification objective function should have.
                Notably, this parameter is called lambda in our manuscript. Due to the fact "lambda" being a key word in Python, we call it impact_classification in the code.
                The impact of the regression function is 1-impact_classification. Default = 0.5. 
            @param sample_info_file: path to the file, where additional sample information should be stored
            @param analysis_name: string indicating the name of the current analysis
            @param leaf_assignment_file_train: path to the file, where the information is stored, in which leaf the training samples ended per tree
            @param feature_imp_output_file: path to the file, where the feature importance scores for all features should be stored
            @param tree_weights: bool, True if SAURON-RF tree weights should be used, False otherwise. Notably, this can only be True, if output_format is set to 'multioutput'
            @param silhouette_score_file: path to the file, where the silhouette score results for the unseen (test) samples should be stored (only used if distance measure is given)
            @param distance_measure: a string indicating, which distance should be used for the cluster analysis
                'pearson': 1-PCC
                'spearman': 1-SCC
                'cosine': cosine distance 
                'euclidean': euclidean distance
                'rank_magnitude': 1-RM
                '': no cluster analysis will be performed
            @param cluster_assignment_file: path to the file, where the most similar training samples for each test sample is stored (only used if distance measure is given)
            @param graph_path: path to the folder where the graphs should be stored (only used if draw_graph is True)
            @param draw_graph: bool indicating whether the RF graphs should be generated (True) or not (False)
            @param silhouette_score_train_file: path to the file, where the silhouette score results for the training samples should be stored (only used if distance measure is given)
        '''
        self.original_Xtrain = copy.deepcopy(X_train)
        self.original_ytrain = copy.deepcopy(y_train)
        self.original_sample_names_train = copy.deepcopy(sample_names_train)
        self.X_train = X_train.reset_index(drop=True)
        self.feature_names = np.array(self.X_train.columns)

        self.y_train = np.array(y_train)
        self.output_format = output_format
        if self.output_format == 'multioutput':
            split = np.hsplit(self.y_train, 2)
            self.y_train_reg = split[0].flatten()
            self.y_train_class = split[1].flatten()
        elif self.output_format == 'classification':
            self.y_train_class = y_train
        else:
            self.y_train_reg = y_train
        self.criterion_class = criterion_class
        self.criterion_reg = criterion_reg
        self.sample_names_train = np.array(sample_names_train)
        # integer ids of the training sample names (position in np.unique(sample_names_train))
        self._unique_train_names, self._train_name_id = np.unique(
            self.sample_names_train, return_inverse=True)
        self._col_of = {f: i for i, f in enumerate(self.feature_names)}
        self.min_number_of_samples_per_leaf = min_number_of_samples_per_leaf
        self.number_of_trees_in_forest = number_of_trees_in_forest
        self.number_of_features_per_split = number_of_features_per_split
        self.tree_weights = tree_weights
        if self.tree_weights and not self.output_format == 'multioutput':
            raise (ValueError(
                    f'Tree weights cannot be calculated for univariate RF.'))
        self.distance_measure = distance_measure
        if not self.output_format == 'regression':
            occurence_count = Counter(self.y_train_class)
            self.majority_class = occurence_count.most_common(1)[0][0]
        self.draw_graph = draw_graph
        self.graph_path = graph_path
        self.min_weight = 3
        if type(number_of_features_per_split) == str:
            if number_of_features_per_split == 'sqrt':
                self.total_number_of_features_per_split = int(
                    math.sqrt(len(self.X_train.columns)))
            elif number_of_features_per_split == 'log2':
                self.total_number_of_features_per_split = int(
                    math.log2(len(self.X_train.columns)))
            else:
                raise (ValueError(
                    f'Unsupported string for number_of_features_per_split {number_of_features_per_split}.'))
        else:
            self.total_number_of_features_per_split = int(
                number_of_features_per_split*len(self.X_train.columns))


        self.class_names = class_names
        self.thresholds = threshold
        self.sample_weights_included = sample_weights_included

        self.max_depth = max_depth
        self.impact_classification = impact_classification
        self.random_state = np.random.RandomState(random_state)

        self.analysis_name = analysis_name
        self.sample_info_file = sample_info_file
        self.time_file = time_file
        self.leaf_assignment_file_train = leaf_assignment_file_train
        self.feature_imp_output_file = feature_imp_output_file
        self.silhouette_score_file = silhouette_score_file
        self.cluster_assignment_file = cluster_assignment_file
        self.silhouette_score_train_file = silhouette_score_train_file
        print("Your input parameters are: ")
        # print("Analysis mode: " + self.analysis_mode)
        print("Number of trees: " + str(self.number_of_trees_in_forest))
        print("Min number of samples per leaf: " +
              str(self.min_number_of_samples_per_leaf))
        print("Number of features per split: " +
              str(self.number_of_features_per_split))
        print(f'Maximum depth of the trees: {self.max_depth}')
        print("The dimensions of your training matrix are: ")
        print("Number of rows (samples): " + str(self.X_train.shape[0]))
        print("Number of columns (features): " + str(self.X_train.shape[1]))
        if self.sample_weights_included in ['linear', 'simple']:
            print(f'sample weights: {self.sample_weights_included}')
        else:
            print('no sample weights')

    def fit(self):
        '''
            fits the trees in the forest using the given weighting scheme

        '''

        # decide which model should be fit: with weights or plain
        self.__dict__.pop('_silhouette_cache', None)
        self.__dict__.pop('_train_silhouette_cache', None)
        self.train_sample_weights = []
        if self.sample_weights_included in ["simple", 'linear']:

            if not math.isnan(self.thresholds[0]):
                if self.output_format == 'multioutput':
                    self.train_sample_weights = np.array(self.calculate_weights(
                        self.thresholds, self.y_train_reg, self.sample_weights_included))
                elif self.output_format == 'regression':
                    self.train_sample_weights = np.array(self.calculate_weights(
                        self.thresholds, self.y_train, self.sample_weights_included))
                elif self.sample_weights_included == 'simple':
                    self.train_sample_weights = self.calculate_simple_weights(sorted_thresholds=self.thresholds, response_values=self.y_train)
                else:
                    self.train_sample_weights = np.ones(len(self.y_train))
                start_time = time.perf_counter()
                self.grow_parallel_trees()
                end_time = time.perf_counter()
                self.elapsed_time = end_time - start_time
            else:

                raise (ValueError(
                    "No Threshold was given to calculate sample weights.\nPlease provide a threshold. Calculations aborted."))
        else:
            self.train_sample_weights = np.ones(len(self.sample_names_train))
            start_time = time.perf_counter()
            self.grow_parallel_trees()
            end_time = time.perf_counter()
            self.elapsed_time = end_time - start_time

        with open(self.sample_info_file, 'w') as sample_info_file:
            sample_info_file.write('Sample_name\tWeight\n')
            for i, sample in enumerate(self.sample_names_train):
                sample_info_file.write(
                    f'{sample}\t{self.train_sample_weights[i]}\n')

        print(f'forest grown in {self.elapsed_time} seconds.')
        with open(self.time_file, 'w') as time_file:
            time_file.write('Name_of_Analysis\tParameters\tTime\n')
            time_file.write(f'{self.analysis_name}\t#Trees:{self.number_of_trees_in_forest},#MinSamplesLeaf:{self.min_number_of_samples_per_leaf},#FeaturesPerSplit:{self.total_number_of_features_per_split},MaxDepth:{self.max_depth}\t{self.elapsed_time}\n')
        self.write_leaf_assignment_to_file()
        self.print_feature_importance_to_file()

    def grow_tree(self, random_object: np.random.RandomState) -> 'tuple[MultivariateDecisionTree, float]':
        ''' 
            grows a MultivariateDecisionTree with the given random object

            @param random_object: a np.random.RandomState instance that is used to randomly sample the bootstrap samples and also to 
                sample the features that are considered per split
            @return: a newly created and fitted MultivariateDecisionTree
        '''
        self.train_sample_weights = np.array(self.train_sample_weights)
        new_tree, seconds = _grow_tree(
            self._train_matrix(), random_object, self._tree_settings())
        new_tree._attach_shared(
            self.X_train, self.feature_names, self._col_of)
        return new_tree, seconds

    def _train_matrix(self) -> np.ndarray:
        '''
            @return: the transposed training matrix (features x samples) as contiguous float64 numpy array
        '''
        return np.ascontiguousarray(self.X_train.to_numpy(dtype=np.float64).T)

    def _tree_settings(self) -> dict:
        return dict(feature_names=self.feature_names, col_of=self._col_of, y_train=self.y_train, sample_weights=np.array(self.train_sample_weights),
                    sample_names_train=self.sample_names_train, class_names=self.class_names, criterion_class=self.criterion_class,
                    criterion_reg=self.criterion_reg, min_number_of_samples_per_leaf=self.min_number_of_samples_per_leaf, max_depth=self.max_depth,
                    number_of_features_per_split=self.number_of_features_per_split, impact_classification=self.impact_classification,
                    output_format=self.output_format, distance_measure=self.distance_measure)

    def grow_parallel_trees(self):
        '''
            grows self.number_of_trees_in_forest in the forest
            the process is parallelized as good as possible, i.e., as many trees as cores are built simultaneously
            and then are added to the tree list
        '''
        self.trees = []
        self.train_sample_weights = np.array(self.train_sample_weights)
        settings = self._tree_settings()
        XT = self._train_matrix()
        shm = None
        try:
            # one copy of the training data in shared memory that is used by all worker processes
            shm = shared_memory.SharedMemory(
                create=True, size=max(XT.nbytes, 1))
            shared = np.ndarray(XT.shape, dtype=XT.dtype, buffer=shm.buf)
            shared[:] = XT
            del shared
            initargs = (shm.name, XT.shape, XT.dtype.str, None, settings)
            del XT
        except OSError:
            # shared memory not available, every worker gets its own copy
            if shm is not None:
                shm.close()
                shm.unlink()
                shm = None
            initargs = (None, None, None, XT, settings)
        try:
            with Pool(initializer=_init_tree_worker, initargs=initargs) as pool:
                seed_list = []
                for random_seed in self.random_state.randint(0, 2**32 - 1, dtype='int64', size=self.number_of_trees_in_forest):
                    # creates random seeds for all treees
                    seed_list.append(np.random.RandomState(random_seed))
                # parallel built of all trees
                results = pool.map_async(_grow_tree_worker, seed_list)
                seconds = []
                for t in results.get():
                    t[0]._attach_shared(
                        self.X_train, self.feature_names, self._col_of)
                    self.trees.append(t[0])
                    seconds.append(t[1])
        finally:
            if shm is not None:
                shm.close()
                shm.unlink()
        print(f'average time for a tree: {np.mean(seconds)}')
        assert (len(self.trees) == self.number_of_trees_in_forest)
        self.trees = np.array(self.trees)

    def _used_features(self) -> set:
        '''
            @return: the set of features that are used in at least one split of the forest
        '''
        used = set()
        for tree in self.trees:
            for leaf in tree.leaves:
                for feature, _ in leaf._path:
                    used.add(feature)
        return used

    def _test_matrix(self, X_test: pd.DataFrame) -> np.ndarray:
        '''
            @return: the transposed feature matrix (features x samples, ordered as the training features) of the test samples
        '''
        columns = list(self.feature_names)
        present = set(X_test.columns)
        missing = [f for f in columns if f not in present]
        if len(missing) > 0:
            used = self._used_features()
            missing_used = [f for f in missing if f in used]
            if len(missing_used) > 0:
                raise KeyError(
                    f'The following features are missing in X_test: {missing_used}')
            frame = X_test.reindex(columns=columns)
        else:
            frame = X_test[columns]
        return np.ascontiguousarray(frame.to_numpy(dtype=np.float64).T)

    def _predict_trees(self, X_test: pd.DataFrame) -> 'tuple[np.ndarray, np.ndarray]':
        '''
            distributes the test samples to the leaves of all trees and collects the predictions of the trees

            @param X_test: a pandas DataFrame with the test samples
            @return: tuple of two numpy arrays (#trees x #test samples) with the regression and the classification predictions
                of each tree (None if not available for the output format)
        '''
        names = np.array(X_test.index)
        XT = self._test_matrix(X_test)
        number_of_trees = len(self.trees)
        tree_predictions_reg = None
        tree_predictions_class = None
        if not self.output_format == 'classification':
            tree_predictions_reg = np.full((number_of_trees, len(names)), np.nan)
        if not self.output_format == 'regression':
            tree_predictions_class = np.empty(
                (number_of_trees, len(names)), dtype=object)
        for t, tree in enumerate(self.trees):
            leaf_ids = tree._route(XT)
            tree._set_test_samples(names, leaf_ids)
            leaf_reg = np.full(len(tree.leaves), np.nan)
            leaf_class = np.empty(len(tree.leaves), dtype=object)
            for j, leaf in enumerate(tree.leaves):
                if len(leaf.sample_ids_test) == 0:
                    continue
                prediction = leaf.predict(class_names=tree.class_names)
                if self.output_format == 'multioutput':
                    leaf_reg[j] = prediction[0]
                    leaf_class[j] = prediction[1]
                elif self.output_format == 'classification':
                    leaf_class[j] = prediction
                else:
                    leaf_reg[j] = prediction
            if tree_predictions_reg is not None:
                tree_predictions_reg[t] = leaf_reg[leaf_ids]
            if tree_predictions_class is not None:
                tree_predictions_class[t] = leaf_class[leaf_ids]
        return tree_predictions_reg, tree_predictions_class

    def predict(self, X_test: pd.DataFrame, quantile: 'list[float]' = None) -> np.array:
        '''
            predicts for a given test set

            @param X_test: a pandas DataFrame with the test samples
            @param quantile: list of floats or none, if not none quantile prediction is performed
            @return: a list containing the predictions, s.t. it matches the output format
        '''
        start = time.perf_counter()
        tree_predictions_reg, tree_predictions_class = self._predict_trees(
            X_test)
        if self.output_format == 'classification':
            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array(self.predict_classification(X_test=X_test, tree_predictions_class=tree_predictions_class))

        elif self.output_format == 'regression':
            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array(self.predict_regression(X_test=X_test, tree_predictions_reg=tree_predictions_reg, quantile=quantile))

        elif self.output_format == 'multioutput':
            forest_predictions_class = np.array(self.predict_classification(
                X_test=X_test, tree_predictions_class=tree_predictions_class))
            forest_predictions_reg = np.array(self.predict_regression(
                X_test=X_test, tree_predictions_reg=tree_predictions_reg, quantile=quantile, class_predictions_forest=forest_predictions_class))
            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            end = time.perf_counter()
            print(f'{end-start} s needed for prediction')
            return np.array([[forest_predictions_reg[i], forest_predictions_class[i]] for i in range(len(X_test.index))])

    def predict_proba(self, X_test: pd.DataFrame, quantile: 'list[float]' = None) -> np.array:
        '''
            used if one wants the prediction probabilites for the classification instead of the majority class

            @param X_test: a pandas DataFrame with the test samples
            @param quantile: list of floats or none, if not none quantile prediction is performed
            @return: a list containing the predictions, s.t. it matches the output format
        '''
        if self.output_format == 'classification':
            _, tree_predictions_class = self._predict_trees(X_test)
            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array(self.predict_proba_classification(X_test=X_test, tree_predictions_class=tree_predictions_class))

        elif self.output_format == 'regression':
            return np.array(self.predict(X_test=X_test, quantile=quantile))

        elif self.output_format == 'multioutput':
            tree_predictions_reg, tree_predictions_class = self._predict_trees(
                X_test)
            forest_predictions_class = np.array(self.predict_proba_classification(
                X_test=X_test, tree_predictions_class=tree_predictions_class))
            class_predictions_forest = np.array(self.predict_classification(
                X_test=X_test, tree_predictions_class=tree_predictions_class))
            forest_predictions_reg = np.array(self.predict_regression(
                X_test=X_test, tree_predictions_reg=tree_predictions_reg, quantile=quantile, class_predictions_forest=class_predictions_forest))
            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array([[forest_predictions_reg[i], forest_predictions_class[i]] for i in range(len(X_test.index))])

    def _class_counts(self, tree_predictions_class: np.array) -> np.ndarray:
        '''
            @param tree_predictions_class: array (#trees x #samples) or (#trees x #samples x 1) with the class predictions of the trees
            @return: numpy array (#classes x #samples) with the number of trees voting for each class
        '''
        predictions = np.asarray(tree_predictions_class)
        if predictions.ndim == 3 and predictions.shape[2] == 1:
            predictions = predictions[:, :, 0]
        counts = np.zeros((len(self.class_names), predictions.shape[1]))
        for k, class_name in enumerate(self.class_names):
            votes = predictions == class_name
            if isinstance(votes, np.ndarray) and votes.shape == predictions.shape:
                counts[k] = votes.sum(axis=0)
        return counts

    def predict_proba_classification(self, X_test: pd.DataFrame, tree_predictions_class: np.array) -> np.array:
        '''
            used to get the prediction probabilites for the classification instead of the majority class

            @param X_test: a pandas DataFrame with the test samples
            @param tree_predictions_class: a list of list with all tree predictions for the test samples
            @return: a list of list containing the predictions, each class is positional encoded as in the given class names list
        '''
        forest_predictions_class = (self._class_counts(
            tree_predictions_class) / self.number_of_trees_in_forest).T
        assert (np.all(forest_predictions_class.sum(axis=1) == 1))
        return np.array(forest_predictions_class)

    def predict_classification(self, X_test: pd.DataFrame, tree_predictions_class: np.array) -> np.array:
        '''
            used to get the prediction for the classification

            @param X_test: a pandas DataFrame with the test samples
            @param tree_predictions_class: a list of list with all tree predictions for the test samples
            @return: a list containing the majority class for each test sample
        '''
        counts = self._class_counts(tree_predictions_class)
        # np.argmax returns the first maximum, i.e., ties are resolved in favour of the first class (as before)
        majority = np.argmax(counts, axis=0)
        maxima = counts.max(axis=0)
        forest_predictions_class = [self.class_names[c] if m > 0 else None
                                    for c, m in zip(majority, maxima)]
        return np.array(forest_predictions_class)

    def predict_regression(self, X_test: pd.DataFrame, tree_predictions_reg: np.array, quantile: 'list[float]' = None, class_predictions_forest: np.array = None) -> np.array:
        '''
            calculates the predication for the regression part

            @param X_test: a pandas DataFrame with the test samples
            @param tree_predictions_reg: a list of list with all tree predictions for the test samples
            @param quantile: list of floats or none, if not none quantile prediction is performed
            @return: a list of lists in case of quantile prediction where the position matches
                the position of the respective alpha in the given quantile list
                or a list containing the "normal" regression results
        '''
        self.calculate_tree_weights(X_test, class_predictions_forest)
        tree_predictions_reg = np.asarray(
            tree_predictions_reg, dtype=np.float64)
        if quantile is None:
            tree_weights = self._tree_weight_matrix
            forest_predictions_reg = np.zeros(len(X_test.index))
            # summed up tree by tree (same order as before)
            for j in range(len(tree_predictions_reg)):
                forest_predictions_reg += tree_weights[:, j] * \
                    tree_predictions_reg[j]
        else:
            self.calculate_weight_efficiently(
                X_test=X_test, class_predictions_forest=class_predictions_forest)
            forest_predictions_reg = np.array(
                self.quantile_prediction(X_test=X_test, quantile=quantile))

        return forest_predictions_reg

    def calculate_weights(self, threshold: 'list[float]', response_values: np.array, weighting_scheme: str) -> np.array:
        '''
            calculates the weights for the given weighting scheme

            @param threshold: the threshold used to discretize the values
            @param response_vlaues: the continuous response values
            @param weighting_scheme: a string defining the weighting scheme, available 'simple', 'linear'
            @return: the calculated sample weights
        '''
        if weighting_scheme == "simple":
            return np.array(self.calculate_simple_weights(threshold, response_values))
        elif weighting_scheme == "linear":
            return np.array(self.calculate_linear_weights(threshold, response_values))
        else:
            print(
                "The given weighting scheme is not supported. Supported schemes are: simple and linear.")
            print("Using weighting sheme simple instead")
            return np.array(self.calculate_simple_weights(threshold, response_values))

    def calculate_linear_weights(self, sorted_thresholds, response_values):
        '''
        @param sorted_thresholds: the thresholds that are used to discretize the response variables (increasingly sorted)
        @param response_values: the continuous response values for each sample
        @return weights: the calculated weights for each sample (in the order of the response values)

        '''

        assert (all(sorted_thresholds[i] <= sorted_thresholds[i+1]
                for i in range(len(sorted_thresholds)-1))), "Thresholds not sorted"

        sum_weights = [0] * (len(sorted_thresholds)+1)

        for value in response_values:

            found_threshold = False
            for i in range(len(sorted_thresholds)+1):

                if not found_threshold:

                    if i == 0 and value < sorted_thresholds[i]:

                        found_threshold = True
                        sum_weights[i] = sum_weights[i] + \
                            abs(value - sorted_thresholds[i])

                    elif i == len(sorted_thresholds):
                        found_threshold = True
                        sum_weights[i] = sum_weights[i] + \
                            abs(value - sorted_thresholds[i-1])

                    elif value < sorted_thresholds[i]:

                        found_threshold = True
                        sum_weights[i] = sum_weights[i] + abs(
                            value - sorted_thresholds[i]) + abs(value - sorted_thresholds[i-1])

        weights = []
        norm_factor = len(sorted_thresholds) + 1

        for value in response_values:
            found_threshold = False
            for i in range(len(sorted_thresholds)+1):

                if not found_threshold:

                    if i == 0 and value < sorted_thresholds[i]:

                        found_threshold = True
                        new_weight = abs(
                            value - sorted_thresholds[i]) / (sum_weights[i]*norm_factor)
                        weights.append(new_weight)

                    elif i == len(sorted_thresholds):
                        found_threshold = True
                        new_weight = abs(
                            value - sorted_thresholds[i - 1]) / (sum_weights[i]*norm_factor)
                        weights.append(new_weight)

                    elif value < sorted_thresholds[i]:

                        found_threshold = True
                        sum_weights[i] = abs(value - sorted_thresholds[i]) + abs(
                            value - sorted_thresholds[i - 1]) / (sum_weights[i]*norm_factor)

        return weights

    def calculate_simple_weights(self, sorted_thresholds, response_values):
        '''
        @param sorted_thresholds: the thresholds that are used to discretize the response variables (increasingly sorted)
        @param response_values: the continuous response values for each sample
        @return weights: the calculated weights for each sample (in the order of the response values)

        '''

        assert (all(sorted_thresholds[i] <= sorted_thresholds[i+1]
                for i in range(len(sorted_thresholds)-1))), "Thresholds not sorted"

        sum_weights = [0] * (len(sorted_thresholds)+1)

        for value in response_values:

            found_threshold = False
            for i in range(len(sorted_thresholds)+1):

                if not found_threshold:

                    if i == len(sorted_thresholds):
                        found_threshold = True
                        sum_weights[i] = sum_weights[i] + 1

                    elif value < sorted_thresholds[i]:

                        found_threshold = True
                        sum_weights[i] = sum_weights[i] + 1

        max_sum = max(sum_weights)
        weights = []

        for value in response_values:
            found_threshold = False
            for i in range(len(sorted_thresholds)+1):

                if not found_threshold:
                    if i == len(sorted_thresholds):
                        found_threshold = True
                        new_weight = float(max_sum)/float(sum_weights[i])
                        weights.append(new_weight)
                    elif value < sorted_thresholds[i]:

                        found_threshold = True
                        new_weight = float(max_sum)/float(sum_weights[i])
                        weights.append(new_weight)
        return weights

    def _tree_leaf_classes(self, number_of_samples: int) -> np.ndarray:
        '''
            @return: numpy array (#trees x #test samples) with the majority class of the leaf each test sample ended in
        '''
        result = np.empty((len(self.trees), number_of_samples), dtype=object)
        for t, tree in enumerate(self.trees):
            majority = np.empty(len(tree.leaves), dtype=object)
            for j, leaf in enumerate(tree.leaves):
                majority[j] = getattr(leaf, 'majority_class', None)
            result[t] = majority[tree._test_leaf_ids]
        return result

    def calculate_tree_weights(self, X_test: pd.DataFrame, class_predictions_forest: np.array):
        '''
            calculates the SAURON-RF weights for each tree
        '''
        test_samples = list(X_test.index)
        number_of_trees = len(self.trees)
        tree_weights = np.full(
            (len(test_samples), number_of_trees), 1/number_of_trees)
        if self.tree_weights:
            X_test['class_predictions'] = class_predictions_forest
            leaf_classes = self._tree_leaf_classes(len(test_samples))
            for i in range(len(test_samples)):
                prediction_test_sample = class_predictions_forest[i]
                if self.majority_class == prediction_test_sample:
                    continue
                agreeing_trees = (
                    leaf_classes[:, i] == prediction_test_sample).astype(np.float64)
                tree_weights[i] = agreeing_trees / np.sum(agreeing_trees)
        self._tree_weight_matrix = tree_weights
        self.tree_weights_dict = {test_sample: tree_weights[i]
                                  for i, test_sample in enumerate(test_samples)}

    def calculate_weight_efficiently(self, X_test: pd.DataFrame, class_predictions_forest: np.array) -> None:
        '''
            calculates the weight of every training sample for every test sample (needed for the quantile regression)
        '''
        number_of_test_samples = len(X_test.index)
        weights = np.zeros(
            (number_of_test_samples, len(self._unique_train_names)))
        tree_weights = self._tree_weight_matrix
        for tree_id, tree in enumerate(self.trees):
            leaf_ids = tree._test_leaf_ids
            bootstrap_name_ids = self._train_name_id[tree.bootstrap_indices]
            order = np.argsort(leaf_ids, kind='stable')
            boundaries = np.searchsorted(
                leaf_ids[order], np.arange(len(tree.leaves) + 1))
            rows, columns, values = [], [], []
            for j, leaf in enumerate(tree.leaves):
                samples = order[boundaries[j]:boundaries[j+1]]
                if len(samples) == 0:
                    continue
                name_ids = bootstrap_name_ids[leaf._rows]
                rows.append(np.repeat(samples, len(name_ids)))
                columns.append(np.tile(name_ids, len(samples)))
                values.append((tree_weights[samples, tree_id][:, None] *
                               np.asarray(leaf.normalized_sample_weights)[None, :]).ravel())
            if len(rows) > 0:
                # np.add.at adds the values one after another (same order as before)
                np.add.at(weights, (np.concatenate(rows), np.concatenate(
                    columns)), np.concatenate(values))
        self._train_weight_matrix = weights
        self.test_samples_to_train_sample_weight = _TrainWeightTable(
            weights, X_test.index, self.sample_names_train, self._train_name_id)

    def quantile_prediction(self, X_test: pd.DataFrame, quantile: 'list[float]'):
        '''
            calculates the quantile prediction as described by Meinshausen et al

            @param X_test: a pandas DataFrame with all test samples
            @param quantile: a list of quantiles
            @return: a list of lists in case of quantile prediction where the position matches
                the position of the respective alpha in the given quantile list
        '''
        if self.output_format == 'multioutput':
            y_train_values = np.asarray(self.y_train_reg)
        else:
            y_train_values = np.asarray(self.y_train)
        # stable sort, i.e., equal values keep their order (as python's sorted)
        order = np.argsort(y_train_values, kind='stable')
        y_train_values_sorted = y_train_values[order]
        # weight of every training sample (by position) for every test sample
        weights = self._train_weight_matrix[:, self._train_name_id[order]]
        cumulative_weights = np.cumsum(weights, axis=1)
        found = []
        positions = []
        for q in quantile:
            # infimum y that fulfills the condition that the sum of the weights are larger than the wanted quantile
            reached = cumulative_weights >= q
            found.append(reached.any(axis=1))
            positions.append(reached.argmax(axis=1))

        quantile_prediction = []
        for sample_id in range(len(X_test.index)):
            quantile_prediction_sample = []
            for i in range(len(quantile)):
                if found[i][sample_id]:
                    quantile_prediction_sample.append(
                        y_train_values_sorted[positions[i][sample_id]])
                else:
                    sum_weight = cumulative_weights[sample_id, -1]
                    print(
                        f'warning weights did not sum up to 1 or quantile > 1? {sum_weight}, {quantile}')
                    quantile_prediction_sample.append(
                        y_train_values_sorted[len(self.y_train)-1])
            quantile_prediction.append(quantile_prediction_sample)
        assert (len(quantile_prediction) == len(X_test.index))
        return quantile_prediction

    def write_leaf_assignment_to_file(self):
        '''
            writes the leaf assignment of the training samples to the specified file
        '''
        with open(self.leaf_assignment_file_train, "w", encoding="utf-8") as leaf_assignment_output:

            unique_names = self._unique_train_names
            leaf_assignment_output.write(
                "Tree" + "\t" + "\t".join([str(x) for x in unique_names]) + "\n")
            nan_string = str(float("NaN"))
            for i, tree in enumerate(self.trees):
                bootstrap_name_ids = self._train_name_id[tree.bootstrap_indices]
                first_leaf = np.full(len(unique_names), -1)
                for j, leaf in enumerate(tree.leaves):
                    name_ids = np.unique(bootstrap_name_ids[leaf._rows])
                    found_before = first_leaf[name_ids] >= 0
                    for _ in range(int(found_before.sum())):
                        print(
                            "Found a sample twice in different leaves of the same tree")
                    first_leaf[name_ids[~found_before]] = j
                current_row = [f'{i}'] + \
                    [nan_string if j < 0 else f'{j}' for j in first_leaf]
                leaf_assignment_output.write("\t".join(current_row) + "\n")

    def print_feature_importance_to_file(self):
        self.feature_imp_fit_model = {}
        importances = np.zeros(len(self.feature_names))
        for tree in self.trees:
            importances += 1 / self.number_of_trees_in_forest * \
                tree.feature_importances.dense(self._col_of)

        with open(self.feature_imp_output_file, "w", encoding="utf-8") as feature_imp_output:

            for i, current_feature_name in enumerate(self.feature_names):
                self.feature_imp_fit_model[current_feature_name] = importances[i]
                feature_imp_output.write(
                    str(current_feature_name) + "\t" + str(self.feature_imp_fit_model[current_feature_name]) + "\n")

    ######################## silhouette scores ########################
    # For a sample, the "cluster" consists of all training samples that share a leaf with it (over all trees).
    # Since these clusters contain samples multiple times, the original implementation built the
    # (#trees * leaf size) x (#trees * leaf size) distance matrix for every sample. Here, the clusters are
    # represented by count vectors and the mean distances are computed with matrix products instead.

    def _silhouette_data(self) -> dict:
        '''
            min-max normalized training data and the pairwise similarities/distances of the training samples
            (rows/columns ordered by the sample name ids)
        '''
        cache = self.__dict__.get('_silhouette_cache')
        if cache is not None and cache['distance_measure'] == self.distance_measure:
            return cache
        if len(self._unique_train_names) != len(self.sample_names_train):
            raise ValueError(
                'The calculation of silhouette scores requires unique training sample names.')
        minima = self.X_train.min().to_numpy(dtype=np.float64)
        maxima = self.X_train.max().to_numpy(dtype=np.float64)
        position_of_id = np.argsort(self._train_name_id)
        normalized = self._min_max_normalize(self.X_train.to_numpy(
            dtype=np.float64)[position_of_id], minima, maxima)
        cache = {'distance_measure': self.distance_measure, 'minima': minima, 'maxima': maxima,
                 'normalized': normalized, 'matrix': distance_matrix(normalized, self.distance_measure)}
        self._silhouette_cache = cache
        return cache

    @staticmethod
    def _min_max_normalize(values: np.ndarray, minima: np.ndarray, maxima: np.ndarray) -> np.ndarray:
        '''
            min-max normalization with the minima and maxima of the training data.
            Features that are constant in the training data are set to 1 for all samples.
        '''
        spans = maxima - minima
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (values - minima) / spans
        normalized[:, spans == 0] = 1.0
        return normalized

    def _leaf_counts(self, tree: MultivariateDecisionTree) -> np.ndarray:
        '''
            @return: numpy array (#leaves x #training sample names): how often each training sample is in each leaf
        '''
        bootstrap_name_ids = self._train_name_id[tree.bootstrap_indices]
        leaf_of_sample = np.empty(len(bootstrap_name_ids), dtype=np.intp)
        for j, leaf in enumerate(tree.leaves):
            leaf_of_sample[leaf._rows] = j
        counts = np.zeros((len(tree.leaves), len(self._unique_train_names)))
        np.add.at(counts, (leaf_of_sample, bootstrap_name_ids), 1)
        return counts

    def _train_silhouette_scores(self) -> 'tuple[np.ndarray, np.ndarray]':
        '''
            @return: silhouette scores of the training samples (without duplicates, with duplicates)
        '''
        cache = self.__dict__.get('_train_silhouette_cache')
        if cache is not None and cache[0] == self.distance_measure:
            return cache[1], cache[2]
        data = self._silhouette_data()
        matrix = data['matrix']
        XT = self._train_matrix()
        number_of_samples = len(self.sample_names_train)
        counts = np.zeros((number_of_samples, len(self._unique_train_names)))
        first_member = None
        for t, tree in enumerate(self.trees):
            leaf_counts = self._leaf_counts(tree)
            # unique sample names: in-bag samples are routed to the leaf they were assigned to during training
            leaf_ids = tree._route(XT)
            counts += leaf_counts[leaf_ids]
            if t == 0:
                bootstrap_name_ids = self._train_name_id[tree.bootstrap_indices]
                first_member = np.array([bootstrap_name_ids[leaf._rows[0]]
                                         for leaf in tree.leaves])[leaf_ids]
        distance = self.distance_measure not in SIMILARITY_MEASURES
        unique_members = (counts > 0).astype(np.float64)
        intra_dup = multiset_intra_cluster_mean(matrix, counts, distance)
        intra_no_dup = multiset_intra_cluster_mean(
            matrix, unique_members, distance)
        # as in the original implementation, the distance of the sample to the cluster is the distance
        # to the first member of the cluster (first sample in its leaf of the first tree or smallest sample name, respectively)
        own = self._train_name_id
        inter_dup = matrix[own, first_member]
        inter_no_dup = matrix[own, np.argmax(unique_members > 0, axis=1)]
        if not distance:
            inter_dup = 1 - inter_dup
            inter_no_dup = 1 - inter_no_dup
        score_no_dup = silhouette_scores(intra_no_dup, inter_no_dup)
        score_dup = silhouette_scores(intra_dup, inter_dup)
        self._train_silhouette_cache = (
            self.distance_measure, score_no_dup, score_dup)
        return score_no_dup, score_dup

    def write_train_sample_silhouette_scores_to_file(self):
        '''
            writes the silhouette score of the training samples to the specified file
        '''
        if self.distance_measure == '':
            print('no distance measure given')
            return
        score_no_dup, score_dup = self._train_silhouette_scores()
        with open(self.silhouette_score_train_file, "w", encoding="utf-8") as silhouette_output:
            silhouette_output.write(
                f'cell_line\tsilhouette_score_no_duplicates\tsilhouette_score_with_duplicates\n')
            for i, sample_name in enumerate(self.sample_names_train):
                silhouette_output.write(
                    f'{sample_name}\t{score_no_dup[i]}\t{score_dup[i]}\n')

    def write_leaf_silhouette_scores_to_file(self, test_sample_names: np.array, X_test):
        '''
            writes the silhouette score for unseen (test) samples to specific file

            @param test_sample_names: the names of the test samples of interest
            @param X_test: the features of the test samples
        '''
        if self.draw_graph:
            self.write_probability_graphs_to_file(test_sample_names)
        if self.distance_measure == '':
            print('no distance measure given')
            return

        data = self._silhouette_data()
        matrix = data['matrix']
        self.write_train_sample_silhouette_scores_to_file()

        unique_names = np.unique(np.array(test_sample_names))
        position_of_name = {}
        for i, name in enumerate(X_test.index):
            position_of_name.setdefault(name, i)
        positions = np.array([position_of_name[name]
                             for name in unique_names], dtype=np.intp)
        # the test data is normalized with the minima and maxima of the training data (X_test itself is not modified)
        test_values = X_test[list(self.feature_names)].to_numpy(
            dtype=np.float64)[positions]
        normalized_test = self._min_max_normalize(
            test_values, data['minima'], data['maxima'])
        distance = self.distance_measure not in SIMILARITY_MEASURES
        test_to_train = distances_to_samples(
            data['normalized'], normalized_test, self.distance_measure)
        if not distance:
            test_to_train = 1 - test_to_train

        counts_dup = np.zeros((len(unique_names), len(self._unique_train_names)))
        counts_no_dup = np.zeros_like(counts_dup)
        cluster_members = [[] for _ in unique_names]
        for tree in self.trees:
            leaf_counts = self._leaf_counts(tree)
            leaf_ids = tree._test_leaf_ids[positions]
            counts_dup += leaf_counts[leaf_ids]
            counts_no_dup += leaf_counts[leaf_ids] > 0
            unique_leaf_members = [np.unique(np.array(leaf.sample_names_train))
                                   for leaf in tree.leaves]
            for i, leaf_id in enumerate(leaf_ids):
                cluster_members[i].append(unique_leaf_members[leaf_id])

        with np.errstate(divide='ignore', invalid='ignore'):
            inter_dup = (counts_dup * test_to_train).sum(axis=1) / \
                counts_dup.sum(axis=1)
            inter_no_dup = (counts_no_dup * test_to_train).sum(axis=1) / \
                counts_no_dup.sum(axis=1)
        score_dup = silhouette_scores(multiset_intra_cluster_mean(
            matrix, counts_dup, distance), inter_dup)
        score_no_dup = silhouette_scores(multiset_intra_cluster_mean(
            matrix, counts_no_dup, distance), inter_no_dup)

        with open(self.silhouette_score_file, "w", encoding="utf-8") as silhouette_output:
            with open(self.cluster_assignment_file, 'w', encoding='utf-8') as cluster_output:
                cluster_output.write('test_sample\ttraining_sample_list\n')
                silhouette_output.write(
                    f'cell_line\tsilhouette_score_no_duplicates\tsilhouette_score_with_duplicates\n')
                for i, sample_name in enumerate(unique_names):
                    x_sample_train_names = np.concatenate(cluster_members[i])
                    silhouette_output.write(
                        f'{sample_name}\t{score_no_dup[i]}\t{score_dup[i]}\n')
                    cluster_output.write(
                        f'{sample_name}\t{list(x_sample_train_names)}\n')

    def write_probability_graphs_to_file(self, test_sample_names: np.array):
        '''
            generates RF graphs for given test samples and writes them to dot files

            @param test_sample_names: names of the test samples of interest
        '''
        unique_names = np.unique(np.array(test_sample_names))
        forest_edge_weight_dict = {}
        forest_feature_score_dict = {}
        for sample_id in unique_names:
            edge_weight_dict = {}
            rank_list_features = {}
            for tree in self.trees:
                path = tree.sample_feature_path[sample_id]
                path_length = len(path)
                for i, feature_rank in enumerate(path):
                    feature_name = feature_rank[0]
                    rank = feature_rank[1]
                    if feature_name in rank_list_features.keys():
                        rank_list_features[feature_name].append(rank)
                    else:
                        rank_list_features[feature_name] = [rank]
                    if i < path_length - 1:
                        edge_name = f'{feature_name.replace("-", "")} -> {path[i+1][0].replace("-", "")}'
                        if edge_name in edge_weight_dict.keys():
                            edge_weight_dict[edge_name] += 1
                            forest_edge_weight_dict[edge_name] += 1
                        else:
                            edge_weight_dict[edge_name] = 1
                            if edge_name in forest_edge_weight_dict.keys():
                                forest_edge_weight_dict[edge_name] += 1
                            else:
                                forest_edge_weight_dict[edge_name] = 1
            average_rank_features = {}
            color_dict = {}

            used_features = rank_list_features.keys()
            for feature in used_features:
                average_rank_features[feature] = np.mean(
                    rank_list_features[feature])
                if feature in forest_feature_score_dict.keys():
                    forest_feature_score_dict[feature].append(
                        average_rank_features[feature])
                else:
                    forest_feature_score_dict[feature] = [
                        average_rank_features[feature]]
                int_rank = round(average_rank_features[feature], 0)
                if (int(int_rank) >= 11):
                    color_dict[feature] = 11
                elif (int(int_rank) < 0):
                    color_dict[feature] = 0
                else:
                    color_dict[feature] = int(int_rank)+1

            with open(f'{self.graph_path}{sample_id}.dot', 'w', encoding='utf-8') as output_file_sample:
                output_file_sample.write('digraph {\n')
                output_file_sample.write('node [colorscheme = spectral11]\n')
                for feature in used_features:
                    output_file_sample.write(
                        f'{feature.replace("-", "")} [label = "{feature}, avg rank = {round(average_rank_features[feature], 2)}", style = filled, color = {color_dict[feature]}]\n')
                for edge in edge_weight_dict.keys():
                    edge_weight = edge_weight_dict[edge]
                    edge = re.sub('[A-z]-[A-z]', string=edge, repl=label_fun)
                    if edge_weight > self.min_weight:
                        output_file_sample.write(
                            f'{edge} [weight = {edge_weight}, label = {edge_weight}, penwidth = {edge_weight}]\n')
                output_file_sample.write('}')

        color_dict = {}
        average_forest_feature_score_dict = {}
        used_features = forest_feature_score_dict.keys()
        for feature in used_features:
            average_forest_feature_score_dict[feature] = np.mean(
                forest_feature_score_dict[feature])

            int_rank = round(average_forest_feature_score_dict[feature], 0)
            if (int(int_rank) > 11):
                color_dict[feature] = 11
            elif (int(int_rank) <= 0):
                color_dict[feature] = 1
            else:
                color_dict[feature] = int(int_rank)

        with open(f'{self.graph_path}_graph_whole_forest.dot', 'w', encoding='utf-8') as output_file_sample:
            output_file_sample.write('digraph {\n')
            output_file_sample.write('node [colorscheme = spectral11]\n')
            for feature in used_features:
                output_file_sample.write(
                    f'{feature.replace("-", "")} [label = "{feature}, avg rank = {round(average_forest_feature_score_dict[feature], 2)}", style = filled, color = {color_dict[feature]}]\n')
            for edge in forest_edge_weight_dict.keys():
                edge_weight = forest_edge_weight_dict[edge]
                edge = re.sub('[A-z]-[A-z]', string=edge, repl=label_fun)
                if edge_weight > (3*len(unique_names)):
                    output_file_sample.write(
                        f'{edge} [weight = {edge_weight}, label = {edge_weight}, penwidth = {edge_weight/len(unique_names)}]\n')
            output_file_sample.write('}')

        with open(f'{self.graph_path}_graph_average_whole_forest.dot', 'w', encoding='utf-8') as output_file_sample:
            output_file_sample.write('digraph {\n')
            output_file_sample.write('node [colorscheme = spectral11]\n')
            for feature in used_features:
                output_file_sample.write(
                    f'{feature.replace("-", "")} [label = "{feature}, avg rank = {round(average_forest_feature_score_dict[feature], 2)}", style = filled, color = {color_dict[feature]}]\n')
            for edge in forest_edge_weight_dict.keys():
                edge_weight = forest_edge_weight_dict[edge] / len(unique_names)
                edge = re.sub('[A-z]-[A-z]', string=edge, repl=label_fun)
                if edge_weight > self.min_weight:
                    output_file_sample.write(
                        f'{edge} [weight = {edge_weight}, label = {edge_weight}, penwidth = {edge_weight/len(unique_names)}]\n')
            output_file_sample.write('}')


def label_fun(match: re.Match) -> str:
    string = match.group()
    return string.replace('-', '')
