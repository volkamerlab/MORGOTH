from collections import Counter
import operator
from typing import Union
import numpy as np
import math
import time
import copy
from morgoth.multivariate_dt import MultivariateDecisionTree, weighted_class_count
import fireducks.pandas as pd
from multiprocessing import Pool
from functools import partial
from similarity_test_train import calculate_silhoutte_score_train_test, precompute_correlation_matrix, calculate_silhouette_score
import re


class MORGOTH:

    def __init__(self, X_train: pd.DataFrame, y_train: np.array, criterion_class: str, criterion_reg: str, sample_names_train: np.array,
                 min_number_of_samples_per_leaf: int, number_of_trees_in_forest: int, number_of_features_per_split: Union[float, str], class_names: list,
                 output_format: str, threshold: float, time_file: str, sample_weights_included: str, random_state: int, max_depth: int,
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
            @param threshold: float indicating the threshold that was used to obtain the binary class labels, needed to calculate sample weights
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
            @param tree_weights: bool, True if SAURON-RF tree weights should be used, False otherwise
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
        self.feature_names = self.X_train.columns
        self.y_train = np.array(y_train)
        split = np.hsplit(self.y_train, 2)
        self.y_train_reg = split[0].flatten()
        self.y_train_class = split[1].flatten()
        self.criterion_class = criterion_class
        self.criterion_reg = criterion_reg
        self.sample_names_train = np.array(sample_names_train)
        self.min_number_of_samples_per_leaf = min_number_of_samples_per_leaf
        self.number_of_trees_in_forest = number_of_trees_in_forest
        self.number_of_features_per_split = number_of_features_per_split
        self.tree_weights = tree_weights
        self.distance_measure = distance_measure
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

        self.output_format = output_format
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
        self.train_sample_weights = []
        if self.sample_weights_included in ["simple", 'linear']:

            if not math.isnan(self.thresholds[0]):
                if self.output_format == 'multioutput':
                    self.train_sample_weights = np.array(self.calculate_weights(
                        self.thresholds, self.y_train_reg, self.sample_weights_included))
                elif self.output_format == 'regression':
                    self.train_sample_weights = np.array(self.calculate_weights(
                        self.thresholds, self.y_train, self.sample_weights_included))
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
        start_time = time.perf_counter()
        self.train_sample_weights = np.array(self.train_sample_weights)

        # draw bootstrap sample
        sample_list = self.X_train.index.to_list()
        # draw with replacement
        index_list_bootstrap = random_object.randint(
            0, len(sample_list), len(sample_list))

        X_bootstrap = self.X_train.iloc[index_list_bootstrap, :]
        X_bootstrap = X_bootstrap.reset_index(drop=True)

        y_bootstrap = self.y_train[index_list_bootstrap]
        sample_names_bootstrap = self.sample_names_train[index_list_bootstrap]
        assert (len(X_bootstrap.index) == len(y_bootstrap))
        # generate Tree with bootstrap sample
        new_tree = MultivariateDecisionTree(X_train=X_bootstrap, y_train=y_bootstrap, class_names=self.class_names, criterion_class=self.criterion_class, criterion_reg=self.criterion_reg, min_number_of_samples_per_leaf=self.min_number_of_samples_per_leaf, max_depth=self.max_depth,
                                            number_of_features_per_split=self.number_of_features_per_split, random_object=random_object, impact_classification=self.impact_classification, output_format=self.output_format, sample_weights=np.array(self.train_sample_weights)[
                                                index_list_bootstrap],
                                            sample_names_train=sample_names_bootstrap, distance_measure=self.distance_measure)

        new_tree.fit()
        end_time = time.perf_counter()
        return new_tree, end_time-start_time

    def grow_parallel_trees(self):
        '''
            grows self.number_of_trees_in_forest in the forest 
            the process is parallelized as good as possible, i.e., as many trees as cores are built simultaneously 
            and then are added to the tree list
        '''
        self.trees = []
        with Pool() as pool:
            seed_list = []
            for random_seed in self.random_state.randint(0, 2**32, size=self.number_of_trees_in_forest):
                # creates random seeds for all treees
                seed_list.append(np.random.RandomState(random_seed))
            # parallel built of all trees
            results = pool.map_async(self.grow_tree, seed_list)
            seconds = []
            for t in results.get():
                self.trees.append(t[0])
                seconds.append(t[1])
            print(f'average time for a tree: {np.mean(seconds)}')
        assert (len(self.trees) == self.number_of_trees_in_forest)
        self.trees = np.array(self.trees)

    def predict(self, X_test: pd.DataFrame, quantile: 'list[float]' = None) -> np.array:
        '''
            predicts for a given test set

            @param X_test: a pandas DataFrame with the test samples
            @param quantile: list of floats or none, if not none quantile prediction is performed
            @return: a list containing the predictions, s.t. it matches the output format
        '''
        # pool.map results are ordered
        start = time.perf_counter()
        if self.output_format == 'classification':
            with Pool() as pool:
                predict_output = pool.map(
                    partial(MultivariateDecisionTree.predict, samples=X_test), self.trees)
            tree_predictions_class = []
            for i, p in enumerate(predict_output):
                tree_predictions_class.append(p[0])
                self.trees[i] = p[1]

            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array(self.predict_classification(X_test=X_test, tree_predictions_class=np.array(tree_predictions_class)))

        elif self.output_format == 'regression':
            with Pool() as pool:
                predict_output = pool.map(
                    partial(MultivariateDecisionTree.predict, samples=X_test), self.trees)
            tree_predictions_reg = []
            for i, p in enumerate(predict_output):
                tree_predictions_reg.append(p[0])
                self.trees[i] = p[1]

            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array(self.predict_regression(X_test=X_test, tree_predictions_reg=np.array(tree_predictions_reg), quantile=quantile))

        elif self.output_format == 'multioutput':
            with Pool() as pool:
                predict_output = pool.map(
                    partial(MultivariateDecisionTree.predict, samples=X_test), self.trees)
            predictions = []
            for i, p in enumerate(predict_output):
                predictions.append(p[0])
                self.trees[i] = p[1]
            tree_predictions_reg = []
            tree_predictions_class = []
            for p in predictions:
                reg_tree_pred = []
                class_tree_pred = []
                for pred in p:
                    reg_tree_pred.append(pred[0])
                    class_tree_pred.append([pred[1]])
                tree_predictions_class.append(class_tree_pred)
                tree_predictions_reg.append(reg_tree_pred)
            forest_predictions_class = np.array(self.predict_classification(
                X_test=X_test, tree_predictions_class=np.array(tree_predictions_class)))
            forest_predictions_reg = np.array(self.predict_regression(X_test=X_test, tree_predictions_reg=np.array(
                tree_predictions_reg), quantile=quantile, class_predictions_forest=forest_predictions_class))
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
            with Pool() as pool:
                predict_output = pool.map(
                    partial(MultivariateDecisionTree.predict, samples=X_test), self.trees)
            tree_predictions_class = []
            for i, p in enumerate(predict_output):
                tree_predictions_class.append(p[0])
                self.trees[i] = p[1]
            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array(self.predict_proba_classification(X_test=X_test, tree_predictions_class=np.array(tree_predictions_class)))

        elif self.output_format == 'regression':
            return np.array(self.predict(X_test=X_test, quantile=quantile))

        elif self.output_format == 'multioutput':
            with Pool() as pool:
                predict_output = pool.map(
                    partial(MultivariateDecisionTree.predict, samples=X_test), self.trees)
            predictions = []
            for i, p in enumerate(predict_output):
                predictions.append(p[0])
                self.trees[i] = p[1]
            tree_predictions_reg = []
            tree_predictions_class = []
            for p in predictions:
                reg_tree_pred = []
                class_tree_pred = []
                for pred in p:
                    reg_tree_pred.append(pred[0])
                    class_tree_pred.append([pred[1]])
                tree_predictions_class.append(class_tree_pred)
                tree_predictions_reg.append(reg_tree_pred)
            forest_predictions_class = np.array(self.predict_proba_classification(
                X_test=X_test, tree_predictions_class=np.array(tree_predictions_class)))
            class_predictions_forest = np.array(self.predict_classification(
                X_test=X_test, tree_predictions_class=np.array(tree_predictions_class)))
            forest_predictions_reg = np.array(self.predict_regression(
                X_test=X_test, tree_predictions_reg=np.array(tree_predictions_reg), quantile=quantile, class_predictions_forest=class_predictions_forest))
            self.write_leaf_silhouette_scores_to_file(
                test_sample_names=X_test.index, X_test=X_test)
            return np.array([[forest_predictions_reg[i], forest_predictions_class[i]] for i in range(len(X_test.index))])

    def predict_proba_classification(self, X_test: pd.DataFrame, tree_predictions_class: np.array) -> np.array:
        '''
            used to get the prediction probabilites for the classification instead of the majority class

            @param X_test: a pandas DataFrame with the test samples
            @param tree_predictions_class: a list of list with all tree predictions for the test samples
            @return: a list of list containing the predictions, each class is positional encoded as in the given class names list
        '''
        start_time = time.perf_counter()
        tree_predictions_class = np.array(tree_predictions_class)
        forest_predictions_class = []
        for i in range(len(X_test.index.to_list())):
            sample_pred_probas = []
            for class_name in self.class_names:
                occurences = weighted_class_count(class_name=class_name, y_class=[
                    y[i] for y in tree_predictions_class])
                sample_pred_probas.append(
                    occurences/self.number_of_trees_in_forest)
            assert (np.sum(sample_pred_probas) == 1)
            forest_predictions_class.append(sample_pred_probas)
        end_time = time.perf_counter()
        return np.array(forest_predictions_class)

    def predict_classification(self, X_test: pd.DataFrame, tree_predictions_class: np.array) -> np.array:
        '''
            used to get the prediction for the classification

            @param X_test: a pandas DataFrame with the test samples
            @param tree_predictions_class: a list of list with all tree predictions for the test samples
            @return: a list containing the majority class for each test sample
        '''
        start_time = time.perf_counter()
        forest_predictions_class = []
        for i in range(len(X_test.index.to_list())):
            max_class = None
            max_occ = 0
            for class_name in self.class_names:
                occurences = weighted_class_count(class_name=class_name, y_class=[
                    y[i] for y in tree_predictions_class])
                if occurences > max_occ:
                    max_class = class_name
                    max_occ = occurences
            forest_predictions_class.append(max_class)
        end_time = time.perf_counter()
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
        start_time = time.perf_counter()
        self.calculate_tree_weights(X_test, class_predictions_forest)
        tree_predictions_reg = np.array(tree_predictions_reg)
        if quantile is None:
            forest_predictions_reg = []
            for i, sample in enumerate(X_test.index):
                tree_weight_array = self.tree_weights_dict[sample]
                assert (math.isclose(
                    np.sum(tree_weight_array), 1.0, abs_tol=10**-2))
                forest_pred = 0
                for j, y in enumerate(tree_predictions_reg):
                    forest_pred += tree_weight_array[j] * y[i]
                forest_predictions_reg.append(forest_pred)
            forest_predictions_reg = np.array(forest_predictions_reg)
        else:
            self.calculate_weight_efficiently(
                X_test=X_test, class_predictions_forest=class_predictions_forest)
            forest_predictions_reg = np.array(
                self.quantile_prediction(X_test=X_test, quantile=quantile))

        end_time = time.perf_counter()
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

    def calculate_tree_weights(self, X_test: pd.DataFrame, class_predictions_forest: np.array):
        '''
            calculates the SAURON-RF weights for each tree
        '''
        if not self.tree_weights:
            tree_weights_dict = {}
            for test_sample in X_test.index:
                tree_weights_dict[test_sample] = np.full(
                    len(self.trees), 1/len(self.trees))
            self.tree_weights_dict = tree_weights_dict
            return
        X_test['class_predictions'] = class_predictions_forest
        tree_weights_dict = {}
        for test_sample in X_test.index:
            tree_weights_dict[test_sample] = np.full(
                len(self.trees), 1/len(self.trees))
        for test_sample in X_test.index:
            prediction_test_sample = X_test.loc[test_sample,
                                                'class_predictions']
            if self.majority_class == prediction_test_sample:
                continue
            for i, tree in enumerate(self.trees):
                for leaf in tree.leaves:
                    if test_sample in leaf.sample_ids_test:
                        if leaf.majority_class == prediction_test_sample:
                            tree_weights_dict[test_sample][i] = 1
                        else:
                            tree_weights_dict[test_sample][i] = 0
                        break
            sum_trees = np.sum(tree_weights_dict[test_sample])
            for i in range(len(self.trees)):
                tree_weights_dict[test_sample][i] /= sum_trees
        self.tree_weights_dict = tree_weights_dict

    def calculate_weight_efficiently(self, X_test: pd.DataFrame, class_predictions_forest: np.array) -> None:
        start_time = time.perf_counter()
        self.test_samples_to_train_sample_weight = {}

        for sample in X_test.index:
            self.test_samples_to_train_sample_weight[sample] = {}
            for train_sample in self.sample_names_train:
                self.test_samples_to_train_sample_weight[sample][train_sample] = 0

        for tree_id, tree in enumerate(self.trees):

            for leaf in tree.leaves:
                for sample in leaf.sample_ids_test:
                    leaf_sample = np.array(leaf.sample_names_train)
                    for i, train_sample in enumerate(leaf_sample):
                        self.test_samples_to_train_sample_weight[sample][train_sample] += self.tree_weights_dict[sample][tree_id] * \
                            leaf.normalized_sample_weights[i]
        end_time = time.perf_counter()
        for weight_dict in self.test_samples_to_train_sample_weight.values():
            assert math.isclose(
                np.sum(list(weight_dict.values())), 1.0, abs_tol=10**-2)

    def quantile_prediction(self, X_test: pd.DataFrame, quantile: 'list[float]'):
        '''
            calculates the quantile prediction as described by Meinshausen et al

            @param X_test: a pandas DataFrame with all test samples
            @param quantile: a list of quantiles 
            @return: a list of lists in case of quantile prediction where the position matches 
                the position of the respective alpha in the given quantile list
        '''
        quantile_prediction = []
        # for sample_id in range(len(X_test.index)):
        for sample_id in X_test.index:
            train_samples_to_weight = self.test_samples_to_train_sample_weight[sample_id]
            quantile_prediction_sample = [None for _ in range(len(quantile))]
            if self.output_format == 'multioutput':
                y_train_values_copy = enumerate(
                    list(self.y_train_reg))
            else:
                y_train_values_copy = enumerate(
                    list(self.y_train))
            y_train_values_sorted = sorted(
                y_train_values_copy, key=operator.itemgetter(1))

            sum_weight = 0.0
            for index in range(0, len(self.y_train)):
                sample_idx = y_train_values_sorted[index][0]
                current_y_value = y_train_values_sorted[index][1]
                current_sample_name = self.sample_names_train[sample_idx]

                # current_weight = train_samples_to_weight[sample_idx]
                current_weight = train_samples_to_weight[current_sample_name]
                sum_weight = sum_weight + current_weight
                continue_loop = False
                for i, q in enumerate(quantile):
                    if sum_weight >= q:  # infimum y that fulfills the condition that the sum of the weights are larger than the wanted quantile
                        if quantile_prediction_sample[i] is None:
                            quantile_prediction_sample[i] = current_y_value
                    if quantile_prediction_sample[i] is None:
                        continue_loop = True
                if not continue_loop:
                    break

            for i, p in enumerate(quantile_prediction_sample):
                if p is None:
                    print(
                        f'warning weights did not sum up to 1 or quantile > 1? {sum_weight}, {quantile}')
                    quantile_prediction_sample[i] = y_train_values_sorted[len(
                        self.y_train)-1][1]
            quantile_prediction.append(quantile_prediction_sample)
        assert (len(quantile_prediction) == len(X_test.index))
        return quantile_prediction

    def write_leaf_assignment_to_file(self):
        '''
            writes the leaf assignment of the training samples to the specified file
        '''
        with open(self.leaf_assignment_file_train, "w", encoding="utf-8") as leaf_assignment_output:

            unique_names = np.unique(np.array(self.sample_names_train))
            leaf_assignment_output.write(
                "Tree" + "\t" + "\t".join([str(x) for x in unique_names]) + "\n")

            for i, tree in enumerate(self.trees):

                current_row = [f'{i}']
                for sample_name in unique_names:
                    found_sample = False
                    current_row_string = str(float("NaN"))
                    for j, leaf in enumerate(tree.leaves):

                        real_current_sample_names = leaf.sample_names_train

                        if sample_name in real_current_sample_names:
                            if found_sample == True:
                                print(
                                    "Found a sample twice in different leaves of the same tree")
                            else:
                                found_sample = True
                                current_row_string = f'{j}'

                    current_row.append(current_row_string)

                leaf_assignment_output.write("\t".join(current_row) + "\n")

    def print_feature_importance_to_file(self):
        self.feature_imp_fit_model = {}

        with open(self.feature_imp_output_file, "w", encoding="utf-8") as feature_imp_output:

            for current_feature_name in self.feature_names:
                self.feature_imp_fit_model[current_feature_name] = 0
                for tree in self.trees:
                    self.feature_imp_fit_model[current_feature_name] += 1 / \
                        self.number_of_trees_in_forest * \
                        tree.feature_importances[current_feature_name]

                feature_imp_output.write(
                    str(current_feature_name) + "\t" + str(self.feature_imp_fit_model[current_feature_name]) + "\n")
        assert (math.isclose(np.sum(list(self.feature_imp_fit_model.values())),
                             1.0, abs_tol=10**-2))

    def write_train_sample_silhouette_scores_to_file(self):
        '''
            writes the silhouette score of the training samples to the specified file
        '''
        if self.distance_measure == '':
            print('no distance measure given')
            return
        if self.distance_measure == 'pearson' or self.distance_measure == 'spearman':
            distance = False
        else:
            distance = True
        normalized_X_train = copy.deepcopy(self.X_train)
        normalized_X_train.index = self.sample_names_train
        # min max normalization
        for column in normalized_X_train.columns:
            min_c = normalized_X_train[column].min()
            max_c = normalized_X_train[column].max()
            normalized_X_train[column] = (
                normalized_X_train[column] - min_c) / (max_c - min_c)

        precomputed_X_train = precompute_correlation_matrix(
            normalized_train_samples=normalized_X_train, distance_measure=self.distance_measure)

        with open(self.silhouette_score_train_file, "w", encoding="utf-8") as silhouette_output:
            silhouette_output.write(
                f'cell_line\tsilhouette_score_no_duplicates\tsilhouette_score_with_duplicates\n')
        for i, sample_name in enumerate(self.sample_names_train):

            x_sample_train = None
            for tree in self.trees:
                if x_sample_train is None:
                    x_sample_train = tree.find_train_sample_leaf_friends(
                        sample_name, self.X_train.iloc[i, :])
                else:
                    x_sample_train = np.concatenate(
                        [x_sample_train, tree.find_train_sample_leaf_friends(sample_name, self.X_train.loc[i, :])])

            x_sample_train_no_duplicates = np.unique(x_sample_train)
            score_dup = calculate_silhouette_score(
                distance_matrix_cluster=precomputed_X_train.loc[x_sample_train, x_sample_train], distance_matrix_sample_to_cluster=precomputed_X_train.loc[sample_name, x_sample_train], distance=distance)
            score_no_dup = calculate_silhouette_score(distance_matrix_cluster=precomputed_X_train.loc[x_sample_train_no_duplicates, x_sample_train_no_duplicates],
                                                      distance_matrix_sample_to_cluster=precomputed_X_train.loc[sample_name, x_sample_train_no_duplicates], distance=distance)
            with open(self.silhouette_score_train_file, "a", encoding="utf-8") as silhouette_output:
                silhouette_output.write(
                    f'{sample_name}\t{score_no_dup}\t{score_dup}\n')

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

        # min max normalization
        normalized_Xtrain = copy.deepcopy(self.original_Xtrain)
        normalized_Xtrain.index = self.sample_names_train

        for column in normalized_Xtrain.columns:
            min_c = normalized_Xtrain[column].min()
            max_c = normalized_Xtrain[column].max()
            normalized_Xtrain[column] = (
                normalized_Xtrain[column] - min_c) / (max_c - min_c)
            X_test[column] = (X_test[column] - min_c) / (max_c - min_c)

        precomputed_pairwise_train_similarities = precompute_correlation_matrix(
            normalized_train_samples=normalized_Xtrain, distance_measure=self.distance_measure)

        self.write_train_sample_silhouette_scores_to_file()
        with open(self.silhouette_score_file, "w", encoding="utf-8") as silhouette_output:
            with open(self.cluster_assignment_file, 'w', encoding='utf-8') as cluster_output:
                cluster_output.write('test_sample\ttraining_sample_list\n')
                unique_names = np.unique(np.array(test_sample_names))
                silhouette_output.write(
                    f'cell_line\tsilhouette_score_no_duplicates\tsilhouette_score_with_duplicates\n')
                for sample_name in unique_names:
                    x_sample_train = None
                    x_sample_train_names = None
                    for tree in self.trees:
                        if x_sample_train is None:
                            x_sample_train = np.array(
                                tree.train_samples_names_in_leaf[sample_name])
                            x_sample_train_names = np.unique(x_sample_train)
                        else:
                            new_sample_names = np.array(
                                tree.train_samples_names_in_leaf[sample_name])
                            x_sample_train = np.concatenate(
                                [x_sample_train, new_sample_names])
                            x_sample_train_names = np.concatenate(
                                [x_sample_train_names, np.unique(new_sample_names)])

                    x_sample_train_no_duplicates = precomputed_pairwise_train_similarities.loc[
                        x_sample_train_names, x_sample_train_names]
                    x_sample_train_with_duplicates = precomputed_pairwise_train_similarities.loc[
                        x_sample_train, x_sample_train]

                    score_dup = calculate_silhoutte_score_train_test(train_samples=normalized_Xtrain.loc[x_sample_train, :].transpose(
                    ), precomputed_similarities=x_sample_train_with_duplicates, test_sample=np.array(X_test.loc[sample_name, :].values), distance_measure=self.distance_measure, precomputed=True)
                    score_no_dup = calculate_silhoutte_score_train_test(train_samples=normalized_Xtrain.loc[x_sample_train_names, :].transpose(
                    ), precomputed_similarities=x_sample_train_no_duplicates, test_sample=np.array(X_test.loc[sample_name, :].values), distance_measure=self.distance_measure, precomputed=True)
                    silhouette_output.write(
                        f'{sample_name}\t{score_no_dup}\t{score_dup}\n')
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
                        f'{feature.replace("-", "")} [label = "{feature}, avg rank = {round(average_rank_features[feature],2)}", style = filled, color = {color_dict[feature]}]\n')
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
                    f'{feature.replace("-", "")} [label = "{feature}, avg rank = {round(average_forest_feature_score_dict[feature],2)}", style = filled, color = {color_dict[feature]}]\n')
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
                    f'{feature.replace("-", "")} [label = "{feature}, avg rank = {round(average_forest_feature_score_dict[feature],2)}", style = filled, color = {color_dict[feature]}]\n')
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
