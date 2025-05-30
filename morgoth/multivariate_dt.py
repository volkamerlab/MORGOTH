from typing import Union
import numpy as np
import math
import sys
if sys.platform.startswith('linux'):
    import fireducks.pandas as pd
else:
    import pandas as pd


def weighted_class_count(class_name: Union[int, str], y_class: list, sample_weights: np.array = None) -> float:
    '''
        calculates the weighted number of samples for a given class

        @param class_name: the name of the class of interest may be a string or an integer
        @param y_class: the vector with the true labels 
        @param sample_weights: the weight vector for the samples, should have same length as y_class

        @return: the weighted number of samples for the given class. Note that this might not be an integer due to the sample weights
    '''
    if sample_weights is None:
        sample_weights = np.ones(len(y_class))
    num_occurences = 0
    for i, y in enumerate(y_class):
        if y == class_name:
            num_occurences += sample_weights[i]
    return num_occurences


def compute_pk(current_class: Union[int, str], y_class: list, sample_weights: np.array) -> float:
    '''
        computes the proportion of occurences for a given class

        @param class_name: the name of the class of interest may be a string or an integer
        @param y_class: the vector with the true labels 
        @param sample_weights: the weight vector for the samples, should have same length as y_class

        @return: the fraction of samples in this class compared to all other classes
    '''
    if sample_weights is None:
        num_occurences_class = y_class.count(current_class)
        pk = num_occurences_class/len(y_class)
    else:
        num_occurences_class = weighted_class_count(
            class_name=current_class, y_class=y_class, sample_weights=sample_weights)
        pk = num_occurences_class/np.sum(sample_weights)

    return pk


def hellinger_distance(class_names: np.array, y_class: list, sample_weights: np.array) -> float:
    '''
        computes the hellinger distance

        @param class_names: array containing all class names
        @param y_class: the vector with the true labels 
        @param sample_weights: the weight vector for the samples, should have same length as y_class

        @return the hellinger distance
    '''
    num_classes = len(class_names)
    hellinger = 0
    for class_name in class_names:
        pk = compute_pk(current_class=class_name,
                        y_class=y_class, sample_weights=sample_weights)
        sqrt_pk = math.sqrt(pk)
        sqrt_1_minus_pk = math.sqrt(1-pk)
        hellinger += math.pow((sqrt_pk-sqrt_1_minus_pk), 2)
    hellinger /= num_classes
    return 1-hellinger


def gini_impurity(class_names: np.array, y_class: list, sample_weights: np.array) -> float:
    '''
        computes the gini-impurity

        @param class_names: array containing all class names
        @param y_class: the vector with the true labels 
        @param sample_weights: the weight vector for the samples, should have same length as y_class

        @return the gini-impurity
    '''
    num_classes = len(class_names)
    gini = 0
    for i in range(num_classes):
        pk = compute_pk(
            current_class=class_names[i], y_class=y_class, sample_weights=sample_weights)
        gini += pk*(1-pk)
    return gini


def entropy(class_names: np.array, y_class: list, sample_weights: np.array) -> float:
    '''
        computes the entropy 

        @param class_names: array containing all class names
        @param y_class: the vector with the true labels 
        @param sample_weights: the weight vector for the samples, should have same length as y_class

        @return the entropy
    '''
    num_classes = len(class_names)
    entropy = 0
    for i in range(num_classes):
        pk = compute_pk(
            current_class=class_names[i], y_class=y_class, sample_weights=sample_weights)
        if pk == 0:
            entropy += 0
        else:
            entropy += pk*math.log(pk)
    return -entropy


def RSS(y_reg: list, sample_weights: np.array) -> float:
    '''
        computes the running sum statistics (RSS)

        @param y_reg: vector with true contiunous labels 
        @param sample_weights: the weight vector for the samples, should have same length as y_reg

        @return the RSS
    '''
    rss = 0
    mean = np.mean(y_reg)
    if sample_weights is None:
        for i, y in enumerate(y_reg):
            rss += (y-mean)*(y-mean)
    else:
        for i, y in enumerate(y_reg):
            rss += (y-mean)*(y-mean) * sample_weights[i]
    return rss


def MSE(y_reg: list, sample_weights: np.array) -> float:
    '''
        calculates the mean squared error (MSE)

        @param y_reg: vector with true contiunous labels 
        @param sample_weights: the weight vector for the samples, should have same length as y_reg

        @return the MSE
    '''
    mse = 0
    mse = RSS(y_reg, sample_weights=sample_weights)
    mse = mse/np.sum(sample_weights)
    return mse


class Split():
    def __init__(self, feature_name: str, threshold: float):
        '''
            construcor of the Split class

            @param feature_name: a str representing the name of the feature we are considering at this split
            @param threshold: float representing the value of the feature used for splitting,
                we send all samples with a value for feature_name < threshold to the left and
                all samples with a value for feature_name >= threshold to the right
        '''
        self.feature_name = feature_name
        self.threshold = threshold
        self.score = 0
        self.feature_index = None

    def calculate_left_right(self, X: pd.DataFrame, epsilon_list: list = None) -> 'tuple[list]':
        '''
            calculates which samples go to the left and which to the right

            @param X: a pandas DataFrame containing the samples to consider
            @param epsilon_list: a list
            @return: a tuple of two lists, where the first one are the indices of the left side
                and the second one is the index list of the right side
        '''
        full_index = set(X.index)
        index_list_left = X.loc[X[self.feature_name]
                                < self.threshold, :].index
        index_list_right = list(full_index - set(index_list_left))
        if epsilon_list is None:
            return index_list_left, index_list_right
        else:
            if self.feature_index is None:
                features = X.columns.to_numpy()
                self.feature_index = np.where(features == self.feature_name)
            epsilon = epsilon_list[self.feature_index][0]
            index_list_adversarial = X.loc[(X[self.feature_name]
                                            >= self.threshold - epsilon) | (X[self.feature_name]
                                                                            <= self.threshold + epsilon), :].index
            return index_list_left, index_list_right, index_list_adversarial

    def calculate_child(self, sample: pd.DataFrame) -> str:
        '''
            tells us for a specific sample whether it belongs to the right or left child

            @param sample: a pandas DataFrame with the sample, column names should correspond to the feature names in the training data
            @return: a string indicating if the sample goes to the 'left' or 'right' child of the node
        '''
        left = not len(sample.loc[sample[self.feature_name]
                                  < self.threshold, :].index) == 0
        if left:
            return 'left'
        else:
            return 'right'


class BinaryTreeNode:

    def __init__(self, parent, level: int, is_leaf: bool, X_train: pd.DataFrame, y_train: np.array, sample_weights: np.array = None, sample_names_train: np.array = None):
        '''
            constructor of a tree node in a binary decision tree

            @param parent: instance of BinaryTreeNode, i.e., the parent of the acutal node or None if we are in root
            @param level: the level of the actual node, i.e., number of steps need from root to here
            @param is_leaf: bool that tells us whether the actual node is a leaf or not
            @param X_train: pd.DataFrame containing the features of the samples that reached this node during training
            @param y_train: numpy array containing the response values for the samples that reached this node during training
            @param sample_weights: corresponding weight vector for each sample
            @param sample_names_train: a list of sample names corresponding to X_train and y_train
        '''
        self.sample_ids_test = []
        self.parent = parent
        self.level = level
        self.is_leaf = is_leaf
        self.X_train = X_train
        self.y_train = y_train
        split = np.hsplit(self.y_train, 2)
        self.already_predicted = False
        self.y_train_reg = split[0].flatten()
        self.y_train_class = split[1].flatten()
        self.sample_names_train = sample_names_train
        self.sample_weights = sample_weights
        self.normalized_sample_weights = None
        self.information_gain_for_split_exists = None

    def find_train_sample_leaf(self, sample: pd.DataFrame,) -> np.array:
        '''
            returns the samples in the same leaf as a specific samples

            @param sample: a dataframe with the features to characterize the sample of interest
            @return a numpy array containing the sample names in the same leaf
        '''
        if self.is_leaf:
            return np.array(self.sample_names_train)
        else:
            left, right = self.split.calculate_left_right(X=sample)
            if len(left) == 0:
                return self.right.find_train_sample_leaf(sample)
            elif len(right) == 0:
                return self.left.find_train_sample_leaf(sample)

    def is_split_possible(self, min_number_of_samples_per_leaf: int):
        '''
            tells whether it is possible to further split this node for a given minimum number of samples per leaf

            @param min_number_of_samples_per_leaf: minimum number of samples that are allowed to be in a leaf
        '''
        if self.information_gain_for_split_exists is None:
            return (len(self.y_train) >= 2 * min_number_of_samples_per_leaf)
        else:
            # we already tried to split but the "best" split was None
            return self.information_gain_for_split_exists

    def split_node(self, features: list, class_criterion_function, regression_criterion_function, root_sample_weights: list, root_class_criterion: float, root_reg_criterion: float, impact_classification: float, class_names: list, min_number_of_samples_per_leaf: int) -> Split:
        '''
            splits the node if possible and if a split gives us better score for the objective function than the node already has

            @param features: a list of features that should be used to select the best split
            @param class_criterion_function: a function that calculates the impurity of a node for the classification task
            @param regression_criterion_function: a function that calculates the impurity of a node for the regression task
            @param root_class_criterion: score of the class_criterion_function applied to all training samples (used for normalization)
            @param root_reg_criterion: score of the class_criterion_function applied to all training samples (used for normalization)
            @param impact_classification: float in [0,1], defining how much impact the classification objective function should have.
                The impact of the regression function is 1-impact_classification.
            @param class_names: list of available classes
            @param min_number_of_samples_per_leaf: int defining minimal number of samples per leaf node
        '''
        self.class_names = class_names
        if not self.is_split_possible(min_number_of_samples_per_leaf):
            return None
        split = self.calculate_best_split(features=features, class_criterion_function=class_criterion_function, root_sample_weights=root_sample_weights,
                                          regression_criterion_function=regression_criterion_function, root_class_criterion=root_class_criterion, root_reg_criterion=root_reg_criterion, impact_classification=impact_classification, class_names=class_names, min_number_of_samples_per_leaf=min_number_of_samples_per_leaf)
        if split is None:
            self.information_gain_for_split_exists = False
            self.is_leaf = True
            return None
        else:
            self.split = split
            left, right = split.calculate_left_right(X=self.X_train)
            self.is_leaf = False
            self.left = BinaryTreeNode(parent=self, level=self.level + 1, is_leaf=True, X_train=self.X_train.iloc[left, :].reset_index(
                drop=True), y_train=self.y_train[left], sample_weights=self.sample_weights[left], sample_names_train=self.sample_names_train[left])
            self.right = BinaryTreeNode(parent=self, level=self.level + 1, is_leaf=True, X_train=self.X_train.iloc[right, :].reset_index(
                drop=True), y_train=self.y_train[right], sample_weights=self.sample_weights[right], sample_names_train=self.sample_names_train[right])
            self.X_train = None
        return self.split

    def calculate_normalized_sample_weights(self) -> None:
        '''
            calculates the sample weights normalized for the bootstrap samples assigned to this node (self)
        '''
        if self.normalized_sample_weights is None:
            sum_weights_leaf = np.sum(self.sample_weights)
            self.normalized_sample_weights = [
                weight/sum_weights_leaf for weight in self.sample_weights]

    def distribute_test_samples_to_leaves(self, samples: pd.DataFrame, sample_feature_path_dict: dict) -> None:
        '''
            distributes unseen (test) samples to the leaves and meanwhile tracks the trace of each sample

            @param samples: a dataframe containing the features for unseen (test) samples
            @param sample_feature_path_dict: a dict with the samples as keys and the trace through the tree as value. Is updated in this function
        '''
        if self.is_leaf:
            self.sample_ids_test = samples.index
        else:
            # self.sample_ids_test = samples.index
            for sample_name in samples.index:
                sample_feature_path_dict[sample_name].append(
                    (self.split.feature_name, self.level))
            left, right = self.split.calculate_left_right(X=samples)
            self.left.distribute_test_samples_to_leaves(
                samples=samples.loc[left, :], sample_feature_path_dict=sample_feature_path_dict)
            self.right.distribute_test_samples_to_leaves(
                samples=samples.loc[right, :], sample_feature_path_dict=sample_feature_path_dict)

    def predict(self, class_names) -> list:
        '''
            predicts for a specific sample,
            if we are in a leaf node: the mean of all continuous valued training samples in the leaf is returned for regression and the majority class for classification
            if not, we calculate whether the sample goes to the left or right child and return the prediction recursively

            @param sample: a pandas DataFrame with the sample, column names should correspond to the feature names in the training data
            @return: a list of length 2 containing at pos 0 the prediction for the regression and at pos 1 the prediction for the classification task
        '''
        if self.is_leaf:
            if not self.already_predicted:
                self.calculate_normalized_sample_weights()
                reg = np.sum([weight * self.y_train_reg[i]
                              for i, weight in enumerate(self.normalized_sample_weights)])
                majority_class_occurences = 0
                majority_class = None
                for class_name in class_names:
                    occ = weighted_class_count(
                        class_name=class_name, y_class=self.y_train_class, sample_weights=self.sample_weights)
                    if occ > majority_class_occurences:
                        majority_class_occurences = occ
                        majority_class = class_name
                self.average = reg
                self.majority_class = majority_class
                self.already_predicted = True

            return [self.average, self.majority_class]
        else:
            print('warning: predict called on no leaf node')

    def calculate_best_split(self, features: list, class_criterion_function, regression_criterion_function, root_class_criterion: float, root_reg_criterion: float, impact_classification: float, class_names: list, min_number_of_samples_per_leaf: int, root_sample_weights: list) -> Split:
        '''
            calculates the best split for this node (or none if we do not gain information by splitting this node up)

            @param features: a list of features that should be used to select the best split
            @param class_criterion_function: a function that calculates the impurity of a node for the classification task
            @param regression_criterion_function: a function that calculates the impurity of a node for the regression task
            @param root_class_criterion: score of the class_criterion_function applied to all training samples (used for normalization)
            @param root_reg_criterion: score of the class_criterion_function applied to all training samples (used for normalization)
            @param impact_classification: float in [0,1], defining how much impact the classification objective function should have.
                The impact of the regression function is 1-impact_classification.
            @param class_names: list of available classes
            @param min_number_of_samples_per_leaf: int defining minimal number of samples per leaf node
            @param root_sample_weights: the sample weight vector for the bootstrap samples assigned to the root node
            @return: the Split instance of the best possible split or None if no split is better than just not splitting
        '''

        sample_weights = self.sample_weights
        if sample_weights is None:
            print('sample weights are None?')
        if not root_class_criterion == 0:
            node_class_criterion = class_criterion_function(
                class_names=class_names, y_class=self.y_train_class, sample_weights=sample_weights)/root_class_criterion
        else:
            node_class_criterion = class_criterion_function(
                class_names=class_names, y_class=self.y_train_class, sample_weights=sample_weights)
        node_reg_criterion = regression_criterion_function(
            y_reg=self.y_train_reg, sample_weights=sample_weights)/root_reg_criterion

        node_score = impact_classification * node_class_criterion + \
            (1-impact_classification) * node_reg_criterion
        best_score = node_score
        best_split = None

        valid_split_found = False
        for feature in features:
            best_score, best_split, valid_split_found = self.calculate_best_split_for_feature(root_sample_weights=root_sample_weights, best_score=best_score, node_score=node_score, min_number_of_samples_per_leaf=min_number_of_samples_per_leaf, class_criterion_function=class_criterion_function,
                                                                                              impact_classification=impact_classification, regression_criterion_function=regression_criterion_function, root_class_criterion=root_class_criterion, root_reg_criterion=root_reg_criterion, feature_name=feature, best_split=best_split, valid_split_found=valid_split_found)

        if best_split is None:
            if not valid_split_found:
                pass
                # print('no valid split found')
        return best_split

    def calculate_best_split_for_feature(self, root_sample_weights: list, min_number_of_samples_per_leaf: int, class_criterion_function, regression_criterion_function, root_reg_criterion, root_class_criterion, impact_classification, feature_name, best_split: Split, best_score: float, valid_split_found: bool, node_score: float):
        '''
            identifies the best split for a given feature

            @param root_sample_weights: the sample weight vector for the bootstrap samples assigned to the root node
            @param min_number_of_samples_per_leaf: int defining minimal number of samples per leaf node
            @param class_criterion_function: a function that calculates the impurity of a node for the classification task
            @param regression_criterion_function: a function that calculates the impurity of a node for the regression task
            @param root_class_criterion: score of the class_criterion_function applied to all training samples (used for normalization)
            @param root_reg_criterion: score of the class_criterion_function applied to all training samples (used for normalization)
            @param impact_classification: float in [0,1], defining how much impact the classification objective function should have.
                The impact of the regression function is 1-impact_classification.
            @param feature_name: the name of the feature that we currently consider
            @param best_split: current best split
            @param best_score: score of the current best_split
            @param valid_split_found: True if a valid split was already found, False otherwise
            @param node_score: score at the current node    

            @return: the Split instance of the best split for the given feature_name
        '''
        thresholds = self.X_train.loc[:, feature_name]
        sorted_values = np.sort(np.unique(thresholds.values.tolist()))

        considered_values = sorted_values[min_number_of_samples_per_leaf-1:len(
            sorted_values)-(min_number_of_samples_per_leaf-1)]
        for i, val in enumerate(considered_values):
            if i+1 == len(considered_values):
                break
            # we can ignore all options where definetly less than the minimal number of samples would end up in the same leaf
            split = Split(feature_name=feature_name,
                          threshold=np.mean([val, considered_values[i+1]]))

            left, right = split.calculate_left_right(X=self.X_train)

            if (len(left) < min_number_of_samples_per_leaf) or (len(right) < min_number_of_samples_per_leaf):
                continue
            valid_split_found = True
            left_weight = np.sum(
                self.sample_weights[left])/np.sum(self.sample_weights)
            right_weight = np.sum(
                self.sample_weights[right])/np.sum(self.sample_weights)

            assert math.isclose(left_weight + right_weight,
                                1.0, abs_tol=10**-2)

            left_class = class_criterion_function(
                class_names=self.class_names, y_class=self.y_train_class[left], sample_weights=self.sample_weights[left])

            right_class = class_criterion_function(
                class_names=self.class_names, y_class=self.y_train_class[right], sample_weights=self.sample_weights[right])
            if root_class_criterion == 0:
                class_intermediate = 0
            else:
                class_intermediate = (
                    left_weight * left_class+right_weight * right_class) / root_class_criterion
            left_reg = regression_criterion_function(
                y_reg=self.y_train_reg[left], sample_weights=self.sample_weights[left])
            right_reg = regression_criterion_function(
                y_reg=self.y_train_reg[right], sample_weights=self.sample_weights[right])

            reg_intermediate = (left_weight * left_reg +
                                right_weight * right_reg) / root_reg_criterion
            act_score = impact_classification * class_intermediate + \
                (1-impact_classification) * reg_intermediate

            split.score = (node_score-act_score) * \
                np.sum(self.sample_weights)/np.sum(root_sample_weights)

            if act_score <= best_score:
                best_score = act_score
                best_split = split

        return best_score, best_split, valid_split_found


class MultivariateDecisionTree:

    def __init__(self, X_train: pd.DataFrame, y_train: np.array, class_names: np.array = None, criterion_class: str = 'gini', criterion_reg: str = 'mse', min_number_of_samples_per_leaf: int = 1, max_depth: int = 20, number_of_features_per_split: Union[str, float] = 'sqrt', random_object: np.random.RandomState = np.random.RandomState(42), impact_classification: float = 0.5, output_format: str = 'multioutput', sample_weights: np.array = None, sample_names_train: np.array = None, distance_measure: str = 'pearson') -> None:
        '''
            constructor of a multivariate decision tree

            @param X_train: a pandas DataFrame with columns as features and rows as samples
            @param y_train: a numpy array with either single outputs or lists of outputs as specified by output_format
            @param class_names: a list containing the names of all available classes or None, only needed if output_format is set to
                'classification', or 'multioutput', default = None
            @param criterion_class: str defining the splitting criterion that should be used to optimize the split for the discrete response,
                only needed if output_format is set to 'classification', or 'multioutput', available criteria: 'gini', 'hellinger', and 'entropy', default = 'gini'
            @param criterion_reg: str defining the splitting criterion that should be used to optimize the split for the continuous response,
                only needed if output_format is set to 'regression', or 'multioutput', available criteria: 'mse', default = 'mse'
            @param min_number_of_samples_per_leaf: int specifying the minimal number of samples per leaf, default = 1
            @param max_depth: int specifying the maximal depth of the tree, default = 20
            @param number_of_features_per_split: float or string, defining the number of features that should be used per split.
                If a float is given, it should be in [0,1], then number_of_features_per_split*number_of_features are drawn randomly.
                If a string is given, we expect it to be in wither 'sqrt' or 'log2'.
                If 'sqrt', then number_of_features_per_split=sqrt(number_of_features).
                If 'log2', then number_of_features_per_split=log2(number_of_features).
                Default = 'sqrt'
            @param random_object: instance of class Random used to draw random features
            @param impact_classification: float in [0,1], defining how much impact the classification objective function should have.
                Notably, this parameter is called lambda in our manuscript. Due to the fact "lambda" being a key word in Python, we call it impact_classification in the code.
                The impact of the regression function is 1-impact_classification. Default = 0.5. 
            @param output_format: string defining the output format, i.e., if we have only classification, or regression data, respectively,
                or a multioutput given by a list of lists with length 2 where position 0 is the continuous response and pos 1 the class
                allowed values: 'regression', 'classification', or 'multioutput', default = 'multioutput'
            @param sample_weights: a numpy array containing weights for each sample
            @param sample_names_train: a list of sample names corresponding to X_train and y_train
        '''

        self.X_train = X_train
        self.y_train = y_train
        self.sample_names_train = np.array(sample_names_train)
        self.train_samples_in_same_leaf = {}
        self.feature_importances = {}
        self.distance_measure = distance_measure

        # check if parameters have supported values
        if not criterion_class in {'gini', 'entropy', 'hellinger', None}:
            raise (ValueError(
                f'Unsupported classifiation criterion {criterion_class}'))
        if not criterion_reg in {'mse', None}:
            raise (ValueError(
                f'Unsupported regression criterion {criterion_reg}'))
        if min_number_of_samples_per_leaf < 0:
            raise (ValueError(
                f'Unsupported number of samples per leaf {min_number_of_samples_per_leaf}'))
        if max_depth <= 0:
            raise (ValueError(f'Unsupported maximum depth {max_depth}'))
        if type(number_of_features_per_split) == str and not number_of_features_per_split in {'sqrt', 'log2'}:
            raise (ValueError(
                f'Unsupported option for number of features per split {number_of_features_per_split}'))
        if type(number_of_features_per_split) == float and not (number_of_features_per_split >= 0 and number_of_features_per_split <= 1):
            raise (ValueError(
                f'Unsupported value for number of features per split {number_of_features_per_split}'))
        if impact_classification > 1 or impact_classification < 0:
            raise (ValueError(
                f'Unsupported value for impact classification {impact_classification}'))
        if not output_format in {'multioutput', 'classification', 'regression'}:
            raise (ValueError(
                f'Unsupported option for output format {output_format}'))

        self.criterion_class = criterion_class
        self.criterion_reg = criterion_reg
        self.min_number_of_samples_per_leaf = min_number_of_samples_per_leaf
        self.max_depth = max_depth
        if type(number_of_features_per_split) == str:
            if number_of_features_per_split == 'sqrt':
                self.number_of_features_per_split = int(
                    math.sqrt(len(self.X_train.columns)))
            elif number_of_features_per_split == 'log2':
                self.number_of_features_per_split = int(
                    math.log2(len(self.X_train.columns)))
            else:
                raise (ValueError(
                    f'Unsupported string for number_of_features_per_split {number_of_features_per_split}.'))
        else:
            self.number_of_features_per_split = int(
                number_of_features_per_split*len(self.X_train.columns))
        self.random_object = random_object

        self.impact_classification = impact_classification
        self.output_format = output_format
        self.sample_weights = sample_weights
        self.root = BinaryTreeNode(
            None, level=0, is_leaf=True, X_train=self.X_train, y_train=self.y_train, sample_weights=self.sample_weights, sample_names_train=self.sample_names_train)

        if not self.output_format == 'multioutput':
            if self.output_format == 'classification':

                if criterion_class == 'gini':
                    self.class_criterion_function = gini_impurity
                elif criterion_class == 'entropy':
                    self.class_criterion_function = entropy
                elif criterion_class == 'hellinger':
                    self.class_criterion_function = hellinger_distance

                self.root_class_criterion = self.class_criterion_function(
                    class_names, list(self.y_train), sample_weights=self.sample_weights)
                self.impact_classification = 1
                self.class_names = class_names
                # dummy regression data
                self.regression_criterion_function = MSE
                self.y_train = np.array([[0, y] for y in self.y_train])
                self.root_reg_criterion = 1
            elif self.output_format == 'regression':
                if criterion_reg == 'mse':
                    self.regression_criterion_function = MSE

                self.root_reg_criterion = self.regression_criterion_function(
                    list(self.y_train), sample_weights=self.sample_weights)
                self.impact_classification = 0

                # dummy classification data
                self.class_criterion_function = gini_impurity
                self.y_train = np.array([[y, 0] for y in self.y_train])
                self.class_names = [0]
                self.root_class_criterion = 1
        else:
            if criterion_class == 'gini':
                self.class_criterion_function = gini_impurity
            elif criterion_class == 'entropy':
                self.class_criterion_function = entropy
            elif criterion_class == 'hellinger':
                self.class_criterion_function = hellinger_distance

            self.root_class_criterion = self.class_criterion_function(
                class_names, self.root.y_train_class, sample_weights=self.sample_weights)
            self.class_names = class_names

            if criterion_reg == 'mse':
                self.regression_criterion_function = MSE

            self.root_reg_criterion = self.regression_criterion_function(
                self.root.y_train_reg, self.sample_weights)

        self.depth = 1
        self.leaves = []
        self.leaves.append(self.root)

    def find_train_sample_leaf_friends(self, train_sample_id: int, x_train: pd.Series):
        '''
            finds the samples in the same leaf as a particular training sample

            @param train_sample_id: the id of the training sample of interest
            @param x_train: the training set 
            @return a numpy array with the training samples in the same leaf of the decision tree
        '''

        if train_sample_id in np.unique(self.root.sample_names_train):
            for leaf in self.leaves:
                leaf_train_sample_names = leaf.sample_names_train
                if train_sample_id in leaf_train_sample_names:
                    indices = np.nonzero(
                        leaf_train_sample_names == train_sample_id)
                    indices = np.array(indices).flatten()
                    leaf_x_train = np.array(leaf.sample_names_train)

                    return leaf_x_train
        else:
            return self.root.find_train_sample_leaf(x_train.to_frame().transpose())

    def fit(self):
        '''
            builds a BinaryDecisionTree for the given training data
        '''
        number_of_features = self.number_of_features_per_split
        feature_list = np.array(self.X_train.columns)
        for feature in feature_list:
            self.feature_importances[feature] = 0

        while self.is_split_possible():

            node = self.find_next_leaf()

            index_list_features = self.random_object.choice(
                len(feature_list), size=number_of_features, replace=False)
            features = np.array(feature_list)[index_list_features]

            split = node.split_node(features=features, root_sample_weights=self.root.sample_weights, class_criterion_function=self.class_criterion_function, regression_criterion_function=self.regression_criterion_function,
                                    root_class_criterion=self.root_class_criterion, root_reg_criterion=self.root_reg_criterion, impact_classification=self.impact_classification, class_names=self.class_names, min_number_of_samples_per_leaf=self.min_number_of_samples_per_leaf)

            if not split is None:
                self.feature_importances[split.feature_name] += split.score
                self.leaves.remove(node)
                self.leaves.append(node.left)
                self.leaves.append(node.right)
                self.update_depth()

        normalizer = np.sum(list(self.feature_importances.values()))

        if not normalizer == 0:
            for f in np.array(list(self.feature_importances.keys())):
                self.feature_importances[f] /= normalizer
            assert (math.isclose(np.sum(list(self.feature_importances.values())),
                                 1.0, abs_tol=10**-2))
        self.leaves = np.array(self.leaves)

    def predict(self, samples: pd.DataFrame) -> list:
        '''
            calculates a prediction for given samples

            @param samples: a pandas DataFrame containing all samples
            @return: a list containing the predictions depending on whether the output should be multivariate or not
        '''
        predictions = np.full(len(samples.index), None)
        self.train_samples_in_same_leaf = {}
        self.train_samples_names_in_leaf = {}
        self.sample_feature_path = {}
        for sample in samples.index:
            self.sample_feature_path[sample] = []
        self.root.distribute_test_samples_to_leaves(
            samples=samples, sample_feature_path_dict=self.sample_feature_path)
        index_list = samples.index.to_list()
        for leaf in self.leaves:
            leaf_test_samples = leaf.sample_ids_test
            for test_sample in leaf_test_samples:
                i = index_list.index(test_sample)
                predictions[i] = leaf.predict(class_names=self.class_names)
                # X_train = copy.deepcopy(leaf.X_train)
                self.train_samples_in_same_leaf[test_sample] = leaf.X_train
                self.train_samples_names_in_leaf[test_sample] = leaf.sample_names_train
        return predictions, self

    def update_depth(self) -> None:
        '''
            updates the depth of the tree
        '''
        for leaf in np.array(self.leaves):
            if leaf.level > self.depth:
                self.depth = leaf.level

    def find_next_leaf(self) -> BinaryTreeNode:
        '''
            returns the next leaf for which a split is possible
        '''
        for leaf in np.array(self.leaves):
            if leaf.is_split_possible(self.min_number_of_samples_per_leaf):
                return leaf

    def is_split_possible(self) -> bool:
        '''
            checks whether a split is possible for any leaf in the tree

            @return bool: True if there is a leaf, which can be split, False otherwise
        '''
        for leaf in np.array(self.leaves):
            if leaf.is_split_possible(self.min_number_of_samples_per_leaf):
                return self.depth < self.max_depth
        return False
