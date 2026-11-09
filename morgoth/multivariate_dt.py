from collections.abc import Mapping, MutableMapping
from typing import Union
import numpy as np
import math
import os
import sys
if sys.platform.startswith('linux'):
    import fireducks.pandas as pd
else:
    import pandas as pd


######################## memory budget of the vectorized split search ########################
# The split search processes a block of (#samples in node x #features) entries at once. The block size is
# derived from the memory that is available on the machine (shared by all worker processes) and can be
# overridden with the environment variable MORGOTH_SPLIT_BLOCK_ELEMENTS.

# fraction of the currently available memory that the split search of all worker processes together may use
_SPLIT_MEMORY_FRACTION = 0.25
# smaller blocks lose the benefit of vectorization, larger blocks do not speed up the computation anymore
_MIN_SPLIT_BLOCK_ELEMENTS = 2 ** 14
_MAX_SPLIT_BLOCK_ELEMENTS = 2 ** 24
# (cached) available memory in bytes of this process, determined on first use
_available_memory_cache = []


def _available_memory_bytes() -> Union[int, None]:
    '''
        @return: the memory (in bytes) that is currently available for new processes or None if it cannot be determined
    '''
    try:
        # linux: MemAvailable includes reclaimable caches (in contrast to the free memory)
        with open('/proc/meminfo') as meminfo:
            for line in meminfo:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
    except (AttributeError, ValueError, OSError):
        return None


def _number_of_parallel_processes() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _split_block_elements(number_of_classes: int) -> int:
    '''
        maximal number of matrix entries (#samples in node * #features) that are processed at once during the
        vectorized split search, adapted to the available memory of the machine

        @param number_of_classes: number of classes (each class needs additional temporary arrays)
    '''
    override = os.environ.get('MORGOTH_SPLIT_BLOCK_ELEMENTS')
    if override:
        return max(1, int(override))
    if not _available_memory_cache:
        _available_memory_cache.append(_available_memory_bytes())
    available = _available_memory_cache[0]
    if available is None:
        return 2 ** 20
    # upper bound for the temporary float64/int64 arrays that exist simultaneously per entry
    bytes_per_element = 8 * (24 + 4 * number_of_classes)
    budget = available * _SPLIT_MEMORY_FRACTION / _number_of_parallel_processes()
    return int(min(_MAX_SPLIT_BLOCK_ELEMENTS, max(_MIN_SPLIT_BLOCK_ELEMENTS, budget // bytes_per_element)))


def _sequential_sum(values: np.ndarray):
    '''
        sums up the values from left to right.
        np.cumsum accumulates strictly sequentially (in contrast to np.sum, which uses pairwise summation),
        hence the result is bit-identical to summing up the values in a python for loop
    '''
    if len(values) == 0:
        return 0
    return np.cumsum(values)[-1]


def _class_mask(y_class, class_name):
    '''
        returns a boolean numpy mask (y_class == class_name) or None if the input cannot be compared element wise
    '''
    try:
        y = np.asarray(y_class)
        mask = y == class_name
    except Exception:
        return None
    if isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (len(y_class),):
        return mask
    return None


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
    mask = _class_mask(y_class, class_name)
    if mask is None:
        num_occurences = 0
        for i, y in enumerate(y_class):
            if y == class_name:
                num_occurences += sample_weights[i]
        return num_occurences
    return _sequential_sum(np.asarray(sample_weights)[mask])


def compute_pk(current_class: Union[int, str], y_class: list, sample_weights: np.array) -> float:
    '''
        computes the proportion of occurences for a given class

        @param class_name: the name of the class of interest may be a string or an integer
        @param y_class: the vector with the true labels
        @param sample_weights: the weight vector for the samples, should have same length as y_class

        @return: the fraction of samples in this class compared to all other classes
    '''
    if sample_weights is None:
        num_occurences_class = list(y_class).count(current_class)
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
    mean = np.mean(y_reg)
    y = np.asarray(y_reg)
    if y.ndim != 1 or y.dtype == object:
        rss = 0
        if sample_weights is None:
            for i, y in enumerate(y_reg):
                rss += (y-mean)*(y-mean)
        else:
            for i, y in enumerate(y_reg):
                rss += (y-mean)*(y-mean) * sample_weights[i]
        return rss
    diff = y - mean
    if sample_weights is None:
        terms = diff * diff
    else:
        terms = diff * diff * np.asarray(sample_weights)
    # sequential summation -> identical to the original python loop
    return _sequential_sum(terms)


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


# criteria for which a vectorized evaluation of all thresholds of a feature exists
_VECTORIZED_CLASS_CRITERIA = {gini_impurity: 'gini',
                              entropy: 'entropy', hellinger_distance: 'hellinger'}
_VECTORIZED_REG_CRITERIA = {MSE: 'mse'}


def _class_criterion_from_proportions(name: str, proportions: list) -> np.ndarray:
    '''
        vectorized version of gini_impurity, entropy and hellinger_distance

        @param name: name of the criterion
        @param proportions: list (one entry per class) of arrays with the class proportions pk
    '''
    result = np.zeros_like(proportions[0])
    if name == 'gini':
        for pk in proportions:
            result += pk * (1 - pk)
        return result
    if name == 'entropy':
        for pk in proportions:
            with np.errstate(divide='ignore', invalid='ignore'):
                result += np.where(pk > 0, pk * np.log(np.where(pk > 0, pk, 1)), 0)
        return -result
    # hellinger
    for pk in proportions:
        with np.errstate(invalid='ignore'):
            result += (np.sqrt(pk) - np.sqrt(np.maximum(1 - pk, 0))) ** 2
    return 1 - result / len(proportions)


class _TreeData:
    '''
        holds the bootstrap sample of a single tree. All nodes of the tree share this object and only store
        integer positions into these arrays instead of copies of the feature matrix.
        Thus, the memory needed for a fitted tree is O(#samples) instead of O(#samples * #features).
    '''

    def __init__(self, XT: np.ndarray, feature_names: np.ndarray, col_of: dict, boot: np.ndarray, y: np.ndarray,
                 weights: np.ndarray, names: np.ndarray, output_format: str, frame=None):
        # transposed feature matrix (features x samples, C-contiguous) of ALL samples; only needed while fitting
        self.XT = XT
        # pandas DataFrame with the same rows as XT (used to lazily create the X_train data frames)
        self.frame = frame
        self.feature_names = feature_names
        self.col_of = col_of
        # positions of the bootstrap samples in XT / frame
        self.boot = boot
        self.y = y
        self.weights = weights
        self.names = names
        if output_format == 'multioutput':
            split = np.hsplit(y, 2)
            self.y_reg = split[0].flatten()
            self.y_class = split[1].flatten()
        elif output_format == 'classification':
            self.y_reg = None
            self.y_class = y
        else:
            self.y_reg = y
            self.y_class = None

    def __getstate__(self):
        state = dict(self.__dict__)
        # the feature matrix is never pickled, it can be restored from the data frame
        state['XT'] = None
        return state

    def frame_rows(self, rows: np.ndarray):
        '''
            returns the data frame for the given bootstrap positions (index reset as in the original implementation)
        '''
        full_rows = self.boot[rows]
        if self.frame is not None:
            return self.frame.iloc[full_rows, :].reset_index(drop=True)
        if self.XT is not None and self.feature_names is not None:
            return pd.DataFrame(self.XT[:, full_rows].T, columns=self.feature_names)
        return None


class _LazyRowAttribute:
    '''
        descriptor for per-sample attributes of a tree node (e.g. y_train). Leaves store their arrays,
        inner nodes drop them after splitting and recompute them from the shared tree data on access.
    '''

    def __init__(self, source: str):
        self.source = source

    def __set_name__(self, owner, name):
        self.slot = '_' + name

    def __get__(self, node, owner=None):
        if node is None:
            return self
        value = node.__dict__.get(self.slot)
        if value is None and node.__dict__.get('_rows') is not None:
            source = getattr(node._data, self.source)
            if source is not None:
                value = source[node._rows]
        return value

    def __set__(self, node, value):
        node.__dict__[self.slot] = value


class _FeatureImportances(MutableMapping):
    '''
        dict-like container for the feature importances of a tree that only stores the non-zero entries.
        Iterating over it yields all features (missing ones have importance 0) as for the original dict.
    '''

    def __init__(self, feature_names=None, index: dict = None):
        self._values = {}
        self._feature_names = feature_names
        # dict feature name -> position, shared between all trees
        self._index = index

    def __getitem__(self, key):
        if key in self._values:
            return self._values[key]
        if self._index is None or key in self._index:
            return 0
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, (int, float, np.floating, np.integer)) and value == 0 and key not in self._values:
            return
        self._values[key] = value

    def __delitem__(self, key):
        del self._values[key]

    def __iter__(self):
        if self._feature_names is None:
            return iter(self._values)
        return iter(self._feature_names)

    def __len__(self):
        if self._feature_names is None:
            return len(self._values)
        return len(self._feature_names)

    def dense(self, index: dict) -> np.ndarray:
        '''
            returns the importances as numpy array

            @param index: dict feature name -> position in the returned array
        '''
        result = np.zeros(len(index))
        for f, v in self._values.items():
            result[index[f]] = v
        return result


class _LeafLookup(Mapping):
    '''
        read-only dict-like view {sample name: function(leaf of sample)}.
        Replaces dictionaries that stored one entry per test sample and tree.
    '''

    def __init__(self, sample_names, leaves, leaf_ids, function):
        self._index = {}
        for i, name in enumerate(sample_names):
            self._index[name] = i
        self._leaves = leaves
        self._leaf_ids = leaf_ids
        self._function = function

    def __getitem__(self, key):
        return self._function(self._leaves[self._leaf_ids[self._index[key]]])

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)


def _leaf_X_train(leaf):
    return leaf.X_train


def _leaf_sample_names(leaf):
    return leaf.sample_names_train


def _leaf_path(leaf):
    return list(leaf._path)


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

    y_train = _LazyRowAttribute('y')
    y_train_reg = _LazyRowAttribute('y_reg')
    y_train_class = _LazyRowAttribute('y_class')
    sample_weights = _LazyRowAttribute('weights')
    sample_names_train = _LazyRowAttribute('names')

    def __init__(self, parent, level: int, is_leaf: bool, X_train: pd.DataFrame, y_train: np.array, sample_weights: np.array = None, sample_names_train: np.array = None, output_format: str = 'multioutput', _data: _TreeData = None, _rows: np.ndarray = None):
        '''
            constructor of a tree node in a binary decision tree

            @param parent: instance of BinaryTreeNode, i.e., the parent of the acutal node or None if we are in root
            @param level: the level of the actual node, i.e., number of steps need from root to here
            @param is_leaf: bool that tells us whether the actual node is a leaf or not
            @param X_train: pd.DataFrame containing the features of the samples that reached this node during training
            @param y_train: numpy array containing the response values for the samples that reached this node during training
            @param sample_weights: corresponding weight vector for each sample
            @param sample_names_train: a list of sample names corresponding to X_train and y_train
            @param _data: internal, the shared data of the tree (if None, it is created from X_train)
            @param _rows: internal, positions of the samples of this node in _data
        '''
        self.sample_ids_test = []
        self.parent = parent
        self.level = level
        self.is_leaf = is_leaf
        self.output_format = output_format
        self.already_predicted = False
        if _data is None:
            # stand-alone node created from a data frame
            feature_names = np.array(X_train.columns)
            XT = np.ascontiguousarray(
                X_train.to_numpy(dtype=np.float64).T)
            _data = _TreeData(XT=XT, feature_names=feature_names, col_of={f: i for i, f in enumerate(feature_names)},
                              boot=np.arange(len(y_train)), y=np.asarray(y_train), weights=None if sample_weights is None else np.asarray(sample_weights),
                              names=None if sample_names_train is None else np.asarray(sample_names_train), output_format=output_format, frame=X_train)
            _rows = np.arange(len(y_train))
        self._data = _data
        self._rows = _rows
        self._X_train_override = False
        self._materialize()

        self.normalized_sample_weights = None
        self.information_gain_for_split_exists = None

    def _materialize(self):
        '''
            stores the per-sample arrays of this node (only done for leaves)
        '''
        self.y_train = self._data.y[self._rows]
        self.sample_weights = None if self._data.weights is None else self._data.weights[self._rows]
        self.sample_names_train = None if self._data.names is None else self._data.names[self._rows]
        if self.output_format == 'multioutput':
            self.y_train_reg = self._data.y_reg[self._rows]
            self.y_train_class = self._data.y_class[self._rows]
        elif self.output_format == 'classification':
            self.y_train_class = self.y_train
        else:
            self.y_train_reg = self.y_train

    def _release(self):
        '''
            drops the per-sample arrays of an inner node (they are recomputed on demand)
        '''
        for attribute in ('_y_train', '_y_train_reg', '_y_train_class', '_sample_weights', '_sample_names_train'):
            self.__dict__[attribute] = None

    @property
    def X_train(self):
        '''
            the features of the training samples in this node as pandas DataFrame (created on demand).
            As in the original implementation, inner nodes return None.
        '''
        if self._X_train_override:
            return self._X_train_value
        if not self.is_leaf:
            return None
        return self._data.frame_rows(self._rows)

    @X_train.setter
    def X_train(self, value):
        if value is None and not self.__dict__.get('_X_train_override', False):
            # the original implementation freed X_train of inner nodes, which happens implicitly here
            return
        self._X_train_override = True
        self._X_train_value = value

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
            return (len(self._rows) >= 2 * min_number_of_samples_per_leaf)
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
            left, right = split.__dict__.pop('_partition')
            self.is_leaf = False
            self.left = BinaryTreeNode(parent=self, level=self.level + 1, is_leaf=True, X_train=None, y_train=None,
                                       output_format=self.output_format, _data=self._data, _rows=self._rows[left])
            self.right = BinaryTreeNode(parent=self, level=self.level + 1, is_leaf=True, X_train=None, y_train=None,
                                        output_format=self.output_format, _data=self._data, _rows=self._rows[right])
            self._release()
        return self.split

    def calculate_normalized_sample_weights(self) -> None:
        '''
            calculates the sample weights normalized for the bootstrap samples assigned to this node (self)
        '''
        if self.normalized_sample_weights is None:
            sum_weights_leaf = np.sum(self.sample_weights)
            self.normalized_sample_weights = np.asarray(
                self.sample_weights) / sum_weights_leaf

    def distribute_test_samples_to_leaves(self, samples: pd.DataFrame, sample_feature_path_dict: dict) -> None:
        '''
            distributes unseen (test) samples to the leaves and meanwhile tracks the trace of each sample

            @param samples: a dataframe containing the features for unseen (test) samples
            @param sample_feature_path_dict: a dict with the samples as keys and the trace through the tree as value. Is updated in this function
        '''
        if self.is_leaf:
            self.sample_ids_test = samples.index
        else:
            for sample_name in samples.index:
                sample_feature_path_dict[sample_name].append(
                    (self.split.feature_name, self.level))
            left, right = self.split.calculate_left_right(X=samples)
            self.left.distribute_test_samples_to_leaves(
                samples=samples.loc[left, :], sample_feature_path_dict=sample_feature_path_dict)
            self.right.distribute_test_samples_to_leaves(
                samples=samples.loc[right, :], sample_feature_path_dict=sample_feature_path_dict)

    def _majority_class(self, class_names):
        majority_class_occurences = 0
        majority_class = None
        for class_name in class_names:
            occ = weighted_class_count(
                class_name=class_name, y_class=self.y_train_class, sample_weights=self.sample_weights)
            if occ > majority_class_occurences:
                majority_class_occurences = occ
                majority_class = class_name
        return majority_class

    def predict(self, class_names) -> list:
        '''
            predicts for a specific sample,
            if we are in a leaf node: the mean of all continuous valued training samples in the leaf is returned for regression and the majority class for classification
            if not, we calculate whether the sample goes to the left or right child and return the prediction recursively

            @param sample: a pandas DataFrame with the sample, column names should correspond to the feature names in the training data
            @return: a list of length 2 containing at pos 0 the prediction for the regression and at pos 1 the prediction for the classification task
        '''

        if self.is_leaf:
            if self.output_format == 'multioutput':
                if not self.already_predicted:
                    self.calculate_normalized_sample_weights()
                    self.average = np.sum(
                        self.normalized_sample_weights * self.y_train_reg)
                    self.majority_class = self._majority_class(class_names)
                    self.already_predicted = True

                return [self.average, self.majority_class]
            elif self.output_format == 'classification':
                if not self.already_predicted:
                    self.calculate_normalized_sample_weights()
                    self.majority_class = self._majority_class(class_names)
                    self.already_predicted = True

                return self.majority_class
            else:
                if not self.already_predicted:
                    self.calculate_normalized_sample_weights()
                    self.average = np.sum(
                        self.normalized_sample_weights * self.y_train_reg)
                    self.already_predicted = True

                return self.average
        else:
            print('warning: predict called on no leaf node')

    def _node_score(self, class_criterion_function, regression_criterion_function, root_class_criterion: float, root_reg_criterion: float, impact_classification: float, class_names: list) -> float:
        sample_weights = self.sample_weights
        if sample_weights is None:
            print('sample weights are None?')

        if not self.output_format == 'regression':
            if not root_class_criterion == 0:
                node_class_criterion = class_criterion_function(
                    class_names=class_names, y_class=self.y_train_class, sample_weights=sample_weights)/root_class_criterion
            else:
                node_class_criterion = class_criterion_function(
                    class_names=class_names, y_class=self.y_train_class, sample_weights=sample_weights)
        else:
            node_class_criterion = 0
        if not self.output_format == 'classification':
            node_reg_criterion = regression_criterion_function(
                y_reg=self.y_train_reg, sample_weights=sample_weights)/root_reg_criterion
        else:
            node_reg_criterion = 0
        return impact_classification * node_class_criterion + \
            (1-impact_classification) * node_reg_criterion

    def calculate_best_split(self, features: list, class_criterion_function, regression_criterion_function, root_class_criterion: float, root_reg_criterion: float, impact_classification: float, class_names: list, min_number_of_samples_per_leaf: int, root_sample_weights: list) -> Split:
        '''
            calculates the best split for this node (or none if we do not gain information by splitting this node up)

            All thresholds of a feature are first scored at once using prefix sums (vectorized). The few thresholds whose
            score is (numerically) close to the best one are then re-evaluated with the exact same arithmetic as the
            original implementation, s.t. the selected split (including tie breaking) is identical.

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
        self.class_names = class_names
        node_score = self._node_score(class_criterion_function, regression_criterion_function,
                                      root_class_criterion, root_reg_criterion, impact_classification, class_names)
        best_score = node_score
        best_split = None
        valid_split_found = False

        self._split_context = dict(class_criterion_function=class_criterion_function, regression_criterion_function=regression_criterion_function,
                                   root_class_criterion=root_class_criterion, root_reg_criterion=root_reg_criterion, impact_classification=impact_classification,
                                   min_number_of_samples_per_leaf=min_number_of_samples_per_leaf, node_score=node_score,
                                   sum_weights=np.sum(self.sample_weights), sum_root_weights=np.sum(root_sample_weights))
        try:
            vectorized = self.sample_weights is not None and \
                (self.output_format == 'regression' or class_criterion_function in _VECTORIZED_CLASS_CRITERIA) and \
                (self.output_format == 'classification' or regression_criterion_function in _VECTORIZED_REG_CRITERIA)
            if not vectorized:
                # unknown (user defined) criterion: evaluate every threshold exactly
                for feature in features:
                    best_score, best_split, valid_split_found = self.calculate_best_split_for_feature(root_sample_weights=root_sample_weights, best_score=best_score, node_score=node_score, min_number_of_samples_per_leaf=min_number_of_samples_per_leaf, class_criterion_function=class_criterion_function,
                                                                                                      impact_classification=impact_classification, regression_criterion_function=regression_criterion_function, root_class_criterion=root_class_criterion, root_reg_criterion=root_reg_criterion, feature_name=feature, best_split=best_split, valid_split_found=valid_split_found)
            else:
                features = list(features)
                number_of_classes = 0 if self.output_format == 'regression' else len(
                    class_names)
                block_size = max(
                    1, _split_block_elements(number_of_classes) // max(1, len(self._rows)))
                for start in range(0, len(features), block_size):
                    best_score, best_split, valid_split_found = self._best_split_for_feature_block(
                        features[start:start+block_size], best_score, best_split, valid_split_found)
        finally:
            del self._split_context
            self.__dict__.pop('_index_set', None)
        return best_split

    def _feature_values(self, feature_name) -> np.ndarray:
        return self._data.XT[self._data.col_of[feature_name], self._data.boot[self._rows]]

    def _complement(self, left: np.ndarray) -> np.ndarray:
        '''
            returns the indices of the right child in the same order as the original implementation
            (list(set(X.index) - set(left))), s.t. the order of samples in the children (and thus all sums) is unchanged
        '''
        index_set = self.__dict__.get('_index_set')
        if index_set is None:
            index_set = set(range(len(self._rows)))
            self.__dict__['_index_set'] = index_set
        return np.array(list(index_set - set(left.tolist())), dtype=np.intp)

    def _evaluate_split_exact(self, feature_name, x: np.ndarray, threshold: float):
        '''
            evaluates a single split with the same arithmetic as the original implementation

            @return: None if the split is not valid, otherwise a tuple (score, split)
        '''
        ctx = self._split_context
        min_number_of_samples_per_leaf = ctx['min_number_of_samples_per_leaf']
        left = np.flatnonzero(x < threshold)
        if (len(left) < min_number_of_samples_per_leaf) or (len(x) - len(left) < min_number_of_samples_per_leaf):
            return None
        right = self._complement(left)
        sample_weights = self.sample_weights
        sum_weights = ctx['sum_weights']
        left_weight = np.sum(sample_weights[left])/sum_weights
        right_weight = np.sum(sample_weights[right])/sum_weights

        assert math.isclose(left_weight + right_weight,
                            1.0, abs_tol=10**-2)

        if not self.output_format == 'regression':
            y_train_class = self.y_train_class
            left_class = ctx['class_criterion_function'](
                class_names=self.class_names, y_class=y_train_class[left], sample_weights=sample_weights[left])
            right_class = ctx['class_criterion_function'](
                class_names=self.class_names, y_class=y_train_class[right], sample_weights=sample_weights[right])
            if ctx['root_class_criterion'] == 0:
                class_intermediate = 0
            else:
                class_intermediate = (
                    left_weight * left_class+right_weight * right_class) / ctx['root_class_criterion']
        else:
            class_intermediate = 0
        if not self.output_format == 'classification':
            y_train_reg = self.y_train_reg
            left_reg = ctx['regression_criterion_function'](
                y_reg=y_train_reg[left], sample_weights=sample_weights[left])
            right_reg = ctx['regression_criterion_function'](
                y_reg=y_train_reg[right], sample_weights=sample_weights[right])
            reg_intermediate = (left_weight * left_reg +
                                right_weight * right_reg) / ctx['root_reg_criterion']
        else:
            reg_intermediate = 0

        impact_classification = ctx['impact_classification']
        act_score = impact_classification * class_intermediate + \
            (1-impact_classification) * reg_intermediate
        split = Split(feature_name=feature_name, threshold=threshold)
        split.score = (ctx['node_score']-act_score) * \
            sum_weights/ctx['sum_root_weights']
        split._col = self._data.col_of[feature_name]
        split._partition = (left, right)
        return act_score, split

    def _best_split_for_feature_block(self, features: list, best_score: float, best_split: Split, valid_split_found: bool):
        '''
            vectorized split search for a block of features (see calculate_best_split)
        '''
        ctx = self._split_context
        min_samples = ctx['min_number_of_samples_per_leaf']
        n = len(self._rows)
        if n < 2:
            return best_score, best_split, valid_split_found
        cols = np.array([self._data.col_of[f] for f in features])
        # features x samples
        values = self._data.XT[np.ix_(cols, self._data.boot[self._rows])]
        order = np.argsort(values, axis=1, kind='stable')
        sorted_values = np.take_along_axis(values, order, axis=1)
        lower = sorted_values[:, :-1]
        upper = sorted_values[:, 1:]
        # candidate thresholds lie between two consecutive distinct values (NaN never forms a boundary)
        boundary = lower < upper
        thresholds = (lower + upper) / 2
        # number of samples with a value < threshold
        n_left = np.broadcast_to(np.arange(1, n), lower.shape)
        degenerate = boundary & (thresholds == lower)
        if degenerate.any():
            # threshold rounded down to the lower value (only for neighbouring floats):
            # then only the values smaller than the lower value go to the left
            is_start = np.ones(sorted_values.shape, dtype=bool)
            is_start[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
            run_start = np.maximum.accumulate(
                np.where(is_start, np.arange(n), 0), axis=1)[:, :-1]
            n_left = np.where(degenerate, run_start, n_left)
        valid = boundary & (n_left >= min_samples) & (
            n - n_left >= min_samples)
        if not valid.any():
            return best_score, best_split, valid_split_found

        def prefix(a):
            result = np.zeros((a.shape[0], n + 1))
            np.cumsum(a, axis=1, out=result[:, 1:])
            return result

        has_degenerate = degenerate.any()

        def left_and_total(a):
            p = prefix(a)
            if has_degenerate:
                left = np.take_along_axis(p, n_left, axis=1)
            else:
                left = p[:, 1:n]
            return left, p[:, n:n+1]

        weights = np.asarray(self.sample_weights, dtype=np.float64)
        w_sorted = weights[order]
        w_left, w_total = left_and_total(w_sorted)
        w_right = w_total - w_left
        with np.errstate(divide='ignore', invalid='ignore'):
            left_weight = w_left / w_total
            right_weight = w_right / w_total
            score = np.zeros(lower.shape)
            impact_classification = ctx['impact_classification']
            if not self.output_format == 'regression':
                y_class_sorted = np.asarray(self.y_train_class)[order]
                proportions_left = []
                proportions_right = []
                for class_name in self.class_names:
                    c_left, c_total = left_and_total(
                        w_sorted * (y_class_sorted == class_name))
                    proportions_left.append(c_left / w_left)
                    proportions_right.append((c_total - c_left) / w_right)
                name = _VECTORIZED_CLASS_CRITERIA[ctx['class_criterion_function']]
                left_class = _class_criterion_from_proportions(
                    name, proportions_left)
                right_class = _class_criterion_from_proportions(
                    name, proportions_right)
                if not ctx['root_class_criterion'] == 0:
                    score += impact_classification * \
                        (left_weight * left_class + right_weight *
                         right_class) / ctx['root_class_criterion']
            if not self.output_format == 'classification':
                y_reg = np.asarray(self.y_train_reg, dtype=np.float64)
                # centering improves the numerical stability of the sum of squares
                y_sorted = (y_reg - np.mean(y_reg))[order]
                wy = w_sorted * y_sorted
                y_left, y_total = left_and_total(y_sorted)
                wy_left, wy_total = left_and_total(wy)
                wyy_left, wyy_total = left_and_total(wy * y_sorted)
                # the original MSE uses the unweighted mean
                mean_left = y_left / n_left
                mean_right = (y_total - y_left) / (n - n_left)
                rss_left = wyy_left - 2 * mean_left * wy_left + mean_left * mean_left * w_left
                rss_right = (wyy_total - wyy_left) - 2 * mean_right * \
                    (wy_total - wy_left) + mean_right * mean_right * w_right
                mse_left = np.maximum(rss_left, 0) / w_left
                mse_right = np.maximum(rss_right, 0) / w_right
                score += (1 - impact_classification) * (left_weight * mse_left +
                                                        right_weight * mse_right) / ctx['root_reg_criterion']
        score = np.where(valid & ~np.isnan(score), score, np.inf)
        minima = score.min(axis=1)
        any_valid = valid.any(axis=1)

        for i, feature_name in enumerate(features):
            if not any_valid[i]:
                continue
            valid_split_found = True
            minimum = minima[i]
            if not np.isfinite(minimum):
                # all scores are NaN, they can never be selected (as in the original implementation)
                continue
            tolerance = 1e-8 * (1 + abs(minimum))
            if minimum > best_score + tolerance:
                continue
            x = values[i]
            for j in np.flatnonzero(score[i] <= minimum + tolerance):
                result = self._evaluate_split_exact(
                    feature_name, x, thresholds[i, j])
                if result is None:
                    continue
                act_score, split = result
                if act_score <= best_score:
                    best_score = act_score
                    best_split = split
        return best_score, best_split, valid_split_found

    def calculate_best_split_for_feature(self, root_sample_weights: list, min_number_of_samples_per_leaf: int, class_criterion_function, regression_criterion_function, root_reg_criterion, root_class_criterion, impact_classification, feature_name, best_split: Split, best_score: float, valid_split_found: bool, node_score: float):
        '''
            identifies the best split for a given feature by evaluating every threshold

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
        own_context = '_split_context' not in self.__dict__
        if own_context:
            self._split_context = dict(class_criterion_function=class_criterion_function, regression_criterion_function=regression_criterion_function,
                                       root_class_criterion=root_class_criterion, root_reg_criterion=root_reg_criterion, impact_classification=impact_classification,
                                       min_number_of_samples_per_leaf=min_number_of_samples_per_leaf, node_score=node_score,
                                       sum_weights=np.sum(self.sample_weights), sum_root_weights=np.sum(root_sample_weights))
        try:
            x = self._feature_values(feature_name)
            sorted_values = np.sort(np.unique(x.tolist()))
            for i, val in enumerate(sorted_values):
                if i+1 == len(sorted_values):
                    continue
                threshold = np.mean([val, sorted_values[i+1]])
                result = self._evaluate_split_exact(feature_name, x, threshold)
                if result is None:
                    continue
                valid_split_found = True
                act_score, split = result
                if act_score <= best_score:
                    best_score = act_score
                    best_split = split
        finally:
            if own_context:
                del self._split_context
                self.__dict__.pop('_index_set', None)
        return best_score, best_split, valid_split_found


class MultivariateDecisionTree:

    def __init__(self, X_train: pd.DataFrame, y_train: np.array, class_names: np.array = None, criterion_class: str = 'gini', criterion_reg: str = 'mse', min_number_of_samples_per_leaf: int = 1, max_depth: int = 20, number_of_features_per_split: Union[str, float] = 'sqrt', random_object: np.random.RandomState = np.random.RandomState(42), impact_classification: float = 0.5, output_format: str = 'multioutput', sample_weights: np.array = None, sample_names_train: np.array = None, distance_measure: str = 'pearson', bootstrap_indices: np.ndarray = None, feature_names: np.ndarray = None, _col_of: dict = None) -> None:
        '''
            constructor of a multivariate decision tree

            @param X_train: a pandas DataFrame with columns as features and rows as samples.
                Alternatively, a transposed numpy feature matrix (features x samples), then feature_names must be given.
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
            @param bootstrap_indices: optional, positions of the training samples (rows of X_train) used by this tree.
                If given, y_train, sample_weights, and sample_names_train correspond to these positions. This avoids
                copying the feature matrix for every tree.
            @param feature_names: names of the features, only needed if X_train is a numpy array
        '''
        if isinstance(X_train, np.ndarray):
            if feature_names is None:
                raise ValueError(
                    'feature_names must be given if X_train is a numpy array')
            XT = X_train
            frame = None
            n_rows = XT.shape[1]
        else:
            feature_names = np.array(X_train.columns)
            XT = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64).T)
            frame = X_train
            n_rows = XT.shape[1]
        self.feature_names = np.asarray(feature_names)
        if _col_of is None:
            _col_of = {f: i for i, f in enumerate(self.feature_names)}
        if bootstrap_indices is None:
            bootstrap_indices = np.arange(n_rows)
        self.bootstrap_indices = np.asarray(bootstrap_indices)

        self.y_train = y_train
        self.sample_names_train = np.array(sample_names_train)
        self.train_samples_in_same_leaf = {}
        self.feature_importances = _FeatureImportances(
            self.feature_names, _col_of)
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
        number_of_features = len(self.feature_names)
        if type(number_of_features_per_split) == str:
            if number_of_features_per_split == 'sqrt':
                self.number_of_features_per_split = int(
                    math.sqrt(number_of_features))
            elif number_of_features_per_split == 'log2':
                self.number_of_features_per_split = int(
                    math.log2(number_of_features))
            else:
                raise (ValueError(
                    f'Unsupported string for number_of_features_per_split {number_of_features_per_split}.'))
        else:
            self.number_of_features_per_split = int(
                number_of_features_per_split*number_of_features)
        self.random_object = random_object

        self.impact_classification = impact_classification
        self.output_format = output_format
        self.sample_weights = sample_weights
        self._data = _TreeData(XT=XT, feature_names=self.feature_names, col_of=_col_of, boot=self.bootstrap_indices,
                               y=np.asarray(self.y_train), weights=None if sample_weights is None else np.asarray(sample_weights),
                               names=self.sample_names_train, output_format=self.output_format, frame=frame)
        self.root = BinaryTreeNode(
            None, level=0, is_leaf=True, X_train=None, y_train=None, output_format=self.output_format, _data=self._data, _rows=np.arange(len(self.bootstrap_indices)))
        self.root._path = []

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

    @property
    def X_train(self):
        '''
            the bootstrap feature matrix of this tree as pandas DataFrame (created on demand)
        '''
        return self._data.frame_rows(np.arange(len(self.bootstrap_indices)))

    def _detach_shared(self):
        '''
            removes references to objects that are shared between all trees (used before sending a tree to another process)
        '''
        self._data.XT = None
        self._data.frame = None
        self._data.feature_names = None
        self._data.col_of = None
        self.feature_names = None
        self.feature_importances._feature_names = None
        self.feature_importances._index = None

    def _attach_shared(self, frame, feature_names, col_of):
        '''
            restores the references removed by _detach_shared
        '''
        self._data.frame = frame
        self._data.feature_names = feature_names
        self._data.col_of = col_of
        self.feature_names = feature_names
        self.feature_importances._feature_names = feature_names
        self.feature_importances._index = col_of

    def _leaf_of_name(self):
        '''
            dict: training sample name -> first leaf (in self.leaves) that contains it
        '''
        lookup = self.__dict__.get('_leaf_of_name_cache')
        if lookup is None:
            lookup = {}
            for leaf in self.leaves:
                for name in leaf.sample_names_train:
                    if name not in lookup:
                        lookup[name] = leaf
            self._leaf_of_name_cache = lookup
        return lookup

    def find_train_sample_leaf_friends(self, train_sample_id: int, x_train: pd.Series):
        '''
            finds the samples in the same leaf as a particular training sample

            @param train_sample_id: the id of the training sample of interest
            @param x_train: the training set
            @return a numpy array with the training samples in the same leaf of the decision tree
        '''
        lookup = self._leaf_of_name()
        if train_sample_id in lookup:
            return np.array(lookup[train_sample_id].sample_names_train)
        else:
            node = self.root
            while not node.is_leaf:
                if x_train[node.split.feature_name] < node.split.threshold:
                    node = node.left
                else:
                    node = node.right
            return np.array(node.sample_names_train)

    def fit(self):
        '''
            builds a BinaryDecisionTree for the given training data
        '''
        number_of_features = self.number_of_features_per_split
        feature_list = self.feature_names

        while self.is_split_possible():

            node = self.find_next_leaf()

            index_list_features = self.random_object.choice(
                len(feature_list), size=number_of_features, replace=False)
            features = feature_list[index_list_features]

            split = node.split_node(features=features, root_sample_weights=self.sample_weights, class_criterion_function=self.class_criterion_function, regression_criterion_function=self.regression_criterion_function,
                                    root_class_criterion=self.root_class_criterion, root_reg_criterion=self.root_reg_criterion, impact_classification=self.impact_classification, class_names=self.class_names, min_number_of_samples_per_leaf=self.min_number_of_samples_per_leaf)

            if not split is None:
                self.feature_importances[split.feature_name] += split.score
                self.leaves.remove(node)
                self.leaves.append(node.left)
                self.leaves.append(node.right)
                node.left._path = node._path + [(split.feature_name, node.level)]
                node.right._path = node.left._path
                if node.left.level > self.depth:
                    self.depth = node.left.level

        dense_importances = self.feature_importances.dense(self._data.col_of)
        normalizer = np.sum(dense_importances)

        if not normalizer == 0:
            for f in list(self.feature_importances._values.keys()):
                self.feature_importances[f] /= normalizer
            assert (math.isclose(np.sum(dense_importances / normalizer),
                                 1.0, abs_tol=10**-2))
        self.leaves = np.array(self.leaves)
        for i, leaf in enumerate(self.leaves):
            leaf._leaf_id = i
        # the feature matrix is not needed anymore
        self._data.XT = None

    def _route(self, XT: np.ndarray, col_of: dict = None) -> np.ndarray:
        '''
            distributes samples to the leaves

            @param XT: transposed feature matrix (features x samples) of the samples, rows ordered as in col_of (default: training features)
            @return: numpy array with the position (in self.leaves) of the leaf of every sample
        '''
        if col_of is None:
            col_of = self._data.col_of
        leaf_ids = np.empty(XT.shape[1], dtype=np.intp)
        stack = [(self.root, np.arange(XT.shape[1]))]
        while stack:
            node, positions = stack.pop()
            if node.is_leaf:
                leaf_ids[positions] = node._leaf_id
                continue
            split = node.split
            col = getattr(split, '_col', None)
            if col is None or col_of is not self._data.col_of:
                col = col_of[split.feature_name]
            go_left = XT[col, positions] < split.threshold
            stack.append((node.right, positions[~go_left]))
            stack.append((node.left, positions[go_left]))
        return leaf_ids

    def _set_test_samples(self, sample_names: np.ndarray, leaf_ids: np.ndarray) -> None:
        '''
            stores which test samples ended in which leaf (as the original predict function did)
        '''
        self._test_leaf_ids = leaf_ids
        order = np.argsort(leaf_ids, kind='stable')
        boundaries = np.searchsorted(
            leaf_ids[order], np.arange(len(self.leaves) + 1))
        for i, leaf in enumerate(self.leaves):
            leaf.sample_ids_test = sample_names[order[boundaries[i]:boundaries[i+1]]]
        self.sample_feature_path = _LeafLookup(
            sample_names, self.leaves, leaf_ids, _leaf_path)
        self.train_samples_in_same_leaf = _LeafLookup(
            sample_names, self.leaves, leaf_ids, _leaf_X_train)
        self.train_samples_names_in_leaf = _LeafLookup(
            sample_names, self.leaves, leaf_ids, _leaf_sample_names)

    def predict(self, samples: pd.DataFrame) -> list:
        '''
            calculates a prediction for given samples

            @param samples: a pandas DataFrame containing all samples
            @return: a list containing the predictions depending on whether the output should be multivariate or not
        '''
        names = np.array(samples.index)
        columns = list(samples.columns)
        col_of = {f: i for i, f in enumerate(columns)}
        XT = np.ascontiguousarray(samples.to_numpy(dtype=np.float64).T)
        leaf_ids = self._route(XT, col_of)
        self._set_test_samples(names, leaf_ids)
        leaf_predictions = [None] * len(self.leaves)
        for i, leaf in enumerate(self.leaves):
            if len(leaf.sample_ids_test) > 0:
                leaf_predictions[i] = leaf.predict(
                    class_names=self.class_names)
        predictions = np.full(len(names), None)
        for i, leaf_id in enumerate(leaf_ids):
            predictions[i] = leaf_predictions[leaf_id]
        return predictions, self

    def update_depth(self) -> None:
        '''
            updates the depth of the tree
        '''
        for leaf in self.leaves:
            if leaf.level > self.depth:
                self.depth = leaf.level

    def find_next_leaf(self) -> BinaryTreeNode:
        '''
            returns the next leaf for which a split is possible
        '''
        for leaf in self.leaves:
            if leaf.is_split_possible(self.min_number_of_samples_per_leaf):
                return leaf

    def is_split_possible(self) -> bool:
        '''
            checks whether a split is possible for any leaf in the tree

            @return bool: True if there is a leaf, which can be split, False otherwise
        '''
        for leaf in self.leaves:
            if leaf.is_split_possible(self.min_number_of_samples_per_leaf):
                return self.depth < self.max_depth
        return False
