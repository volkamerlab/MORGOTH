# MORGOTH
This is the implementation of our novel random forest (RF)-based approach for Multivariate classificatiOn and Regression increasinG trustwOrTHiness (MORGOTH). A detailed description and application of the model can be found in [our pre-print `Increasing trustworthiness of machine learning-based drug sensitivity prediction with a multivariate random forest approach'](https://doi.org/10.26434/chemrxiv-2025-ml78s). MORGOTH can be used to simultaneously perform classification and regression using a novel objective function during the training, which is a linear combination of classification and regression error. Moreover, it offers the possibility to perform conformal prediction (CP), which can be used to obtain reliable classification and regression results. A more detailed explanation of CP and the framework we use can be found in [our article 'Reliable anti-cancer drug sensitivity prediction and prioritization'](https://doi.org/10.1038/s41598-024-62956-6). Additionally, MORGOTH provides a graph representation of the random forest to address model interpretability, and a cluster analysis of the leaves to measure the dissimilarity of new inputs from the training data to account for its reliability. 



For issues and questions, please contact Lisa-Marie Rolli (lisa-marie.rolli[at]uni-saarland.de) or Kerstin Lenhof (research[at]klenhof.de).

## Installation

You can install our morgoth package using pip:
```
pip install morgoth
```
used python3 libraries: fireducks pandas numpy typing math bisect operator copy sklearn time scipy collections multiprocessing functools re

## Usage

### Command line usage

An exemplary use is running our provided main as a module, which you can call after downloading the `Example_Data` folder from our GitHub.

```
python3 -m morgoth Example_Data/example_Json_config.json
```
Note that the directory tree should be kept and the path to the output folder should be edited in the file `Example_Data/example_JSON_config.json`. The prediction results for classification will be found in ```<output_dir><analysis_name>_ClassificationResultsFile1.txt ``` and the regression results are stored in ```<output_dir><analysis_name>_<1-error_rate>_RegressionResultsFile1.txt```. If if the field swap_test_calibration in the config file is set to 'True' there will be one additional file per task, respectively, where the '1' in the file name is replaced by a '2'. If a distance measure is given in the config, ```<output_dir><analysis_name>_SilhouetteScoresTrainSamples_<distance>.txt``` and ```<output_dir><analysis_name>_SilhouetteScoresTestSamples_<distance>.txt``` will contain the silhouette scores for the training and test samples, respectively. If draw_graph is set to True, the files ```<output_dir>/<analysis_name>_<sample_name>.dot``` contain the sample specific graphs and ```<output_dir><analysis_name>__graph_whole_forest.dot``` and  ```<output_dir><analysis_name>__graph_average_whole_forest.dot``` contain the graph for the whole test set with either the raw count across all samples as edge weight or averaged by the number of test samples, respectively.

### Example Python code

```python
import pandas as pd
from morgoth import MORGOTH
import numpy as np
from sklearn.metrics import r2_score, matthews_corrcoef

seed = 42
# change to your input data directory
input_dir = 'Example_Data/'
# change to your output data directory
output_dir = 'Example_Data/output/'
# full feature matrix
X = pd.read_csv(f'{input_dir}/expression_matrix.txt', index_col=0, sep='\t')
# regression data is split in train and test
y_reg_train = pd.read_csv(
    f'{input_dir}/Training_Irinotecan_scores.txt', index_col=0, sep='\t')
y_reg_test = pd.read_csv(
    f'{input_dir}/Test_Irinotecan_scores.txt', index_col=0, sep='\t')
# binary contains all samples
y_binary_full = pd.read_csv(
    f'{input_dir}/Irinotecan_Discretized.txt', index_col=0, sep='\t')
# thus we split it into train and test
y_binary_train = y_binary_full.loc[y_reg_train.index, :]
y_binary_test = y_binary_full.loc[y_reg_test.index, :]

# for multioutput we need an array of arrays of size 2
y_train = np.array([np.array(
    [y_reg_train.values[i], y_binary_train.values[i]]) for i in range(len(y_reg_train))])
y_test = np.array([np.array([y_reg_test.values[i], y_binary_test.values[i]])
                  for i in range(len(y_reg_test))])

# create output_files
time_file = f'{output_dir}/ElapsedTimeFitting.txt'
sample_info_file = f'{output_dir}/Additional_Sample_Information.txt'
leaf_assignment_file_train = f'{output_dir}/Training_Set_LeafAssignment.txt'
feature_imp_output_file = f'{output_dir}/Feature_Importance.txt'
silhouette_score_file = f'{output_dir}/Silhouette_Score.txt'
silhouette_score_train_file = f'{output_dir}Silhouette_Score_Train.txt'
cluster_assignment_file = f'{output_dir}/Cluster_Assignment.txt'

# subset source to the 100 genes with the highest score
gene_scores = pd.read_csv(
    f'{input_dir}/gene_score_list.txt', sep='\t', names=['gene', 'score'])

# find top 100 genes
gene_scores.sort_values(by='score', ascending=False, inplace=True)
gene_scores.set_index('gene', inplace=True, drop=True)
top_100 = gene_scores.index.values[:100]

# subset the feature matrix
X_train_subset = X.loc[y_reg_train.index, top_100]
X_test_subset = X.loc[y_reg_test.index, top_100]

# construct morgoth object
mult_rf = MORGOTH(X_train=X_train_subset, y_train=y_train, sample_names_train=X_train_subset.index,
                  criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=500, analysis_name='example_usage',
                  number_of_features_per_split='sqrt', class_names=[0, 1], output_format='multioutput', threshold=[0.499358], time_file=time_file,
                  sample_weights_included='simple', random_state=seed, max_depth=20, impact_classification=0,
                  sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file,
                  tree_weights=True, silhouette_score_file=silhouette_score_file, distance_measure='', cluster_assignment_file=cluster_assignment_file,
                  draw_graph=False, graph_path=output_dir,
                  silhouette_score_train_file=silhouette_score_train_file)
# fit and predict
mult_rf.fit()
y_pred = mult_rf.predict(X_test=X_test_subset)
# split output in regression and classification data (in case you use multioutput)
split = np.hsplit(y_pred, 2)
y_pred_reg = split[0].flatten()
y_pred_class = split[1].flatten()

print(f'MCC: {matthews_corrcoef(y_pred=y_pred_class, y_true=y_binary_test.values)}')
print(f'R2: {r2_score(y_pred=y_pred_reg, y_true=y_reg_test.values)}')

```