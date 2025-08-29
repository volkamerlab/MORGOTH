import pandas as pd
from morgoth import MORGOTH
import numpy as np
from sklearn.metrics import r2_score, matthews_corrcoef

seed = 42
# change to your input data directory
input_dir = 'Example_Data/'
# change to your output data directory
output_dir = 'Example_Data/'
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
                  criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=500, analysis_name='benchmark',
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
