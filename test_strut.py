import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef, mean_absolute_error
from morgoth import MORGOTH
import warnings
warnings.filterwarnings("ignore")
from morgoth.multivariate_dt import MultivariateDecisionTree

# Split into SOURCE and TARGET by stalk-shape
# There are two stalk-shape values: "e" and "t"
source_df = pd.read_csv('Example_Data/wine_quality/winequality-white.csv', sep = ';')
target_df = pd.read_csv('Example_Data/wine_quality/winequality-red.csv', sep = ';')


def prepare_xy(d):
    X = d.drop(columns=["quality"])
    y = d["quality"].values
    return X, y

Xs_raw, ys = prepare_xy(source_df)
Xt_raw, yt = prepare_xy(target_df)


# Use a LabelEncoder PER COLUMN, trained on combined source+target
X_all = pd.concat([Xs_raw, Xt_raw], axis=0)

encoders = {}
for col in X_all.columns:
    le = LabelEncoder()
    le.fit(X_all[col])
    encoders[col] = le
    Xs_raw[col] = le.transform(Xs_raw[col])
    Xt_raw[col] = le.transform(Xt_raw[col])

Xs = Xs_raw.astype(int)
Xt = Xt_raw.astype(int)


seed = 42
# change to your input data directory
input_dir = 'Example_Data/'
# change to your output data directory
output_dir = 'Example_Data/output/'
# full feature matrix

# create output_files
time_file = f'{output_dir}/ElapsedTimeFitting.txt'
sample_info_file = f'{output_dir}/Additional_Sample_Information.txt'
leaf_assignment_file_train = f'{output_dir}/Training_Set_LeafAssignment.txt'
feature_imp_output_file = f'{output_dir}/Feature_Importance.txt'
silhouette_score_file = f'{output_dir}/Silhouette_Score.txt'
silhouette_score_train_file = f'{output_dir}Silhouette_Score_Train.txt'
cluster_assignment_file = f'{output_dir}/Cluster_Assignment.txt'
# rf only trained on the source dataset


kf = KFold(n_splits=20, shuffle=True, random_state=42)


for fold, (test_idx, train_idx) in enumerate(kf.split(Xt, yt), 1):
    print(f'----- Fold {fold} -----')

    Xt_train, Xt_test = Xt.iloc[train_idx, :], Xt.iloc[test_idx, :]
    yt_train, yt_test = yt[train_idx], yt[test_idx]

    # -----------------------------------------
    # TARGET ONLY
    # -----------------------------------------
    rf_tgt = MORGOTH(X_train=Xs, y_train=ys, sample_names_train=Xs.index, threshold=[0.5],
                     criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=10, analysis_name=f'target_only_fold_{fold}',
                     number_of_features_per_split='sqrt', class_names=[0, 1], output_format='regression', time_file=time_file,
                     sample_weights_included='', random_state=seed, max_depth=20, impact_classification=1,
                     sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file,
                     tree_weights='wma', silhouette_score_file=silhouette_score_file, distance_measure='', cluster_assignment_file=cluster_assignment_file,
                     draw_graph=False, graph_path=output_dir,
                     silhouette_score_train_file=silhouette_score_train_file)

    rf_tgt.fit()
    pred = rf_tgt.predict(X_test=Xt_test)
    print(f'MAE {mean_absolute_error(pred, yt_test)}')
    rf_tgt.fine_tune(X_target=Xt_train, y_target=yt_train)
    pred = rf_tgt.predict(X_test=Xt_test)
    print(f'MAE {mean_absolute_error(pred, yt_test)}')



