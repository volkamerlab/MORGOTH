import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef, r2_score, mean_absolute_error, balanced_accuracy_score
from morgoth import MORGOTH
import warnings
warnings.filterwarnings("ignore")




# Split into SOURCE and TARGET by stalk-shape
# There are two stalk-shape values: "e" and "t"
source_df = pd.read_csv('Example_Data/wine_quality/winequality-white.csv', sep = ';')
target_df = pd.read_csv('Example_Data/wine_quality/winequality-red.csv', sep = ';')


def prepare_xy(d):
    X = d.drop(columns=["quality"])
    d["quality_reg"] = d['quality'].values
    y = d.loc[:, ['quality', 'quality_reg']].values
    y_one_column = d["quality"]
    return X, y, y_one_column

Xs_raw, ys, _ = prepare_xy(source_df)
Xt_raw, yt, yt_one_column = prepare_xy(target_df)


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
tree_weight_file = f'{output_dir}/Tree_Weights_Wine_Classification.txt'
# rf only trained on the source dataset
rf_source = MORGOTH(X_train=Xs, y_train=ys, sample_names_train=Xs.index, threshold=[3.5, 4.5, 5.5, 6.6, 7.5, 8.5],
                    criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=50, analysis_name='source_only',
                    number_of_features_per_split='sqrt', class_names=[3, 4, 5, 6, 7, 8, 9], output_format='multioutput', time_file=time_file,
                    sample_weights_included='simple', random_state=seed, max_depth=20, impact_classification=1,
                    sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file,
                    tree_weights=None, silhouette_score_file=silhouette_score_file, distance_measure='', cluster_assignment_file=cluster_assignment_file,
                    draw_graph=False, graph_path=output_dir,
                    silhouette_score_train_file=silhouette_score_train_file)
rf_source.fit()

kf = KFold(n_splits=20, shuffle=True, random_state=42)
# Segev et al say they train on 5% of the target for their baseline
results_target_ba = []
results_source_ba = []
results_transfer_ba = []
results_target_mae = []
results_source_mae = []
results_transfer_mae = []

for fold, (test_idx, train_idx) in enumerate(kf.split(Xt, yt), 1):

    print(f"\n--- Fold {fold} ---")

    Xt_train, Xt_test = Xt.iloc[train_idx], Xt.iloc[test_idx]
    yt_train, yt_test = yt[train_idx], yt[test_idx]
    y_true = yt_one_column[test_idx]
    # -----------------------------------------
    # TARGET ONLY
    # -----------------------------------------
    rf_tgt = MORGOTH(X_train=Xt_train, y_train=yt_train, sample_names_train=Xt_train.index, threshold=[3.5, 4.5, 5.5, 6.6, 7.5],
                     criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=50, analysis_name=f'target_only_fold_{fold}',
                     number_of_features_per_split='sqrt', class_names=[3, 4, 5, 6, 7, 8], output_format='multioutput', time_file=time_file,
                     sample_weights_included='', random_state=seed, max_depth=20, impact_classification=1,
                     sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file,
                     tree_weights=None, silhouette_score_file=silhouette_score_file, distance_measure='', cluster_assignment_file=cluster_assignment_file,
                     draw_graph=False, graph_path=output_dir,
                     silhouette_score_train_file=silhouette_score_train_file)

    rf_tgt.fit()
    preds = rf_tgt.predict(Xt_test)
    split = np.hsplit(preds, 2)
    y_pred_reg = split[0].flatten()
    y_pred_class = split[1].flatten()


    ba = balanced_accuracy_score(y_pred_class, y_true)
    print(f"Target-only BA: {ba}")
    mae = mean_absolute_error(y_pred=y_pred_reg, y_true=y_true)
    results_target_ba.append(ba)
    results_target_mae.append(mae)
    print(f"Target-only MAE: {mae}")
    # -----------------------------------------
    # SOURCE ONLY
    # -----------------------------------------
    preds = rf_source.predict(Xt_test)
    split = np.hsplit(preds, 2)
    y_pred_reg_src = split[0].flatten()
    y_pred_class_src = split[1].flatten()


    ba = balanced_accuracy_score(y_pred_class_src, y_true)
    print(f"Source-only BA: {ba}")
    mae = mean_absolute_error(y_pred=y_pred_reg_src, y_true=y_true)
    results_source_ba.append(ba)
    results_source_mae.append(mae)
    print(f"Source-only MAE: {mae}")

    # -----------------------------------------
    # TRANSFER LEARNING
    # -----------------------------------------
    rf_tl = MORGOTH(X_train=Xs, y_train=ys, sample_names_train=Xs.index, threshold=[3.5, 4.5, 5.5, 6.6, 7.5, 8.5],
                    criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=50, analysis_name=f'target_only_fold_{fold}',
                    number_of_features_per_split='sqrt', class_names=[3, 4, 5, 6, 7, 8, 9], output_format='multioutput', time_file=time_file,
                    sample_weights_included='simple', random_state=seed, max_depth=20, impact_classification=1,
                    sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file,
                    tree_weights='wma', silhouette_score_file=silhouette_score_file, distance_measure='', cluster_assignment_file=cluster_assignment_file,
                    draw_graph=False, graph_path=output_dir,
                    silhouette_score_train_file=silhouette_score_train_file,  X_target_train=Xt_train, y_target_train=yt_train, loss_wma_regression='abs',
                    loss_wma_classification='sum',
                    beta=0.2, labda_wma=0.5, tree_weight_file=tree_weight_file)

    rf_tl.fit()                # Pretrain
    preds = rf_tl.predict(Xt_test)
    split = np.hsplit(preds, 2)
    y_pred_reg = split[0].flatten()
    y_pred_class = split[1].flatten()


    ba = balanced_accuracy_score(y_pred_class, y_true)
    print(f"Transfer BA: {ba}")
    mae = mean_absolute_error(y_pred=y_pred_reg, y_true=y_true)
    results_transfer_ba.append(ba)
    results_transfer_mae.append(mae)
    print(f"Transfer MAE: {mae}")


    ba = balanced_accuracy_score(y_pred_class, y_pred_class_src)
    print(f"Transfer BA vs Source: {ba}")
    mae = mean_absolute_error(y_pred=y_pred_reg, y_true=y_pred_reg_src)
    print(f"Transfer MAE vs Source: {mae}")

print("\n====================== FINAL RESULTS ======================")
print("Target-only BA: mean=%.4f" % np.mean(results_target_ba))
print("Source-only BA: mean=%.4f" % np.mean(results_source_ba))
print("Transfer BA:    mean=%.4f" % np.mean(results_transfer_ba))
print("Target-only MAE: mean=%.4f" % np.mean(results_target_mae))
print("Source-only MAE: mean=%.4f" % np.mean(results_source_mae))
print("Transfer MAE:    mean=%.4f" % np.mean(results_transfer_mae))