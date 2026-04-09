import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef, r2_score, mean_absolute_error
from morgoth import MORGOTH
import warnings
warnings.filterwarnings("ignore")




# Split into SOURCE and TARGET by stalk-shape
# There are two stalk-shape values: "e" and "t"
target_df = pd.read_csv('Example_Data/ToxCastERa/NVS_NR_hER-2026-04-08.csv', sep = ',', index_col=0)
source_df = pd.read_csv('Example_Data/ToxCastERa/TOX21_ERa_BLA_Agonist_ratio-2026-04-08.csv', sep = ',', index_col=0)
properties = pd.read_csv('Example_Data/ToxCastERa/physchem_properties.csv', sep = '\t', index_col=0)


def prepare_xy(features:pd.DataFrame, d:pd.DataFrame):
    intersection_compounds = sorted(set(features.index).intersection(set(d.index)))
    X = features.loc[intersection_compounds, :]
    d['HIT CALL'] = d.apply(lambda x: 1 if x['HIT CALL'] == 'Active' else 0, axis=1)
    y = d.loc[intersection_compounds, ['CONTINUOUS HIT CALL', 'HIT CALL']]
    y_reg = d.loc[intersection_compounds, 'CONTINUOUS HIT CALL']
    y_class = d.loc[intersection_compounds, 'HIT CALL']
    return X, y, y_reg, y_class

Xs_raw, ys, _ , _= prepare_xy(d = source_df, features=properties)
Xt_raw, yt, y_reg_true, y_class_true = prepare_xy(d = target_df, features=properties)





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



# Segev et al say they train on 5% of the target for their baseline
results_target_mcc = []
results_source_mcc = []
results_transfer_mcc = []
results_target_mae = []
results_source_mae = []
results_transfer_mae = []

for fold in range(5):
    
    with open(f'/home/lisa-marie-rolli/ToxCastBenchmark/model_inputs/steroidal/NVS_NR_hER/fold{fold}/train.txt', 'r') as train_file:
        train_samples = train_file.read().splitlines()
    with open(f'/home/lisa-marie-rolli/ToxCastBenchmark/model_inputs/steroidal/NVS_NR_hER/fold{fold}/test.txt', 'r') as test_file:
        test_samples = test_file.read().splitlines()
    with open(f'/home/lisa-marie-rolli/ToxCastBenchmark/model_inputs/steroidal/NVS_NR_hER/fold{fold}/physchem_feature_names_mrmr.txt', 'r') as feature_file:
        features = feature_file.read().splitlines()
    print(f"\n--- Fold {fold} ---")

    Xt_train, Xt_test = Xt.loc[train_samples, features], Xt.loc[test_samples, features]
    yt_train, yt_test = yt.loc[train_samples, :].values, yt.loc[test_samples, :].values
    y_reg = y_reg_true[test_samples]
    y_class = y_class_true[test_samples]
    # -----------------------------------------
    # TARGET ONLY
    # -----------------------------------------
    rf_tgt = MORGOTH(X_train=Xt_train, y_train=yt_train, sample_names_train=Xt_train.index, threshold=[0.9],
                     criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=50, analysis_name=f'target_only_fold_{fold}',
                     number_of_features_per_split='sqrt', class_names=[0,1], output_format='multioutput', time_file=time_file,
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

    
    mcc = matthews_corrcoef(y_pred_class, y_class)
    print(f"Target-only MCC: {mcc}")
    mae = mean_absolute_error(y_pred=y_pred_reg, y_true=y_reg)
    results_target_mcc.append(mcc)
    results_target_mae.append(mae)
    print(f"Target-only MAE: {mae}")
    # -----------------------------------------
    # SOURCE ONLY
    # -----------------------------------------

    rf_source = MORGOTH(X_train=Xs.loc[:, features], y_train=ys.values, sample_names_train=Xs.index, threshold=[0.9],
                    criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=50, analysis_name='source_only',
                    number_of_features_per_split='sqrt', class_names=[0, 1], output_format='multioutput', time_file=time_file,
                    sample_weights_included='simple', random_state=seed, max_depth=20, impact_classification=1,
                    sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file,
                    tree_weights=None, silhouette_score_file=silhouette_score_file, distance_measure='', cluster_assignment_file=cluster_assignment_file,
                    draw_graph=False, graph_path=output_dir,
                    silhouette_score_train_file=silhouette_score_train_file)
    rf_source.fit()
    preds = rf_source.predict(Xt_test)
    split = np.hsplit(preds, 2)
    y_pred_reg_src = split[0].flatten()
    y_pred_class_src = split[1].flatten()


    mcc = matthews_corrcoef(y_pred_class_src, y_class)
    print(f"Source-only MCC: {mcc}")
    mae = mean_absolute_error(y_pred=y_pred_reg_src, y_true=y_reg)
    results_source_mcc.append(mcc)
    results_source_mae.append(mae)
    print(f"Source-only MAE: {mae}")

    # -----------------------------------------
    # TRANSFER LEARNING
    # -----------------------------------------
    rf_tl = MORGOTH(X_train=Xs.loc[:, features], y_train=ys.values, sample_names_train=Xs.index, threshold=[0.9],
                    criterion_class='gini', criterion_reg='mse', min_number_of_samples_per_leaf=10, number_of_trees_in_forest=50, analysis_name=f'target_only_fold_{fold}',
                    number_of_features_per_split='sqrt', class_names=[0,1], output_format='multioutput', time_file=time_file,
                    sample_weights_included='simple', random_state=seed, max_depth=20, impact_classification=1,
                    sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file,
                    tree_weights='wma', silhouette_score_file=silhouette_score_file, distance_measure='', cluster_assignment_file=cluster_assignment_file,
                    draw_graph=False, graph_path=output_dir,
                    silhouette_score_train_file=silhouette_score_train_file,  X_target_train=Xt_train, y_target_train=yt_train, loss_wma_regression='abs',
                    loss_wma_classification='cost_sensitive',
                    beta=0.2, labda_wma=0.5, tree_weight_file=tree_weight_file)

    rf_tl.fit()                # Pretrain
    preds = rf_tl.predict(Xt_test)
    split = np.hsplit(preds, 2)
    y_pred_reg = split[0].flatten()
    y_pred_class = split[1].flatten()


    mcc = matthews_corrcoef(y_pred_class, y_class)
    print(f"Transfer MCC: {mcc}")
    mae = mean_absolute_error(y_pred=y_pred_reg, y_true=y_reg)
    results_transfer_mcc.append(mcc)
    results_transfer_mae.append(mae)
    print(f"Transfer MAE: {mae}")


    '''mcc = matthews_corrcoef(y_pred_class, y_pred_class_src)
    print(f"Transfer MCC vs Source: {mcc}")
    mae = mean_absolute_error(y_pred=y_pred_reg, y_true=y_pred_reg_src)
    print(f"Transfer MAE vs Source: {mae}")'''

print("\n====================== FINAL RESULTS ======================")
print("Target-only MCC: mean=%.4f" % np.mean(results_target_mcc))
print("Source-only MCC: mean=%.4f" % np.mean(results_source_mcc))
print("Transfer MCC:    mean=%.4f" % np.mean(results_transfer_mcc))
print("Target-only MAE: mean=%.4f" % np.mean(results_target_mae))
print("Source-only MAE: mean=%.4f" % np.mean(results_source_mae))
print("Transfer MAE:    mean=%.4f" % np.mean(results_transfer_mae))