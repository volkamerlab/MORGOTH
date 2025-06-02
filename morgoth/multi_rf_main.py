import copy
import json
from morgoth.morgoth import MORGOTH
import sys
if sys.platform.startswith('linux'):
    import fireducks.pandas as pd
else:
    import pandas as pd
import numpy as np
from morgoth.conformal_prediction import *


def read_gene_expression_matrix(gene_expression_matrix_file: str) -> pd.DataFrame:
    '''
        reads the gene expression matrix

        @param gene_expression_matrix_file: path to a file with the gene expression matrix
        @return: a pandas data frame containing the gene expression matrix
    '''
    gene_expression_matrix = pd.read_csv(
        gene_expression_matrix_file, sep='\t', index_col=0)
    return gene_expression_matrix


def filter_gene_expression_features(wanted_names: list, gene_expression_matrix: pd.DataFrame) -> list:
    '''
        filters the gene expression matrix st only selected genes are considered as features

        @param wanted_genes: a list with the wanted/selected genes to be considered for this analysis
        @param gene_expression_matrix:  a pandas data frame containing the gene expression matrix
        @return:  a pandas data frame only containing the gene expression for the wanted gene list
    '''
    updated_gene_expression_matrix = copy.deepcopy(
        gene_expression_matrix.loc[:, wanted_names])
    return updated_gene_expression_matrix


def read_wanted_gene_list(wanted_gene_list_file: str, number_of_wanted_genes: int) -> list:
    '''
        reads which genes are wanted

        @param wanted_gene_list_file: path to a file with all genes that should be considered 
            in each line of the file we expect the gene name followed by a tab and its score
        @param number_of_wanted_genes: int, number of genes to consider
        @return: a list with the names of the wanted genes
    '''
    gene_score_list = pd.read_csv(wanted_gene_list_file, names=[
                                  'gene', 'score'], sep='\t')
    gene_score_list.sort_values(by=['score'], ascending=False, inplace=True)
    gene_score_list = gene_score_list.reset_index(drop=True)
    gene_names = gene_score_list.loc[:number_of_wanted_genes - 1, 'gene']
    return gene_names


def read_classification_file(classification_filename: str) -> dict:
    '''
        @param classification_filename: path to a file containing the classification responses, 
            we expect each line to contain the  tab separated cell line name and respective class 
        @return: a pandas data frame with the cell line names and the class 
    '''
    y_disc = pd.read_csv(classification_filename, names=[
                         'sample', 'class'], header=0, sep='\t')
    y_disc.dropna(how='any', inplace=True, axis=0)
    return y_disc


def read_drug_file(drug_filename: str) -> tuple:
    '''
        @param drug_filename: path to a file containing the continuous drug response,
            we expect each line to contain the  tab separated cell line name and sensitivity measure (e.g., cmax, IC50)
        @return sample_list: list of all sample/cell line names
        @return y_train_reg: pd dataframe containing the sample names their continuous response values
    '''
    y_train_reg = pd.read_csv(drug_filename, names=[
                              'sample', 'response'], header=0, sep='\t')
    y_train_reg.dropna(how='any', inplace=True, axis=0)
    sample_list = y_train_reg.loc[:, 'sample'].values.tolist()
    return sample_list, y_train_reg


def split_classification_file(training_samples: list, test_samples: list, calibration_samples: list, classification_matrix: pd.DataFrame) -> list:
    '''
        @param training_samples: list of all sample names that are used for training
        @param test_samples: list of all sample names that are used for testing
        @param calibration_samples: list of all sample names that are used for calibration
        @param classification_matrix: a pandas dataframe with cell line names and respective classes

        @return training_classification_samples: list containing the classes for the samples that are used for training
        @return test_classification_samples: list containing the classes for the samples that are used for testing
        @return calibration_classification_samples: list containing the classes for the samples that are used for calibration
    '''
    training_classification_samples = classification_matrix.loc[
        classification_matrix['sample'].isin(training_samples), :]
    test_classification_samples = classification_matrix.loc[
        classification_matrix['sample'].isin(test_samples), :]
    calibration_classification_samples = classification_matrix.loc[
        classification_matrix['sample'].isin(calibration_samples), :]
    return [training_classification_samples, test_classification_samples, calibration_classification_samples]


def split_gene_expression_matrix(gene_expression_matrix: pd.DataFrame, training_samples: list, calibration_samples: list, test_samples: list):
    '''
        @param gene_expression_matrix: a pandas data frame with the cell line names as index and the gene expression 
        @param training_samples: list of all sample names that are used for training
        @param test_samples: list of all sample names that are used for testing
        @param calibration_samples: list of all sample names that are used for calibration

        @return gene_expression_training_matrix: list containing the gene expression values for the samples that are used for training
        @return gene_expression_test_matrix: list containing the gene expression values for the samples that are used for testing
        @return gene_expression_calibration_matrix: list containing the gene expression values for the samples that are used for calibration
    '''
    gene_expression_training_matrix = gene_expression_matrix.loc[training_samples, :]
    gene_expression_test_matrix = gene_expression_matrix.loc[test_samples, :]
    gene_expression_calibration_matrix = gene_expression_matrix.loc[calibration_samples, :]

    return [gene_expression_training_matrix, gene_expression_test_matrix, gene_expression_calibration_matrix]


def generate_epsilon_list(X_train: pd.DataFrame):
    var = X_train.p_var()
    epsilon_list = var.values
    return epsilon_list


def perform_conformal_prediction(fitted_model: MORGOTH, X_cal: pd.DataFrame, y_cal: np.array, X_test: pd.DataFrame, y_test: np.array,  error_rate: float, results_file_classification: str, results_file_regression: str, score_classification: str, output_format: str, samples_names_calibration: np.array, class_names: np.array, samples_names_test: np.array):
    '''
        performs the conformal prediction for the given error rate
    '''
    if score_classification == 'summation':
        class_function = get_pred_score_summation
        eval_class_function = eval_classification_summation
    elif score_classification == 'true_class':
        class_function = get_pred_score_true_class
        eval_class_function = eval_classification_true_class
    elif score_classification == 'mondrian':
        class_function = get_pred_score_mondrian
        eval_class_function = eval_classification_mondrian
    else:
        print(
            f'score {score_classification} not known. We use summation instead.')
        class_function = get_pred_score_summation
        eval_class_function = eval_classification_summation

    if not output_format == 'multioutput':
        print('no multioutput, we skip conformal prediction')
        return

    q_class, q_reg = conformal_prediction(
        estimator=fitted_model, X_cal=X_cal, y_cal=[y[0] for y in y_cal], y_cal_discretized=[y[1] for y in y_cal], sample_names=samples_names_calibration, score_function=class_function, minimal_certainty=1-error_rate, SAURON=False, num_classes=len(class_names), class_names=class_names, multivariate=True)

    test_set_predictions = fitted_model.predict_proba(
        X_test=X_test, quantile=[error_rate/2, 1-(error_rate/2)])

    plain_regression_test_results = fitted_model.predict(
        X_test=X_test, quantile=None)
    plain_regression_test_results = [y[0]
                                     for y in plain_regression_test_results]

    y_probas = np.array([y[1] for y in test_set_predictions])
    quantile_predictions = np.array([y[0] for y in test_set_predictions])

    minimal_certainty = 1-error_rate
    y_pred_test_reg = eval_regression(
        predictions=quantile_predictions, true_y=[y[0] for y in y_test], sample_names=samples_names_test, minimal_certainty=minimal_certainty, q=q_reg)
    y_pred_test_reg['plain_rf'] = plain_regression_test_results

    y_pred_test_class = eval_class_function(y_pred_proba=y_probas, y_test=[
                                            y[1] for y in y_test], sample_names=samples_names_test, minimal_certainty=minimal_certainty, q=q_class, num_classes=len(class_names), class_names=class_names)

    y_pred_test_class.to_csv(
        results_file_classification, sep='\t', index=False)
    y_pred_test_reg.to_csv(results_file_regression, sep='\t', index=False)

    return y_pred_test_class, y_pred_test_reg


def perform_multi_prediction(json_dict: json.JSONDecoder):
    '''
        performs the multivariate prediction for the settings given in the json config file

        @param json_dict: the config file
    '''
    training_samples_file = json_dict["tr_matrix_file"]
    test_samples_file = json_dict["te_matrix_file"]
    calibration_samples_file = json_dict["cal_matrix_file"]
    classification_matrix_file = json_dict["cl_matrix_file"]
    gene_expression_matrix_file = json_dict["ge_matrix_file"]
    wanted_gene_list_file = json_dict["wanted_genes"]
    nr_of_wanted_genes = int(json_dict["nr_of_w_genes"])
    already_sorted = json_dict['already_sorted'] in ['True', 'true']

    training_samples_and_drug_response_values = read_drug_file(
        training_samples_file)
    test_samples_and_drug_response_values = read_drug_file(test_samples_file)
    calibration_samples_and_drug_response_values = read_drug_file(
        calibration_samples_file)

    classification_dataframe = read_classification_file(
        classification_matrix_file)
    if not already_sorted:
        wanted_genes = read_wanted_gene_list(
            wanted_gene_list_file, nr_of_wanted_genes)
    else:
        with open(wanted_gene_list_file, 'r') as wanted_gene_list:
            wanted_genes = wanted_gene_list.readlines()
            for i, sample in enumerate(wanted_genes):
                wanted_genes[i] = sample.strip()
            wanted_genes = wanted_genes[:nr_of_wanted_genes+1]

    samples_names_train = training_samples_and_drug_response_values[0]
    samples_names_test = test_samples_and_drug_response_values[0]
    samples_names_calibration = calibration_samples_and_drug_response_values[0]

    classifications_training_testing_calibration = split_classification_file(
        training_samples=samples_names_train, test_samples=samples_names_test, calibration_samples=samples_names_calibration, classification_matrix=classification_dataframe)

    gene_expression_matrix_data_frame = read_gene_expression_matrix(
        gene_expression_matrix_file)
    wanted_feature_names_and_filtered_gene_expression_matrix_dict = filter_gene_expression_features(
        wanted_genes, gene_expression_matrix_data_frame)
    filtered_gene_expression_matrix_dict = wanted_feature_names_and_filtered_gene_expression_matrix_dict
    gene_expression_training_testing_calibration = split_gene_expression_matrix(
        gene_expression_matrix=filtered_gene_expression_matrix_dict, training_samples=samples_names_train, test_samples=samples_names_test, calibration_samples=samples_names_calibration)

    X_train = gene_expression_training_testing_calibration[0]
    X_test = gene_expression_training_testing_calibration[1]
    X_cal = gene_expression_training_testing_calibration[2]

    y_train_reg = training_samples_and_drug_response_values[1]
    y_disc_train = classifications_training_testing_calibration[0]

    y_train = np.array([[y_train_reg.loc[y_train_reg['sample'] == sample, 'response'].values[0],
                         int(y_disc_train.loc[y_disc_train['sample'] == sample, 'class'].values[0])] for sample in samples_names_train])

    X_train = X_train.reset_index(drop=True)

    y_test_reg = test_samples_and_drug_response_values[1]
    y_disc_test = classifications_training_testing_calibration[1]
    y_test = np.array([[y_test_reg.loc[y_test_reg['sample'] == sample, 'response'].values[0], int(
        y_disc_test.loc[y_disc_test['sample'] == sample, 'class'].values[0])] for sample in samples_names_test])

    y_cal_reg = calibration_samples_and_drug_response_values[1]
    y_disc_cal = classifications_training_testing_calibration[2]
    y_cal = np.array([[y_cal_reg.loc[y_cal_reg['sample'] == sample, 'response'].values[0], int(
        y_disc_cal.loc[y_disc_cal['sample'] == sample, 'class'].values[0])] for sample in samples_names_calibration])

    classes = np.unique(y_disc_train.loc[:, 'class'])
    classes = np.sort(classes)

    min_number_of_samples_per_leaf = int(json_dict["samples_per_leaf"])
    number_of_features_per_split = json_dict["number_of_features_per_split"]

    if not number_of_features_per_split in ['sqrt', 'log2']:
        number_of_features_per_split = float(number_of_features_per_split)

    name_of_analysis = json_dict['analysis_name']
    number_of_trees_in_forest = int(json_dict["number_of_trees"])

    output_directory = json_dict["output_dir"]

    threshold_g = json_dict["threshold"]
    threshold = 0.0
    if threshold_g != "":

        threshold = [float(t) for t in threshold_g.strip().split(',')]
    else:
        threshold = [float("NaN")]

    sample_weights_included = json_dict["sample_weights"]
    distance_measure = json_dict['distance_measure']
    leaf_assignment_file_train = output_directory + \
        name_of_analysis + "_Training_Set_LeafAssignment.txt"
    sample_info_file = output_directory + name_of_analysis + \
        "_Additional_Sample_Information.txt"
    feature_imp_output_file = output_directory + \
        name_of_analysis + "_Feature_Importance.txt"
    time_file = output_directory + name_of_analysis + "_ElapsedTimeFitting.txt"
    cluster_assignment_file = output_directory + \
        name_of_analysis + "_ClusterAssignment.txt"
    silhouette_score_file = output_directory + name_of_analysis + \
        f'_SilhouetteScores_{distance_measure}.txt'
    silhouette_score_train_file = output_directory + name_of_analysis + \
        f'_SilhouetteScoresTrainSamples_{distance_measure}.txt'
    results_file_class1 = output_directory + \
        name_of_analysis + "_ClassificationResultsFile1.txt"
    results_file_class2 = output_directory + \
        name_of_analysis + "_ClassificationResultsFile2.txt"
    results_file_reg1 = output_directory + \
        name_of_analysis + "_RegressionResultsFile1.txt"
    results_file_reg2 = output_directory + \
        name_of_analysis + "_RegressionResultsFile2.txt"

    criterion_class = json_dict['criterion_class']
    criterion_reg = json_dict['criterion_reg']
    max_depth = int(json_dict['max_depth'])
    impact_classification = float(json_dict['impact_classification'])
    output_format = json_dict['output_format']
    tree_weight = json_dict['tree_weight']
    draw_graph = json_dict['draw_graph']
    if tree_weight in ['true', 'True']:
        tree_weight = True
    else:
        tree_weight = False
    if draw_graph in ['true', 'True']:
        draw_graph = True
    else:
        draw_graph = False

    mult_rf = MORGOTH(X_train=X_train, y_train=y_train, criterion_class=criterion_class, criterion_reg=criterion_reg, sample_names_train=samples_names_train, min_number_of_samples_per_leaf=min_number_of_samples_per_leaf, number_of_trees_in_forest=number_of_trees_in_forest, analysis_name=name_of_analysis,
                      number_of_features_per_split=number_of_features_per_split, class_names=classes, output_format=output_format, threshold=threshold, time_file=time_file, sample_weights_included=sample_weights_included, random_state=42, max_depth=max_depth, impact_classification=impact_classification,
                      sample_info_file=sample_info_file, leaf_assignment_file_train=leaf_assignment_file_train, feature_imp_output_file=feature_imp_output_file, tree_weights=tree_weight, silhouette_score_file=silhouette_score_file, distance_measure=distance_measure, cluster_assignment_file=cluster_assignment_file,
                      draw_graph=draw_graph, graph_path=f'{output_directory}/{name_of_analysis}_', silhouette_score_train_file=silhouette_score_train_file)
    print('fitting forest')
    mult_rf.fit()

    if json_dict['conformal_prediction'] in ['true', 'True']:
        error_rate = float(json_dict['error_rate'])
        perform_conformal_prediction(fitted_model=mult_rf, X_cal=X_cal, y_cal=y_cal, error_rate=error_rate, results_file_classification=results_file_class1, results_file_regression=results_file_reg1,
                                     score_classification=json_dict['score_classification'], output_format=output_format, class_names=classes, samples_names_calibration=samples_names_calibration, samples_names_test=samples_names_test, X_test=X_test, y_test=y_test)
        if json_dict['swap_test_calibration'] in ['true', 'True']:
            perform_conformal_prediction(fitted_model=mult_rf, X_cal=X_test, y_cal=y_test, error_rate=error_rate, results_file_classification=results_file_class2, results_file_regression=results_file_reg2,
                                         score_classification=json_dict['score_classification'], output_format=output_format, class_names=classes, samples_names_calibration=samples_names_test, samples_names_test=samples_names_calibration, X_test=X_cal, y_test=y_cal)
    else:
        pred1 = mult_rf.predict(X_test=X_test)
        pred2 = None
        if json_dict['swap_test_calibration'] in ['true', 'True']:
            pred2 = mult_rf.predict(X_test=X_cal)
        with open(results_file_class1, 'w') as output:
            output.write('sample\tactual\tpredicted\n')
            for i, p in enumerate(pred1):
                output.write(
                    f'{test_samples_and_drug_response_values[0][i]}\t{y_test[i][1]}\t{p[1]}\n')

        if not pred2 is None:
            with open(results_file_class2, 'w') as output:
                output.write('sample\tactual\tpredicted\n')
                for i, p in enumerate(pred2):
                    output.write(
                        f'{calibration_samples_and_drug_response_values[0][i]}\t{y_cal[i][1]}\t{p[1]}\n')

        with open(results_file_reg1, 'w') as output:
            output.write('sample\tactual\tplain_rf\n')
            for i, p in enumerate(pred1):
                output.write(
                    f'{test_samples_and_drug_response_values[0][i]}\t{y_test[i][0]}\t{p[0]}\n')

        if not pred2 is None:
            with open(results_file_reg2, 'w') as output:
                output.write('sample\tactual\tplain_rf\n')
                for i, p in enumerate(pred2):
                    output.write(
                        f'{calibration_samples_and_drug_response_values[0][i]}\t{y_cal[i][0]}\t{p[0]}\n')


################ Main function ##################################


def main(config_filename):

    # print(config_filename)
    json_file = open(config_filename)
    data = json.load(json_file)
    print(data)
    perform_multi_prediction(data)
