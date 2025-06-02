import sys
from morgoth.multi_rf_main import main
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("This program needs the following arguments:\
                    \n- Config json file with information about \
                    \n a) number of trees (number_of_trees)\
                    \n b) number of samples per leaf (samples_per_leaf)\
                    \n c) number of features per split (number_of_features_per_split)\
                    \n d) analysis mode for HARF (analysis_mode, accepted: binary_no_weights, binary_weights, majority_weights, binary_no_weights_sensitive)\
                    \n e) output directory for results (output_dir)\
                    \n f) Should MSE and PCC be included (mse_included, accepted: true, false)\
                    \n g) Should classification errors be included (classification_errors_included, accepted: true, false)\
                    \n h) Training matrix cell line file name (tr_matrix_file)\
                    \n i) Test matrix cell line file name (te_matrix_file)\
                    \n j) Gene Expression matrix (ge_matrix_file)\
                    \n k) Classification info file (cl_matrix_file)\
                    \n l) File with gene names that should be used for analysis (wanted_genes)\
                    \n m) Number of wanted genes from the sorted list (nr_of_w_genes)\
                    \n n) Name for the analysis (analysis_name, e.g. Fold0, CompleteTraining, etc.)\
                    \n o) Threshold(s) for calculating weights (threshold). Can be used with sample weights.\
                    \n p) Should training data be upsampled for minority class (upsample, accepted: simple, linear). Cannot be combined with sample_weights\
                    \n q) Should sample weights be used for fitting trees? (sample_weights, accepted: simple, no)\
                    \n r) Information whether classification or regression trees should be fit (regression_classification, accepted: regression, classification not yet implemented)\
                    \n s) quantile (a number between 0 and 1, can also be an empty string, then usual RF is fitted)\
                    \n t) score classification (summation, mondrian, or true_class)")
    print(sys.argv[1])
    main(sys.argv[1])
