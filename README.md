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

An exemplary use is running our provided main as a module, which you can call after downloading the `Example_Data` folder from our GitHub.

```
python3 -m morgoth Example_Data/example_Json_config.json
```
Note that the directory tree should be kept and the path to the output folder should be edited in the file `Example_Data/example_JSON_config.json`. The prediction results for classification will be found in ```<output_dir><analysis_name>_ClassificationResultsFile1.txt ``` and the regression results are stored in ```<output_dir><analysis_name>_<1-error_rate>_RegressionResultsFile1.txt```. If if the field swap_test_calibration in the config file is set to 'True' there will be one additional file per task, respectively, where the '1' in the file name is replaced by a '2'. If a distance measure is given in the config, ```<output_dir><analysis_name>_SilhouetteScoresTrainSamples_<distance>.txt``` and ```<output_dir><analysis_name>_SilhouetteScoresTestSamples_<distance>.txt``` will contain the silhouette scores for the training and test samples, respectively. If draw_graph is set to True, the files ```<output_dir>/<analysis_name>_<sample_name>.dot``` contain the sample specific graphs and ```<output_dir><analysis_name>__graph_whole_forest.dot``` and  ```<output_dir><analysis_name>__graph_average_whole_forest.dot``` contain the graph for the whole test set with either the raw count across all samples as edge weight or averaged by the number of test samples, respectively.
