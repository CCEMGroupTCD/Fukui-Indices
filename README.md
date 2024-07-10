# Prediction of selectivity and Fukui indices

This is the repository for the paper XXXXXXXXXXX.

It is used for predicting which atomic site in an organic molecule is the one with the highest activity. 

## Machine Learning
We trained machine learning classifiers on predicting the most active site. To check which active site is predicted, we used the predict_proba() method of the classifier to output the probability, instead of just the predicted class. Then, for all atomic sites in a molecule, the site with the highest probability was selected as the predicted active site. In order to get a fair estimate of the model comparison, we also used a leave-one-molecule-out cross-validation.

### Scripts
There are two main scripts in this repository:
- ``src/get_features_from_smiles.py``: Generates SOAP features from SMILES strings and optionally from DFT structures as well. The input is a csv file with SMILES strings, the output is a csv file in which additionally to all original columns the SOAP features and PCA-compressed SOAP features are added.
- ``src/run_simple_ML.py``: Runs machine learning models on the csv file with SOAP features. The options are explained in the script.

### Data
Data is contained in the ``data`` directory. This directory contains three subdirectories:
- ``generate_features``: Contains the original csv file with SMILES string and the generated csv file with SOAP features.
- ``test_new_molecules``: Contains the same csv files, but with additional 4 molecules that were tested at a later stage of the project.
- ``ml_results``: Contains the results of the machine learning models. The file ``ml_results.xlsx`` in the main directory contains an overview of these machine learning experiments.

## Conda environment
The exact conda environment used for the results in the paper can be found in the ``conda_env.yml`` file. To create the environment, run the following command:
```
conda env create -f conda_env.yml
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 