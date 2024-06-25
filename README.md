# MyFirstML
This repo is intended as a first starting point for beginners in machine learning. It provides essential scripts with the basic components in a machine learning pipeline. For simplicity, it uses sklearn models on tabular data, but can of course be generalized. Many comments are included in the code to help understanding what is going on and how different machine learning models can be used.

## Installation:
### Clone the repo:
```
git clone https://github.com/TimoSommer/MyFirstML
```

### Install the conda environment:
```
cd MyFirstML
conda env create -f MyFirstML.yaml
conda activate MyFirstML
```  

### Use as PyCharm project:
Open the project in PyCharm and set the interpreter to the conda environment you just created. As of now, this is important to be able to import the MyFirstML package. Therefore, simply running `python ...` would not work at the moment. The package must be imported as PyCharm project.

## How to use
The main machine learning script is located at MyFirstML/machine_learning/run_simple_ML.py. Try to execute it and see if everything runs smoothly. Then, use this script as starting point for all your machine learning.

### Input and output data
Data is inputted in form of a .csv file and put into a pandas DataFrame(). This dataframe contains all the per-sample information, such as features, targets, predictions,  if a data point belongs to the train or test split etc. This makes it easy to add more columns if necessary. Because this script is supposed to be a first starting point for machine learning, tabular data is a good choice. However, the script can be easily generalized to CNNs or GNNs by providing a path to an image or graph and then let this model read in the data from the path.

### Models
As a simple start into machine learning, the script uses sklearn models and XGB in it's sklearn implementation and makes use of their .fit() and .predict() api. It can be easily adapted to models built using tensorflow, jax or pytorch by writing a wrapper providing the .fit() and .predict() api.

### Train and test data splits
In usual ML workflows, there is three types of data: the train set, the validation set and the test set. Models are trained on the train set, hyperparameters are optimized using the validation set and the final model performance is reported for the test set. For the sake of simplicity, MyFirstML restricts itself to train and test data (called 'train' and 'test'): models are trained on the train data and tested on the test data. In usual applications, a user would therefore first split off the real test data, and then use the rest of the data for trying out the machine learning pipeline and optimize hyperparameters. Only at the very end, the user would read in the entire dataset, label the test and train data accordingly and run MyFirstML a final time to get the performance on the test metrics.

### Hyperparameter optimization
MyFirstML is not intended to be a pipeline for optimizing hyperparameters, but to try out models. Therefore, hyperparameter optimization needs to be sourced out into one out of many existing frameworks for this purpose, e.g. SigOpt. The MyFirstML script can then be called from those other frameworks in an outer loop to do the model training and to report the performance on the validation dataset.

### Saving and loading models and data
Data is currently saved by saving the main dataframe (ml.df) and the scores dataframe (ml.df_all_scores) to csv. Models are currently saved using joblib. This means it is very easy to save and load models, but if you change the sklearn version, you might not be able to load the model anymore. If this is an issue, you could either save the model weights and load them into a new model, or use a different saving method.

### Performance metrics
The script is currently set up for regression problems and reports the r^2 value and the Mean Absolute Error (MAE) for test and train set. Usually, you would expect that a model has a slightly lower performance on the test set than the train set. If the difference between the two is very big, this can mean that your model is overfitting. In this case, you can try to play around with the hyperparameters of the model to reduce the overfitting. If the difference is very small, this can mean that your model is underfitting. In this case, you might want to try out a more complex model or add more expressive features to your dataset.  

### Grouping
Grouping of your dataset can make a lot of sense if you expect your dataset to be very clustered. In this case, it might make sense to not have random splis in test and train data, but to train on n-1 groups and to test on the nth group, to get an insight into the performance of the model in predicting completely new data. This is especially important for machine learning in natural sciences, where the aim often is to detect novel materials which are quite dissimilar to existing data points. To support this, you can provide the name of a column in your dataset to the option 'group'. Please make sure that the split into train and test split really works exactly the way you want it, since this is a non-trivial situation.

### Data scaling
Most machine learning models perform best if the data is scaled such that each feature/target has a normal distribution with a mean of 0 and a variance of 1. This is done by the StandardScaler() in sklearn. However, if you have data which varies on a logarithmic scale, it might also make sense to first the logarithm of the data and then to standard scale it. Currently the script just supports providing a single scaler for all features and one for all targets, but you can easily implement it so that you can provide a different scaler for each feature/target. As a note, models like Neural Networks and Gaussian Processes benefit a lot by the correct scaling, while other models like Linear Regression and Random Forest derivatives are insensitive to linear scaling (but not to non-linear scaling).

## Current limitations
- In its current implementation, the script is made for regression problems, but classification problems are easy to add.

## Interesting links
### Python packages
- [scikit-learn](https://scikit-learn.org/stable/): An amazing package for everything around machine learning and machine learning workflows. It is used in this project as well. Very widely used!
- [List of scikit-learn related projects](https://scikit-learn.org/stable/related_projects.html): Very interesting list of python packages working together with scikit-learn.
- [pandas](https://pandas.pydata.org/): A package for everything around tabular data. Widely used!
- [dScribe](https://singroup.github.io/dscribe/latest/): A package for computing features of molecules and crystal structures. Most importantly, contains the widely used SOAP features. 
- [Yellowbrick](https://www.scikit-yb.org/en/latest/): A package for visualizing everything around machine learning, based on scikit-learn.
- [Seaborn](https://seaborn.pydata.org/): A package for plotting data. Uses matplotlib, but provides a lot of high-level functions and makes beautiful plots. Widely used!

### Papers
- [Retrospective on a decade of machine learning for chemical discovery](https://www.nature.com/articles/s41467-020-18556-9)
- [Recent advances and applications of deep learning methods in materials science](https://www.nature.com/articles/s41524-022-00734-6)
- [Machine Learning for Materials Scientists: An Introductory Guide toward Best Practices](https://pubs.acs.org/doi/10.1021/acs.chemmater.0c01907)
- [Materials Data toward Machine Learning: Advances and Challenges](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c00576)
- [Materials Science in the AI age: high-throughput library generation, machine learning and a pathway from correlations to the underpinning physics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7067066/)
- [Methods for comparing uncertainty quantifications for material property predictions](https://iopscience.iop.org/article/10.1088/2632-2153/ab7e1a)
